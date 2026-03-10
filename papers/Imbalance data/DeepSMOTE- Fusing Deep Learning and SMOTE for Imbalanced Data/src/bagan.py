"""
BAGAN: Balancing GAN (Mariani et al. 2018) — used as the GAN-based baseline.

Key idea (matching paper §V-A-3):
  Phase 1 — Train class-conditional autoencoder on all (imbalanced) training data.
  Phase 2 — Initialise GAN generator from the AE decoder; train conditional GAN.
  Phase 3 — Generate synthetic minority-class images to balance the training set.

Class conditioning: one-hot label is concatenated with z (generator)
                    and broadcast as extra image channels (discriminator).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from typing import Tuple

from config import BAGANConfig
from architecture import Encoder, Decoder, BAGANGenerator, BAGANDiscriminator


class BAGAN:
    """
    Wraps autoencoder pre-training, conditional GAN training, and generation.

    Usage:
        bagan = BAGAN(cfg, n_classes, device)
        bagan.train(X_train, y_train)
        X_balanced, y_balanced = bagan.generate_balanced(X_train, y_train)
    """

    def __init__(self, cfg: BAGANConfig, n_classes: int, device: torch.device):
        self.cfg       = cfg
        self.n_classes = n_classes
        self.device    = device

        # Autoencoder (phase 1)
        self.ae_enc = Encoder(cfg.enc_dec).to(device)
        self.ae_dec = Decoder(cfg.enc_dec).to(device)

        # Generator & Discriminator (phase 2, init from AE)
        self.G = BAGANGenerator(cfg.enc_dec, n_classes=n_classes).to(device)
        self.D = BAGANDiscriminator(cfg.enc_dec, n_classes=n_classes).to(device)

        self._ae_opt = torch.optim.Adam(
            list(self.ae_enc.parameters()) + list(self.ae_dec.parameters()),
            lr=cfg.learning_rate_g, betas=cfg.adam_betas,
        )
        self._g_opt = torch.optim.Adam(
            self.G.parameters(), lr=cfg.learning_rate_g, betas=cfg.adam_betas
        )
        self._d_opt = torch.optim.Adam(
            self.D.parameters(), lr=cfg.learning_rate_d, betas=cfg.adam_betas
        )
        self._bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              verbose: bool = True) -> dict:
        """
        Full BAGAN training: AE pre-train → copy weights → GAN train.

        Returns dict with 'ae_loss', 'g_loss', 'd_loss' histories.
        """
        ae_hist = self._train_autoencoder(X_train, y_train, verbose)
        self._init_generator_from_ae()
        gan_hist = self._train_gan(X_train, y_train, verbose)
        return {"ae_loss": ae_hist, "g_loss": gan_hist["g"], "d_loss": gan_hist["d"]}

    def generate_balanced(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic images for minority classes and return balanced dataset.
        """
        self.G.eval()
        counts   = np.bincount(y_train)
        target_n = int(counts.max())

        X_syn_list, y_syn_list = [], []

        with torch.no_grad():
            for cls in range(self.n_classes):
                n_needed = target_n - int(counts[cls])
                if n_needed <= 0:
                    continue

                z = torch.randn(n_needed, self.cfg.enc_dec.latent_dim,
                                device=self.device)
                one_hot = self._one_hot(
                    torch.full((n_needed,), cls, dtype=torch.long,
                               device=self.device)
                )
                imgs = self.G(z, one_hot).cpu().numpy()
                X_syn_list.append(imgs)
                y_syn_list.append(
                    np.full(n_needed, cls, dtype=np.int64)
                )

        if not X_syn_list:
            return X_train.copy(), y_train.copy()

        X_syn = np.concatenate(X_syn_list, axis=0).astype(np.float32)
        y_syn = np.concatenate(y_syn_list, axis=0).astype(np.int64)
        X_bal = np.concatenate([X_train, X_syn], axis=0)
        y_bal = np.concatenate([y_train, y_syn], axis=0)
        return X_bal, y_bal

    def reset(self) -> None:
        """Re-initialise all weights (call before each CV fold)."""
        def _init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for net in [self.ae_enc, self.ae_dec, self.G, self.D]:
            net.apply(_init)

        self._ae_opt = torch.optim.Adam(
            list(self.ae_enc.parameters()) + list(self.ae_dec.parameters()),
            lr=self.cfg.learning_rate_g, betas=self.cfg.adam_betas,
        )
        self._g_opt = torch.optim.Adam(
            self.G.parameters(), lr=self.cfg.learning_rate_g,
            betas=self.cfg.adam_betas
        )
        self._d_opt = torch.optim.Adam(
            self.D.parameters(), lr=self.cfg.learning_rate_d,
            betas=self.cfg.adam_betas
        )

    # ------------------------------------------------------------------
    # Phase 1: Autoencoder pre-training
    # ------------------------------------------------------------------

    def _train_autoencoder(
        self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool
    ) -> list:
        self.ae_enc.train(); self.ae_dec.train()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train)),
            batch_size=self.cfg.batch_size, shuffle=True, num_workers=0,
        )

        history = []
        epochs = tqdm(range(self.cfg.n_epochs_autoencoder),
                      desc="BAGAN AE pre-train", leave=False, disable=not verbose)
        for _ in epochs:
            ep_loss = 0.0; n = 0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                self._ae_opt.zero_grad()
                z     = self.ae_enc(x_batch)
                x_rec = self.ae_dec(z)
                loss  = F.mse_loss(x_rec, x_batch)
                loss.backward()
                self._ae_opt.step()
                ep_loss += loss.item(); n += 1
            ep_loss /= n
            history.append(ep_loss)
            epochs.set_postfix(loss=f"{ep_loss:.4f}")

        return history

    # ------------------------------------------------------------------
    # Phase 2: Initialise generator from AE decoder
    # ------------------------------------------------------------------

    def _init_generator_from_ae(self) -> None:
        """Copy AE decoder conv-transpose weights into G."""
        self.G.init_from_decoder(self.ae_dec)

    # ------------------------------------------------------------------
    # Phase 3: Conditional GAN training
    # ------------------------------------------------------------------

    def _train_gan(
        self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool
    ) -> dict:
        self.G.train(); self.D.train()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train),
                          torch.from_numpy(y_train)),
            batch_size=self.cfg.batch_size, shuffle=True, num_workers=0,
        )
        latent_dim = self.cfg.enc_dec.latent_dim

        g_hist, d_hist = [], []
        epochs = tqdm(range(self.cfg.n_epochs_gan),
                      desc="BAGAN GAN train", leave=False, disable=not verbose)

        for _ in epochs:
            ep_g = ep_d = 0.0; n = 0

            for x_real, y_real in loader:
                bs        = x_real.size(0)
                x_real    = x_real.to(self.device)
                y_real_oh = self._one_hot(y_real.to(self.device))

                real_label = torch.ones(bs, 1, device=self.device)
                fake_label = torch.zeros(bs, 1, device=self.device)

                # ── Train Discriminator ──────────────────────────────
                self._d_opt.zero_grad()
                d_real = self.D(x_real, y_real_oh)
                d_loss_real = self._bce(d_real, real_label)

                z = torch.randn(bs, latent_dim, device=self.device)
                # Use same class labels as real batch for conditional generation
                x_fake = self.G(z, y_real_oh).detach()
                d_fake = self.D(x_fake, y_real_oh)
                d_loss_fake = self._bce(d_fake, fake_label)

                d_loss = (d_loss_real + d_loss_fake) * 0.5
                d_loss.backward()
                self._d_opt.step()

                # ── Train Generator ──────────────────────────────────
                self._g_opt.zero_grad()
                z        = torch.randn(bs, latent_dim, device=self.device)
                # Sample random minority class for generator training
                rand_cls = torch.randint(0, self.n_classes, (bs,),
                                         device=self.device)
                rand_oh  = self._one_hot(rand_cls)
                x_fake   = self.G(z, rand_oh)
                d_gen    = self.D(x_fake, rand_oh)
                g_loss   = self._bce(d_gen, real_label)   # G wants D to output 1
                g_loss.backward()
                self._g_opt.step()

                ep_d += d_loss.item(); ep_g += g_loss.item(); n += 1

            ep_d /= n; ep_g /= n
            g_hist.append(ep_g); d_hist.append(ep_d)
            epochs.set_postfix(G=f"{ep_g:.4f}", D=f"{ep_d:.4f}")

        return {"g": g_hist, "d": d_hist}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros(labels.size(0), self.n_classes, device=self.device)
        oh.scatter_(1, labels.unsqueeze(1), 1.0)
        return oh
