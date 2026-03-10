"""
DeepSMOTE: encoder/decoder training + SMOTE-in-embedding-space generation.

Implements Algorithm 1 from the paper exactly:
  Train phase : reconstruction loss + permutation penalty loss
  Generate phase: encode minority → SMOTE embeddings → decode → synthetic images
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from typing import Tuple

from config import DeepSMOTEConfig
from architecture import Encoder, Decoder


class DeepSMOTE:
    """
    Wraps the encoder/decoder training and synthetic image generation.

    Usage:
        ds = DeepSMOTE(cfg, device)
        ds.train(X_train, y_train)
        X_balanced, y_balanced = ds.generate_balanced(X_train, y_train)
    """

    def __init__(self, cfg: DeepSMOTEConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device

        self.encoder = Encoder(cfg.enc_dec).to(device)
        self.decoder = Decoder(cfg.enc_dec).to(device)

        self._opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=cfg.learning_rate,
            betas=cfg.adam_betas,
        )

    # ------------------------------------------------------------------
    # Training (Algorithm 1 — Train phase)
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> list:
        """
        Train encoder/decoder with reconstruction + permutation penalty loss.

        Args:
            X_train : (N, 1, 28, 28)  float32 in [-1, 1]
            y_train : (N,)            int64
            verbose : show tqdm progress bar

        Returns:
            loss_history: list of (total_loss, rl, pl) per epoch
        """
        self.encoder.train()
        self.decoder.train()

        data_by_class = self._build_class_dict(X_train, y_train)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )

        loss_history = []
        epochs = tqdm(range(self.cfg.n_epochs), desc="DeepSMOTE enc/dec",
                      leave=False, disable=not verbose)

        for epoch in epochs:
            ep_tl = ep_rl = ep_pl = 0.0
            n_batches = 0

            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                tl, rl, pl = self._train_step(x_batch, data_by_class)
                ep_tl += tl; ep_rl += rl; ep_pl += pl
                n_batches += 1

            ep_tl /= n_batches; ep_rl /= n_batches; ep_pl /= n_batches
            loss_history.append((ep_tl, ep_rl, ep_pl))
            epochs.set_postfix(loss=f"{ep_tl:.4f}", rl=f"{ep_rl:.4f}",
                               pl=f"{ep_pl:.4f}")

        return loss_history

    def _train_step(
        self,
        x_batch: torch.Tensor,
        data_by_class: dict,
    ) -> Tuple[float, float, float]:
        self._opt.zero_grad()
        n = x_batch.size(0)

        # ── Reconstruction loss ──────────────────────────────────────
        z      = self.encoder(x_batch)
        x_recon = self.decoder(z)
        rl     = F.mse_loss(x_recon, x_batch)

        # ── Penalty loss (Algorithm 1, lines 9-14) ──────────────────
        # 1. Randomly select a class
        cls = random.choice(list(data_by_class.keys()))
        cls_imgs = data_by_class[cls]       # (N_cls, 1, 28, 28) tensor

        # 2. Sample |batch| images from that class
        n_cls = cls_imgs.size(0)
        idx = torch.randint(0, n_cls, (n,))
        cls_batch = cls_imgs[idx].to(self.device)

        # 3. Encode
        z_cls = self.encoder(cls_batch)

        # 4. Permute encoded order
        perm     = torch.randperm(n, device=self.device)
        z_perm   = z_cls[perm]

        # 5. Decode permuted encodings
        x_perm_recon = self.decoder(z_perm)

        # 6. Compare decoded-permuted to ORIGINAL class images (CDi in paper)
        pl = F.mse_loss(x_perm_recon, cls_batch)

        tl = rl + pl
        tl.backward()
        self._opt.step()

        return tl.item(), rl.item(), pl.item()

    # ------------------------------------------------------------------
    # Generation (Algorithm 1 — Generate phase)
    # ------------------------------------------------------------------

    def generate_balanced(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic minority-class images via SMOTE in embedding space.

        Steps:
          1. Encode all training images → Z_train
          2. For each minority class: apply SMOTE in Z_train space
          3. Decode synthetic embeddings → synthetic images
          4. Concatenate with originals

        Returns:
            X_balanced : (N_balanced, 1, 28, 28)  float32
            y_balanced : (N_balanced,)              int64
        """
        self.encoder.eval()
        self.decoder.eval()

        # 1. Encode all training images
        Z_train = self._encode_all(X_train)

        # 2. SMOTE in embedding space
        target_n = int(np.bincount(y_train).max())
        Z_syn, y_syn = self._smote_embeddings(Z_train, y_train, target_n)

        if len(Z_syn) == 0:
            return X_train.copy(), y_train.copy()

        # 3. Decode synthetic embeddings
        X_syn = self._decode_embeddings(Z_syn)

        # 4. Concatenate
        X_bal = np.concatenate([X_train, X_syn], axis=0)
        y_bal = np.concatenate([y_train, y_syn], axis=0)

        return X_bal.astype(np.float32), y_bal.astype(np.int64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_all(self, X: np.ndarray) -> np.ndarray:
        """Encode all images in batches; returns numpy (N, latent_dim)."""
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X)),
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        zs = []
        with torch.no_grad():
            for (x_batch,) in loader:
                z = self.encoder(x_batch.to(self.device))
                zs.append(z.cpu().numpy())
        return np.concatenate(zs, axis=0)

    def _decode_embeddings(self, Z: np.ndarray) -> np.ndarray:
        """Decode embedding array; returns numpy (N, 1, 28, 28)."""
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Z.astype(np.float32))),
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        imgs = []
        with torch.no_grad():
            for (z_batch,) in loader:
                x = self.decoder(z_batch.to(self.device))
                imgs.append(x.cpu().numpy())
        return np.concatenate(imgs, axis=0)

    def _smote_embeddings(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        target_n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE in the latent embedding space (paper §IV-E).

        For each minority class:
          - Find k nearest neighbours in Z-space
          - Generate synthetic z = z_i + λ * (z_nn - z_i),  λ ~ U[0,1]

        Returns synthetic embeddings and labels only (originals excluded).
        """
        k  = self.cfg.smote_k_neighbors
        Z_syn_list, y_syn_list = [], []

        for cls in np.unique(y):
            mask  = y == cls
            Z_cls = Z[mask]
            n_cls = len(Z_cls)
            n_needed = target_n - n_cls

            if n_needed <= 0:
                continue

            # Ensure k does not exceed available neighbours
            k_actual = min(k, n_cls - 1)
            if k_actual < 1:
                # Tiny class: duplicate with small Gaussian perturbation
                noise = np.random.randn(n_needed, Z_cls.shape[1]) * 1e-3
                Z_syn_list.append(
                    Z_cls[np.random.randint(0, n_cls, n_needed)] + noise
                )
                y_syn_list.append(np.full(n_needed, cls, dtype=np.int64))
                continue

            nbrs = NearestNeighbors(n_neighbors=k_actual + 1,
                                    algorithm="auto").fit(Z_cls)
            _, indices = nbrs.kneighbors(Z_cls)  # (n_cls, k_actual+1); col 0 = self

            synthetic = np.empty((n_needed, Z_cls.shape[1]), dtype=np.float32)
            for i in range(n_needed):
                base_idx = np.random.randint(0, n_cls)
                nn_col   = np.random.randint(1, k_actual + 1)
                nn_idx   = indices[base_idx, nn_col]
                lam      = np.random.random()
                synthetic[i] = (Z_cls[base_idx]
                                + lam * (Z_cls[nn_idx] - Z_cls[base_idx]))

            Z_syn_list.append(synthetic)
            y_syn_list.append(np.full(n_needed, cls, dtype=np.int64))

        if not Z_syn_list:
            return np.empty((0, Z.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)

        return (
            np.concatenate(Z_syn_list, axis=0),
            np.concatenate(y_syn_list, axis=0),
        )

    @staticmethod
    def _build_class_dict(X: np.ndarray, y: np.ndarray) -> dict:
        """Return {class_idx: tensor_of_images} for penalty-loss sampling."""
        return {
            int(c): torch.from_numpy(X[y == c])
            for c in np.unique(y)
        }

    def reset(self) -> None:
        """Re-initialise weights (call before each CV fold)."""
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

        self.encoder.apply(_init)
        self.decoder.apply(_init)
        self._opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.cfg.learning_rate,
            betas=self.cfg.adam_betas,
        )
