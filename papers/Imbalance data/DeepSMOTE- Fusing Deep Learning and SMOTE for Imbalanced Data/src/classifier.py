"""
ResNet-18 classifier trainer and evaluator (paper §V-A-4).

The same ResNet-18 is used as the downstream classifier for ALL resampling
methods to ensure a fair comparison — only the training data differs.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from config import ClassifierConfig
from architecture import get_resnet18


class ClassifierTrainer:
    """Train and evaluate ResNet-18 on a (potentially balanced) dataset."""

    def __init__(self, cfg: ClassifierConfig, n_classes: int,
                 device: torch.device):
        self.cfg       = cfg
        self.n_classes = n_classes
        self.device    = device

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = False,
    ) -> nn.Module:
        """
        Train ResNet-18 from scratch on the given (balanced) dataset.

        Args:
            X_train : (N, 1, 28, 28)  float32 in [-1, 1]
            y_train : (N,)            int64

        Returns:
            Trained nn.Module (moved to device, in eval mode after training).
        """
        model = get_resnet18(n_classes=self.n_classes, in_channels=1).to(self.device)
        opt   = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )

        model.train()
        epochs = tqdm(range(self.cfg.n_epochs), desc="  ResNet-18",
                      leave=False, disable=not verbose)

        for _ in epochs:
            ep_loss = 0.0; n_batches = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                opt.zero_grad()
                logits = model(x_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                opt.step()
                ep_loss += loss.item(); n_batches += 1
            epochs.set_postfix(loss=f"{ep_loss/n_batches:.4f}")

        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Return class predictions for the test set.

        Args:
            model  : trained ResNet-18
            X_test : (N, 1, 28, 28) float32

        Returns:
            preds : (N,) int64 predicted labels
        """
        model.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test)),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        all_preds = []
        for (x_batch,) in loader:
            logits = model(x_batch.to(self.device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)
