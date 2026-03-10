"""
Data utilities for the DeepSMOTE replication experiment.

Handles:
  - MNIST loading and normalisation to [-1, 1]   (matches decoder Tanh output)
  - Creating the artificially imbalanced training set (paper §V-A-2)
  - Stratified 5-fold CV splits                  (paper §V-A-6)
  - Torch DataLoader creation
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple

from config import DataConfig


class MNISTDataManager:
    """Loads MNIST, applies imbalance, and provides CV-ready data splits."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self._X_all: np.ndarray | None = None  # (N, 1, 28, 28), float32 in [-1,1]
        self._y_all: np.ndarray | None = None  # (N,), int64

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and load the full MNIST training split."""
        transform = transforms.Compose([
            transforms.ToTensor(),                          # → [0, 1]
            transforms.Normalize((0.5,), (0.5,)),           # → [-1, 1]
        ])
        mnist = datasets.MNIST(
            root=self.cfg.data_dir, train=True, download=True, transform=transform
        )
        loader = DataLoader(mnist, batch_size=512, shuffle=False, num_workers=0)

        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy())
            ys.append(y.numpy())
        self._X_all = np.concatenate(xs, axis=0)   # (60000, 1, 28, 28)
        self._y_all = np.concatenate(ys, axis=0)   # (60000,)

    def get_imbalanced_dataset(
        self, seed: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample MNIST to the configured imbalance distribution.

        Returns:
            X: (N_total, 1, 28, 28)  float32 in [-1, 1]
            y: (N_total,)            int64
        """
        if self._X_all is None:
            self.load()

        rng = np.random.default_rng(seed if seed is not None else self.cfg.random_seed)
        counts = self.cfg.imbalance_counts
        assert len(counts) == self.cfg.n_classes, "One count per class required."

        xs, ys = [], []
        for cls_idx, n_samples in enumerate(counts):
            cls_mask = self._y_all == cls_idx
            cls_X = self._X_all[cls_mask]
            cls_y = self._y_all[cls_mask]

            if len(cls_X) < n_samples:
                raise ValueError(
                    f"Class {cls_idx} has only {len(cls_X)} samples; "
                    f"requested {n_samples}."
                )
            chosen = rng.choice(len(cls_X), size=n_samples, replace=False)
            xs.append(cls_X[chosen])
            ys.append(cls_y[chosen])

        X = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        return X, y

    def get_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return list of (train_indices, test_indices) for stratified k-fold CV.

        Stratified split preserves the imbalance ratio in every fold.
        """
        skf = StratifiedKFold(
            n_splits=self.cfg.n_folds,
            shuffle=True,
            random_state=self.cfg.random_seed,
        )
        return list(skf.split(X, y))

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_loader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Wrap numpy arrays in a DataLoader."""
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    @staticmethod
    def data_by_class(
        X: np.ndarray, y: np.ndarray
    ) -> dict:
        """
        Return {class_idx: X_cls_tensor} for all classes.
        Used during the penalty-loss phase to sample same-class batches.
        """
        classes = np.unique(y)
        return {
            int(c): torch.from_numpy(X[y == c])
            for c in classes
        }

    @staticmethod
    def majority_class_count(y: np.ndarray) -> int:
        """Return the count of the most frequent class."""
        return int(np.bincount(y).max())

    @staticmethod
    def class_counts(y: np.ndarray) -> np.ndarray:
        """Return per-class sample counts."""
        return np.bincount(y)
