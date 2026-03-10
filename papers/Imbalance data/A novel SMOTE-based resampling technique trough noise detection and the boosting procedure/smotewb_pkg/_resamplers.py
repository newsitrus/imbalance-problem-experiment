"""Resampling methods — internal module."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from ._core import smotewb_resample


def _ros(X, y, random_state=None):
    """Random Oversampling: duplicate minority instances until balanced."""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    n_to_gen = int(np.sum(y == 0)) - len(pos_idx)
    if n_to_gen <= 0:
        return X.copy(), y.copy()
    chosen = rng.choice(pos_idx, size=n_to_gen, replace=True)
    return np.vstack([X, X[chosen]]), np.concatenate([y, np.ones(n_to_gen, dtype=int)])


def _smote(X, y, k=5, random_state=None):
    """SMOTE with k=5 nearest minority neighbors (paper default)."""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    n_pos = len(pos_idx)
    n_to_gen = int(np.sum(y == 0)) - n_pos
    if n_to_gen <= 0 or n_pos < 2:
        return X.copy(), y.copy()

    X_pos = X[pos_idx]
    k_actual = min(k, n_pos - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X_pos)
    _, indices = nn.kneighbors(X_pos)
    neighbor_indices = indices[:, 1:]  # exclude self

    synthetics = []
    for _ in range(n_to_gen):
        i = rng.randint(n_pos)
        j = rng.randint(k_actual)
        lam = rng.rand()
        synthetics.append(X_pos[i] + lam * (X_pos[neighbor_indices[i, j]] - X_pos[i]))

    X_syn = np.array(synthetics)
    return np.vstack([X, X_syn]), np.concatenate([y, np.ones(n_to_gen, dtype=int)])


def resample(X, y, method='smotewb', random_state=None):
    """
    Apply a resampling method to training data (minority class must be label 1).

    Parameters
    ----------
    method : str — 'none' | 'ros' | 'smote' | 'smotewb'
    random_state : int

    Returns
    -------
    X_resampled, y_resampled
    """
    if method == 'none':
        return X.copy(), y.copy()
    elif method == 'ros':
        return _ros(X, y, random_state=random_state)
    elif method == 'smote':
        return _smote(X, y, k=5, random_state=random_state)
    elif method == 'smotewb':
        return smotewb_resample(X, y, M=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method '{method}'. "
                         f"Choose from: none, ros, smote, smotewb")
