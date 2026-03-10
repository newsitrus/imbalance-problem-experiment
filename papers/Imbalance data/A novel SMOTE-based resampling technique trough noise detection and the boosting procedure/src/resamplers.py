"""
Unified resampling interface: none, ROS, SMOTE, SMOTEWB.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .smotewb import smotewb


def _ros(X, y, random_state=None):
    """Random Oversampling: duplicate minority instances until balanced."""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    n_pos = len(pos_idx)
    n_neg = int(np.sum(y == 0))
    n_to_generate = n_neg - n_pos

    if n_to_generate <= 0:
        return X.copy(), y.copy()

    chosen = rng.choice(pos_idx, size=n_to_generate, replace=True)
    X_res = np.vstack([X, X[chosen]])
    y_res = np.concatenate([y, np.ones(n_to_generate, dtype=int)])
    return X_res, y_res


def _smote(X, y, k=5, random_state=None):
    """Standard SMOTE with k nearest minority neighbors."""
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    n_pos = len(pos_idx)
    n_neg = int(np.sum(y == 0))
    n_to_generate = n_neg - n_pos

    if n_to_generate <= 0 or n_pos < 2:
        return X.copy(), y.copy()

    X_pos = X[pos_idx]
    k_actual = min(k, n_pos - 1)

    nn = NearestNeighbors(n_neighbors=k_actual + 1)
    nn.fit(X_pos)
    _, indices = nn.kneighbors(X_pos)
    neighbor_indices = indices[:, 1:]  # exclude self

    synthetics = []
    for _ in range(n_to_generate):
        i = rng.randint(n_pos)
        j = rng.randint(k_actual)
        neighbor = X_pos[neighbor_indices[i, j]]
        lam = rng.rand()
        synthetics.append(X_pos[i] + lam * (neighbor - X_pos[i]))

    X_syn = np.array(synthetics)
    X_res = np.vstack([X, X_syn])
    y_res = np.concatenate([y, np.ones(n_to_generate, dtype=int)])
    return X_res, y_res


def resample(X, y, method='none', random_state=None):
    """
    Apply resampling to training data.

    Parameters
    ----------
    X, y : training features and labels
    method : str — 'none', 'ros', 'smote', 'smotewb'
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
        return smotewb(X, y, M=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: '{method}'")
