"""
ecoli3 dataset loader.

The KEEL 'ecoli3' dataset (336 samples, 7 features, IR ≈ 8.6) is retrieved
via imbalanced-learn's fetch_datasets() as the 'ecoli' entry, which provides
the same binary split: minority class (imU, 35 samples) vs. all others (301).
Labels are converted from {-1, +1} to {0, 1}.
"""

import numpy as np


def load_ecoli3():
    """
    Load the ecoli3 binary imbalanced dataset.

    Returns
    -------
    X : ndarray of shape (336, 7), float64
    y : ndarray of shape (336,), int — 1 = minority (35 samples), 0 = majority
    """
    from imblearn.datasets import fetch_datasets

    raw = fetch_datasets()["ecoli"]
    X = raw.data.astype(np.float64)
    y = (raw.target == 1).astype(int)   # −1 → 0, +1 → 1
    return X, y
