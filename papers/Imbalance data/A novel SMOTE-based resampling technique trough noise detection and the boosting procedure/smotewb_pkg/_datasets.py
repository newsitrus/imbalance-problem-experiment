"""Dataset loading and auto-setup — internal module."""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

_PKG_DIR = Path(__file__).resolve().parent
_DATASETS_DIR = _PKG_DIR.parent / 'datasets'

# Fallback locations to search for datasets if not in _DATASETS_DIR
_FALLBACK_ROOTS = [
    # Sibling paper's extracted datasets
    _PKG_DIR.parent.parent
    / 'A Robust Oversampling Approach for Class Imbalance Problem With Small Disjuncts'
    / 'datasets' / 'extracted',
]

DATASET_REGISTRY = {
    'pima': {
        'description': 'Pima Indians Diabetes',
        'filename': 'pima-indians-diabetes.csv',
        'n_features': 8,
        'minority_label': 1,  # label in the raw file that indicates minority class
    },
}


def _ensure_dataset(filename):
    """Copy dataset from fallback location if not present in datasets dir."""
    target = _DATASETS_DIR / filename
    if target.exists():
        return target
    for root in _FALLBACK_ROOTS:
        candidate = root / filename
        if candidate.exists():
            _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            return target
    raise FileNotFoundError(
        f"Dataset '{filename}' not found.\n"
        f"Place the CSV file at: {target}"
    )


def load_dataset(name='pima'):
    """
    Load a dataset by name.

    Returns
    -------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) — 0=majority, 1=minority
    info : dict — dataset metadata
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    cfg = DATASET_REGISTRY[name]
    path = _ensure_dataset(cfg['filename'])

    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    raw_labels = df.iloc[:, -1].values
    y = (raw_labels == cfg['minority_label']).astype(int)

    n_min = int(np.sum(y == 1))
    n_maj = int(np.sum(y == 0))
    info = {
        'name': name,
        'description': cfg['description'],
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_minority': n_min,
        'n_majority': n_maj,
        'imbalance_ratio': round(n_maj / n_min, 3) if n_min > 0 else float('inf'),
    }
    return X, y, info


def list_datasets():
    """Return DataFrame with available dataset names and descriptions."""
    rows = [{'name': k, 'description': v['description']} for k, v in DATASET_REGISTRY.items()]
    return pd.DataFrame(rows).set_index('name')
