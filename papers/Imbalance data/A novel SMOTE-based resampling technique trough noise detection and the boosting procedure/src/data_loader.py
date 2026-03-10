import os
import numpy as np
import pandas as pd
from pathlib import Path

_DATASETS_DIR = Path(__file__).resolve().parent.parent / 'datasets'

DATASET_REGISTRY = {
    'pima': {
        'description': 'Pima Indians Diabetes',
        'filename': 'pima-indians-diabetes.csv',
        'minority_label': 1,
        'majority_label': 0,
    },
}


def list_datasets():
    """List available datasets."""
    rows = [{'name': k, 'description': v['description']}
            for k, v in DATASET_REGISTRY.items()]
    return pd.DataFrame(rows)


def load_dataset(name='pima', datasets_dir=None):
    """
    Load dataset by name.

    Returns (X, y, info) where y: 0=majority, 1=minority.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    cfg = DATASET_REGISTRY[name]
    base_dir = Path(datasets_dir) if datasets_dir else _DATASETS_DIR
    path = base_dir / cfg['filename']

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Download from UCI and place CSV at: {path}"
        )

    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    labels = df.iloc[:, -1].values
    y = np.where(labels == cfg['minority_label'], 1, 0)

    n_min = int(np.sum(y == 1))
    n_maj = int(np.sum(y == 0))
    info = {
        'description': cfg['description'],
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_minority': n_min,
        'n_majority': n_maj,
        'ir': round(n_maj / n_min, 2) if n_min > 0 else float('inf'),
    }
    return X, y, info
