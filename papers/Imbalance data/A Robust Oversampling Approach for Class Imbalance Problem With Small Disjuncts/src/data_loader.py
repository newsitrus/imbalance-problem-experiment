"""
Dataset loader for UCI datasets used in the DROS paper.
Handles extraction from zip files and class selection per Table 1.
"""

import os
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

# Base path to datasets directory (relative to this file)
_DATASETS_DIR = Path(__file__).resolve().parent.parent / 'datasets'

# ---------------------------------------------------------------------------
# Dataset registry: maps name -> loading config
# Table 1 from the paper: dataset, dims, labels, min:maj, IR
# "-" for labels means already binary
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    'haberman': {
        'zip': 'haberman.zip',
        'description': 'Survival < 5yr',
        'minority_label': 2,        # died within 5 years
        'majority_label': 1,        # survived 5+ years
        'loader': '_load_haberman',
    },
    'wpbc': {
        'zip': 'wpbc.zip',
        'description': 'Cancer wpbc ret',
        'minority_label': 'R',      # recur
        'majority_label': 'N',      # non-recur
        'loader': '_load_wpbc',
    },
    'diabetes': {
        'zip': 'diabetes.zip',
        'description': 'Diabetes absent',
        'minority_label': 1,        # tested positive
        'majority_label': 0,        # tested negative
        'loader': '_load_diabetes',
    },
    'hepatitis': {
        'zip': 'hepatitis.zip',
        'description': 'Hepatitis normal',
        'minority_label': 1,        # die
        'majority_label': 2,        # live
        'loader': '_load_hepatitis',
    },
    'housing': {
        'zip': None,                # not a zip, directory
        'description': 'Housing MEDV > 35',
        'minority_label': 1,        # MEDV > 35
        'majority_label': 0,        # MEDV <= 35
        'loader': '_load_housing',
    },
    'spectf': {
        'zip': 'spectf.zip',
        'description': 'Spectf 0',
        'minority_label': 0,        # abnormal
        'majority_label': 1,        # normal
        'loader': '_load_spectf',
    },
    'iris': {
        'zip': 'iris.zip',
        'description': 'Iris setosa',
        'minority_label': 'Iris-setosa',
        'majority_label': 'NOT-setosa',   # combined versicolor+virginica
        'loader': '_load_iris',
    },
    'abalone_5_6': {
        'zip': 'abalone.zip',
        'description': 'Abalone_{5-6}',
        'minority_label': 5,
        'majority_label': 6,
        'loader': '_load_abalone',
    },
    'abalone_4_11': {
        'zip': 'abalone.zip',
        'description': 'Abalone_{4-11}',
        'minority_label': 4,
        'majority_label': 11,
        'loader': '_load_abalone',
    },
    'ecoli_4_2': {
        'zip': 'ecoli.zip',
        'description': 'Ecoli_{4-2}',
        'minority_label': 'imU',        # class 4 (35 samples)
        'majority_label': 'im',         # class 2 (77 samples)
        'loader': '_load_ecoli',
    },
    'ecoli_5_1': {
        'zip': 'ecoli.zip',
        'description': 'Ecoli_{5-1}',
        'minority_label': 'om',         # class 5 (20 samples)
        'majority_label': 'cp',         # class 1 (143 samples)
        'loader': '_load_ecoli',
    },
    'glass_7_2': {
        'zip': 'glass.zip',
        'description': 'Glass_{7-2}',
        'minority_label': 7,
        'majority_label': 2,
        'loader': '_load_glass',
    },
    'glass_5_1': {
        'zip': 'glass.zip',
        'description': 'Glass_{5-1}',
        'minority_label': 5,
        'majority_label': 1,
        'loader': '_load_glass',
    },
    'pageblocks_3_1': {
        'zip': 'pageblocks.zip',
        'description': 'Pageblocks_{3-1}',
        'minority_label': 3,
        'majority_label': 1,
        'loader': '_load_pageblocks',
    },
    'pageblocks_5_2': {
        'zip': 'pageblocks.zip',
        'description': 'Pageblocks_{5-2}',
        'minority_label': 5,
        'majority_label': 2,
        'loader': '_load_pageblocks',
    },
    'yeast_5_3': {
        'zip': 'yeast.zip',
        'description': 'Yeast_{5-3}',
        'minority_label': 'ME2',        # class 5
        'majority_label': 'MIT',        # class 3
        'loader': '_load_yeast',
    },
    'yeast_9_4': {
        'zip': 'yeast.zip',
        'description': 'Yeast_{9-4}',
        'minority_label': 'ERL',        # class 9
        'majority_label': 'ME1',        # class 4
        'loader': '_load_yeast',
    },
}


def list_datasets():
    """List all available dataset names and descriptions."""
    rows = []
    for name, cfg in DATASET_REGISTRY.items():
        rows.append({'name': name, 'description': cfg['description']})
    return pd.DataFrame(rows)


def load_dataset(name, datasets_dir=None):
    """
    Load a dataset by name. Returns (X, y) where minority=1, majority=0.

    Parameters
    ----------
    name : str
        Dataset name from DATASET_REGISTRY.
    datasets_dir : str or Path, optional
        Override default datasets directory.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), values in {0, 1}
    info : dict with 'description', 'n_minority', 'n_majority', 'ir'
    """
    if name not in DATASET_REGISTRY:
        available = ', '.join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    cfg = DATASET_REGISTRY[name]
    base_dir = Path(datasets_dir) if datasets_dir else _DATASETS_DIR

    # Extract zip if needed
    if cfg['zip'] is not None:
        zip_path = base_dir / cfg['zip']
        extract_dir = base_dir / 'extracted'
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

    loader_fn = globals()[cfg['loader']]
    X, y = loader_fn(base_dir, cfg)

    n_min = int(np.sum(y == 1))
    n_maj = int(np.sum(y == 0))
    ir = n_maj / n_min if n_min > 0 else float('inf')

    info = {
        'description': cfg['description'],
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_minority': n_min,
        'n_majority': n_maj,
        'ir': round(ir, 2),
    }

    return X, y, info


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def _load_haberman(base_dir, cfg):
    path = base_dir / 'extracted' / 'haberman.data'
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    labels = df.iloc[:, -1].values
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_wpbc(base_dir, cfg):
    path = base_dir / 'extracted' / 'wpbc.data'
    df = pd.read_csv(path, header=None)
    labels = df.iloc[:, 1].values
    # Drop ID column (0) and label column (1)
    X = df.iloc[:, 2:].values
    # Handle '?' missing values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
    # Drop rows with NaN
    valid = ~np.isnan(X).any(axis=1)
    X, labels = X[valid], labels[valid]
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X.astype(float), y


def _load_diabetes(base_dir, cfg):
    # Pima Indians Diabetes: 768 samples, 8 features, binary (0/1)
    path = base_dir / 'extracted' / 'pima-indians-diabetes.csv'
    if not path.exists():
        path = base_dir / 'extracted' / 'diabetes.csv'
    if not path.exists():
        for candidate in (base_dir / 'extracted').rglob('*diabetes*.csv'):
            path = candidate
            break
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    labels = df.iloc[:, -1].values
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_hepatitis(base_dir, cfg):
    path = base_dir / 'extracted' / 'hepatitis.data'
    df = pd.read_csv(path, header=None, na_values='?')
    df = df.dropna()
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(float)
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_housing(base_dir, cfg):
    path = base_dir / 'housing' / 'housing.data'
    df = pd.read_csv(path)
    # MEDV is the target column (last column or named 'medv'/'MEDV')
    target_col = df.columns[-1]
    X = df.iloc[:, :-1].values.astype(float)
    medv = df[target_col].values.astype(float)
    y = np.where(medv > 35, 1, 0)  # minority: MEDV > 35
    return X, y


def _load_spectf(base_dir, cfg):
    extract = base_dir / 'extracted'
    # UCI zip contains SPECT.train/SPECT.test (not SPECTF); combine for own CV
    parts = []
    for fname in ['SPECTF.train', 'SPECTF.test', 'SPECT.train', 'SPECT.test']:
        path = extract / fname
        if path.exists():
            parts.append(pd.read_csv(path, header=None))
    if not parts:
        for p in extract.rglob('*SPECT*'):
            if p.is_file() and p.suffix in ('.train', '.test'):
                parts.append(pd.read_csv(p, header=None))
    df = pd.concat(parts, ignore_index=True)
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(float)
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_iris(base_dir, cfg):
    path = base_dir / 'extracted' / 'iris.data'
    if not path.exists():
        for p in (base_dir / 'extracted').rglob('iris*data*'):
            path = p
            break
    df = pd.read_csv(path, header=None)
    df = df[df.iloc[:, -1].str.strip() != '']
    labels = df.iloc[:, -1].values.astype(str)
    labels = np.array([l.strip() for l in labels])
    X = df.iloc[:, :-1].values.astype(float)
    # Setosa (minority=1) vs rest (majority=0)
    y = np.where(labels == 'Iris-setosa', 1, 0)
    return X, y


def _load_abalone(base_dir, cfg):
    path = base_dir / 'extracted' / 'abalone.data'
    if not path.exists():
        for p in (base_dir / 'extracted').rglob('abalone*data*'):
            path = p
            break
    df = pd.read_csv(path, header=None)
    # Column 0 is Sex (categorical M/F/I) — encode numerically
    sex_map = {'M': 0, 'F': 1, 'I': 2}
    df[0] = df[0].map(sex_map)
    labels = df.iloc[:, -1].values  # Rings (integer)
    X = df.iloc[:, :-1].values.astype(float)
    # Filter to only the two selected classes
    mask = (labels == cfg['minority_label']) | (labels == cfg['majority_label'])
    X, labels = X[mask], labels[mask]
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_ecoli(base_dir, cfg):
    path = base_dir / 'extracted' / 'ecoli.data'
    if not path.exists():
        for p in (base_dir / 'extracted').rglob('ecoli*data*'):
            path = p
            break
    df = pd.read_csv(path, header=None, sep=r'\s+')
    labels = df.iloc[:, -1].values.astype(str)
    labels = np.array([l.strip() for l in labels])
    X = df.iloc[:, 1:-1].values.astype(float)  # drop name col 0
    mask = (labels == cfg['minority_label']) | (labels == cfg['majority_label'])
    X, labels = X[mask], labels[mask]
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_glass(base_dir, cfg):
    path = base_dir / 'extracted' / 'glass.data'
    if not path.exists():
        for p in (base_dir / 'extracted').rglob('glass*data*'):
            path = p
            break
    df = pd.read_csv(path, header=None)
    labels = df.iloc[:, -1].values
    X = df.iloc[:, 1:-1].values.astype(float)  # drop ID col 0
    mask = (labels == cfg['minority_label']) | (labels == cfg['majority_label'])
    X, labels = X[mask], labels[mask]
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_pageblocks(base_dir, cfg):
    extract = base_dir / 'extracted'
    # Look for decompressed .data file first, then .data.Z
    path = extract / 'page-blocks.data'
    if not path.exists():
        # Try to decompress .Z if it exists
        z_path = extract / 'page-blocks.data.Z'
        if z_path.exists():
            import subprocess
            subprocess.run(['uncompress', str(z_path)], check=False)
            if not path.exists():
                subprocess.run(['gzip', '-d', '-S', '.Z', str(z_path)], check=False)
    if not path.exists():
        for p in extract.rglob('page-blocks*'):
            if p.is_file() and not p.name.endswith(('.names', '.Z')):
                path = p
                break
    if path is None or not path.exists():
        raise FileNotFoundError("page-blocks data not found in extracted/")
    df = pd.read_csv(path, header=None, sep=r'\s+')
    labels = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values.astype(float)
    mask = (labels == cfg['minority_label']) | (labels == cfg['majority_label'])
    X, labels = X[mask], labels[mask]
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y


def _load_yeast(base_dir, cfg):
    path = base_dir / 'extracted' / 'yeast.data'
    if not path.exists():
        for p in (base_dir / 'extracted').rglob('yeast*data*'):
            path = p
            break
    df = pd.read_csv(path, header=None, sep=r'\s+')
    labels = df.iloc[:, -1].values.astype(str)
    labels = np.array([l.strip() for l in labels])
    X = df.iloc[:, 1:-1].values.astype(float)  # drop name col 0
    mask = (labels == cfg['minority_label']) | (labels == cfg['majority_label'])
    X, labels = X[mask], labels[mask]
    y = np.where(labels == cfg['minority_label'], 1, 0)
    return X, y
