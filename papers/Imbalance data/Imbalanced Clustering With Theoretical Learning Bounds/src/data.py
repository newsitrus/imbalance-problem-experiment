"""
data.py — KEEL dataset downloader, parser, and preprocessor.

Downloads imbalanced datasets from the KEEL repository, parses their .dat format,
applies L2 normalization as required by the ICTLB paper (Section IV-A):
    "the feature vectors are normalized, and their L2 norm equals 1."
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, normalize

# ─── Dataset Registry ──────────────────────────────────────────────────────────
# Metadata matched to Table II of the paper.
# keel_file: the .dat filename inside the downloaded zip.
_KEEL_BASE = "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced"

DATASETS: dict[str, dict] = {
    "haberman": {
        "keel_url":  f"{_KEEL_BASE}/imb_IRlowerThan9/haberman.zip",
        "keel_file": "haberman.dat",
        "n": 306, "n_features": 3, "k": 2,
        "IR": 2.78,
        "minority_class_size": 81,
        "majority_class_size": 225,
        "description": "Haberman survival (< 5 yr post-surgery = minority)",
    },
    "ecoli3": {
        "keel_url":  f"{_KEEL_BASE}/imb_IRlowerThan9/ecoli3.zip",
        "keel_file": "ecoli3.dat",
        "n": 336, "n_features": 7, "k": 2,
        "IR": 8.60,
        "minority_class_size": 35,
        "majority_class_size": 301,
        "description": "E.coli protein localisation site (imb3 variant)",
    },
    "vowel0": {
        "keel_url":  f"{_KEEL_BASE}/imb_IRhigherThan9p1/vowel0.zip",
        "keel_file": "vowel0.dat",
        "n": 988, "n_features": 10, "k": 2,
        "IR": 9.98,
        "minority_class_size": 90,
        "majority_class_size": 898,
        "description": "Vowel recognition (vowel 0 vs rest)",
    },
    "dermatology6": {
        "keel_url":  f"{_KEEL_BASE}/imb_IRhigherThan9p1/dermatology-6class.zip",
        "keel_file": "dermatology-6class.dat",
        "n": 358, "n_features": 34, "k": 2,
        "IR": 16.90,
        "minority_class_size": 20,
        "majority_class_size": 338,
        "description": "Dermatology (class 6 vs rest)",
    },
    "shuttle-2vs5": {
        "keel_url":  f"{_KEEL_BASE}/imb_IRhigherThan9p3/shuttle-2_vs_5.zip",
        "keel_file": "shuttle-2_vs_5.dat",
        "n": 3316, "n_features": 9, "k": 2,
        "IR": 66.67,
        "minority_class_size": 49,
        "majority_class_size": 3267,
        "description": "NASA shuttle (class 2 vs 5)",
    },
}

# Alternative filenames to try inside the zip (KEEL naming is inconsistent).
_ALT_NAMES: dict[str, list[str]] = {
    "shuttle-2vs5": [
        "shuttle-2_vs_5.dat",
        "shuttle-2vs5.dat",
        "shuttle-2-vs-5.dat",
    ],
    "dermatology6": [
        "dermatology-6class.dat",
        "dermatology-6.dat",
        "dermatology6.dat",
    ],
}

# Dataset-specific column drops applied AFTER parsing (before normalisation).
# vowel0: KEEL provides 13 columns (TT, SpeakerNumber, Sex + F0–F9).
# The paper reports 10 features → keep only the 10 acoustic formant columns F0–F9
# (indices 3–12), dropping the 3 metadata columns.
_DROP_COLS: dict[str, list[int]] = {
    "vowel0": [0, 1, 2],   # drop TT, SpeakerNumber, Sex
}


# ─── KEEL .dat Parser ──────────────────────────────────────────────────────────

def _parse_keel_dat(content: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a KEEL .dat file string into (X, y) arrays.

    KEEL format:
        @relation <name>
        @attribute <name> <type>
        ...
        @inputs <attr1>, <attr2>, ...
        @outputs <label_attr>
        @data
        v1,v2,...,label
        ...
    """
    lines = content.splitlines()

    attributes: list[str] = []
    output_attr: Optional[str] = None
    data_lines: list[str] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        low = stripped.lower()
        if low.startswith("@data"):
            in_data = True
            continue
        if in_data:
            data_lines.append(stripped)
            continue
        if low.startswith("@attribute"):
            # @attribute <name> <type>
            parts = stripped.split(None, 2)
            if len(parts) >= 2:
                attributes.append(parts[1])
        elif low.startswith("@outputs"):
            # @outputs <attr_name>
            output_attr = stripped.split(None, 1)[1].strip()

    if not data_lines:
        raise ValueError("No @data section found in KEEL file.")
    if output_attr is None and attributes:
        output_attr = attributes[-1]

    # ── Parse data rows ──
    rows: list[list[str]] = []
    for dl in data_lines:
        if dl:
            rows.append([v.strip() for v in dl.split(",")])

    if not rows:
        raise ValueError("No data rows found.")

    n_cols = len(rows[0])
    if len(attributes) != n_cols:
        # Fallback: assume last column is label
        output_idx = n_cols - 1
    else:
        output_idx = attributes.index(output_attr) if output_attr in attributes else n_cols - 1

    feature_indices = [i for i in range(n_cols) if i != output_idx]

    X_raw: list[list[float]] = []
    y_raw: list[str] = []
    for row in rows:
        try:
            X_raw.append([float(row[i]) for i in feature_indices])
            y_raw.append(row[output_idx])
        except (ValueError, IndexError):
            continue  # skip malformed rows

    X = np.array(X_raw, dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)
    return X, y


# ─── Downloader ────────────────────────────────────────────────────────────────

def _download_zip(url: str, timeout: int = 30) -> bytes:
    """Download a zip file from URL and return raw bytes."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _extract_dat_from_zip(zip_bytes: bytes, preferred_name: str,
                           alt_names: Optional[list[str]] = None) -> str:
    """Extract the .dat file from a zip archive, trying preferred and alt names."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        # Try exact preferred name
        candidates = [preferred_name] + (alt_names or [])
        for candidate in candidates:
            # Match by basename (zip may contain subdirectory)
            matches = [n for n in names
                       if os.path.basename(n).lower() == candidate.lower()]
            if matches:
                return zf.read(matches[0]).decode("utf-8", errors="replace")
        # Fallback: any .dat file in the zip
        dat_files = [n for n in names if n.lower().endswith(".dat")]
        if dat_files:
            return zf.read(dat_files[0]).decode("utf-8", errors="replace")
    raise FileNotFoundError(
        f"Could not find a .dat file in zip. "
        f"Tried: {candidates}. Available: {names}"
    )


# ─── Public API ────────────────────────────────────────────────────────────────

def load_dataset(
    name: str,
    data_dir: str | Path = "data",
    force_download: bool = False,
    timeout: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a KEEL imbalanced dataset by name.

    Downloads from KEEL on first call and caches as .npz.  Applies L2
    normalisation to features as required by the ICTLB paper.

    Parameters
    ----------
    name : str
        Dataset key from DATASETS registry.
    data_dir : str | Path
        Directory for cached .npz files.
    force_download : bool
        Re-download even if cache exists.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    X : np.ndarray, shape (n, d)
        L2-normalised feature matrix.
    y : np.ndarray, shape (n,)
        Integer class labels (0 = majority, 1 = minority by convention).
    info : dict
        Metadata from DATASETS registry plus actual shape.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")

    meta = DATASETS[name]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{name}.npz"

    if cache_path.exists() and not force_download:
        cached = np.load(cache_path)
        X, y = cached["X"], cached["y"]
    else:
        print(f"  Downloading {name} from KEEL…", end=" ", flush=True)
        zip_bytes = _download_zip(meta["keel_url"], timeout=timeout)
        dat_content = _extract_dat_from_zip(
            zip_bytes,
            preferred_name=meta["keel_file"],
            alt_names=_ALT_NAMES.get(name),
        )
        X, y = _parse_keel_dat(dat_content)
        # Dataset-specific column drops (e.g. vowel0: drop metadata cols 0-2)
        drop = _DROP_COLS.get(name)
        if drop:
            keep = [i for i in range(X.shape[1]) if i not in drop]
            X = X[:, keep]
        # Ensure minority class = label 1 (majority = 0)
        counts = np.bincount(y)
        if len(counts) == 2 and counts[0] < counts[1]:
            y = 1 - y  # swap so minority = 1
        np.savez_compressed(cache_path, X=X, y=y)
        print("done.")

    # L2 normalisation — required by paper (||x|| = 1)
    X = normalize(X, norm="l2")

    info = {**meta, "actual_n": int(X.shape[0]), "actual_d": int(X.shape[1])}
    return X, y, info


def load_all_datasets(
    names: list[str],
    data_dir: str | Path = "data",
    force_download: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    """Load multiple datasets by name. Returns {name: (X, y, info)}."""
    result: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for name in names:
        result[name] = load_dataset(name, data_dir=data_dir,
                                    force_download=force_download)
    return result


def dataset_summary(names: Optional[list[str]] = None) -> None:
    """Print a summary table of dataset metadata."""
    import pandas as pd
    names = names or list(DATASETS)
    rows = []
    for n in names:
        m = DATASETS[n]
        rows.append({
            "Dataset": n,
            "Samples": m["n"],
            "Features": m["n_features"],
            "k": m["k"],
            "IR": m["IR"],
            "Minority": m["minority_class_size"],
            "Majority": m["majority_class_size"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
