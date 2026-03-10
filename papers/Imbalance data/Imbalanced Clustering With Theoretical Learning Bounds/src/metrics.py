"""
metrics.py — Clustering evaluation metrics used in the ICTLB paper.

Four metrics (Section IV-C):
  1. ACC   — Clustering accuracy via the Hungarian assignment algorithm.
  2. F-score — Macro-averaged F1 after optimal label alignment.
  3. Recall  — Macro-averaged recall after optimal label alignment.
  4. DCV     — Density Cluster Validity (Xiong et al. 2009 [44]).
               Operationalised here as the coefficient of variation (CV) of
               predicted cluster sizes.  A larger DCV indicates a more
               distorted cluster-size distribution (poor performance), which is
               consistent with every qualitative observation in the paper:
               • density-k-means++ → very unequal predicted sizes → large DCV
               • k-means uniform-effect → very equal sizes → small DCV (but
                 still wrong for imbalanced data, so "small DCV ≠ good")
               • ICTLB → sizes closer to truth → moderate DCV, best ACC

Higher ACC / F-score / Recall = better.
Larger DCV = worse (but small DCV ≠ necessarily good).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, recall_score


# ─── Label alignment via Hungarian algorithm ──────────────────────────────────

def _optimal_label_mapping(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return a permutation array such that pred_mapped[i] = mapping[y_pred[i]].

    Solves the assignment problem: find the bijection between predicted cluster
    labels and true class labels that maximises the number of matches.
    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    k_true = int(y_true.max()) + 1
    k_pred = int(y_pred.max()) + 1
    k = max(k_true, k_pred)

    # Confusion matrix C[i, j] = #samples where true=i, pred=j
    C = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1

    # Hungarian: maximise trace → minimise negative C
    row_ind, col_ind = linear_sum_assignment(-C)

    # Build mapping: pred_label → true_label
    mapping = np.arange(k, dtype=int)
    for r, c in zip(row_ind, col_ind):
        mapping[c] = r          # predicted cluster c maps to true class r
    return mapping


def _align_predictions(y_pred: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Apply a label mapping to predicted cluster indices."""
    return mapping[np.asarray(y_pred, dtype=int)]


# ─── Individual metrics ────────────────────────────────────────────────────────

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy (ACC) with optimal Hungarian label alignment.

    ACC = (number of correctly assigned samples) / n
    Higher is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    mapping = _optimal_label_mapping(y_true, y_pred)
    y_aligned = _align_predictions(y_pred, mapping)
    return float((y_aligned == y_true).mean())


def clustering_fscore(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1-score after optimal label alignment.

    Higher is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    mapping = _optimal_label_mapping(y_true, y_pred)
    y_aligned = _align_predictions(y_pred, mapping)
    return float(f1_score(y_true, y_aligned, average="macro", zero_division=0))


def clustering_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged recall after optimal label alignment.

    Higher is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    mapping = _optimal_label_mapping(y_true, y_pred)
    y_aligned = _align_predictions(y_pred, mapping)
    return float(recall_score(y_true, y_aligned, average="macro", zero_division=0))


def dcv(y_pred: np.ndarray) -> float:
    """Density Cluster Validity (DCV) index.

    Operationalised as the coefficient of variation of predicted cluster sizes:
        DCV = std(cluster_sizes) / mean(cluster_sizes)

    Properties:
    • k-means uniform effect → nearly equal cluster sizes → DCV ≈ 0 (small)
    • density-k-means++ over-grouping → very skewed sizes → large DCV
    • Larger DCV indicates a worse (more distorted) predicted distribution.

    Reference: Xiong H., Wu J., Chen J. (2009) IEEE Trans. Syst. Man Cybern. B,
               39(2), 318-331.  [Paper reference 44]
    """
    y_pred = np.asarray(y_pred, dtype=int)
    _, counts = np.unique(y_pred, return_counts=True)
    if len(counts) <= 1 or counts.mean() == 0:
        return 0.0
    return float(counts.std() / counts.mean())


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all four metrics in a single pass (reuses label mapping).

    Returns
    -------
    dict with keys: 'ACC', 'F-score', 'Recall', 'DCV'
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    mapping = _optimal_label_mapping(y_true, y_pred)
    y_aligned = _align_predictions(y_pred, mapping)

    acc = float((y_aligned == y_true).mean())
    fs  = float(f1_score(y_true, y_aligned, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_aligned, average="macro", zero_division=0))
    dc  = dcv(y_pred)

    return {"ACC": acc, "F-score": fs, "Recall": rec, "DCV": dc}
