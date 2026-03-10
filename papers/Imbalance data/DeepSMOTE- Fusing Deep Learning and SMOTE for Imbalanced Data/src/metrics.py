"""
Evaluation metrics matching the paper:
  - ACSA: Average Class-Specific Accuracy (mean per-class recall)
  - GM:   Macro-averaged Geometric Mean of per-class recall
  - FM:   Macro-averaged F1 measure

Reference: Sokolova & Lapalme (2009) — skew-insensitive metrics (§V-A-5).
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute ACSA, GM, and FM from true and predicted labels.

    Args:
        y_true: Ground-truth labels, shape (N,)
        y_pred: Predicted labels,    shape (N,)

    Returns:
        dict with keys 'acsa', 'gm', 'fm' (all in [0, 1])
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    per_class_recall = []
    per_class_precision = []
    per_class_f1 = []

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # row sum minus TP
        fp = cm[:, i].sum() - tp          # col sum minus TP

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class_recall.append(recall)
        per_class_precision.append(precision)
        per_class_f1.append(f1)

    # ACSA: mean per-class recall
    acsa = float(np.mean(per_class_recall))

    # GM: geometric mean of per-class recalls
    # Clip tiny values to avoid log(0)
    recalls_clipped = np.clip(per_class_recall, 1e-10, None)
    gm = float(np.exp(np.mean(np.log(recalls_clipped))))

    # FM: macro-averaged F1
    fm = float(np.mean(per_class_f1))

    return {"acsa": acsa, "gm": gm, "fm": fm}
