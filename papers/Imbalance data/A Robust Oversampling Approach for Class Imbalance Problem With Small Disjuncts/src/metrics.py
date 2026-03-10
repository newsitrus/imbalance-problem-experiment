"""
Evaluation metrics from Section 4.1, Equation 26.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute paper metrics. Minority class = positive (1).

    Returns dict with: precision, recall, f_measure, g_mean, auc
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Eq. 26
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f_measure = (2 * recall * precision / (recall + precision)
                 if (recall + precision) > 0 else 0.0)

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    g_mean = np.sqrt(tpr * tnr)

    auc = 0.0
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f_measure': f_measure,
        'g_mean': g_mean,
        'auc': auc,
    }
