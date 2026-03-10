"""
Evaluation metrics used in the paper: G-mean and AUC.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def g_mean_score(y_true, y_pred):
    """
    Geometric mean of sensitivity (recall of minority) and specificity.

    G-mean = sqrt( TP/(TP+FN)  ×  TN/(TN+FP) )

    Returns 0.0 when a class is missing from y_true.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))


def auc_score(y_true, y_proba):
    """
    ROC-AUC for binary classification.

    Parameters
    ----------
    y_true  : 1-D int array
    y_proba : 2-D float array of shape (n, 2) — column 1 is P(minority)

    Returns 0.5 on error.
    """
    try:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    except Exception:
        return 0.5
