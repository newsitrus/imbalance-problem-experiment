from sklearn.metrics import matthews_corrcoef


def compute_mcc(y_true, y_pred):
    """Compute Matthews Correlation Coefficient."""
    return matthews_corrcoef(y_true, y_pred)
