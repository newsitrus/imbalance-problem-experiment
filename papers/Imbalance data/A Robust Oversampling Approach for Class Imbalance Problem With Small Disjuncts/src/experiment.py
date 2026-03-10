"""
Experiment pipeline matching Section 4.2 of the DROS paper.
Stratified 5-fold CV x 10 repeats = 50 runs, SVM classifier.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dros import dros
from .metrics import compute_metrics


def run_experiment(
    X,
    y,
    dros_params=None,
    n_splits=5,
    n_repeats=10,
    random_state=42,
    verbose=True,
):
    """
    Run the full DROS + SVM experiment pipeline.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), values {0, 1} (1=minority)
    dros_params : dict, optional
        Override DROS parameters. Defaults: rho=0.5, k=7, delta=-0.7660, g=1.0
    n_splits : int
        Number of CV folds. Default 5.
    n_repeats : int
        Number of CV repeats. Default 10.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    results_df : pd.DataFrame
        Per-fold metrics (50 rows x 5 metric columns).
    summary : dict
        Mean ± std for each metric.
    """
    if dros_params is None:
        dros_params = {'rho': 0.5, 'k': 7, 'delta': -0.7660, 'g': 1.0}

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    total_folds = n_splits * n_repeats
    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        if verbose and (fold_idx + 1) % 10 == 0:
            print(f"  Fold {fold_idx + 1}/{total_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Z-score standardization (Section 4.2)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Split into majority / minority
        S_min = X_train[y_train == 1]
        S_maj = X_train[y_train == 0]

        # DROS oversampling
        S_new = dros(S_maj, S_min, **dros_params)

        # Combine original + synthetic
        if len(S_new) > 0:
            X_train_aug = np.vstack([X_train, S_new])
            y_train_aug = np.concatenate([
                y_train, np.ones(len(S_new), dtype=int)
            ])
        else:
            X_train_aug = X_train
            y_train_aug = y_train

        # SVM classifier (Section 4.2: Gaussian kernel, MATLAB fitcsvm defaults)
        # MATLAB fitcsvm: BoxConstraint=1, KernelScale=auto-heuristic
        # sklearn gamma='scale' = 1/(n_features * X.var()) is closest match
        # after z-score standardization (empirically verified vs paper results)
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train_aug, y_train_aug)

        # Predict
        y_pred = clf.predict(X_test)
        y_score = clf.decision_function(X_test)

        # Compute metrics (Eq. 26)
        fold_metrics = compute_metrics(y_test, y_pred, y_score)
        fold_metrics['fold'] = fold_idx + 1
        all_results.append(fold_metrics)

    # Aggregate
    results_df = pd.DataFrame(all_results)
    metric_cols = ['precision', 'recall', 'f_measure', 'g_mean', 'auc']

    summary = {}
    for col in metric_cols:
        mean = results_df[col].mean()
        std = results_df[col].std()
        summary[col] = {'mean': round(mean, 4), 'std': round(std, 4)}

    return results_df, summary
