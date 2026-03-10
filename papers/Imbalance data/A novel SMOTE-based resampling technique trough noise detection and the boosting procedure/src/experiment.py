"""
Experiment pipeline: 10x10-fold CV, 4 resamplers x 9 classifiers, MCC + ranking.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from .resamplers import resample
from .classifiers import get_classifier, CLASSIFIER_REGISTRY
from .metrics import compute_mcc

RESAMPLER_NAMES = ['none', 'ros', 'smote', 'smotewb']
RESAMPLER_DISPLAY = {
    'none': 'No Resampling',
    'ros': 'ROS',
    'smote': 'SMOTE',
    'smotewb': 'SMOTEWB',
}


def run_experiment(
    X, y,
    resampler_names=None,
    classifier_names=None,
    n_splits=10,
    n_repeats=10,
    random_state=42,
    verbose=True,
):
    """
    Run the full experiment pipeline.

    Parameters
    ----------
    X, y : dataset (original scale, y: 0=majority, 1=minority)
    resampler_names : list of str, default all 4
    classifier_names : list of str, default all from registry
    n_splits, n_repeats : CV config (default 10x10)
    random_state : int
    verbose : bool

    Returns
    -------
    fold_results : pd.DataFrame
        Columns: fold, classifier, resampler, mcc
    rank_summary : pd.DataFrame
        Average rank per classifier x resampler
    """
    if resampler_names is None:
        resampler_names = RESAMPLER_NAMES
    if classifier_names is None:
        classifier_names = list(CLASSIFIER_REGISTRY.keys())

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    total_folds = n_splits * n_repeats

    all_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        if verbose and (fold_idx + 1) % 10 == 0:
            print(f"  Fold {fold_idx + 1}/{total_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for res_name in resampler_names:
            # Resample training data
            X_res, y_res = resample(
                X_train, y_train, method=res_name,
                random_state=random_state + fold_idx
            )

            # Scale to [0,1] after resampling
            scaler = MinMaxScaler()
            X_res_scaled = scaler.fit_transform(X_res)
            X_test_scaled = scaler.transform(X_test)

            for clf_name in classifier_names:
                clf = get_classifier(clf_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        clf.fit(X_res_scaled, y_res)
                        y_pred = clf.predict(X_test_scaled)
                        mcc = compute_mcc(y_test, y_pred)
                    except Exception:
                        mcc = 0.0

                all_records.append({
                    'fold': fold_idx,
                    'classifier': clf_name,
                    'resampler': res_name,
                    'mcc': mcc,
                })

    fold_results = pd.DataFrame(all_records)

    # Compute ranks: per (fold, classifier), rank resamplers by MCC (1=best)
    def _rank_group(group):
        group = group.copy()
        group['rank'] = group['mcc'].rank(ascending=False, method='average')
        return group

    ranked = fold_results.groupby(
        ['fold', 'classifier'], group_keys=False
    ).apply(_rank_group)

    rank_summary = ranked.groupby(
        ['classifier', 'resampler']
    )['rank'].mean().unstack()

    # Reorder columns
    col_order = [r for r in resampler_names if r in rank_summary.columns]
    rank_summary = rank_summary[col_order]
    rank_summary.columns = [RESAMPLER_DISPLAY.get(c, c) for c in col_order]

    return fold_results, rank_summary
