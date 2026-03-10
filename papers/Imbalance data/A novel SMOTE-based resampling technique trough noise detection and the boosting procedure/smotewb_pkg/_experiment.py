"""
Experiment pipeline — internal module.

Implements 10×10 stratified CV with resampling + scaling + classification,
reporting per-fold MCC and average rank table.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef

from ._resamplers import resample
from ._classifiers import get_classifier, CLASSIFIER_REGISTRY

RESAMPLER_NAMES = ['none', 'ros', 'smote', 'smotewb']
RESAMPLER_DISPLAY = {
    'none': 'No Resampling',
    'ros': 'ROS',
    'smote': 'SMOTE',
    'smotewb': 'SMOTEWB',
}


class ResamplingPipeline(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible pipeline: resample → scale → classify.

    Enables use with cross_validate and other sklearn utilities.
    Resampling applies only during fit (training fold), not predict.
    """

    def __init__(self, resampler_name='smotewb', classifier_name='J48', random_state=42):
        self.resampler_name = resampler_name
        self.classifier_name = classifier_name
        self.random_state = random_state

    def fit(self, X, y):
        X_res, y_res = resample(X, y, method=self.resampler_name,
                                random_state=self.random_state)
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X_res)
        self.clf_ = get_classifier(self.classifier_name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.clf_.fit(X_scaled, y_res)
        return self

    def predict(self, X):
        return self.clf_.predict(self.scaler_.transform(X))


def run_experiment(
    X,
    y,
    resamplers=None,
    classifiers=None,
    n_splits=10,
    n_repeats=10,
    random_state=42,
    progress=True,
):
    """
    Run the full replication experiment.

    Performs n_splits × n_repeats stratified CV for every
    (resampler, classifier) combination and reports MCC scores
    and average rank per (classifier, resampler).

    Parameters
    ----------
    X, y       : dataset arrays — y must be 0 (majority) / 1 (minority)
    resamplers : list of str — subset of RESAMPLER_NAMES (default: all 4)
    classifiers: list of str — subset of CLASSIFIER_REGISTRY (default: all)
    n_splits   : int — CV folds (paper: 10)
    n_repeats  : int — CV repeats (paper: 10)
    random_state : int — master seed
    progress   : bool — print fold progress

    Returns
    -------
    fold_df : pd.DataFrame
        Columns: fold, classifier, resampler, mcc
    rank_table : pd.DataFrame
        Index: classifier, Columns: resampler display names, Values: avg rank
    """
    if resamplers is None:
        resamplers = RESAMPLER_NAMES
    if classifiers is None:
        classifiers = list(CLASSIFIER_REGISTRY.keys())

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    total_folds = n_splits * n_repeats
    records = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        if progress and (fold_idx + 1) % 10 == 0:
            print(f"  Fold {fold_idx + 1:3d}/{total_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Per-fold random seed varies so resamplers produce different samples each fold
        fold_seed = random_state + fold_idx

        for res_name in resamplers:
            X_res, y_res = resample(X_train, y_train, method=res_name,
                                    random_state=fold_seed)
            scaler = MinMaxScaler()
            X_res_sc = scaler.fit_transform(X_res)
            X_test_sc = scaler.transform(X_test)

            for clf_name in classifiers:
                clf = get_classifier(clf_name)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        clf.fit(X_res_sc, y_res)
                        y_pred = clf.predict(X_test_sc)
                        mcc = matthews_corrcoef(y_test, y_pred)
                    except Exception:
                        mcc = 0.0

                records.append({
                    'fold': fold_idx,
                    'classifier': clf_name,
                    'resampler': res_name,
                    'mcc': mcc,
                })

    fold_df = pd.DataFrame(records)

    # Rank resamplers per (fold, classifier): 1=best MCC, 4=worst
    # Use transform to preserve all rows while adding the rank column
    fold_df['rank'] = fold_df.groupby(['fold', 'classifier'])['mcc'].transform(
        lambda s: s.rank(ascending=False, method='average')
    )

    rank_table = (
        fold_df.groupby(['classifier', 'resampler'])['rank']
        .mean()
        .unstack()
        .reindex(columns=[r for r in resamplers if r in fold_df['resampler'].unique()])
    )
    rank_table.columns = [RESAMPLER_DISPLAY.get(c, c) for c in rank_table.columns]

    return fold_df, rank_table
