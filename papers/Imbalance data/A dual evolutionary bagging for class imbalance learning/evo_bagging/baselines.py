"""
Baseline classifiers compared in the paper.

Three baselines are replicated:
  - OverBagging    : Bagging + ROS per bootstrap bag (DT base estimator)
  - EE-Bagging     : EasyEnsemble (imbalanced-learn)
  - Random Forest  : scikit-learn RandomForestClassifier

Each has a run_cv helper; run_all_baselines runs all three.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import RandomOverSampler

from .metrics import g_mean_score, auc_score


# ── OverBagging ───────────────────────────────────────────────────────────────

class OverBagging(BaseEstimator, ClassifierMixin):
    """
    Bagging with Random Over-Sampling applied inside each bootstrap bag.
    Uses Decision Tree as base estimator (standard bagging base).

    Parameters
    ----------
    n_estimators : int, default=30   (matches n_sub_datasets in Evo-Bagging)
    random_state : int, default=42
    """

    def __init__(self, n_estimators: int = 30, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: list = []
        self.classes_: np.ndarray = None  # type: ignore[assignment]

    def fit(self, X, y):
        self.estimators_ = []
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            seed = int(rng.randint(0, 100_000))
            # Bootstrap sample
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[idx], y[idx]
            # Balance minority within this bag
            if len(np.unique(y_boot)) > 1:
                ros = RandomOverSampler(
                    sampling_strategy="minority", random_state=seed
                )
                X_res, y_res = ros.fit_resample(X_boot, y_boot)
            else:
                X_res, y_res = X_boot, y_boot
            clf = DecisionTreeClassifier(random_state=seed)
            clf.fit(X_res, y_res)
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X):
        return np.mean(
            [clf.predict_proba(X) for clf in self.estimators_], axis=0
        )

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── CV runner ─────────────────────────────────────────────────────────────────

def _run_cv(clf_factory, X, y, cv: int = 10,
            random_state: int = 42) -> dict:
    """
    Stratified k-fold CV for one classifier factory.

    Returns dict with auc_mean, auc_std, gmean_mean, gmean_std,
    fold_aucs, fold_gmeans.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    aucs, gmeans = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        pred = np.argmax(proba, axis=1)
        aucs.append(auc_score(y_te, proba))
        gmeans.append(g_mean_score(y_te, pred))

    return {
        "auc_mean":   float(np.mean(aucs)),
        "auc_std":    float(np.std(aucs)),
        "gmean_mean": float(np.mean(gmeans)),
        "gmean_std":  float(np.std(gmeans)),
        "fold_aucs":  aucs,
        "fold_gmeans": gmeans,
    }


def run_all_baselines(X, y, cv: int = 10,
                      random_state: int = 42) -> dict:
    """
    Run all three paper baselines with stratified k-fold CV.

    Returns
    -------
    dict keyed by method name, each value is a metrics dict from _run_cv.
    """
    factories = {
        "OverBagging": lambda: OverBagging(
            n_estimators=30, random_state=random_state
        ),
        "EE-Bagging": lambda: EasyEnsembleClassifier(
            n_estimators=30, random_state=random_state
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    results = {}
    for name, factory in factories.items():
        results[name] = _run_cv(factory, X, y, cv=cv,
                                random_state=random_state)
    return results
