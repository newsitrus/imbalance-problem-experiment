"""
Inner Ensemble — base classifier generation via grid search.

For each of n_classifiers sub-datasets (created by ROS oversampling):
  1. Run grid search (5-fold CV) for MLP, DT, and SVM.
  2. Select the sub-classifier with the highest CV accuracy.
  3. Return it as a fitted base classifier.

Grid-search spaces are taken verbatim from Paper Section 6.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler


# ── Grid-search parameter spaces (reduced subset for feasible runtime) ────────
# Paper Section 6 originals → reduced (approx. 4x speedup on grid search alone)
#   MLP: 10 configs → 5  |  DT: 30 configs → 10  |  SVM: 5 configs → 3

_MLP_PARAMS = {
    "hidden_layer_sizes": [(50,), (150,), (300,), (450,), (500,)],  # 5 configs (was 10, spans 50–500)
    # activation=logistic (sigmoid) — fixed; solver default (adam)
}

_DT_PARAMS = {
    "max_depth": [1, 5, 10, 15, 19],      # 5 configs (was 10, spans shallow–deep)
    "min_samples_leaf": [1, 10],          # 2 configs (was 3, covers extreme/moderate)
}

_SVM_PARAMS = {
    "C": [2, 6, 10],                      # 3 configs (was 5, spans low/mid/high regularisation)
    # kernel=rbf, probability=True — fixed
}


def _best_classifier(X_sub, y_sub, inner_cv: int, seed: int):
    """
    Run grid search for MLP, DT, SVM on one sub-dataset.
    Return the fitted estimator with the highest cross-validation accuracy.
    """
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)

    candidates = [
        (
            MLPClassifier(
                activation="logistic", max_iter=2000,
                random_state=seed,
            ),
            _MLP_PARAMS,
        ),
        (
            DecisionTreeClassifier(random_state=seed),
            _DT_PARAMS,
        ),
        (
            SVC(kernel="rbf", probability=True, random_state=seed),
            _SVM_PARAMS,
        ),
    ]

    best_clf, best_score = None, -1.0
    for base_clf, param_grid in candidates:
        gs = GridSearchCV(
            base_clf, param_grid,
            cv=cv, scoring="accuracy",
            n_jobs=1, refit=True,
        )
        gs.fit(X_sub, y_sub)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_clf = gs.best_estimator_

    return best_clf


# ── InnerEnsemble ─────────────────────────────────────────────────────────────

class InnerEnsemble:
    """
    Generates n_classifiers base classifiers from a training set.

    Parameters
    ----------
    n_classifiers : int, default=30
        Number of balanced sub-datasets (and resulting base classifiers).
    inner_cv : int, default=5
        Number of folds for grid-search cross-validation.
    random_state : int, default=42
    """

    def __init__(self, n_classifiers: int = 30, inner_cv: int = 5,
                 random_state: int = 42):
        self.n_classifiers = n_classifiers
        self.inner_cv = inner_cv
        self.random_state = random_state
        self.classifiers_: list = []

    def fit(self, X_train, y_train):
        """
        Build n_classifiers base classifiers.

        For each sub-dataset:
          - Apply ROS (100 % oversampling of minority → balanced)
          - Select the best sub-classifier via grid search
        """
        self.classifiers_ = []
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_classifiers):
            seed = int(rng.randint(0, 100_000))
            ros = RandomOverSampler(
                sampling_strategy="minority",   # minority → same size as majority
                random_state=seed,
            )
            X_sub, y_sub = ros.fit_resample(X_train, y_train)
            clf = _best_classifier(X_sub, y_sub, self.inner_cv, seed)
            self.classifiers_.append(clf)

        return self

    def get_classifiers(self) -> list:
        """Return the list of fitted base classifiers."""
        return self.classifiers_
