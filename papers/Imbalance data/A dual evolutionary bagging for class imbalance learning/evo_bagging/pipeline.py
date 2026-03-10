"""
Evo-Bagging end-to-end pipeline.

EvoBagging wraps InnerEnsemble + MultiModalGA and exposes:
  fit(X_train, y_train)   — train on one fold
  predict_proba(X_test)   — equal-weight averaged predictions
  run_cv(X, y, cv=10)     — full 10-fold cross-validation
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .inner_ensemble import InnerEnsemble
from .outer_ga import MultiModalGA
from .metrics import g_mean_score, auc_score


class EvoBagging:
    """
    Dual Evolutionary Bagging for class imbalance learning.

    Replicates Paper (ecoli3 subset, ROS oversampling).

    Parameters — all defaults match Paper Section 6
    -----------------------------------------------
    n_sub_datasets  : 30      number of ROS-balanced sub-datasets
    inner_cv        : 5       CV folds for inner grid search
    ga_pop_size     : 50
    ga_max_iter     : 100
    ga_pc           : 0.6     crossover probability
    ga_pm           : 0.08    mutation probability
    ga_niche_radius : 2.0     sharing radius L (inferred)
    ga_archive_size : 50
    random_state    : 42
    """

    def __init__(
        self,
        n_sub_datasets: int = 30,
        inner_cv: int = 5,
        ga_pop_size: int = 50,
        ga_max_iter: int = 100,
        ga_pc: float = 0.6,
        ga_pm: float = 0.08,
        ga_niche_radius: float = 2.0,
        ga_archive_size: int = 50,
        random_state: int = 42,
    ):
        self.n_sub_datasets = n_sub_datasets
        self.inner_cv = inner_cv
        self.ga_pop_size = ga_pop_size
        self.ga_max_iter = ga_max_iter
        self.ga_pc = ga_pc
        self.ga_pm = ga_pm
        self.ga_niche_radius = ga_niche_radius
        self.ga_archive_size = ga_archive_size
        self.random_state = random_state

        # Populated by fit()
        self.base_classifiers_: list = []
        self.selected_indices_: np.ndarray = np.array([], dtype=int)
        self.ga_: MultiModalGA = None  # type: ignore[assignment]

    # ── Core interface ────────────────────────────────────────────────────────

    def fit(self, X_train, y_train):
        """
        Train on one (train) fold.

        Step 1 — Inner ensemble: generate n_sub_datasets base classifiers.
        Step 2 — Outer GA: find the simplest optimal ensemble structure.
        """
        inner = InnerEnsemble(
            n_classifiers=self.n_sub_datasets,
            inner_cv=self.inner_cv,
            random_state=self.random_state,
        )
        inner.fit(X_train, y_train)
        self.base_classifiers_ = inner.get_classifiers()

        self.ga_ = MultiModalGA(
            pop_size=self.ga_pop_size,
            max_iter=self.ga_max_iter,
            pc=self.ga_pc,
            pm=self.ga_pm,
            niche_radius=self.ga_niche_radius,
            archive_size=self.ga_archive_size,
            random_state=self.random_state,
        )
        self.selected_indices_ = self.ga_.run(
            self.base_classifiers_, X_train, y_train
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Equal-weight averaging of selected base classifiers."""
        selected = [self.base_classifiers_[i] for i in self.selected_indices_]
        if not selected:
            # Fallback: use all classifiers
            selected = self.base_classifiers_
        return np.mean(
            [clf.predict_proba(X) for clf in selected], axis=0
        )

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Cross-validation ─────────────────────────────────────────────────────

    def run_cv(self, X, y, cv: int = 10, verbose: bool = True) -> dict:
        """
        10-fold stratified cross-validation.

        Each fold uses a fresh EvoBagging instance with
        random_state = self.random_state + fold_index to ensure
        reproducibility while varying randomness across folds.

        Parameters
        ----------
        X, y    : full dataset
        cv      : number of folds (default 10, matches paper)
        verbose : print fold-level progress

        Returns
        -------
        dict with:
          auc_mean, auc_std, gmean_mean, gmean_std  — aggregate metrics
          fold_aucs, fold_gmeans                    — per-fold lists
          ensemble_sizes                            — #classifiers used per fold
          ga_histories                              — best GA fitness per fold
        """
        skf = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=self.random_state
        )
        aucs, gmeans, sizes, histories = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if verbose:
                print(f"  Fold {fold + 1}/{cv} ...", end=" ", flush=True)

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = EvoBagging(
                n_sub_datasets=self.n_sub_datasets,
                inner_cv=self.inner_cv,
                ga_pop_size=self.ga_pop_size,
                ga_max_iter=self.ga_max_iter,
                ga_pc=self.ga_pc,
                ga_pm=self.ga_pm,
                ga_niche_radius=self.ga_niche_radius,
                ga_archive_size=self.ga_archive_size,
                random_state=self.random_state + fold,
            )
            model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_te)
            pred  = model.predict(X_te)

            fold_auc   = auc_score(y_te, proba)
            fold_gmean = g_mean_score(y_te, pred)
            aucs.append(fold_auc)
            gmeans.append(fold_gmean)
            sizes.append(int(len(model.selected_indices_)))
            histories.append(model.ga_.best_fitness_history_)

            if verbose:
                print(
                    f"AUC={fold_auc:.4f}  G-mean={fold_gmean:.4f}"
                    f"  ensemble_size={sizes[-1]}"
                )

        return {
            "auc_mean":      float(np.mean(aucs)),
            "auc_std":       float(np.std(aucs)),
            "gmean_mean":    float(np.mean(gmeans)),
            "gmean_std":     float(np.std(gmeans)),
            "fold_aucs":     aucs,
            "fold_gmeans":   gmeans,
            "ensemble_sizes": sizes,
            "ga_histories":  histories,
        }
