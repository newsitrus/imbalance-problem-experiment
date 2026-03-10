# Implementation Plan: Dual Evo-Bagging Replication

## Experimental Subset

| Setting | Value | Source |
|---|---|---|
| Dataset | ecoli3 (336 samples, 7 features, IR=8.6, 35 minority) | KEEL via imbalanced-learn |
| Oversampling | ROS (Random Over-Sampling) at 100% rate | Paper Section 6 |
| Sub-datasets (n) | 30 | Paper Section 5.1 (empirically stabilises at 30) |
| Inner CV | 5-fold stratified | Paper Algorithm 1 |
| Outer evaluation | 10-fold stratified CV | Paper Section 6 |
| Baselines | OverBagging, EE-Bagging, Random Forest | Paper Table 9/10 |
| Metrics | AUC, G-mean (mean ± std across 10 folds) | Paper Section 6 |

## GA Parameters (all explicit from Paper Section 6)

| Parameter | Value |
|---|---|
| Population size (Ps) | 50 |
| Max iterations (T) | 100 |
| Crossover probability (pc) | 0.6 |
| Mutation probability (pm) | 0.08 |
| Niche radius (L) | 2.0 (inferred; paper uses symbol L, not stated numerically) |
| Archive size (NA) | 50 |
| Fitness function | G-mean on original (imbalanced) training set |
| Encoding | Binary, length = n_sub_datasets = 30 |
| Selection | Roulette wheel |
| Crossover | Single-point |
| Mutation | Single-point bit-flip |
| Niching | Fitness sharing (Euclidean distance) |
| Final selection | Simplest structure (min active genes) from archive |

## Grid Search Ranges (Paper Section 6)

| Classifier | Parameter | Range |
|---|---|---|
| MLP | hidden_layer_sizes | {(50,), (100,), ..., (500,)} — 10 values |
| MLP | activation | logistic (sigmoid) — fixed |
| DT | max_depth | {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} — 10 values |
| DT | min_samples_leaf | {1, 5, 10} — 3 values |
| SVM | C | {2, 4, 6, 8, 10} — 5 values (paper: 0–10 step 2; C=0 invalid) |
| SVM | kernel | rbf — fixed |

## Paper Reference Results (ecoli3, ROS / best variant)

| Method | AUC | G-mean | Source |
|---|---|---|---|
| Evo-Bagging (ROS) | — | 0.9441 ± 0.0214 | Table 6 |
| Evo-Bagging (best) | 0.940 ± 0.031 | 0.947 ± 0.025 | Tables 9, 10 |
| OverBagging | 0.757 ± 0.070 | 0.719 ± 0.065 | Tables 9, 10 |
| EE-Bagging | 0.827 ± 0.089 | 0.923 ± 0.023 | Tables 9, 10 |
| Random Forest | 0.729 ± 0.104 | 0.757 ± 0.096 | Tables 9, 10 |

## Package Structure

```
evo_bagging/
├── __init__.py          # Public API
├── dataset.py           # ecoli3 loader (imblearn → binary labels)
├── metrics.py           # g_mean_score, auc_score
├── inner_ensemble.py    # InnerEnsemble: 30 base classifiers via grid search
├── outer_ga.py          # MultiModalGA: niche-based structure optimizer
├── baselines.py         # OverBagging, EE-Bagging, Random Forest
└── pipeline.py          # EvoBagging: end-to-end pipeline + run_cv
experiment.ipynb         # User-facing notebook
```

## Implementation Steps

1. **dataset.py** — load ecoli3, convert labels −1/+1 → 0/1
2. **metrics.py** — `g_mean_score(y_true, y_pred)`, `auc_score(y_true, y_proba)`
3. **inner_ensemble.py** — `InnerEnsemble.fit(X, y)` returns 30 fitted base classifiers
4. **outer_ga.py** — `MultiModalGA.run(classifiers, X_train, y_train)` returns selected indices
5. **baselines.py** — `OverBagging`, EE-Bagging (imbalanced-learn), RF wrappers + `run_all_baselines`
6. **pipeline.py** — `EvoBagging` with `fit / predict_proba / run_cv`
7. **experiment.ipynb** — config → data → run → results → paper comparison

## Key Implementation Notes

- **Prediction caching**: base classifier predictions on X_train are precomputed once per fold before the GA loop, reducing GA evaluation cost from O(T × Ps × n) forward passes to near-zero.
- **Vectorised niche**: pairwise distance matrix computed with NumPy broadcasting (no Python loops over pairs).
- **Reproducibility**: each 10-fold CV fold uses `random_state = base_seed + fold_index`.
- **OverBagging**: manually implemented (DT base estimator + ROS per bootstrap bag) to avoid imbalanced-learn API version issues.
