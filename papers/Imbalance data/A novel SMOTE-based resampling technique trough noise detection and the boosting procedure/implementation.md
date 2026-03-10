# SMOTEWB Replication — Implementation Plan

**Paper:** A novel SMOTE-based resampling technique through noise detection and the boosting procedure
**Authors:** Fatih Salam, Mehmet Ali Cengiz
**Venue:** Expert Systems with Applications, 2022

---

## 1. Objective

Replicate the core experiment from the paper: compare four resampling methods across multiple classifiers using 10×10 stratified cross-validation, measured by MCC (Matthews Correlation Coefficient) average rank.

The experiment design is fixed to match the paper. **The only user-configurable settings are:**
- Which dataset to run on
- Which resampling methods to include
- Which classifiers to include
- CV splits/repeats (default: 10×10 as in paper)

---

## 2. Paper Experiment Design (Fixed Defaults)

### Resampling Methods (4)
| Key | Method | Details |
|-----|---------|---------|
| `none` | No Resampling | Baseline |
| `ros` | ROS | Duplicate minority instances uniformly at random |
| `smote` | SMOTE | k=5 minority neighbors, interpolation |
| `smotewb` | SMOTEWB | AdaBoost noise detection + adaptive k_i per instance |

### Classifiers (9, matching paper's R packages)
| Key | Paper Name | sklearn equivalent | Key params |
|-----|-----------|-------------------|-----------|
| `J48` | J48 (C4.5) | `DecisionTreeClassifier` | criterion=entropy, min_samples_leaf=2 |
| `KNN` | KNN | `KNeighborsClassifier` | n_neighbors=5 |
| `CART` | CART | `DecisionTreeClassifier` | criterion=gini, ccp_alpha=0.01 |
| `Radial_SVM` | Radial SVM | `SVC` | kernel=rbf, gamma=1.0, C=0.25 |
| `Linear_SVM` | Linear SVM | `SVC` | kernel=linear, C=0.25 |
| `NNet` | NNet | `MLPClassifier` | hidden=(3,), alpha=0.001, solver=lbfgs |
| `NaiveBayes` | Naive Bayes | `GaussianNB` | defaults |
| `LogisticReg` | Logistic Reg | `LogisticRegression` | max_iter=1000, lbfgs |
| `MARS` | MARS | `pyearth.Earth` (optional) | max_terms=10, degree=1 |

### Evaluation
- **CV:** 10-fold stratified × 10 repeats = 100 folds
- **Metric:** MCC (Matthews Correlation Coefficient)
- **Ranking:** Per (fold, classifier): rank 4 resamplers from 1 (best MCC) to 4 (worst); report average rank

### Dataset (default)
- **Pima Indians Diabetes** — 768 samples, 8 numeric features, IR ≈ 1.87 (500 neg / 268 pos)
- Source: UCI repository (file placed in `datasets/pima-indians-diabetes.csv`)

---

## 3. SMOTEWB Algorithm (Paper Equations)

1. Scale X to [0,1] via MinMaxScaler
2. Run AdaBoost for M=100 rounds (decision stumps) → per-sample weights W
3. Compute class thresholds:
   - T_pos = 2 · n_neg / n²  (for minority/positive class)
   - T_neg = 2 · n_pos / n²  (for majority/negative class)
4. Label instance i as noise if W[i] > T_class[i]
5. k_max = floor(n_neg / n_pos)
6. For each minority instance x_i:
   - Walk through non-noise neighbors (sorted by distance)
   - k_i = count of minority non-noise neighbors before first majority non-noise neighbor (≤ k_max)
   - Label: **good** (k_i > 0), **lonely** (k_i = 0, not noise), **bad** (k_i = 0, noise)
7. Generate n_neg − n_pos synthetic samples (round-robin over good/lonely):
   - **good** → x_syn = x_i + λ · (neighbor − x_i), λ ~ U(0,1)
   - **lonely** → copy x_i
   - **bad** → skip
8. Descale, return balanced dataset

---

## 4. Repository Structure

```
papers/A novel SMOTE-based resampling technique.../
├── datasets/
│   └── pima-indians-diabetes.csv     ← auto-copied on first import
├── smotewb_pkg/                       ← Python package (user-imported)
│   ├── __init__.py                    ← clean public API
│   ├── _core.py                       ← SMOTEWB algorithm internals
│   ├── _resamplers.py                 ← none / ROS / SMOTE / SMOTEWB
│   ├── _classifiers.py                ← classifier registry
│   ├── _datasets.py                   ← dataset loading + auto-setup
│   └── _experiment.py                 ← pipeline: CV loop + MCC ranking
├── src/                               ← original implementation (reference)
├── experiment.ipynb                   ← user-facing Jupyter notebook
└── implementation.md                  ← this file
```

---

## 5. Package API (smotewb_pkg)

```python
from smotewb_pkg import (
    load_dataset,       # (X, y, info) from dataset name
    list_datasets,      # DataFrame of available datasets
    list_classifiers,   # dict of available classifiers
    RESAMPLER_NAMES,    # ['none','ros','smote','smotewb']
    run_experiment,     # main entry point
)

# Run the experiment
fold_df, rank_table = run_experiment(
    X, y,
    resamplers=RESAMPLER_NAMES,       # user subset
    classifiers=list(CLASSIFIER_REGISTRY),
    n_splits=10,
    n_repeats=10,
    random_state=42,
)
```

---

## 6. Pipeline Design

Since resampling must be applied only to each training fold (not the test fold), a standard `sklearn.Pipeline` cannot natively wrap the resampler. The implementation uses a **custom `ResamplingPipeline`** estimator:

```
ResamplingPipeline(resampler, classifier)
  └── fit(X_train, y_train):
        1. resample(X_train, y_train) → X_res, y_res
        2. MinMaxScaler.fit_transform(X_res) → X_scaled
        3. classifier.fit(X_scaled, y_res)
  └── predict(X_test):
        1. MinMaxScaler.transform(X_test)
        2. classifier.predict(X_scaled_test)
```

This estimator is compatible with sklearn's `cross_validate`, enabling clean cross-validation without data leakage.

The experiment loop uses `RepeatedStratifiedKFold` with per-fold random seeds (seed + fold_index) to ensure resampling varies across folds while remaining reproducible.

---

## 7. Notebook Design (experiment.ipynb)

| # | Cell | Content |
|---|------|---------|
| 1 | Setup | `sys.path` injection, imports |
| 2 | Dataset | Load and display dataset summary |
| 3 | **Settings** | User-editable: dataset, resamplers, classifiers, CV |
| 4 | Run | `run_experiment(...)` with progress bar |
| 5 | Rank Table | Pivot table: classifier × resampler average rank |
| 6 | MCC Plot | Boxplot: MCC distribution per resampler |
| 7 | Rank Heatmap | Heatmap: rank per classifier × resampler |

---

## 8. Reproducibility Notes

- All random seeds fixed via `random_state` parameter
- Per-fold seed = `random_state + fold_index` (same as original experiment.py)
- MARS classifier is optional (requires `pyearth`); experiment runs without it if unavailable
- Results will differ slightly from the R-based paper due to:
  - Different random number generators (Python vs. R)
  - Sklearn vs. R package parameter equivalences (approximated)
  - No exact replication of R's `kernlab` sigma or `rpart` cp parameter
