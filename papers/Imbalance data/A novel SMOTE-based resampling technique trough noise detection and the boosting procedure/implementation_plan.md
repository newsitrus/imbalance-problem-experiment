# SMOTEWB Replication — Implementation Plan

**Paper:** A novel SMOTE-based resampling technique through noise detection and the boosting procedure
**Authors:** Fatih Salam, Mehmet Ali Cengiz
**Venue:** Expert Systems with Applications, 2022

## Objective

Replicate SMOTEWB on **Pima Indians Diabetes** (n=768, 500 neg / 268 pos, IR=1.87) with 4 resampling methods × 9 classifiers, using 10×10-fold CV and MCC rank-based comparison.

## Scope

### Resampling Methods (4)
1. No Resampling — baseline
2. ROS — Random Oversampling (duplicate minority instances)
3. SMOTE — standard SMOTE (k=5)
4. SMOTEWB — proposed method (AdaBoost noise detection, adaptive k_i)

### Classifiers (9)

| Paper Name | Python (sklearn) | Parameters |
|------------|------------------|------------|
| J48 | DecisionTreeClassifier | criterion='entropy', min_samples_leaf=2 |
| KNN | KNeighborsClassifier | n_neighbors=5 |
| CART | DecisionTreeClassifier | criterion='gini', ccp_alpha=0.01 |
| Radial SVM | SVC | kernel='rbf', gamma=1.0, C=0.25 |
| Linear SVM | SVC | kernel='linear', C=0.25 |
| NNet | MLPClassifier | hidden_layer_sizes=(3,), alpha=0.001, max_iter=1000 |
| Naive Bayes | GaussianNB | default |
| Logistic Reg | LogisticRegression | max_iter=1000 |
| MARS | pyearth.Earth | nprune=10, degree=1 (skip if unavailable) |

### Dataset
- Pima Indians Diabetes (UCI), 768 samples, 8 numeric features, binary

### Evaluation
- 10-fold stratified CV × 10 repeats = 100 folds
- Metric: MCC (Matthews Correlation Coefficient)
- Comparison: rank 4 resamplers per fold per classifier (1=best), average ranks

## SMOTEWB Algorithm

1. Scale features to [0,1] (MinMaxScaler)
2. Run AdaBoost for M=100 iterations (decision stumps) → final instance weights W
3. Noise thresholds: T_pos = 2·n_neg/n², T_neg = 2·n_pos/n²
4. Label noise: positive instance noisy if W[i] > T_pos; negative if W[i] > T_neg
5. k_max = floor(n_neg / n_pos)
6. For each positive instance: find non-noise neighbors sorted by distance; k_i = number of positive non-noise neighbors before first negative non-noise neighbor (capped at k_max)
7. Classify: good (k_i>0), lonely (k_i=0 and not noise), bad (k_i=0 and noise)
8. Generate: good → SMOTE interpolation with k_i neighbors; lonely → copy; bad → skip
9. Descale, return balanced dataset

## File Structure

```
src/
├── __init__.py       # Public API exports
├── smotewb.py        # Core SMOTEWB algorithm
├── resamplers.py     # Unified interface: none/ros/smote/smotewb
├── classifiers.py    # 9 classifier registry
├── data_loader.py    # Pima dataset loading
├── metrics.py        # MCC computation
└── experiment.py     # CV pipeline + ranking
```

## Key Design Decisions

- **AdaBoost**: Custom weight-update loop to extract per-sample weights (sklearn's AdaBoostClassifier does not cleanly expose these)
- **SMOTE**: Custom implementation (no dependency on imbalanced-learn)
- **Scaling**: MinMaxScaler [0,1] applied after resampling, before classifier training
- **MARS**: Try pyearth import; if unavailable, run 8 classifiers with warning
- **R sigma=1 → sklearn gamma=1.0**: kernlab's RBF uses exp(−σ·‖x−y‖²), same as sklearn's gamma

## Expected Runtime

~10–12 minutes on single-core CPU for full experiment (Pima is small: 768 samples, 8 features).
