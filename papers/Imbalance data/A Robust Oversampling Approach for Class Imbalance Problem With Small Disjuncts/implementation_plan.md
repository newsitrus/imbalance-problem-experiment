# Implementation Plan: DROS Replication

## Scope
- **Method:** DROS (Disjuncts-Robust Oversampling) — Algorithms 1–4 from the paper
- **Classifiers:** SVM (Gaussian/RBF kernel) + NN (1 hidden layer, 10 nodes)
- **Dataset:** 1 UCI dataset (user's choice from downloaded datasets)
- **Evaluation:** Stratified 5-fold CV × 10 repeats = 50 runs
- **Metrics:** Precision, Recall, F-measure, G-mean, AUC
- **Estimated time:** ~10–15 hours

---

## File Structure

```
project/
├── dros.py                  # DROS algorithm (Algorithms 1-4)
├── experiment.py            # Main experiment runner
├── metrics.py               # Evaluation metrics
├── data_loader.py           # Dataset loading + z-score + class selection
└── results/
    └── <dataset_name>/      # Output results per dataset
```

---

## Step 1: Data Loading & Preprocessing (`data_loader.py`)

**Reference:** Section 4.2, Table 1

```python
# Pseudocode
def load_dataset(name, minority_label, majority_label):
    # 1. Load raw data from UCI zip (use pandas/scipy)
    # 2. Select two classes by label (Table 1 "Labels of selected classes")
    #    - If "-", dataset is already binary
    #    - Otherwise, filter rows matching minority_label and majority_label
    # 3. Separate features (X) and target (y)
    # 4. Ensure all features are numerical
    # 5. Return X, y (minority=1, majority=0)
```

**Z-score standardization** (Section 4.2: "pre-processed by the standardized z-scores"):
```python
# Applied per-fold on training data; transform test data with training stats
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Important:** Standardize AFTER train-test split to avoid data leakage.

---

## Step 2: DROS Algorithm (`dros.py`)

### 2.1 Main Function — Algorithm 1 (p.6)

```python
def dros(S_maj, S_min, rho=0.5, k=7, delta=-0.7660, g=1.0):
    """
    Input:
        S_maj: np.ndarray, shape (|S_maj|, H) — majority training samples
        S_min: np.ndarray, shape (|S_min|, H) — minority training samples
        rho:   float, 0.5 — light-cone aperture parameter (Eq. 1)
        k:     int, 7 — number of nearest majority neighbors (Eq. 8)
        delta: float, -0.7660 — direct-interlinked threshold (Eq. 14)
        g:     float, 1.0 — boundary proximity control (Eq. 25)
    Output:
        S_new: np.ndarray, shape (|S_maj|-|S_min|, H) — synthetic samples
    """
    # Step 1: Compute relationships
    I = relationships(S_maj, S_min, delta)

    # Step 2: Compute light-cone structures
    S2 = structures(S_maj, S_min, rho, k, I)

    # Step 3: Generate synthetic samples
    S_new = data_generation(len(S_maj), len(S_min), S2, rho, g)

    return S_new
```

### 2.2 Relationships — Algorithm 2 (p.6)

**Reference:** Equations 12–14

```python
def relationships(S_maj, S_min, delta):
    """
    Compute direct-interlinked relationship matrix between minority pairs.

    Output: I — symmetric matrix, shape (|S_min|, |S_min|)
            I[i][j] = 1 if direct-interlinked, else 0
    """
    n_min = len(S_min)
    n_maj = len(S_maj)
    I = np.zeros((n_min, n_min), dtype=int)

    for i in range(n_min - 1):
        for j in range(i + 1, n_min):
            # Compute D_k for all majority points (Eq. 12)
            min_D = np.inf
            for kk in range(n_maj):
                z_k = S_maj[kk]
                diff_i = S_min[i] - z_k
                diff_j = S_min[j] - z_k
                norm_i = np.linalg.norm(diff_i)
                norm_j = np.linalg.norm(diff_j)

                if norm_i == 0 or norm_j == 0:
                    D_k = 0.0
                else:
                    D_k = np.dot(diff_i / norm_i, diff_j / norm_j)

                min_D = min(min_D, D_k)  # Eq. 13

            # Eq. 14
            if min_D >= delta:
                I[i][j] = 1
                I[j][i] = 1

    return I
```

**Optimization note:** The triple nested loop is O(|S_min|² × |S_maj| × H). For small UCI datasets this is tractable. Can vectorize the inner loop over majority points using NumPy broadcasting if needed.

### 2.3 Structures — Algorithm 3 (p.6)

**Reference:** Equations 8–11, 15–23

```python
def structures(S_maj, S_min, rho, k, I):
    """
    Compute light-cone structure for each minority sample.

    Output: S2 — list of dicts, each: {'a': base unit vector,
                                        'v': vertex,
                                        'r': radius,
                                        'rho': rho}
    """
    from sklearn.neighbors import NearestNeighbors
    n_min = len(S_min)
    n_maj = len(S_maj)
    S2 = []

    # Precompute k nearest majority neighbors for all minority points
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(S_maj)

    for i in range(n_min):
        x_i = S_min[i]

        # --- Base unit vector (Eqs. 8-11) ---
        distances, indices = knn.kneighbors(x_i.reshape(1, -1))
        S_knn = S_maj[indices[0]]           # k nearest majority neighbors (Eq. 8)
        z_bar = np.mean(S_knn, axis=0)      # mean center (Eq. 9)

        diff = x_i - z_bar
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0:
            continue  # improper: a == 0

        c = diff / norm_diff                # Eq. 10
        a = -c                              # Eq. 11

        # --- Vertex (Eqs. 15-21) ---
        sum_positive_p = 0.0
        count_positive = 0

        for j in range(n_min):
            if I[i][j] != 1:
                continue
            d_j = S_min[j] - x_i            # Eq. 16
            p_j = np.dot(d_j, c)            # Eq. 18
            if p_j > 0:                     # Eq. 20
                sum_positive_p += p_j
                count_positive += 1

        if count_positive == 0:
            continue  # improper: no interlinked points with positive projection

        p_bar = sum_positive_p / count_positive  # Eq. 19
        v = p_bar * c + x_i                      # Eq. 21

        # --- Radius (Eqs. 22-23) ---
        S1 = []  # distances from vertex to illuminated majority points
        for kk in range(n_maj):
            z_k = S_maj[kk]
            diff_zv = z_k - v
            norm_zv = np.linalg.norm(diff_zv)
            if norm_zv == 0:
                continue
            # Eq. 22: check if majority point is illuminated
            inner = np.dot(diff_zv / norm_zv, c)
            if inner >= rho:
                L_v_zk = norm_zv
                S1.append(L_v_zk)

        if len(S1) == 0:
            continue  # improper: no illuminated majority points

        L_v_g = min(S1)                          # nearest illuminated majority
        L_v_x = np.linalg.norm(v - x_i)         # distance vertex to seed

        # Eq. 23
        r = min(L_v_x, L_v_g) + 0.5 * (L_v_g - L_v_x)

        if r <= 0:
            continue  # improper: non-positive radius

        # Valid light-cone
        S2.append({'a': a, 'v': v, 'r': r, 'rho': rho})

    return S2
```

### 2.4 Data Generation — Algorithm 4 (p.7)

**Reference:** Equations 24–25

```python
def data_generation(n_maj, n_min, S2, rho, g):
    """
    Generate |S_maj| - |S_min| synthetic samples from valid light-cones.

    Output: S_new — np.ndarray, shape (n_maj - n_min, H)
    """
    n_generate = n_maj - n_min
    if len(S2) == 0 or n_generate <= 0:
        return np.empty((0, len(S2[0]['a']))) if S2 else np.empty((0, 0))

    H = len(S2[0]['a'])  # dimensionality
    S_new = []

    for _ in range(n_generate):
        # Randomly pick a light-cone structure
        s = S2[np.random.randint(len(S2))]
        a = s['a']
        v = s['v']
        r = s['r']

        # Eq. 25: random scalar in [g, 1]
        xi = g + np.random.rand() * (1 - g)

        # Generate random unit vector d satisfying <d, a> >= rho
        d_vec = generate_valid_direction(a, rho, H)

        # Eq. 24
        new_sample = v + (xi * r) * d_vec
        S_new.append(new_sample)

    return np.array(S_new)


def generate_valid_direction(a, rho, H):
    """
    Generate a random unit vector d such that <d, a> >= rho.

    Paper approach (Section 3.4): generate random unit vector, add to a,
    normalize, repeat N1 times until constraint satisfied.
    Pre-generate N2 valid vectors, then pick one randomly.
    """
    # Strategy from paper: iteratively try until valid
    max_attempts = 1000  # N1 in paper
    for _ in range(max_attempts):
        # Generate random unit vector
        rand_vec = np.random.randn(H)
        # Add to a, then normalize (paper's approach for high dimensions)
        candidate = rand_vec + a
        norm = np.linalg.norm(candidate)
        if norm == 0:
            continue
        d_vec = candidate / norm

        if np.dot(d_vec, a) >= rho:
            return d_vec

    # Fallback: return a itself (guaranteed to satisfy <a, a> = 1 >= rho)
    return a.copy()
```

---

## Step 3: Classifiers (`experiment.py`)

**Reference:** Section 4.2

### 3.1 SVM — Matching MATLAB `fitcsvm` defaults

```python
from sklearn.svm import SVC

# MATLAB fitcsvm with 'Gaussian' kernel defaults:
# - KernelFunction: 'gaussian' (RBF)
# - KernelScale: auto (uses heuristic based on data)
# - BoxConstraint: 1
# - Standardize: false (we already z-score externally)
svm_clf = SVC(
    kernel='rbf',
    gamma='scale',   # sklearn default, similar to MATLAB auto KernelScale
    C=1.0,
    random_state=42
)
```

### 3.2 NN — Matching MATLAB `patternnet(10)` defaults

```python
from sklearn.neural_network import MLPClassifier

# MATLAB patternnet(10) defaults:
# - 1 hidden layer, 10 nodes
# - Hidden activation: 'tansig' (equivalent to 'tanh')
# - Output activation: 'softmax'
# - Training: scaled conjugate gradient (closest sklearn: 'lbfgs' or 'adam')
# - Max epochs: 1000
nn_clf = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='tanh',        # tansig equivalent
    solver='lbfgs',           # closest to MATLAB's trainscg
    max_iter=1000,
    random_state=42
)
```

---

## Step 4: Experiment Pipeline (`experiment.py`)

**Reference:** Section 4.2

```python
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_experiment(X, y, classifier_name='svm', n_splits=5, n_repeats=10):
    """
    Stratified 5-fold CV repeated 10 times = 50 runs.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42
    )

    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Z-score standardization (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Split into majority/minority
        S_min = X_train[y_train == 1]  # minority
        S_maj = X_train[y_train == 0]  # majority

        # DROS oversampling
        S_new = dros(S_maj, S_min, rho=0.5, k=7, delta=-0.7660, g=1.0)

        # Combine: original training data + synthetic minority samples
        if len(S_new) > 0:
            X_train_aug = np.vstack([X_train, S_new])
            y_train_aug = np.concatenate([
                y_train,
                np.ones(len(S_new), dtype=int)  # synthetic = minority class
            ])
        else:
            X_train_aug = X_train
            y_train_aug = y_train

        # Train classifier
        clf = get_classifier(classifier_name)
        clf.fit(X_train_aug, y_train_aug)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        y_score = get_scores(clf, X_test)  # for AUC
        fold_metrics = compute_metrics(y_test, y_pred, y_score)
        all_results.append(fold_metrics)

    # Average over 50 runs
    return aggregate_results(all_results)
```

---

## Step 5: Evaluation Metrics (`metrics.py`)

**Reference:** Section 4.1, Equation 26

```python
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute paper's metrics. Minority class = positive (1).
    """
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # Eq. 26
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f_measure = (2 * recall * precision / (recall + precision)
                 if (recall + precision) > 0 else 0.0)
    g_mean    = np.sqrt((TP / (TP + FN)) * (TN / (TN + FP))
                 ) if (TP + FN) > 0 and (TN + FP) > 0 else 0.0

    auc = roc_auc_score(y_true, y_score) if y_score is not None else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f_measure': f_measure,
        'g_mean': g_mean,
        'auc': auc
    }
```

---

## Step 6: Reporting

For each classifier (SVM, NN), report:
```
Metric     | Mean ± Std (over 50 runs)
-----------|---------------------------
Precision  | x.xxxx ± x.xxxx
Recall     | x.xxxx ± x.xxxx
F-measure  | x.xxxx ± x.xxxx
G-mean     | x.xxxx ± x.xxxx
AUC        | x.xxxx ± x.xxxx
```

Match Table 2 format from the paper.

---

## Implementation Order

| # | Task | Est. Time | Dependencies |
|---|------|-----------|-------------|
| 1 | `data_loader.py` — load + z-score + class split | 1 hr | None |
| 2 | `dros.py` — Algorithm 2 (`relationships`) | 1–1.5 hr | None |
| 3 | `dros.py` — Algorithm 3 (`structures`) | 1.5–2 hr | Step 2 |
| 4 | `dros.py` — Algorithm 4 (`data_generation` + `generate_valid_direction`) | 1–1.5 hr | Step 3 |
| 5 | `dros.py` — Algorithm 1 (`dros` main) | 0.5 hr | Steps 2–4 |
| 6 | `metrics.py` — Eq. 26 metrics | 0.5 hr | None |
| 7 | `experiment.py` — SVM + NN + CV pipeline | 1.5–2 hr | Steps 1, 5, 6 |
| 8 | Debugging + validation on small data | 2–3 hr | Steps 1–7 |
| 9 | Run full experiment (1 dataset, 50 runs, 2 classifiers) | <1 hr | Step 8 |
| **Total** | | **~10–13 hr** | |

---

## Key Parameters (Paper Section 4.2)

```
DROS:
  rho   = 0.5
  delta = -0.7660
  k     = 7
  g     = 1.0

SVM:
  kernel = 'rbf' (Gaussian)
  C      = 1.0 (MATLAB default)
  gamma  = 'scale' (MATLAB auto KernelScale equivalent)

NN:
  hidden_layers = (10,)
  activation    = 'tanh' (MATLAB tansig)
  output        = softmax (handled internally by sklearn for classification)
  solver        = 'lbfgs' (closest to MATLAB trainscg)
  max_iter      = 1000

CV:
  n_splits  = 5 (stratified)
  n_repeats = 10
  total runs = 50

Distance metric: Euclidean (for all kNN operations)
```

---

## Dataset Selection Guide

Any downloaded UCI dataset works. Suggested starting point based on small size (fast iteration):

| Dataset | Dims | Samples | IR | Good for |
|---------|------|---------|-----|----------|
| Iris setosa | 4 | 150 | 2.0 | Quick sanity check |
| Ecoli_{4-2} | 7 | 112 | 2.2 | Small, low-dim |
| Glass_{7-2} | 9 | 105 | 2.6 | Small, low-dim |
| Haberman (Survival) | 3 | 306 | 2.8 | Low-dim, moderate size |
| Diabetes | 8 | 768 | 1.9 | Medium, well-known |
