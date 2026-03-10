---
paper_id: sha256_3c5ec21f
exported_at: 2026-02-22T19:44:06
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_3c5ec21f/report.md/analysis_report.md
format: markdown
---

# A novel sMOTE-based resampling technique trough noise detection and the boosting procedure


## Executive Summary

The paper addresses problems of standard oversampling methods (ROS and SMOTE) which can create noisy synthetic samples and use a fixed neighbor parameter across minority instances. The authors propose SMOTEWB, which uses AdaBoost-derived instance weights for noise detection and computes a per-instance neighbor count (k_i) based on non-noise neighbors and the distance to the first non-noise majority neighbor. 'Good' minority instances are oversampled via SMOTE with adapted k_i, 'lonely' instances are copied, and 'bad' (noisy) instances are not used for synthesis. Experiments on two simulated datasets and 16 benchmark datasets (UCI/KEEL) using multiple classifiers show SMOTEWB yields better average MCC ranks than ROS and SMOTE, with overall statistical significance reported (pairwise p < 0.0001).


## Metadata
- **Authors:** Fatih Salam, Mehmet Ali Cengiz
- **Venue:** Expert Systems with Applications
- **Year:** 2022
- **Keywords:** Oversampling, SMOTE, Class imbalance, Noisy data

### Abstract
Keywords: Oversampling SMOTE Class imbalance Noisy data

Mos o theclafication methoassue that he nuber  ca bservaions a balanc. In sue, me r preic ias whheeas wimostnsThereorehe eas wi alu satins nhemaori smak a recns. Th some advised performance measures to be used in datasets, as well as recommended approaches to solve class imbalance problem. One of the most widely used methods is resampling method. In this study, the difficulties relevant to random oversampling (ROS) and synthetic minority oversampling technique (SMOTE), which are some of the oversampling methods, are discussed. This study aims to propose a combination of a new noise detection method and sMOTE to overcome those difficulties. Using the boosting procedure in ensemble algorithms, noise detection is possible with the proposed SMOTE with boosting (SMOTEWB) method, which makes us o this information to determine the appropriate number of neighbors or each observation within SMTE algorithm.


## Problem Statement
**Problem:** Standard oversampling methods (ROS, SMOTE) for class-imbalanced classification can generate noise and use a fixed neighborhood parameter (k) for all minority instances, which can harm classifier performance. The paper addresses detecting noisy observations and adapting neighborhood sizes per instance to avoid creating synthetic samples that lie in majority regions.

**Motivation:** Class imbalance is common in real datasets and causes classifiers to be biased toward the majority class, overlooking minority instances. SMOTE and ROS are widely used but can create noisy/invalid synthetic samples and exacerbate overfitting; improving resampling quality (especially in presence of noise) can yield better predictive performance on minority class and overall robustness.

**Confidence:** high

### Existing Limitations
- Random oversampling (ROS) duplicates minority examples causing overfitting and very specific classifier rules.
- SMOTE can generate synthetic samples in majority-class regions, producing noise.
- SMOTE uses a fixed k (number of neighbors) for all minority samples; k does not adapt per-instance.
- SMOTE does not consider the quality (safety/noisiness) of neighbors when generating synthetic data.


## Proposed Method

### SMOTEWB (SMOTE with boosting) (Main Method)

**Algorithm Steps:**
1. Scale features to [0,1].
2. Run AdaBoost for M rounds to obtain final instance weights W^M.
3. Compute class-specific noise thresholds T_pos = (1/n) * (2 * n_neg / n) and T_neg = (1/n) * (2 * n_pos / n) and label instances as noise/not-noise.
4. Compute k_max = floor(n_neg / n_pos).
5. For each positive instance, determine k_i by counting positive non-noise neighbors until the first non-noise negative neighbor; label instance as 'good' (k_i>0), 'lonely' (k_i=0 and not-noise), or 'bad' (k_i=0 and noise).
6. Generate synthetic samples: 'good' instances use adapted SMOTE with k_i neighbors; 'lonely' instances are copied; 'bad' instances are skipped.
7. Descale features back and output balanced dataset X_new.

**Pseudocode:**
```
Algorithm (summary):
1. Scale X to [0,1].
2. Run AdaBoost(X,Y,M) to get final weights W.
3. Split W into W_pos and W_neg. Compute thresholds T_pos and T_neg. Label instances as noise if weight > class threshold (per Eqns (2),(3)).
4. k_max := floor(n_neg / n_pos).
5. For each positive instance x_i: find up to k_max nearest non-noise positive neighbors; find nearest non-noise negative neighbor; set k_i := number of positive neighbors before first negative; label as 'good'(k_i>0), 'lonely'(k_i==0 & not-noise), 'bad'(k_i==0 & noise).
6. For each positive instance:
   if 'good': select random neighbor among k_i and create x_syn = x_i + (neighbor - x_i) * lambda, append to X_syn;
   if 'lonely': copy x_i to X_syn;
   if 'bad': do nothing.
7. X_new := bind rows of X and X_syn; descale; output X_new.
```

**Parameters:**
- `k_max`: floor(n_neg / n_pos) - Maximum number of nearest neighbors to consider on average; determines upper bound of synthetic samples per positive instance.
- `M`: user-specified (number of AdaBoost iterations) - Number of boosting rounds used to compute instance weights for noise detection.
- `T_pos`: 2 * n_neg / n^2 - Threshold for labeling positive-class instances as noisy based on AdaBoost weight and imbalance ratio.
- `T_neg`: 2 * n_pos / n^2 - Threshold for labeling negative-class instances as noisy.
- `lambda`: Uniform(0,1) - Random interpolation factor used when generating a synthetic sample between an instance and a selected neighbor.
- `k_id`: Random integer in [1, k_i] - Index of randomly selected neighbor among k_i neighbors for interpolation.

**Inputs:** Training features X (numeric, scaled to [0,1]), Training labels Y (binary: positive/minority, negative/majority), Number of boosting iterations M
**Outputs:** X_new: resampled (balanced) training dataset containing original and synthetic minority samples, Noise labels per instance (noise / not noise), Per-instance neighbor counts k_i and instance classification (good/lonely/bad)

### Noise detection via boosting

**Algorithm Steps:**
1. Run AdaBoost for M iterations on the training set to update instance weights.
2. Split weights by class (positive and negative).
3. Compute class-specific thresholds T_pos and T_neg and label instances as 'noise' if weight exceeds threshold (or below, per class scheme).

### Determine per-instance neighbor counts

**Algorithm Steps:**
1. Compute k_max = floor(n_neg / n_pos).
2. For each positive instance, sort distances to non-noise positive neighbors up to k_max and find the position of the first non-noise negative neighbor.
3. Set k_i equal to number of positive neighbors before first non-noise negative; label instance 'good' if k_i>0, 'lonely' if k_i==0 and not noise, 'bad' if k_i==0 and noisy.

### Synthetic sample generation (sMOTE with adapted neighbors)

**Algorithm Steps:**
1. For each 'good' instance, randomly select one of the k_i neighbors and generate x_syn = x_i + (neighbor - x_i) * lambda where lambda ~ Uniform(0,1).
2. For each 'lonely' instance, copy the instance directly (no interpolation).
3. Do not create synthetic samples from 'bad' instances.


## Evaluation

### Metrics
- **Matthews Correlation Coefficient (MCC):** Average rank: No Resampling 2.727, ROS 2.541, SMOTE 2.414, SMOTEWB 2.317 average rank (lower is better as reported in paper)

### Datasets
- **Simulation linear (synthetic):** n_neg=1000, n_pos=50 (plus noise added: 15 positives flipped)
  - Source: simulated by authors
- **Simulation non-linear (synthetic):** 1000 samples (approx 1:21 imbalance)
  - Source: simulated by authors
- **Yeast4:** n=1484
  - Source: KEEL repository
- **Banana:** n=2640
  - Source: UCI repository
- **Pima Indians Diabetes:** n=768
  - Source: UCI repository
- **Ionosphere:** n=351
  - Source: UCI repository
- **Complete set of 16 benchmark datasets (incl. Ecoli, Cleveland, Statlog, Glass0, Seeds, Vehicle.* etc.):** various (see Table 1 in paper)
  - Source: KEEL and UCI repositories

### Baseline Comparisons
- **No Resampling**
- **ROS (Random Oversampling)**
- **SMOTE (original)**


## Limitations & Future Work

### Limitations
- [Explicit] The proposed method aims to overcome only the problems caused by SMOTE; other classification issues (e.g., variable selection) are not addressed.
- [Explicit] Thresholds for noise detection (T_pos,T_neg) are heuristic and not absolute; the authors note it is not possible to specify an absolute threshold.
- [Explicit] Classifier parameters were not optimized in experiments (authors used typical/default R package parameters), so reported performance does not represent fully optimized models.
- [Implicit] Experiments exclude categorical features (authors removed categorical predictors) — method demonstrated only on numeric features.
- [Implicit] Computational cost of running boosting + neighbor searches for large datasets is not analyzed.
- [Implicit] Evaluation is limited to a set of UCI/KEEL datasets and two synthetic simulations; generalization to large-scale or domain-specific data is untested.
- [Implicit] Threshold formulas and k_max heuristic may be sensitive to extreme imbalance ratios; robustness across very high imbalance is not thoroughly explored.

### Future Work
- Modify the boosting + noise detection procedure to operate on the majority class and design a corresponding undersampling method.
- Combine majority-class and minority-class adaptations to form a hybrid resampling method.
- Explore further adaptations of SMOTEWB as undersampling and hybrid methods and investigate parameter tuning and scalability.


## Cross-Reference Validation

### Unreferenced Figures
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/25d996b783d9da1b5716777a4c13682fec4fd4a531c9e2b6316818e0757fed12.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/0b2814fa1de2091deb71aaab767b4d4068837c4f69db96a9fb3f4d9dc3650a3d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/2e655aa9b807821a4f80eaee485bd064ca4435cc534dc5ffcc5be6dce9da2f76.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/2b00a8a75f73e29ed33543471a0a5b8737b32d9bdeca0aaf508d7ef7b83725e0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/448142d5e7c923e6eeaeccf4ab0767502250a7b3d12c59e8895b7da01d6ac37f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/556b6e525df90963d55bcd22afa84ed6bc4e0502dfbe168e91d61553ec51333d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/60f9c9e005796aa8b091da2edc4d288a8b1afb956940c13eb036a30a034acc38.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/761ed9f6a51e4dca650d40d273072f94a6085bbae2ea282da916a6abe879ec95.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/74fead0f33f3b418762f7067d498594c223ce15c2f4d59f6fdc3834d741fa209.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/7db2197336a7c6f246131e3888732b4ce8b6a032a839a6d1d1a9f8f8d0b192b2.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/813f4d50242e7df8295f16ad2ef6ca4508e735085179133481d7852c8db0aba6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/7710eb68a9477d9df26e7df4bfebf88bd55408d3e1f0caf4259a226ffe2b47b2.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/8255ea141be21f5d33f06368be7a9e8d6772b263c88cd11581bbc1701b7859e7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/9a3c429e6676a3f6587ade8810505c8cf4f275198633ae80008d3c4bf943b1db.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/d13c00f25d6071d9b3c2835bccb5cdc0c1c574a6510978d511217b45a0c2f41e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/a759e3e982e64abcd8359d37e3fc19bb83f4f7e6dcb84b9f47dc1e9a2bca78b8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/dba7ee2c61263e9a37ba3e79228f3cdbed8f2c5d3ca704c292416be49f9ff924.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/a1be7208d41a2d5cf5231b7f2e31e7c676d4d9a458edd99ac093a9f08335828c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/dcc3d98d1b3d27b5264e2c1d285021b4f7b32b34697d2817b94438c05b84d661.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/dc3aa4cbe8b15a61e9c616b721f20dc56556e20002acdf937434cc12b0149faf.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_3c5ec21f/parsed/9ede060ee078d169085091b74684cdf958a3a29f6294fd5bb6daf477588cac82.jpg

*5 figure references validated as consistent.*


## Analysis Warnings
- ⚠️ Evaluation section not detected
- ⚠️ Methodology section not detected


---

*Analysis cost: $1.03 (26 images analyzed)*