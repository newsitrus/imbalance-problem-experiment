---
paper_id: sha256_5b435fbf
exported_at: 2026-02-09T11:27:31
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_5b435fbf/report.md/analysis_report.md
format: markdown
---

# Sentiment Analysis of Imbalanced Tourism News Dataset Using Random Forest with Particle Swarm Optimization and Synthetic Minority Oversampling


## Executive Summary

This study addresses sentiment classification of an imbalanced Indonesian tourism news dataset (200 Madura-related articles) by combining Particle Swarm Optimization (PSO) for feature selection, SMOTE for minority-class oversampling, and Random Forest for classification. Preprocessing and TF vectorization are followed by PSO-based feature selection; experiments compare Random Forest+PSO with and without SMOTE across multiple PSO population sizes. Results show Random Forest with PSO (no SMOTE) achieved an average accuracy of 91% (peak 95%), while PSO+SMOTE+RF averaged 83.75% (peak 90%). The authors report that PSO reduced computation time and that SMOTE increased data preparation time and did not improve accuracy in their setup.


## Metadata
- **Authors:** Husni, Arif Muntasa, Vina Angelina Savitri
- **Venue:** N/A
- **Year:** 2024
- **Keywords:** imbalanced tourism news, particle swarm optimization, random forest, sentiment analysis, synthetic minority oversampling

### Abstract
This paper reports the results of research on the classification of Indonesian tourism news texts based on their sentiment using the Random Forest method combined with feature selection and sampling techniques. The dataset comprised lots of tourism news related to Madura, East Java. Feature selection in the classification process was performed using Particle Swarm Optimization (PSO) to identify influential features. Subsequently, the dataset was divided into two parts: training and testing data. In the training data, an imbalance in the number of positive and negative classes led to classification results biased towards the majority class. Therefore, this study employed the Synthetic Minority Oversampling Technique (SMOTE) to balance class numbers in the training data. Following that, the classification process utilized the Random Forest method to determine the accuracy of this study. The obtained results revealed an accuracy rate of $91 \%$ as the average accuracy when using the PSO and Random Forest methods. The PSO feature selection method contributed to accelerating computation time compared to not using feature selection methods.


## Problem Statement
**Problem:** Classification of Indonesian tourism news sentiment is challenged by (1) class imbalance (majority positive, minority negative) that biases classifiers toward majority class, and (2) high computational cost and potential overfitting when using Random Forest on high-dimensional text feature sets; the study addresses balancing minority classes and reducing feature dimensionality to improve accuracy and efficiency.

**Motivation:** Tourism news sentiment influences tourist behavior and destination image; accurate sentiment classification helps stakeholders gauge public perception and inform decision-making. Improving classifier performance on imbalanced tourism news (Madura, East Java) and reducing computation time for text-based models are therefore important for reliable sentiment analysis in this domain.

**Confidence:** high

### Existing Limitations
- Decision tree classifiers (e.g., C4.5) are prone to overfitting, sensitive to noise, and can perform poorly on imbalanced datasets.
- Traditional feature selection methods (information gain, gain ratio, mutual information) do not handle redundancy well, can be computationally expensive on large/high-dimensional datasets, and may select noisy/irrelevant features.
- Random Forest on large feature sets consumes significant computational resources and time.
- Class imbalance leads to biased models favoring the majority class if not addressed.


## Proposed Method

### Random Forest with Particle Swarm Optimization (PSO) feature selection and SMOTE oversampling (Main Method)

**Algorithm Steps:**
1. Preprocess raw news text (case folding, filtering, tokenizing, stop-word removal, stemming).
2. Compute term frequency representations for documents.
3. Run binary PSO to select a subset of features based on fitness (classification score and feature count).
4. Apply SMOTE on the training partition to balance minority class by synthesizing new minority samples (when used).
5. Train Random Forest (n_estimators=200, max_depth=10) on the prepared training data.
6. Evaluate classifier on test data and record accuracy and running time across PSO population settings.

**Parameters:**
- `C1`: 2 - PSO cognitive acceleration constant
- `C2`: 2 - PSO social acceleration constant
- `n_estimator`: 200 - Number of trees in Random Forest
- `Population`: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 - PSO population sizes evaluated in experiments
- `max_depth`: 10 - Maximum tree depth for Random Forest
- `PSO parameters (general)`: population, iterations, c1, c2, w, position, velocity - PSO algorithm parameters (some initialized randomly as described)

**Inputs:** Raw Indonesian tourism news articles about Madura (200 articles), Preprocessed TF feature vectors for each document, Training/testing split of the dataset (not numerically specified)
**Outputs:** Predicted sentiment labels for news articles (positive / negative), Evaluation metrics: accuracy and running time per experiment/population

### Text preprocessing

**Algorithm Steps:**
1. case folding
2. filtering
3. tokenizing
4. stop-word removal
5. stemming

### Term Frequency computation

**Algorithm Steps:**
1. calculate TF variants (binary, tf, 1+log(tf), etc.)
2. construct TF vectors for documents

### Feature selection using PSO

**Algorithm Steps:**
1. initialize PSO parameters (population, velocity, position, c1, c2, w, iterations)
2. evaluate fitness for particles
3. update Pbest and Gbest and particle velocities/positions until convergence
4. select features indicated by final particle positions

### Class balancing using SMOTE

**Algorithm Steps:**
1. compute k-nearest neighbors (Euclidean distance) for minority instances
2. interpolate between minority instances and neighbors to create synthetic samples

### Classification with Random Forest

**Algorithm Steps:**
1. initialize number of trees (n_estimator)
2. build trees using bagging and Gini impurity for splits
3. aggregate tree votes for final class prediction
4. measure accuracy and runtime


## Evaluation

### Metrics
- **Accuracy (PSO + Random Forest):** 91 %
- **Accuracy (PSO + SMOTE + Random Forest) average:** 83.75 %
- **Highest observed accuracy (PSO+RF):** 95 %
- **Highest observed accuracy (PSO+SMOTE+RF):** 90 %

### Datasets
- **Madura tourism news dataset:** 200 articles
  - Source: Manually collected from various online news outlets (including kompas.com); dataset is exclusive and not publicly available

### Baseline Comparisons
- **Random Forest + PSO + SMOTE (experiment 1)**
- **Random Forest + PSO (experiment 2)**
- **C4.5 decision tree (discussed in related work, not used as experimental baseline)**


## Limitations & Future Work

### Limitations
- [Explicit] Dataset is small (200 articles) and was manually collected, not publicly available, limiting reproducibility.
- [Explicit] PSO and SMOTE require additional data preparation time (though testing time impact is small).
- [Explicit] Addition of SMOTE did not improve accuracy in this study; in experiments PSO+RF without SMOTE yielded higher average accuracy than with SMOTE.
- [Implicit] Limited generalizability due to small, region-specific (Madura, East Java) Indonesian-only dataset.
- [Implicit] Manual data collection and labeling potentially introduces bias; labeling procedure not detailed.
- [Implicit] Train/test split and validation methodology (e.g., cross-validation) are not fully specified, reducing clarity on robustness of results.
- [Implicit] Reproducibility limited because dataset is not publicly available and some experimental details (e.g., exact train/test split, PSO iteration count) are not explicitly reported.


## Cross-Reference Validation

### Unreferenced Figures
- 51f341f82328f3d0cc683f57d3c68a59f0944d658aedbadb51ba3efb04250e82.jpg
- 94de1a526c165ea41f7d6877aa57a06e69088a6cfd31999aace88fcf56c76b38.jpg
- 569ff8685d5b6f79180fc4c88fb86c5fb7d5a90d73d41e03284799e6b417fc0d.jpg
- fd93298cc60c465cea705fe4c5bfb5b215614a489a941f967187707a9da983da.jpg
- 52cc4db84023d982c8c00acce0e56e01b1550c2f7c3cec911849414f0c07684b.jpg
- 0d2fb9f16b91b1ccc897e6c0a3407250861cb3a7b5ebf8344258260f45832d82.jpg
- 73b872b0f264e8e815ccf0c81949294d83de720d7b4be87bfc406aaef8ff2564.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/2db119dc03f03b6aab20beb97f767ac015c0a78c94bada2d6426610a339fe53a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/08b826749ab9ba9814fcfc8fe9afa1d778179f4d6329a4df721abf8644a19e9a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/2a32ca088a775cd062c928e59e303e3fdf91353ab372baad4ac761e5ff7beb13.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/23452094a87ecba80e9b80f0ffc368a2814e7ce6b5cb3df2f62e92d63810fcc4.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/47e5b82e29fd82e3a2dbee6582123cc80ba6baab6af6d7320a2d0cbc96776aed.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/3f5cd44b94a3ba0ac99e9ef78a9c51bb12758ad3830468e6581a89ef80fb0e0c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/59757addb85e50c6bc0fc0c5681753fecdd5a0602339c503e98c05ec76fba2fa.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/66aeae0b06531e4ca4893b8027567cd05c0bf423762ad2ef4dba6967a14e7de8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/7488570bb87cee9586bd004f9f5287d1af69c3ad747ca5a8300d1036abef1afc.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/7c46a389f8c08a0010d279f6a7affbb91622b6238e919b8fb3899e6b30f47f80.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/978d1065d93682a1dc271ae954e50bef90854fd84755af88194a28c711227aa5.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/ccd852d75d7d029b46e0644098aa9777be54e90dc271877f3244a170638fcc78.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/f28cfcbea30dfec57881ccbd8ea4ea11e2d16914a208058a9a0fe8d139e20c4a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_5b435fbf/parsed/e00fa4c0dc18d0db2d4820b4396bfc26715fc8614b851ad86c404fc6359a5077.jpg


## Analysis Warnings
- ⚠️ Methodology section not detected
- ⚠️ Evaluation section not detected


---

*Analysis cost: $0.80 (21 images analyzed)*