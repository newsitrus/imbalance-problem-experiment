# Implementation Plan: Traditional ML Only
## Tourism Sentiment Analysis (Simplified Version)

**Scope:** Naive Bayes, Logistic Regression, SVM with oversampling techniques

---

## 1. Overview

### Models to Implement
| Model | Description |
|-------|-------------|
| Naive Bayes (NB) | MultinomialNB with TF-IDF features |
| Logistic Regression (LR) | Linear classifier with L2 regularization |
| Support Vector Machine (SVM) | Linear kernel SVM |

### Oversampling Methods
| Method | Description |
|--------|-------------|
| None | Baseline (imbalanced) |
| SMOTE | Synthetic minority oversampling |
| ADASYN | Adaptive synthetic sampling |
| RandomOverSampler | Duplicate minority samples |

### Total Experiments: 12
- 3 models × 4 oversampling conditions = 12 experiments

### Estimated Total Time: ~15-20 minutes

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  Raw Data                                                   │
│      ↓                                                      │
│  Data Cleaning (remove duplicates, missing values)          │
│      ↓                                                      │
│  Text Preprocessing (lowercase, remove punct, lemmatize)    │
│      ↓                                                      │
│  Sentiment Labeling (1-3→Negative, 4-5→Positive)           │
│      ↓                                                      │
│  Train/Test Split (70/30, stratified)                       │
│      ↓                                                      │
│  TF-IDF Vectorization (ngram=1-2, max_features=5000)       │
│      ↓                                                      │
│  Oversampling (SMOTE/ADASYN/ROS) - Training set only       │
│      ↓                                                      │
│  Model Training (NB, LR, SVM)                              │
│      ↓                                                      │
│  Evaluation (Accuracy, F1, Kappa, Balanced Accuracy)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Expected Results (Target from Paper)

### Baseline (No Oversampling)
| Model | Accuracy | F1-Score | Cohen's Kappa | Balanced Acc |
|-------|----------|----------|---------------|--------------|
| NB | 82.35% | 90.24% | 0.0054 | 50.17% |
| LR | 90.25% | 94.28% | 0.6150 | 76.73% |
| SVM | 83.40% | 93.07% | 0.1041 | 53.31% |

### With SMOTE
| Model | Accuracy | F1-Score | Cohen's Kappa | Balanced Acc |
|-------|----------|----------|---------------|--------------|
| NB | 86.08% | 88.67% | 0.7692 | 86.67% |
| LR | 93.47% | 94.31% | 0.8237 | 89.73% |
| SVM | 95.42% | 96.08% | 0.8464 | 92.62% |

---

## 4. Configuration Parameters

```yaml
# Data Split
train_ratio: 0.7
test_ratio: 0.3
random_state: 42

# TF-IDF
ngram_range: (1, 2)
max_features: 5000

# Oversampling
smote_k_neighbors: 5
adasyn_n_neighbors: 5
random_state: 42
```

---

## 5. Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Positive prediction quality |
| Recall | TP/(TP+FN) | Positive detection rate |
| F1-Score | 2×(P×R)/(P+R) | Balance of P and R |
| Balanced Accuracy | (TPR+TNR)/2 | Fairness across classes |
| Cohen's Kappa | (p₀-pₑ)/(1-pₑ) | Agreement beyond chance |

---

## 6. Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn nltk pyyaml

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run experiments
python run_ml_experiments.py --data path/to/data.csv
```

---

## 7. Output Files

```
outputs/
├── results/
│   └── ml_results_YYYYMMDD_HHMMSS.csv
├── figures/
│   ├── confusion_matrices.png
│   ├── performance_comparison.png
│   └── oversampling_effect.png
└── models/
    ├── nb_baseline.pkl
    ├── nb_smote.pkl
    ├── lr_baseline.pkl
    └── ...
```
