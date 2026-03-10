# Summary List: Sentiment Analysis of Customers' Reviews Using a Hybrid Evolutionary SVM-Based Approach in an Imbalanced Data Distribution

## Data Preprocessing
- Remove symbols, special characters, non-Arabic letters, emoticons
- Arabic stopword removal (e.g., انا، ثم، جدا، اذا، الذي، قد، هذا)
- Text normalization
- Stemming using Lovins stemmer

## Feature Preprocessing
- 4 tokenization variants producing 4 dataset versions:
  - Data 1: 1-gram → 3,439 features
  - Data 2: 2-gram → 8,985 features
  - Data 3: 3-gram → 14,233 features
  - Data 4: Bag-of-Words → 2,916 features
- PSO-based per-feature weighting (continuous weights multiplied element-wise to feature vectors)
- 4 oversampling techniques with PSO-optimized k parameter:
  - SMOTE
  - SVM-SMOTE
  - ADASYN
  - Borderline-SMOTE

## Model
- **Main model:** PSO-SVM — PSO optimizes feature weight vector + oversampling k, SVM classifies
  - PSO: 100 particles, 100 iterations, 30 runs, fitness function = G-mean
- **Baselines:** SVM, XGBoost, Decision Tree, Random Forest, Naive Bayes, k-NN, Logistic Regression
- **External comparisons:** TF-IDF + Bidirectional LSTM, TF-IDF + GBDT

## Train-Test Split
- Not explicitly specified in the paper
- Standard train/test hold-out split (no cross-validation mentioned)
- Oversampling applied to training set; optimized weights + k applied to test set

## Evaluation Metrics
- Accuracy
- F1-score (positive class) — F1P
- F1-score (negative class) — F1N
- G-mean (primary metric for imbalanced evaluation) = sqrt(Recall_pos × Recall_neg)
- AUC (Area Under ROC Curve)

## Estimated Time (RTX 2060)
- Dataset preparation (using OCLAR substitute): 1–2 hours
- Text preprocessing (Arabic NLP): 2–3 hours
- Feature extraction (4 tokenization variants): 1–2 hours
- PSO-SVM implementation (100 particles × 100 iters × 30 runs × 4 datasets × 4 oversampling): 8–16 hours
- 7 baseline classifiers: 2–4 hours
- Deep learning baselines (BiLSTM, GBDT): 2–3 hours
- Visualization & analysis: 2–3 hours
- In total: ~3–5 days

## Dataset Description
| Dataset | Size | Positive | Negative | Language | Source | Public? |
|---------|------|----------|----------|----------|--------|---------|
| Jeeran (Our-data) | 2,790 | 2,150 (77%) | 640 (23%) | Arabic (Jordanian) | Crawled from jeeran.com via C# script, labeled by 290 crowdsource annotators | No |
| Public-data | 3,916 | 3,465 (88.5%) | 451 (11.5%) | Arabic | Referenced in paper Table 1 | Unclear |
| OCLAR (comparison) | 3,916 | Mixed | Mixed | Arabic (Lebanese) | UCI ML Repository | Yes — downloaded to `datasets/` |
