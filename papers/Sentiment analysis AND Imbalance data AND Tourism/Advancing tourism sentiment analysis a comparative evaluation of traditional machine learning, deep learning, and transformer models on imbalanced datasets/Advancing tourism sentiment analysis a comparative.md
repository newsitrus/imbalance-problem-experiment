---
paper_id: sha256_61dfa6ca
exported_at: 2026-02-09T11:33:09
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_61dfa6ca/report.md/analysis_report.md
format: markdown
---

# Advancing tourism sentiment analysis: a comparative evaluation of traditional machine learning, deep learning, and transformer models on imbalanced datasets


## Executive Summary

This paper evaluates and compares traditional ML (NB, LR, SVM), deep learning (CNN, LSTM, GRU), and transformer-based (RoBERTa pretrained and fine-tuned) approaches for binary sentiment classification on a large (505,980) English TripAdvisor attraction review dataset that is highly imbalanced (82.11% positive). The pipeline includes cleaning, TF-IDF/vectorized and embedding-based modeling, and applying oversampling methods (SMOTE, ADASYN, RandomOverSampler) to training data. Results show deep learning models (LSTM, GRU) outperform traditional methods on fairness and reliability, SMOTE most effectively mitigates class bias for weaker classifiers, and fine-tuned RoBERTa achieves the best overall performance (92.31% accuracy, 95.34% F1, Cohen's kappa 0.7321) at the expense of very high training cost (~110,830 s).


## Metadata
- **Authors:** Sawitree Srianan, Aziz Nanthaamornphong, Chayanon Phucharoen
- **Venue:** N/A
- **Year:** 2025
- **Keywords:** Sentiment analysis, Natural language processing (NLP), Class imbalance, Deep learning, RoBERTa, Online review

### Abstract
Tourism sentiment analysis faces substantial challenges due to class imbalance and the complex linguistic features of user-generated content. This study systematically compares eight sentiment classification models, spanning traditional machine learning (naïve Bayes, support vector machines, logistic regression), deep learning (convolutional neural networks, long short-term memory networks [LSTMs], gated recurrent units [GRUs]), and transformer-based architectures (RoBERTa in two configurations: pretrained and fine-tuned), using a dataset of 505,980 TripAdvisor reviews. We evaluate model performance under imbalanced class conditions and examine the effectiveness of three oversampling techniques—SMOTE, ADASYN, and RandomOverSampler—in mitigating class bias. The results reveal significant performance disparities across architectures. Deep learning models, particularly LSTM (91.06% accuracy, Cohen’s kappa = 0.6846) and GRU (90.82% accuracy, Cohen’s kappa = 0.6781), consistently outperform traditional approaches. Fine-tuned RoBERTa achieved the highest performance, with 92.31% accuracy, a 95.34% F1-score, and Cohen’s kappa = 0.7321. Traditional models showed notable limitations; for example, naïve Bayes exhibited strong majority-class bias, despite an accuracy of 82.35% (Cohen’s kappa = 0.0054). Among the oversampling methods, SMOTE was the most effective in improving the fairness of traditional models, while RoBERTa’s fine-tuning process inherently mitigated class imbalance. A computational analysis highlights key trade-offs: traditional models train quickly but require oversampling, deep learning offers a balanced trade-off between performance and efficiency, and transformer models provide state-of-the-art accuracy at the cost of high computational resources. These findings offer evidence-based guidance for selecting appropriate models for tourism sentiment analysis.


## Problem Statement
**Problem:** How to achieve accurate, fair, and reliable sentiment classification of imbalanced tourism reviews given strong class imbalance and complex linguistic properties of user-generated content.

**Motivation:** Tourism user-generated reviews strongly influence consumer choices and business decisions; however, review datasets are heavily skewed toward positive sentiment, which can bias classifiers and produce unfair or misleading insights—addressing class imbalance and model choice is crucial for reliable, actionable sentiment analysis in tourism.

**Confidence:** high

### Existing Limitations
- Prior reliance on traditional ML methods (e.g., Naïve Bayes) that assume feature independence and perform poorly on imbalanced, complex text.
- Limited empirical comparisons across modern DL and transformer architectures under controlled, identical experimental conditions in the tourism domain.
- Insufficient systematic evaluation of oversampling techniques (SMOTE, ADASYN, RandomOverSampler) across different model families for tourism sentiment data.
- Limited use of transformer-based models (e.g., RoBERTa) specifically evaluated and fine-tuned on tourism review datasets.


## Proposed Method

### Comparative evaluation of traditional ML, deep learning, and RoBERTa transformer models with oversampling for imbalanced tourism sentiment analysis (Main Method)

**Algorithm Steps:**
1. Scrape 505,980 English TripAdvisor attraction reviews and extract text and ratings
2. Clean and preprocess text (lowercase, remove punctuation/stopwords, lemmatize, tokenize)
3. Label reviews: ratings 1–3 = negative, 4–5 = positive
4. Split into 70% train and 30% test (random_state=42)
5. Apply oversampling (SMOTE, ADASYN, RandomOverSampler) to training folds only
6. Train traditional ML models (NB, LR, SVM) with TF-IDF features
7. Train DL models (CNN, LSTM, GRU) on padded word sequences
8. Run RoBERTa pretrained inference and fine-tune RoBERTa on task-specific data
9. Evaluate all models using accuracy, precision, recall, F1, balanced accuracy, Cohen's kappa, confusion matrices, and training time
10. Compare results across architectures and oversampling techniques and analyze trade-offs

**Parameters:**
- `TF-IDF ngram range`: (1, 2) - N-gram range used in TF-IDF vectorization for traditional ML models
- `TF-IDF max features`: 5000 - Maximum number of TF-IDF features
- `random_state (general)`: 42 - Random seed used for reproducibility
- `learning rate (DL)`: 1e-3 - Initial learning rate for DL models (CNN/LSTM/GRU)
- `batch size (DL)`: 64 - Batch size for training DL models
- `number of epochs (DL)`: 5 - Number of training epochs for DL models
- `optimizer (DL)`: Adam - Optimizer used for DL models
- `loss function (DL)`: CrossEntropyLoss - Loss function for DL classification
- `embedding dimension`: 128 - Word embedding dimensionality for DL models
- `hidden dimension (LSTM/GRU)`: 128 - Hidden layer size for LSTM and GRU models
- `RandomOverSampler random_state`: 42 - Random seed for RandomOverSampler
- `ADASYN n_neighbors`: 5 - Number of neighbors for ADASYN
- `ADASYN random_state`: 42 - Random seed for ADASYN
- `SMOTE k_neighbors`: 5 - Number of neighbors for SMOTE
- `SMOTE random_state`: 42 - Random seed for SMOTE
- `RoBERTa base model`: cardiffnlp/twitter-roberta-base-sentiment - Pretrained RoBERTa checkpoint used for fine-tuning and zero-shot inference
- `RoBERTa max input length`: 512 - Maximum token length for RoBERTa inputs
- `RoBERTa batch size`: 64 - Batch size used during RoBERTa fine-tuning
- `RoBERTa learning rate`: 2e-5 - Learning rate used for fine-tuning RoBERTa
- `RoBERTa dropout (encoder/pooler)`: 0.1 - Dropout probability for RoBERTa encoder/pooler layers
- `RoBERTa optimizer`: AdamW - Optimizer used for fine-tuning RoBERTa

**Inputs:** 505,980 English-language TripAdvisor tourist attraction reviews (review text and 1–5 rating) collected 2010–2023 focusing on Bali and Phuket
**Outputs:** Binary sentiment labels (positive/negative) per review, Evaluation results: accuracy, precision, recall, F1-score, balanced accuracy, Cohen's kappa, confusion matrices, and training times for each model and oversampling condition

### Data collection

**Algorithm Steps:**
1. Web-scrape TripAdvisor reviews
2. Filter to English-language reviews
3. Collect review text, rating, year, and metadata

### Data cleaning and preprocessing

**Algorithm Steps:**
1. Remove duplicate rows
2. Remove records with missing text or rating
3. Convert rating to numeric and year to datetime
4. Lowercase, remove punctuation and extra spaces, remove stopwords, lemmatize, tokenize

### Labeling and dataset preparation

**Algorithm Steps:**
1. Map ratings to binary sentiment labels
2. Compute descriptive statistics and class distribution
3. Split dataset into train (70%) and test (30%)

### Handling imbalanced data

**Algorithm Steps:**
1. Apply SMOTE to training set
2. Apply ADASYN to training set
3. Apply RandomOverSampler to training set

### Model implementation and training

**Algorithm Steps:**
1. Vectorize text with TF-IDF for traditional ML models
2. Train CNN, LSTM, GRU with PyTorch on padded sequences
3. Run RoBERTa pretrained inference and fine-tune on labeled dataset

### Evaluation

**Algorithm Steps:**
1. Compute accuracy, precision, recall, F1, balanced accuracy, Cohen’s kappa
2. Produce confusion matrices before and after SMOTE
3. Record and compare training times


## Evaluation

### Metrics
- **Accuracy (RoBERTa fine-tuned):** 92.31 %
- **F1-score (RoBERTa fine-tuned):** 95.34 %
- **Cohen's kappa (RoBERTa fine-tuned):** 0.7321
- **Accuracy (LSTM):** 91.06 %
- **Cohen's kappa (LSTM):** 0.6846
- **Accuracy (GRU):** 90.82 %
- **Cohen's kappa (GRU):** 0.6781
- **Accuracy (Naïve Bayes):** 82.35 %
- **Cohen's kappa (Naïve Bayes):** 0.0054
- **Majority-class baseline accuracy:** 82.11 %
- **Training time (RoBERTa fine-tune):** 110829.76 s
- **Training time (NB):** 0.38 s

### Datasets
- **TripAdvisor tourist attraction reviews (Bali and Phuket):** 505,980 reviews
  - Source: Web-scraped from TripAdvisor; English-language reviews collected between 2010 and 2023 focusing on Bali (Indonesia) and Phuket (Thailand)

### Baseline Comparisons
- **Majority-class baseline**
- **Naïve Bayes (NB)**
- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**


## Limitations & Future Work

### Limitations
- [Explicit] Document-level sentiment classification may obscure mixed sentiments within single reviews.
- [Explicit] Rating-based labeling (1–3 negative, 4–5 positive) can misalign with textual sentiment and conflate neutral with negative.
- [Explicit] Study limited to English-language reviews.
- [Explicit] High computational cost for fine-tuning transformer models (RoBERTa).
- [Implicit] Geographic focus on Bali and Phuket may limit generalizability to other destinations or cultures.
- [Implicit] No manual human-annotated ground truth was used; labels derived solely from numeric ratings (potential label noise).
- [Implicit] Binary sentiment mapping ignores neutral/mixed sentiment granularity.
- [Implicit] Contradictory data availability statement in declarations suggests potential reproducibility/data-sharing limitations.

### Future Work
- Adopt sentence-level sentiment analysis with manual annotation to capture mixed sentiments and enable fine-grained/multi-label modeling.
- Explore lightweight transformer variants (e.g., DistilBERT, TinyBERT) to balance performance and computational cost.
- Extend to multilingual sentiment classification and emotion detection for cross-cultural tourism contexts.
- Investigate multimodal sentiment analysis combining textual, visual, and audio inputs.


## Cross-Reference Validation

### Unreferenced Figures
- e3c8ffe25559ec996fdeb54a599ac6d74b5d14cddbe2df74a1622dd4fe8325cb.jpg
- 1cab10e896582ae8089908c56f6436391795ada1b035edff08c01224c59a20df.jpg
- b3cf9d9f130cbe4118cac984e122a356db7897811054fdebd23748c9d8656b0c.jpg
- 101a77f9b1bfd2f437161910e407062ba39429d93110e7aeabfc3525a4d98a43.jpg
- e747b5094b39c3971583cacc65470181690a1b5400eb89b5f66095f24cbe4219.jpg
- 949581ba7f43371efa5ecf7924c2d7570e2605f86389205b2b39390efb9e8788.jpg
- 4dc60678f0fca04e9304e0662741ff349da662d148bbff753cdba163dca4d9d1.jpg
- 009f855788fa8eaf7266920234b434b49b1a1ee3406a7e5e6ef8e8e85379c0ca.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/57989b65266dc752f2e882cf3780742b9b7956b32348383748484b587fee3b42.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/6680e283548568e4249266e62cf8a174334a0d218fbbf7c7d770e5254880328a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/642ca9941aa363766de775793d28254cbcd8a222fae952d6e69ccd2bfc51ab06.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/76940e86486a70f32199e441d5217ba2aac60a809c3257bdbd6c2cd51b459cc7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/8ad4b1d9ee32efa8a9102a2aeec4f82df272ef507bb79c2341451cfa27ec5658.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/ad01e92a30ef22aae899d2cf340ed1108c8dbc1dbf2aaf13c349e2d416642110.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/b8676965710a25d046cda66925f2e61cba4e39149c924f5ebf95fe40e481c1b0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/cb9591c104b457da96ab01488f71a2efe7ec83676db3d9fa75fcebaef7e6f93f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/ec96a900c9855d145f9f8a16f61301ee697173d7a985ef7aac9102634c3f78c8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/fbfa6e7657948156b73304dbaabe9faf55f23b7c5cb0a36201324270e2c72f70.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61dfa6ca/parsed/f521d6506645ce7fcc747f0025ae5497694d4af5761ed4c69aef3a0411a48282.jpg


---

*Analysis cost: $1.25 (19 images analyzed)*