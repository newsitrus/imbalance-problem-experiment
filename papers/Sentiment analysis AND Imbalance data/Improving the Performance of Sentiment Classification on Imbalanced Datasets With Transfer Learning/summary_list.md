# Summary List: Improving the Performance of Sentiment Classification on Imbalanced Datasets With Transfer Learning

## Data Preprocessing
- Remove reviews with score = 3 (ambiguous sentiment)
- Remove duplicate/redundant reviews
- Truncate/pad to max 60 words (sentence-level classification)
- Ratings 1–2 = negative (minority), ratings 4–5 = positive (majority)
- Minority class augmentation: randomly concatenate 2 negative reviews, remove random words, scramble order (2,154 → 6,000 negative samples)

## Feature Preprocessing
- Word2vec embeddings (skip-gram, 200-dim, trained on Chinese Wikipedia)
- CNN-static mode: word vectors frozen during classification training
- No TF-IDF or n-gram bag-of-words — uses raw embedding sequences

## Model
- **Main model:** TL-based-US (Transfer Learning-based Under-Sampling) with dual-convolution TextCNN framework
  - Step 1: Pre-train TextCNN on balanced source domain (MinChnCorp) → model1
  - Step 2: Fine-tune model1 on balanced mini-subset of target domain → model2 (freeze E+C layers, tune H+O)
  - Step 3: Use model2 to classify majority class; misclassified positives become under-sampled set (Tag-neg1)
  - Step 4: Fine-tune model1 on Tag-neg1 + negatives with dual convolution (transferred filters 2,3,4,5 + new filters 5,6,7) → model3
  - Step 5: Two-stage prediction: model2 on test → Tag-pos2/Tag-neg2; model3 on Tag-neg2 → Tag-pos3/Tag-neg3; final = Tag-pos2 + Tag-pos3 + Tag-neg3
- **Baselines:** Non-US (TextCNN), Random-US, NearMiss-US, RENN-US, BalanceCascade, Multi-model Fusion (NB+DT+LR+SVM)

## Train-Test Split
- 1/10 of target domain as test set (Td-test), rest as development set
- 5-fold cross-validation on development set for hyperparameter optimization
- Final evaluation on Td-test

## Evaluation Metrics
- Precision (minority class)
- Recall (minority class)
- F1-value (minority class) — primary metric

## Estimated Time (single-core CPU + RTX 2060 GPU)
- Word2vec training on Chinese Wikipedia (200-dim, skip-gram): 4–8 hours
- Data preprocessing & minority augmentation: 1–2 hours
- Step 1 — Pre-train TextCNN on source domain (53K samples): 1–2 hours
- Step 2 — Fine-tune on balanced mini target subset: 15–30 minutes
- Step 3 — TL-based under-sampling (inference on 125K majority samples): 10–20 minutes
- Step 4 — Dual-convolution fine-tuning on under-sampled set: 30 minutes–1 hour
- Step 5 — Two-stage prediction on test set: < 10 minutes
- Baseline under-sampling methods (Random-US, NearMiss, RENN-US): 1–2 hours
- Baseline models (BalanceCascade, Multi-model Fusion): 1–2 hours
- Evaluation & visualization: 1–2 hours
- In total: ~1–2 days

## Dataset Description
```
Dataset: MinChnCorp (Source Domain)
Source: Chinese hotel reviews (Lin et al. 2015)
Raw size: 1,037,337 reviews (799,565 positive / 78,477 negative / 159,295 score-3)
Cleaned size: 53,336 reviews (27,610 positive / 25,726 negative)
Language: Chinese
Imbalance ratio: ~1:1 (balanced after cleaning)
Public: No — not found on any public repository

Dataset: Tour-review (Target Domain)
Source: DataFountain competition #283 (Chinese tourist attraction reviews)
Raw size: 220,000 reviews (194,790 positive / 3,246 negative / 21,964 score-3)
Cleaned size: 127,725 reviews (125,331 positive / 2,394 negative)
Language: Chinese
Imbalance ratio: 52:1 (extremely imbalanced)
Public: Partially — preliminary round (100K train + 30K test) downloaded to datasets/
```
