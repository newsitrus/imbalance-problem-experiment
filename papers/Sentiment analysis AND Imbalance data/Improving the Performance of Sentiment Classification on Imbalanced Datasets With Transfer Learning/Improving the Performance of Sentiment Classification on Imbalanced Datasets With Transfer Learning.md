---
paper_id: sha256_d1cd3e54
exported_at: 2026-02-09T11:36:40
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_d1cd3e54/report.md/analysis_report.md
format: markdown
---

# Improving the Performance of Sentiment Classification on Imbalanced Datasets With Transfer Learning


## Executive Summary

This paper addresses poor performance of sentiment classifiers on highly imbalanced text datasets by proposing a transfer learning based under-sampling method (TL-based-US) combined with a dual-convolution fine-tuning framework. The approach pre-trains a TextCNN on a balanced source dataset, fine-tunes on a balanced subset of the target, uses that model to select informative majority-class examples (misclassified positives) as an under-sampled set, and then fine-tunes further to produce a final model ensemble. On Chinese real-world datasets (MinChnCorp as source; Tour-review as target), TL-based-US improves minority-class F1 from ~35% (baseline TextCNN) to 57% (after under-sampling) and the full framework reaches 63% F1 (precision 63%, recall 64%). The authors note requirements for a similar external dataset and that generalization beyond sentiment classification remains to be validated.


## Metadata
- **Authors:** Z. XIAO, L. WANG, J. Y. DU
- **Venue:** N/A
- **Year:** N/A
- **Keywords:** Convolutional neural network, sentiment classification, transfer learning, under-sampling

### Abstract
In recent years, many sentiments classification models, such as deep learning models and traditional machine learning models, claim that they can achieve state-of-the-art performance in sentiment analysis problems. Admittedly, this is based on the premise that the training samples are class balanced. However, in the real world, the training data sets we can get are often imbalanced, which will cause the trained classifier to tend to predict the test samples into a majority, making the recall of minority very low. In order to minimize the influence of the imbalanced data class on the model performance, a transfer learning method based on a convolution neural network is proposed in this paper. First, we use a CNN-based model for pre-training in the class-balanced source domain data set, before transferring the model to the target domain for fine-tuning to improve the recall of minority class; furthermore, we propose a transfer learning-based under-sampling technique, which can under-sample the majority class in the target domain. In the data set after under-sampling, we again fine-tune the pre-trained model, so that the recall and precision of the minority class have been greatly improved. The experiments on real-world data sets show that our proposed under-sampling method has obvious advantages compared with others.


## Problem Statement
**Problem:** Sentiment classifiers (especially deep models) perform poorly on highly imbalanced text datasets: they are biased toward the majority class and have very low recall for the minority class (important negative reviews).

**Motivation:** Real-world sentiment datasets (e.g., reviews) are often extremely class-imbalanced; minority-class examples (negative reviews) are more informative for improving services/products, so improving recall and overall performance on minority classes is important.

**Confidence:** high

### Existing Limitations
- Common re-sampling methods for text use vector space models (VSMs) which ignore word order and word correlations, reducing effectiveness for text.
- Under-sampling and nearest-neighbor based sampling can remove useful majority-class information (loss of information).
- Over-sampling (random copy) can cause overfitting; SMOTE-style interpolation may be ill-suited for textual data.
- Deep models trained on small/imbalanced target data lack robustness without additional data or transfer.


## Proposed Method

### Transfer Learning-based Under-Sampling (TL-based-US) (Main Method)

**Algorithm Steps:**
1. Pre-train TextCNN on balanced source domain S_d to get model1.
2. Sample a class-balanced mini T_d from target and fine-tune model1 (freeze E and C) to get model2.
3. Use model2 to classify Td-train-pos; collect misclassified positive samples as under-sampling candidates (Tag-neg1).
4. Fine-tune model1 on Tag-neg1 combined with Td-train-neg (retraining additional convolution layers as described) to obtain model3.
5. For Td-test, predict with model2 obtaining Tag-pos2 and Tag-neg2; reclassify Tag-neg2 with model3 to produce Tag-pos3 and Tag-neg3; combine outputs for final prediction.

**Parameters:**
- `Embedding size`: 200 - Dimensionality of pre-trained word2vec embeddings (skip-gram on Chinese Wikipedia).
- `Max document length`: 60 - Maximum sentence length (padding/truncation) used as CNN input.
- `Filter sizes (pre-trained convolution)`: (2,3,4,5) - Filter (kernel) window sizes in transferred convolution module.
- `Filter sizes (new convolution)`: (5,6,7) - Filter sizes in newly trained convolution module during final fine-tuning.
- `Number of filters`: 128 (pre-trained), 64 (new) - Number of filters per size/channel counts used in convolutional layers.
- `Fully-connected size`: 300 (pre-train/fine-tune), 500 (final fine-tune) - Size of fully connected hidden layer(s).
- `Batch size`: 64 (pre-train/fine-tune), 128 (final fine-tune) - Training mini-batch size.
- `Learning rate`: 0.001 (pre-train), 0.0001 (fine-tune) - Initial learning rates used in respective stages.
- `Regularization`: Batch Normalization (pre-train/fine-tune), Dropout (final fine-tune) - Regularization techniques used in different training steps.

**Inputs:** S_d: source domain dataset (MinChnCorp) — class-balanced subset used for pre-training, T_d: target domain dataset (Tour-review) — highly imbalanced target data, Pre-trained word2vec embeddings (200-dim) trained on Chinese Wikipedia
**Outputs:** Final sentiment predictions (positive/negative) on T_d-test with improved minority-class (negative) recall and F1, Under-sampled subset of majority-class samples selected via TL-based supervised selection

### Stage 1 — Pre-training on source domain

**Algorithm Steps:**
1. Train word2vec (skip-gram) on Chinese Wikipedia (200-dimension).
2. Pre-train CNN (TextCNN variant: filter sizes 2,3,4,5; 128 channels) on balanced S_d to get model1.

### Stage 2 — Fine-tune on balanced subset of target (mini T_d)

**Algorithm Steps:**
1. Randomly sample mini T_d with balanced positive/negative ratio.
2. Fine-tune only the fully connected and output layers of model1 on mini T_d to obtain model2.

### Stage 3 — Transfer-learning based under-sampling and final fine-tuning

**Algorithm Steps:**
1. Apply model2 to Td-train-pos; collect samples predicted as negative (Tag-neg1) as under-sampling set.
2. Fine-tune model1 on Tag-neg1 plus Td-train-neg (train H and O layers, and additionally retrain a new convolution module) to obtain model3.
3. Predict Td-test: first with model2 to obtain Tag-pos2/Tag-neg2; then classify Tag-neg2 with model3 to obtain Tag-pos3/Tag-neg3; final labels are Tag-pos2, Tag-pos3, Tag-neg3.


## Evaluation

### Metrics
- **F1-value (minority class):** 0.57
- **Precision (minority class) — TL-based-US (comparison):** 0.56
- **Recall (minority class) — TL-based-US (comparison):** 0.58
- **Non-US baseline (TextCNN) — precision:** 0.35
- **Non-US baseline (TextCNN) — recall:** 0.34
- **Final framework (after TL-based-US + final fine-tune) — precision:** 0.63
- **Final framework (after TL-based-US + final fine-tune) — recall:** 0.64
- **Final framework (after TL-based-US + final fine-tune) — f1-value:** 0.63

### Datasets
- **MinChnCorp (S_d):** Cleaned: 53,336 (27,610 positive; 25,726 negative)
  - Source: MinChnCorp dataset (hotel reviews) — used as source domain for pre-training
- **Tour-review (T_d):** Cleaned: 127,725 (125,331 positive; 2,394 negative); raw total 220,000
  - Source: Reviews on tourist attractions — target domain (highly imbalanced, ~52:1 positive:negative)

### Baseline Comparisons
- **Non-US (TextCNN baseline)**
- **Random-US**
- **NearMiss-US**
- **RENN-US**
- **BalanceCascade**
- **Multi-model Fusion**


## Limitations & Future Work

### Limitations
- [Explicit] Requires a similar external dataset with sufficient samples (S_d) to pre-train the model.
- [Explicit] Method validated only for sentiment classification; unknown whether it generalizes to other text classification tasks.
- [Implicit] Experiments are conducted on Chinese datasets only; cross-lingual or English performance is not shown.
- [Implicit] Minority-class augmentation (concatenation and random perturbation) was used to expand few-shot negatives, which may affect realism/generalization.
- [Implicit] Dependence on similarity between source and target domains (performance may degrade if domains differ substantially).
- [Implicit] No statistical significance testing reported for comparative improvements.

### Future Work
- Conduct more comprehensive research to validate TL-based-US across more tasks and domains.
- Test generalization of the approach to other text classification tasks and datasets (different languages/domains).


## Cross-Reference Validation

### Unreferenced Figures
- 5a892c82cee062d384a225b40c1f4e2c05a17372fd6094c5491c085b34998f0f.jpg
- 703aa6f5f9e38210b730a79c8bcb4b7af927bd6f2c9f2d252be394e613695599.jpg
- 99a75fba6de7b4a20967ad72a58241812c3f762135ae622abc07bd23233834bb.jpg
- 7e0bc3d4e31cd72f775951cd8dcbcb7a2e982c82a539dc674f9f8bb0e5e8cca8.jpg
- 93fc232fe990a7d16ef96363f6af9ba3fb3d3aa230dab618decd1f14b2910704.jpg
- a8ea643d73414ff41cddf163109ef7e9fa7006f6cb6190bb33ceca1ce441c221.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/49ccefab5ee8ee185a30390a5f8cf007e003773ad1d6bdcd2991a6d4652aba33.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/6e5102a319d02abc6702d09d57f4e702c064cd2ffa7289f0c2eef5cb6a6d4e14.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/51df2fc3433412b3c4603a85be3f3ffb1a943f0183d7d8eebbae424f7053de9e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/93dc20e8f9b3eca4654bc0f411602810c05bcc1e1e3b35e44b5030cf66c58798.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/84d8bea6babc1c60b966b813e46e76828a69d110423de1b9ee46dd9396ff13d5.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/31cb7fa2efc2148c211feba382f0f526fa5918c0557c1af8ad2c76aaf42c4a85.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/b125ee35caeb92b639448a3dad8b694cf505d82855c9db4a55fe7023a819805b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/a9b5155901620585262a5fa900d4c1c6c5fefaf9ae50c2eac811431470401e25.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/e86506af94263f098f1da14169658b63dbbf42b655c3c267efd9c1081cf0cc91.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/b6fb68b7d7516c0894aee4caec5e3a20832fdfef1a3b422e4931f66793a944fd.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/bc526ca98fa602e01317340ee25ed0b6e758b7f3489cbfb2199259184b40482b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d1cd3e54/parsed/c19a1f6c18dc8092c2484ce738f511d18f642aecac800fbb0d4a42ed2b5c3689.jpg


## Analysis Warnings
- ⚠️ Evaluation section not detected
- ⚠️ Methodology section not detected


---

*Analysis cost: $0.87 (18 images analyzed)*