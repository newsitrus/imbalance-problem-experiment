---
paper_id: sha256_e8218977
exported_at: 2026-02-09T11:35:17
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_e8218977/report.md/analysis_report.md
format: markdown
---

# Sentiment Analysis of Customers’ Reviews Using a Hybrid Evolutionary SVM-Based Approach in an Imbalanced Data Distribution


## Executive Summary

This paper addresses imbalanced Arabic restaurant-review sentiment analysis by proposing a hybrid evolutionary method that uses Particle Swarm Optimization to learn per-feature weights and to tune oversampling parameters (k) for SMOTE-family techniques, combined with an SVM classifier. The authors collect and crowd-label ≈2.8k reviews from Jeeran, prepare four tokenization variants (1-/2-/3-grams and BoW), and use PSO to optimize weights and oversampling settings using G-mean as the fitness. Experiments show that PSO-SVM with borderline-SMOTE (and SMOTE variants) outperforms standard classifiers (SVM, RF, NB, XGBoost, etc.) on accuracy, F1 (positive and negative), G-mean and AUC across dataset variants, demonstrating improved balanced performance on imbalanced Arabic review data.


## Metadata
- **Authors:** Ruba Obiedat, Raneem Qaddoura, Ala’ M. Al-Zoubi, Laila Al-Qaisi, Osama Harfoushi, Mo’ath Alrefai, Hossam Faris
- **Venue:** N/A
- **Year:** N/A
- **Keywords:** Sentiment analysis, SVM, PSO, SMOTE, oversampling, feature extraction, features weighting

### Abstract
Online media has an increasing presence on the restaurants’ activities through social media websites, coinciding with an increase in customers’ reviews of these restaurants. These reviews become the main source of information for both customers and decision-makers in this field. Any customer who is seeking such places will check their reviews first, which usually affect their final choice. In addition, customers’ experiences can be enhanced by utilizing other customers’ suggestions. Consequently, customers’ reviews can influence the success of restaurant business since it is considered the final judgment of the overall quality of any restaurant. Thus, decision-makers need to analyze their customers’ underlying sentiments in order to meet their expectations and improve the restaurants’ services, in terms of food quality, ambiance, price range, and customer service. The number of reviews available for various products and services has dramatically increased these days and so has the need for automated methods to collect and analyze these reviews. Sentiment Analysis (SA) is a field of machine learning that helps analyze and predict the sentiments underlying these reviews. Usually, SA for customers’ reviews face imbalanced datasets challenge, as the majority of these sentiments fall into supporters or resistors of the product or service. This work proposes a hybrid approach by combining the Support Vector Machine (SVM) algorithm with Particle Swarm Optimization (PSO) and different oversampling techniques to handle the imbalanced data problem. SVM is applied as a machine learning classification technique to predict the sentiments of reviews by optimizing the dataset, which contains different reviews of several restaurants in Jordan. Data were collected from Jeeran, a well-known social network for Arabic reviews. A PSO technique is used to optimize the weights of the features, as well as four different oversampling techniques, namely, the Synthetic Minority Oversampling Technique (SMOTE), SVM-SMOTE, Adaptive Synthetic Sampling (ADASYN) and borderline-SMOTE were examined to produce an optimized dataset and solve the imbalanced problem of the dataset. This study shows that the proposed PSO-SVM approach produces the best results compared to different classification techniques in terms of accuracy, F-measure, G-mean and Area Under the Curve (AUC), for different versions of the datasets.


## Problem Statement
**Problem:** Automatic sentiment analysis of Arabic restaurant reviews faces class imbalance (many more positive than negative reviews), high-dimensional textual features, and the need to tune oversampling parameters and feature weights; the paper addresses improving classification under imbalanced data by combining PSO for feature-weight and oversampling parameter optimization with SVM classification.

**Motivation:** Online reviews strongly influence customers' choices and restaurant success; the volume of Arabic reviews is large and growing, making automated, accurate sentiment analysis essential. Imbalanced class distributions and Arabic language challenges (dialects, morphology) reduce classifier performance, motivating methods to handle imbalance and optimize features/parameters.

**Confidence:** high

### Existing Limitations
- Many prior studies focus on English; fewer evolutionary approaches have been applied to Arabic sentiment analysis.
- Several previous hybrid/evolutionary studies ignored class imbalance in datasets (leading to biased models).
- Parameter tuning for oversampling techniques (e.g., SMOTE k) and feature weighting is not automated or optimized in many prior works.
- High-dimensional feature representations (many n-grams) make feature selection/weighting necessary; some prior methods do not address this adequately.


## Proposed Method

### PSO-SVM (PSO-optimized SVM with oversampling variants) (Main Method)

**Algorithm Steps:**
1. Collect and label Arabic restaurant reviews; preprocess and extract features (n-grams, BoW).
2. Initialize PSO: create population of particles where each particle encodes feature weights and an oversampling k value.
3. For each particle: apply its weights to training features to obtain weighted training data.
4. Apply selected oversampling technique with the particle's k to generate balanced training set.
5. Train SVM on the oversampled weighted training data and compute fitness using G-mean.
6. Update particle velocities/positions (weights and k) using PSO rules; update pbest and gbest.
7. Iterate PSO for specified iterations; keep best particle.
8. Use best particle to weight and oversample test data, classify with SVM and report final metrics.

**Pseudocode:**
```
Algorithm 1 (PSO): Initialize D-dimensional swarm and velocities; for t=1..maxIterations do for each particle i do for each dimension d do update velocity via v_{i,d}^{t+1} = v_{i,d}^t + c1*r1*(p_{i,d}^t - x_{i,d}^t) + c2*r2*(p_{g,d}^t - x_{i,d}^t); update position x_{i,d}^{t+1} = x_{i,d}^t + v_{i,d}^{t+1}; end; compute fitness (G-mean) of updated position; update pbest and gbest if improved; end; terminate if gbest meets requirements; end. Algorithm 2 (SMOTE): For each minority example, compute k nearest neighbors, generate N synthetic samples by interpolating between example and selected neighbors.
```

**Parameters:**
- `PSO iterations`: 100 - Maximum number of PSO iterations used for optimization
- `PSO population size`: 100 - Number of particles in the PSO swarm
- `PSO runs`: 30 - Number of repeated runs reported in experiments
- `Oversampling k`: optimized (variable) - Number of nearest neighbors (k) for SMOTE-family oversampling; tuned by PSO per particle
- `Feature weights`: optimized (vector of length = #features) - Per-feature weight vector optimized by PSO and applied multiplicatively to feature values
- `Evaluation fitness`: G-mean - Fitness function used by PSO to guide optimization (balance between class recalls)

**Inputs:** Arabic restaurant reviews (text) collected from Jeeran, Labeled training set (positive / negative) created via crowdsourcing, Tokenized feature representations (1-gram, 2-gram, 3-gram, Bag-of-Words)
**Outputs:** Predicted sentiment labels (positive / negative) for test reviews, Optimized feature weight vector and optimized oversampling k value (particle solution), Evaluation metrics (Accuracy, F1 positive, F1 negative, G-mean, AUC)

### Data collection and preparation

**Algorithm Steps:**
1. Crawl reviews (~3000) from Jeeran using a C# script
2. Crowdsource labeling (positive/negative) and assign labels by majority vote
3. Text cleaning: remove symbols, non-Arabic characters, stopwords, normalize and stem
4. Feature extraction: generate 1-gram, 2-gram, 3-gram and bag-of-words representations

### Training and evolutionary optimization

**Algorithm Steps:**
1. Initialize PSO population with random weight vectors and random k values
2. Apply weights to training features to create weighted training data
3. Perform oversampling (SMOTE / SVM-SMOTE / ADASYN / borderline-SMOTE) using k
4. Train SVM on the oversampled weighted data and compute G-mean
5. Update PSO pbest/gbest and iterate to improve fitness

### Testing and evaluation

**Algorithm Steps:**
1. Generate weighted test data using optimized weights
2. Oversample test data using optimized k and chosen oversampling variant
3. Classify with the trained SVM and compute evaluation metrics


## Evaluation

### Metrics
- **Accuracy (Data 1, SVM-PSO+BorderlineSMOTE):** 0.8970
- **F1 (positive) (Data 1, SVM-PSO+BorderlineSMOTE):** 0.9396
- **F1 (negative) (Data 1, SVM-PSO+BorderlineSMOTE):** 0.6507
- **G-mean (Data 1, SVM-PSO+BorderlineSMOTE):** 0.8006
- **AUC (Data 3, SVM-PSO+BorderlineSMOTE):** 0.82

### Datasets
- **Jeeran restaurant reviews (Our-data):** 2790 reviews (2150 positive, 640 negative)
  - Source: Jeeran website (collected by authors via a C# crawler)
- **Public-data:** 3916 reviews (3465 positive, 451 negative)
  - Source: public dataset referenced in paper (table 1)
- **OCLAR (for comparison):** not specified in paper (dataset available in UCI repository)
  - Source: UCI repository (Opinion Corpus for Lebanese Arabic Reviews) used for comparison

### Baseline Comparisons
- **SVM**
- **XGBoost**
- **Decision Tree**
- **Random Forest**
- **Naïve Bayes**
- **k-NN**
- **Logistic Regression**
- **Bidirectional LSTM**
- **GBDT**


## Limitations & Future Work

### Limitations
- [Explicit] Imbalanced datasets are common in customer reviews (majority positive) and are a challenge to address.
- [Explicit] Parameter tuning of the oversampling technique is a challenge (authors aim to automate via PSO).
- [Explicit] Large number of features from tokenization requires feature analysis/weighting; computational cost implied.
- [Explicit] Authors note they plan to employ different metaheuristic algorithms in future work and test other application domains.
- [Implicit] Dataset is limited geographically (restaurants in Jordan) and may not generalize to other Arabic dialects or regions.
- [Implicit] Dataset size is modest (~3k reviews), which may limit robustness/generalization compared with large-scale corpora.
- [Implicit] Approach focuses on SVM as classifier backbone; results may differ with modern deep-learning models on larger data.
- [Implicit] Oversampling produces synthetic examples which can introduce bias if minority subspace is poorly represented.

### Future Work
- Employ different metaheuristic algorithms for optimization (beyond PSO).
- Apply the approach to other product domains (e.g., medical and engineering product reviews).
- Explore additional oversampling or generative techniques and larger/more diverse Arabic datasets.
- Investigate integration with or comparison to deep learning models on balanced/augmented data.


## Cross-Reference Validation

### Unreferenced Figures
- 6d162958185bb2f6748da6449eb93c043157cb859af511411249dbf1939b7630.jpg
- 019dbc148fa3bd5817dcdc4772cbeef7f4f621d1608c7082dc7e001ce48984cc.jpg
- 6cbbdc9d24ec32923061bfb3d31ad3980688234c366f5146c5e8a0a02b7c3409.jpg
- 8cd2fc109ca03b0d96346b6ca57950de7a3fb6ccaa1f21f3af31c91d4fade51a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/1202920a43c14afbedf58cdfeff2917a905c08111e57cdb02e6c4d4cee36754d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/23dc66465a77421b9f1a79c236bf704d4c089652ebc6dd59267c02869218595f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/128fd40af473d33ac6f7a89bdea7350746d97e6ce2792a049200c25465d1c64d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/2c67b8f08edc27b9dfccd7f8aa563790fd5b871e7a624e8bef124dfeee805ead.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/1bbe6bf6029feb28c0093cbcbb2c054fb6c8e6d036d85ec3e08a880db6624ea0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/3e5ca41a31cbfb99c3f0fca50f7f792de8c1c6638391d09db0c8d7dfeaac329c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/358e29e1c0e159eb8d2f52992c573ad85c6a3d39bbc44b231f6ddfafdeeab01c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/4555fc5058ac0f418ff719f4475245488964e8ee1502aa323edd43990101da40.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/45b182c825c0054850dd445cb1ae3e4e4dd7311d5ef49473cf080494e5e82dca.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/43922e380f4743daa870d6c6012d5676f59e028fb2a693a8905b580de801c9ad.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/31ab171b24223500f22db1626e656342b1ea30606108b9217118502b05b773cd.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/5d8def6fd21ac49ddb70d0e024be4f0e72d49503a7912624672bf50ddaf6c84c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/560190cd5584e80aff7f97a0066bc8b773c429f14eb8cc52bd572a9d3df067ac.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/0c8b85898b73d4d2b71f5b1cf4a2687789edd3dee06b821722d3f1da245dc212.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/adbfe34dc22c4227f0acb52bc08f651c346eb4c415ecd8c6dd3ffe4339ad9409.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/bcf5a535290439673f85af5d476745c4851955c9946bda7ef13d14fcf6b55fd8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/cd7ce155a676820963407a44325d305849e05d41b197617d62aa5fffcae18598.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/8f2e30d6c5d3a9fa091e694abd5ae50191bea1968775ecbd750c88eb7ba9515b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/fecf6bcd355ce030d1ae50a1df7076e911a0e712c281fc3124e61b1766500132.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/c47be31945f4f97e579ab59b0a67d90dd9d225da801fc36de75b751941fbb313.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/d28558dd270a1da427c29fc3c76030785d9ec5a794a4df747ab7c27c3be32955.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/e316ac67c1ea79267f34145f5b6918ef86439364feb71b1c00512bef33184f30.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/ffaf09cadf53ab7e941d56c7dd5950eadd547dcf04a794e0fc6e52d70dfe149b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_e8218977/parsed/b43161462d6917ff3bef1276f55512e15abab05db9b6ea2400dfd05c7db8c440.jpg


## Analysis Warnings
- ⚠️ Evaluation section not detected
- ⚠️ Methodology section not detected


---

*Analysis cost: $1.36 (28 images analyzed)*