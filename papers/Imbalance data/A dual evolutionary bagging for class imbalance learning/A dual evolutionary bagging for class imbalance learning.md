---
paper_id: sha256_d95ba683
exported_at: 2026-02-22T19:44:17
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_d95ba683/report.md/analysis_report.md
format: markdown
---

# A dual evolutionary bagging for class imbalance learning


## Executive Summary

This paper addresses ensemble-structure selection for class-imbalanced classification by proposing Dual Evo-Bagging (Evo-Bagging): an inner ensemble that produces optimized heterogeneous base classifiers (MLP, DT, SVM) per oversampled sub-dataset via grid search, and an outer ensemble that uses a niche-based multi-modal genetic algorithm to find multiple high-quality ensemble combinations (optimized by G-mean). From the identified optima the authors choose the simplest ensemble (fewest base classifiers) and aggregate outputs by equal-weight averaging. Experiments on 40 KEEL imbalanced datasets (and a coal-burst case) report strong AUC/G-mean performance (e.g., AUC ~0.9791 for ROS) and improvements over several baselines, while acknowledging higher computational cost due to grid search and GA.


## Metadata
- **Authors:** uo Ji Fe Bot Ja  iui  SheanZeu
- **Venue:** N/A
- **Year:** N/A
- **Keywords:** Imbalance learning, Multi-modal genetic algorithm, Oversampling, Ensemble structure

### Abstract
Bagging, as a commonly-used class imbalance learning method, combines resampling techniques with ensemble leang  provide  srong classir wh hig generalization or  kew dataset. However,intea differet uber basclassir maybtahe sam cassificatio permance, almulodaliy. To seek the most compac ensemble sructur wih the highe accuracy, a dual evolutonary bggin fmework composedof inner and outer ensemble models is proposed. In inner ensemble model, three sub-classifiers are built by SVM, MLP and DT, respectively, with the purpose of enhancing he diversity among them. For each su-ataset, a classr w the best perrmanc is electe asa bas lass outeenseblee. Followingthat, all optimal combinationsof base classfiers is found by a multi-odal gnetialgorithm with anie ratyters  heaverag -eanA cobination ha aegate he mallestumbe classifiers by the weighte su forms the al ensemble ructure Experenal esult n 0 EL bn datasets and a practical one of coal burst show that dual ensemble framework proposed in the paper provides the simples ensemble sructure wih the best classication accuracy o mbalance datasets and outpeorms the state-of-the-art ensemble learning methods.


## Problem Statement
**Problem:** Design an ensemble method that (1) finds compact ensemble structures with the highest classification accuracy for class-imbalanced datasets, (2) generates better base classifiers and increases diversity among them, and (3) avoids manual pre-setting of ensemble size by treating ensemble-structure selection as a multi-modal optimization problem.

**Motivation:** Class-imbalance problems occur in many real-world tasks (fault diagnosis, medical diagnosis, fraud detection) where minority class detection is critical but traditional classifiers biased to majority class produce suboptimal decision boundaries; ensembles and resampling help but existing ensemble designs often lack diversity and require manual setting of ensemble size, leading to redundant classifiers and higher cost.

**Confidence:** high

### Existing Limitations
- Traditional ML driven by overall accuracy tends to misclassify minority class and produce sub-optimal boundaries.
- Bagging with independent sub-dataset training may produce similar base classifiers, reducing diversity.
- Ensemble size is often preset by humans, which can lead to redundant classifiers and unnecessary computation.
- Prior work focuses on applying resampling to ensembles but rarely addresses optimal multi-classifier structure selection (multi-modal nature).


## Proposed Method

### Dual Evo-Bagging (Evo-Bagging) (Main Method)

**Algorithm Steps:**
1. Oversample the imbalanced training data to produce multiple balanced sub-datasets.
2. For each sub-dataset, train MLP, DT and SVM sub-classifiers and optimize hyperparameters by grid search with cross-validation.
3. Select the best sub-classifier per sub-dataset as a base classifier for outer ensemble.
4. Encode candidate ensembles as binary individuals over base classifiers.
5. Use a multi-modal genetic algorithm with niching (sharing) to search for multiple optima maximizing G-mean.
6. From the set of optimal solutions, select the one with the smallest number of base classifiers (simplest structure).
7. Aggregate selected base classifiers by equal-weight averaging to form final predictions.

**Pseudocode:**
```
Algorithm 1 To optimize the parameters of a classifier by grid search

Input: Sub - classi f ier: MLP, DT and SVM Parameter: the range of each parameter's value Dataset: a sub-dataset

1: Initialize the parameters of grid search
2: 3:
for each value in Parameter do Initialize a Sub  classi f ier
4: K-fold cross-validation of Sub  classi f ier on Dataset
5: Save values that has the least error
6:end for

Output: The optimal values of parameters
```

**Parameters:**
- `mutation probability (p_m)`: not fixed (paper references p_m) - Probability for single-point mutation flipping a gene in GA
- `crossover probability (p_c)`: not fixed (paper references p_c) - Probability for crossover between two parents in GA
- `maximum iterations (T)`: 100 - Termination iteration count for the GA (empirical value)
- `population size (P_s)`: not fixed (paper uses P_s symbol) - GA population size
- `niche radius (L)`: preset (symbol L) - Radius used in sharing strategy to determine similarity and penalize crowded niches
- `grid search parameter combinations (P_n)`: not fixed (symbol P_n) - Number of candidate parameter combinations evaluated in grid search
- `objective function`: G-mean - Fitness used to evaluate ensemble individuals in GA
- `ensemble weight (omega_i)`: 1/|E| - Final ensemble uses equal weights for selected base classifiers

**Inputs:** Imbalanced dataset (training and test splits), Ranges of hyperparameters for grid search for MLP, DT, SVM, Choice of oversampling strategy (ROS, SMOTE, SMOTE-borderline1, SMOTE-borderline2, SMOTE-SVM)
**Outputs:** Final ensemble classifier E composed of selected base classifiers (simplest optimal combination), Predictions on test data (aggregated by equal-weight averaging)

### Inner ensemble model (base classifier optimization)

**Algorithm Steps:**
1. Create n balanced sub-datasets D = {D1,...,Dn} by oversampling (e.g., ROS, SMOTE, SMOTE-borderline1/2, SMOTE-SVM).
2. For each Di, train three sub-classifiers SC_i = {MLP, DT, SVM}.
3. Optimize hyperparameters of each sub-classifier by grid search with K-fold cross-validation.
4. Select the sub-classifier with least testing error / highest AUC as base classifier ci (weight 1, others 0).

### Outer ensemble model (structure optimization via multi-modal GA)

**Algorithm Steps:**
1. Translate the set of base classifiers C = {C1,...,Cn} into binary-encoded individuals; gene = 1 means include base classifier.
2. Use G-mean as objective to evaluate each individual on the original training set.
3. Evolve population with crossover, single-point mutation, roulette-wheel selection; apply sharing niching to maintain multiple optima.
4. Collect Pareto best/optimal solutions in an external archive; when archive exceeds size, remove worst.
5. From obtained optimal solutions, pick xc_out that minimizes the number of selected base classifiers.
6. Form final ensemble E and predict by equal-weight averaging of selected base classifiers.


## Evaluation

### Metrics
- **G-mean:** 
- **AUC:** 0.9791 (Evo-Bagging, ROS, average reported in Table 7)
- **AUC:** 0.9283 (Evo-Baggingu (without GA), ROS, average reported in Table 7)

### Datasets
- **KEEL binary-class imbalanced benchmark datasets (40 datasets):** 
  - Source: KEEL repository (as stated in Table 1)
- **glass6 (example dataset reported in tables):** 
  - Source: KEEL benchmark (Table 1)

### Baseline Comparisons
- **OverBagging**
- **ANASYN-Bagging**
- **Adaboost**
- **Random Forest**
- **EasyEnsemble-Bagging (EE-Bagging)**
- **SMOTEBoost**
- **RUSBoost**


## Limitations & Future Work

### Limitations
- [Explicit] Grid-search hyperparameter optimization provides full exploration but is high time-consumption (mentioned by authors).
- [Explicit] Generating a large number of alternative base classifiers increases computational complexity and may reduce difference among them.
- [Explicit] Overall computational complexity includes costly GA evaluation and grid-search steps (complexity analysis provided).
- [Implicit] Evaluation focuses on KEEL binary-class benchmarks; generalization to large-scale, multi-class, or real-world varied domains beyond reported coal-burst case may be uncertain.
- [Implicit] Relies on oversampling strategies and grid search; scalability and runtime may be problematic for very large datasets.
- [Implicit] Selection of the simplest ensemble among optima (fewest base classifiers) may discard slightly higher-performing but larger ensembles.


## Cross-Reference Validation

### Unreferenced Figures
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/00b826fa7a24e33c1525f250a8942d202b00b0fb06c9b49e9667ee4180b6fdcd.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/238d68e53ecc9d202e2a87eae1b1ddf1ac28d8e054931fded33e235985db7c59.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/1c96163bab7f673c6bdb33eed487d8c7496405295da9bfc56191345eaedac1fc.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/30ed8f416b68a49b339af4f48dabb5b996cdfdeaa425d11d645f69aca15befb3.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/24b3da9ae3e794a7e3f4a4a11f32356307c99847e748b6114e56f47e5e10b071.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/3283972c01e15544b933d63f60b7477a57240fc4c58264a446032d145b44ecf8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/332b7e36f35811c4dca73dec9081c91c00e166f6e9b8fb3c45dc1a7508c34f4e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/1e8210f653f8ed8d95aeebe0a3ffacacba250cf4bec97d5be93c5315e9010577.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/4eb8a0f1588948844c747e095824228a685476400ba58c60036c709613fe5088.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/5841edb390bb91c9e14cc97c2de5254cc2300e87e602b50e84ee97ef64470c02.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/5eff3d64803e29042ba0f5cd3cade7aef32713215e7f428176513901eac3b8cf.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/5e93b6ff1f0b89ddfc32cb05d48bfa82b36359e96b511996d19a35f06b50c704.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/3362503f79249feb170e1ea08815422f423287dd6c4cd966098ccd8ff7229d58.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/5fa66073637edee465383bfb502f6c0b9c520f716c00752864068f45d35dd9b8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/68a9c2370a6854c238cd23e8addeb20d35d27e23ab2af6a45a9e24ddb292752d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/5fee24f9f74096a5bb98403acb5c6292df750dac0222fcbb64cc17129d6be129.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/6aae273a4a14c836a5277ea8ec9e90dae01fae7ac07d597cff118ef4e5ea6f98.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/79e5ee2a706eb03e0e4737efc56b42e02ae2a055c561f1b779de17269b9278cc.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/8332885d59c423b6b9af32e33e7fbe6d682ca46e7a9d55ee7b3a9e3a2234b866.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/a9a3cc76f04f9d93f50dd9f02b09c3d369b17793b865012e7bf56a6b8452f5e6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/b368652f4ec5209c2288be46745ff2aa19c65297e0c105d2cc9f857414ea1330.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/b31940395bb4ed648bc13f27518b1c2a2adf34eef682d95603c29a500107f1de.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/b722880ecd66827491e8c1053503b6360b752045952838111c5017be066bb24e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/8f6412b6d60ff1b70cb0928409a67668504e0250b258e6e2f7182545e816bf66.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/c32674dda207052650e93aa2f48151ef58a133f5a09cf6bdd549d44403d3f390.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/cf99fdc17d7ee84171db075d839973d126152c9334287c276549d9c0b027a1c0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/ab121487d13960c0bd9853e1e073c2fc887d9d3bc388384b97a7bfe2d8a6564a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/dd9de6e8a2a35805ae0ed4abde8d827ce988d7b357da9719b1d45fa50eacf780.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/e457b4c9ee2ac7c9ee848a984e8cabd51d9f21b0a37b927b1a0e1c8009a09bb4.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/e2b5d3ac72e38c07982e9dcf3919c4cb608fefe4606855189047d88ff8ba5577.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/ebdc1622dfe2ec8cee776a697778beba8fb613c319dfa5dee59dcb6b1993c1c7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/cf9b631efd1ee2bbaf2c7cfe270361d537b1334f758f828f8fb5d1a81c0b863c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_d95ba683/parsed/d341029b87db348af7f3c63088abede1e7e3d0c75c34e5f74e6f7139b86f0a1d.jpg

*10 figure references validated as consistent.*


## Analysis Warnings
- ⚠️ Evaluation section not detected
- ⚠️ Methodology section not detected


---

*Analysis cost: $1.52 (43 images analyzed)*