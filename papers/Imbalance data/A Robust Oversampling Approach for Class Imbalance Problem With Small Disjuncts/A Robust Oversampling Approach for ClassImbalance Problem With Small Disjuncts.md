---
paper_id: sha256_61a28e96
exported_at: 2026-02-22T19:44:36
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_61a28e96/report.md/analysis_report.md
format: markdown
---

# A Robust Oversampling Approach for ClassImbalance Problem With Small Disjuncts


## Executive Summary

This paper addresses minority-class oversampling for imbalanced datasets in the presence of small disjuncts. The authors propose DROS (Disjuncts-Robust Oversampling), which models local minority regions as geometric light-cones: for each minority seed point they compute a base unit vector, a vertex (inner minority launch point) and a radius by analyzing majority neighbours and direct-interlinked minority points, then generate synthetic samples inside these light-cones while avoiding majority areas. Experiments on multiple 2D emulated datasets and 31 real-world (mostly UCI) datasets show DROS achieves the best mean ranks on recall, f-measure and g-mean (SVM classifier), demonstrating improved handling of small disjuncts; however, the method is limited to numerical features, is computationally intensive, and primarily evaluated for binary classification.


## Metadata
- **Authors:** Yi Sun, Lijun Cai, Bo Liao, Wen Zhu, Junlin Xu
- **Venue:** IEEE Transactions on Knowledge and Data Engineering
- **Year:** 2022
- **Keywords:** Imbalance problem, oversampling, small disjuncts, area illumination

### Abstract
Class imbalance is one of the important challenges for machine learning because of it’s learning to bias toward the majority classes. The oversampling method is a fundamental imbalance-learning technique with many real-world applications. However, when the small disjuncts problem occurs, how to effectively avoiding the negative oversampling results rather than using clusters previously, remains a challenging task. Thus, this study introduces a disjuncts-robust oversampling (DROS) method. The novel method shows that the data filling of new synthetic samples to the minority class areas in data space can be thought of as the searchlight illuminating with light cones to the restricted areas in real life. In the first step, DROS computes a series of light-cone structures that is first started from the inner minority class area, then passes through the boundary minority class area, last is stopped by the majority class area. In the second step, DROS generates new synthetic samples in those light-cone structures. Experiments considering both real-world and 2D emulational datasets demonstrate that our method outperforms the current state-of-the-art oversampling methods and suggest that our method is able to deal with the small disjuncts.


## Problem Statement
**Problem:** How to perform effective minority-class oversampling in imbalanced datasets when small disjuncts (minority subregions with few samples) are present, avoiding negative effects (noisy or overlapping synthetic samples) produced by existing oversampling approaches.

**Motivation:** Class imbalance causes learning algorithms to bias toward majority classes and leads to poor minority-class performance in important real-world applications (e.g., fraud detection). Existing oversampling methods often produce noisy or overlapping synthetic samples especially when data distributions are abnormal (outliers, overlapping, small disjuncts). A robust oversampling method that respects minority-area structure (small disjuncts) can improve classification on minority classes.

**Confidence:** high

### Existing Limitations
- Cost-sensitive and some algorithm-level methods are application-specific and sensitive to noisy data.
- Interpolation-based oversampling (e.g., SMOTE) can generate noisy samples when the interpolation line-segment crosses majority-class areas.
- Clustering-based techniques (e.g., used in MWMOTE) may not adaptively group far minority samples of the same minority area and may group near samples from different minority areas together.
- Structure-preserving methods (INOS, MDO, AMDO, SWIM, GDO) do not account for abnormal distributions like small disjuncts and can generate overly extended synthetic data including noisy/overlapped points.
- High time complexity / scalability issues for large datasets.


## Proposed Method

### Disjuncts-Robust Oversampling (DROS) (Main Method)

**Algorithm Steps:**
1. Step 1: Compute direct-interlinked relationships I(x_i, x_j) between every pair of minority samples using majority points to test whether the line-segment between two minority points goes through majority areas (via inner-product-based angle threshold delta).
2. Step 2: For each minority sample, compute its light-cone structure components (base unit vector a, vertex v, radius r) using k nearest majority neighbours, direct-interlinked minority samples, and illuminated majority samples; discard improper light-cones.
3. Step 3: Iteratively generate |S_maj| - |S_min| synthetic minority samples by randomly selecting valid light-cones and sampling points within them controlled by parameter g and rho.

**Pseudocode:**
```
Algorithm 1 DROS (from paper): Input: S_maj, S_min, rho, k, d (delta), g. Step 1: I = Relationships(S_maj, S_min, |S_maj|, |S_min|, delta). Step 2: S2 = Structures(S_maj, S_min, |S_maj|, |S_min|, rho, k, I). Step 3: S_new = DataGeneration(|S_maj|, |S_min|, S2) (generate |S_maj|-|S_min| samples by randomly choosing light-cones and sampling new = v + (xi * r) * d where xi in [g,1] and d is random unit vector satisfying <d,a> >= rho). Return S_new.
```

**Parameters:**
- `rho`: 0.5 - Scalar parameter controlling cone aperture / inner-product threshold for light-cone membership (component of light-cone structure).
- `delta`: -0.7660 - Threshold for minimum inner product (cosine of max angle) used to decide direct-interlinked relationship between two minority points; ranges in [-1,1].
- `k`: 7 - Number of nearest majority neighbours used to compute base unit vector (and mean majority centre).
- `g`: 1 - Lower bound parameter in [0,1] controlling how near generated points are placed to boundaries inside the light-cone (xi sampled in [g,1]).
- `N1,N2`: not fixed - Internal parameters controlling how many attempts (N1) or pre-generated direction vectors (N2) to use when sampling a direction unit vector that satisfies the inner-product constraint; used to handle high-dimensional sampling.

**Inputs:** S_maj: training majority class samples (feature vectors), S_min: training minority class samples (feature vectors), user parameters: rho, k, delta (d), g (and optional sampling parameters N1,N2)
**Outputs:** S_new: synthetic minority class samples generated by DROS

### Light-cone structure definition

**Algorithm Steps:**
1. Define cone via inner product threshold with base unit vector a and scalar rho
2. Intersect cone with ball of radius r to form light-cone structure S

### Compute light-cone components for each seed minority sample

**Algorithm Steps:**
1. Compute k nearest majority neighbours of the seed point and set base unit vector a = - (x - mean(z_i))/||x - mean(z_i)||
2. Determine direct-interlinked minority samples using minimum inner products over majority points (compare M(x_i,x_j) with threshold delta) and compute vertex by projecting vectors to direct-interlinked points onto the inward direction
3. Find illuminated majority samples (those satisfying angle criterion with rho), select nearest majority point g, and compute radius r as a function of distances from vertex to seed and to g

### Generate synthetic samples inside light-cones

**Algorithm Steps:**
1. Randomly pick a light-cone structure s = {a, v, r, rho}
2. Generate random scalar xi in [g,1] (g controls closeness to boundaries)
3. Generate random unit vector d satisfying inner product constraint with base vector a
4. Compute new sample new = v + (xi * r) * d and add to synthetic set S_new


## Evaluation

### Metrics
- **precision:** 
- **recall:** 
- **f-measure:** 
- **g-mean:** 
- **AUC:** 

### Datasets
- **Ring (2D emulational):** 
- **Curve1 (2D emulational):** 
- **Survival < 5yr:** 81:225 (minority:majority)
  - Source: UCI repository
- **Pageblocks3-1:** 28:4913 (minority:majority)
  - Source: UCI repository

### Baseline Comparisons
- **SMOTE**
- **ADASYN**
- **MWMOTE**
- **INOS**
- **SWIM**
- **GDO**
- **Ori (raw data)**


## Limitations & Future Work

### Limitations
- [Explicit] Method applies only to instances with numerical dimensions (cannot handle categorical features directly).
- [Explicit] Method is specialized in addressing small disjuncts; not primarily designed for outliers or overlapping.
- [Explicit] High time complexity; not desirable for very large datasets (tens or hundreds of thousands of instances).
- [Explicit] Primarily evaluated for binary classification; multi-class use requires repeating per minority class which can be time-consuming.
- [Implicit] Potential sensitivity to user-defined parameters (rho, delta, k, g) though authors report empirical tuning; may require dataset-specific tuning.
- [Implicit] Scalability concerns in high-dimensional or very large sample-size settings due to O(|S_maj| |S_min|^2 + ...) complexity.
- [Implicit] Evaluations mainly on UCI and small emulated datasets; limited evidence on very large-scale or real-world industrial datasets.
- [Implicit] Dependence on quality of majority-sample geometry for deriving base vectors and vertices; noisy majority distributions may affect light-cone construction.

### Future Work
- Improve robustness towards noisy and overlapped data points.
- Reduce time complexity to handle big datasets more efficiently.
- Extend or adapt method for multi-class contexts more efficiently (beyond per-class repeated application).


## Cross-Reference Validation

### Unreferenced Figures
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/027c905ad22f903b08aa052e586a7b15c570be2efd466256c01d69116ab43488.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/00e90a868b211cc92e389db38347286d2ae0d457e86ee20db8a430df9dded1c1.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/041c4872b5c1f8dc9517280556cfef76b2b645d3738232323bb2dba172a7b399.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/0592d0e0062f4795676d023983d7eb2f9f68adf0abea7b7f99c077896e59dc3a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/095feeb672a81291712f91d6de69616f794f77b901124451961888dc4197dd03.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/192ac4eec7c5f4eed10239002afec6231870b28f2a5515828a91c21fd604aca0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/068c1e6b1108159701a1f59e66241616746ca45b7bea634ef186a0c4b9ea289c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/093e9ef0ec28250d22b581503f5494f8d3f44cba233d7c9c5a257e261b07e7fb.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/0ab80126ce69cc62c093cdafc6613354202c430a0227078f9a0fa9f02a2bfb29.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/27c4813bf60530c68277bf635bf5e64bb0847967f56f964e386acc45bfc18b6f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/09bcabc9a5e33b2dfcc1420751cdf47c4fa70ee264cc16439c022acd1ccf51c1.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/3a5f86abdc7938988769a6e2778ca13b3f4be5cc7d4c4a87abb8552ae06bb0c7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/449484ab30f3c38f85ca755af040f5baac8a471475cd1d75292f95476da81399.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/4a2643d854608a79fed0de3d7335d906da644e399fda39a2ee5ae37201f34944.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/4dfbf470d5ef96ffc9d8c688248281e7033e4424a655b7dcfac676452f5035dc.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/5b1e8e62986102a17b3f554cf52e4786a4309d4172f94857ea9e64e4f0c9d143.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/6401f68616366bf5e69e2a6978f2a8380f427f3b956c646092d3ed96e25048c7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/692c6fad84210426b8d8f65e1ccf1db04cb77335659fe2082e215bc0e9941efe.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/1984b57589ea4ca8926ee278dcfe0beb70cc815391e17645d5f0d8e3621b8326.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/69ddeb60ab5262742dbe668509b6a87f2cbe8bfe1da9450519d5a9b7b60f6702.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/711f30129fafb0c479e9af1206e62d43557a772721b7cf92401826d1d51274c3.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/44cf1b2398b10b69aff0c3e610cb41201a6f90cf40d2cc44db612368353851c9.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/721d20a69612542f017c8ead421419b155189b75390b395c5cfe5f8f8476000c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/72cd6d1215872aa6a08cab7ef6a4e288f6cd440af92630bcf2d8815b559c3318.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/74e973e52b1c12733db5bd45365e6b2c1b516a1598ea09dcabb779228818ff6d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/820736a442dfd8ad20a11d5d9cd768728789e12f18b4f5c2f4487fec55ba451a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/936473397060fdfb51703ce321eb5dadb60c727f28d1e8b8361cc49aaf0fc269.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/9c78a0e40363d364e8991058b99dbb0b1a8779576e68fc69339325d896bf09e4.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/a2656a62de81e21c126b51dc3c22fe29955734fd9df7b8fa22f05a71576464d7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/b099b235d6e1f2c9e29c5d35f3c006493936aba8dfb74e1d600b17957ab59b6a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/d2cb73e972c09b508129a107c1539fbd9de97eb3fa301fe0cbb19b06e23981c8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/de53fdd04fbfd2466bd2b1e4927e73066c0d60bdd4df23eb73387ae9838a2694.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/9718c372d949ee28be4fd16de418eda9572a8e5aad0501928bdabcc5c2a2861e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/c510e01345635427e32a525c366159796041a8fde39c05239e69e0abf20f8635.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/ff57f988c6c304dd94bf6f9a0497ca7f77950a992bbb9268e0a20b20637d50ed.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/ffc50ff98764dd9410b4ac79278f1f31a154b872e82e462f3d97f78efdd959ab.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/e9c773dcf1d11ade543607c7e1de0a0384f8823ca2f24d1756bcb7768fe146c1.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_61a28e96/parsed/312a795e901f31cc668adc4c7169b70600d0b89f28f76dd565bf51e6b626e5b5.jpg

*6 figure references validated as consistent.*


## Analysis Warnings
- ⚠️ Methodology section not detected


---

*Analysis cost: $1.15 (44 images analyzed)*