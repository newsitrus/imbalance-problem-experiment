---
paper_id: sha256_525375a4
exported_at: 2026-02-22T19:43:57
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_525375a4/report.md/analysis_report.md
format: markdown
---

# Imbalanced Clustering With Theoretical Learning Bounds


## Executive Summary

This paper addresses imbalanced clustering (unequal cluster sizes) and the lack of theoretical excess-risk guarantees in existing methods. The authors propose MACW, a k-means variant that introduces adaptive per-cluster weights, and derive an excess clustering risk bound for MACW. Building on a finer per-cluster excess risk analysis, they propose ICTLB, which sets per-cluster weights by optimizing the derived bound (λ_r^α ∝ |θ_r|^{α/(2(1-α))} scaled by W). Experiments on 14 imbalanced datasets (KEEL/UCI variants) show ICTLB often outperforms baselines (k-means variants, DEC/IDEC, SMCL) across ACC, F-score, Recall and DCV. The authors note that a fuller theoretical explanation for ICTLB remains future work.


## Metadata
- **Authors:** Jing Zhang, Hong Tao, Chenping Hou
- **Venue:** N/A
- **Year:** N/A
- **Keywords:** Clustering, excess risk, imbalanced data, learning bound

### Abstract
Imbalanced clustering, where the number of samples varies in different clusters, has arisen from many real data mining applications. It has gained increasing attention. Nevertheless, due to its unsupervised nature, imbalanced clustering is more challenging than its supervised counterpart, i.e., imbalanced classification. Furthermore, existing imbalanced clustering methods are empirically designed and they often lack solid theoretical guarantees, e.g., the excess risk estimation. To solve these important but rarely studied problems, we first propose a novel k-Means algorithm for imbalanced clustering problem with Adaptive Cluster Weight (MACW), together with its excess clustering risk bound analysis. Inspired by this theoretical result, we further propose an improved algorithm called Imbalanced Clustering with Theoretical Learning Bounds (ICTLB). It refines the weights and encourages the optimal trade-off among per-cluster weights by optimizing the excess clustering risk bound. A theoretically-principled justification of ICTLB is provided for verification. Comprehensive experiments on many imbalanced datasets verify the effectiveness of ICTLB in solving cluster imbalanced problems.


## Problem Statement
**Problem:** How to perform clustering on imbalanced (unequal cluster sizes) unlabeled data while avoiding the uniform-size bias of k-means and providing theoretical guarantees (excess clustering risk bounds); design algorithms that adapt cluster weights and optimize theoretical learning bounds to improve imbalanced clustering.

**Motivation:** Imbalanced clustering occurs frequently in real applications (e.g., fraud detection, network security, bioinformatics) where labels are unavailable; standard k-means tends to produce clusters of roughly uniform size (the 'uniform effect'), harming minority-cluster detection; existing imbalanced clustering methods are mostly empirical and lack theoretical excess risk guarantees, motivating methods with principled bounds and improved performance.

**Confidence:** high

### Existing Limitations
- Existing imbalanced clustering methods are empirically designed and often lack solid theoretical guarantees (e.g., excess risk analysis).
- k-means exhibits a uniform effect that biases toward equal-sized clusters and can misassign majority-cluster points to minority clusters.
- Prior theoretical analyses for k-means do not directly extend to data-dependent (imbalanced) distributions or provide per-cluster excess risk bounds.


## Proposed Method

### Imbalanced Clustering with Theoretical Learning Bounds (ICTLB) (Main Method)

**Algorithm Steps:**
1. Initialize cluster centers C^0 and set initial cluster weights λ_r = 1/k.
2. For each data point, assign it to cluster r minimizing ||x_i - c_r||^2 * λ_r^α.
3. Update cluster weights λ_r (MACW: via equation based on within-cluster SSE; ICTLB: set λ_r^α = W * |θ_r|^{α/(2(1-α))}).
4. Update cluster centers as the mean of assigned points.
5. Repeat assignment, weight update, and center update until convergence.

**Pseudocode:**
```
Input: dataset S = {x_i}_{i=1}^n, k. Initialize centers C^0, set λ_r = 1/k for r=1..k. repeat: 1) For each i assign r = argmin_j ||x_i - c_j||^2 * λ_j^α; 2) Update λ_r: MACW: λ_r = (sum_{x in θ_r} ||x - c_r||^2)^{1/(1-α)} / sum_{s=1..k} (sum_{x in θ_s} ||x - c_s||^2)^{1/(1-α)}; ICTLB: λ_r^α = W * |θ_r|^{α/(2(1-α))}; 3) Update c_r = (1/|θ_r|) sum_{x in θ_r} x; until convergence. Output: cluster labels and centers.
```

**Parameters:**
- `α (alpha)`: constant (paper discusses general α; in ICTLB experiments they set α = -1/2) - exponent controlling influence of cluster weight in weighted distance (λ_r^α)
- `W`: {0.1, 0.2, 0.3, 1, 2, 3} (tuned) - global hyperparameter in ICTLB scaling the weight formula λ_r^α = W * |θ_r|^{α/(2(1-α))}
- `λ_r (initial)`: 1/k - initial per-cluster weights
- `k`: user-specified (number of clusters) - number of clusters for k-means framework

**Inputs:** Dataset S = {x_i} (normalized so ||x|| ≤ 1), number of clusters k
**Outputs:** Cluster assignments (labels) for data points, cluster centers C, learned per-cluster weights λ_r

### Stage 1 — MACW (Adaptive cluster-weighted k-means)

**Algorithm Steps:**
1. Initialize centers C^0 and set initial λ_r = 1/k.
2. Assign each point to the cluster minimizing weighted squared distance ||x_i - c_j||^2 * λ_j^α.
3. Update cluster weights λ_r using a closed-form expression based on within-cluster squared errors (equation (13)).
4. Update centers as cluster means; iterate until convergence.

### Stage 2 — Theoretical analysis (excess clustering risk)

**Algorithm Steps:**
1. Formulate empirical and expected clustering risk with weights.
2. Define clustering Rademacher complexity and derive upper bounds (Lemma 1–3).
3. Prove Theorem 1: an excess risk bound for MACW.

### Stage 3 — ICTLB (optimize per-cluster weights via fine-grained bound)

**Algorithm Steps:**
1. Derive per-cluster excess risk bound (Theorem 2) averaging cluster-wise bounds.
2. Optimize simplified bound (binary case) to obtain λ_r^α ∝ |θ_r|^{α/(2(1-α))}.
3. Set λ_r using the derived formula (with hyperparameter W) and run weighted k-means updates (assignment and center updates).


## Evaluation

### Metrics
- **Accuracy:** 
- **F-score:** 
- **Recall:** 
- **DCV:** 

### Datasets
- **glass1:** 214
  - Source: KEEL / UCI variants (from Table II)
- **contraceptive:** 1473
  - Source: KEEL / UCI variants (from Table II)
- **haberman:** 306
  - Source: KEEL / UCI variants (from Table II)
- **wpbc:** 198
  - Source: KEEL / UCI variants (from Table II)
- **newthyroid:** 215
  - Source: KEEL / UCI variants (from Table II)
- **ecoli3:** 336
  - Source: KEEL / UCI variants (from Table II)
- **pageblocks0:** 5472
  - Source: KEEL / UCI variants (from Table II)
- **vowel0:** 988
  - Source: KEEL / UCI variants (from Table II)
- **balance_uni:** 625
  - Source: KEEL / UCI variants (from Table II)
- **dermatology6:** 358
  - Source: KEEL / UCI variants (from Table II)
- **shuttle-2vs5:** 3316
  - Source: KEEL / UCI variants (from Table II)
- **WDBC:** 
  - Source: UCI (used in application to breast cancer)
- **BCC:** 
  - Source: UCI (Breast Cancer Coimbra; used in application)

### Baseline Comparisons
- **k-means**
- **k-means++**
- **density-k-means++**
- **DEC (Deep Embedded Clustering)**
- **IDEC (Improved DEC)**
- **SMCL (Self-adaptive Multiprototype-based Competitive Learning)**
- **MACW** [Proposed]


## Limitations & Future Work

### Limitations
- [Explicit] The theoretical analysis of ICTLB’s success is not fully clear (acknowledged by authors).
- [Explicit] This work only provides excess clustering risk bounds for k-means; theoretical properties of other clustering methods are not studied.
- [Implicit] ICTLB requires tuning hyperparameter W (sensitivity and selection needed).
- [Implicit] Evaluation limited to selected imbalanced datasets from KEEL/UCI; generalization to other domains not shown.
- [Implicit] Potential sensitivity to initialization (they use density-k-means++ initialization) and to choice of α.

### Future Work
- Provide deeper theoretical analysis explaining ICTLB’s empirical success.
- Study theoretical properties and excess risk bounds for other clustering algorithms beyond k-means.
- Explore extensions and validations on broader datasets and application domains.


## Cross-Reference Validation

### Unreferenced Figures
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/0472890e2381878dcb352d17e51c2d99b1c913a76e30424fea3488522edd80c4.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/0ab0084833d824451729516f8ba553de64a450f95fd122be712e4971a435e61a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/0d29fa6c4ca7dc5e1ca2748929e65d410a33bccf686b64ef02b32c1d8c5bec29.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/0f72868c1147054b6c9374427daf5fe67ffce852923242f0a1eea296c611ef17.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/10ad1eb2115b738beddf492dab727f95d651885a6aafce5483a9aafefa331850.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/15761c32f6948f8ac66bfe00f685db9ece296cc8c8006bff3ab06dacf0f87680.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/14220db2620180c46197168899410f9272d2d4699c997de246002e6d82ea8d75.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/164199ab1aeb0090f126f6abf6ad736a9de6f93eec235118a418ff681fdb4075.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/10e57ee36a07bb2f420a206f245804add9d25aa71e5595c42e0a90ea83e64b3e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/17b4564926d1ede9ffd1337da7ba766d076ef979b8b8a0e393651e437f5e5779.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/18abfa31235c25a4c74375897a8fa08df2f9d51992407b278695a97ebc819968.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/1be7d20b4ef434f9d68bc23c0a159d26a434c5822adb946386f995eb0e062fd1.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/1d1e885eea4113f12e9adae7d0aef77743350a201148449288c84b543bbc6b5e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/18de9ecdb69c99f46a0d4ba05b8adb6417b95a4193dbb74b834ca86f1c234e08.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/1f7730b28fc1bd91fed3dec4897a932bf76e8716bc29b5f096594f5b1d132171.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/208b1243581ddcfbe843d7e06c92500ea5aa77f9406a460886313ca2f932177c.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/28a719bdf5e635dd6420e06ab424c992b7bbf5609ad17cbb052f089043d041eb.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/227efd32cab51c25ab91d3c0b8200ab3f6469a5bc9eb3ddd34bdf9277a6b3533.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/2a9096e585c86a12f938405a1f81565acb93fe34a5a527251c03b37cc046eea2.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/2b367510137872547ab305eed7f8fafab0b8ccde7d3827724425da6f6d95a9a6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/2def150190f4f6e664b0f76a7fe56a1c813c3abd420648e67595d910bfa54a78.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/3454ebea51b6aa22a94ede3991125e6bedb1bc9aa27b79fc74bf1d4cb832728b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/338d15659cef1bdc736a6dfba1fc23a60bcc290b9b270e8940328095d732c256.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/3bbf951d2846ed21a99391f2930838d0e463cb2e01d8e16692e25888c4b1a842.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/3bd00e7214544ba9836a2192130c80b4ca88f55f00af75a0cff459f8f3ca1234.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/3dd7dda54e4afefe908c0e49425c764ee4af90f6a5d307b932402dfa36dc6d4b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/3c90338a76227386e3b21eacb45935d2c1eb14df26c37b3060507a2df77dccb7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/420f10924379ac37ad1d9c6ab44ecf3ffe4887aca20407abfbd70a6aa1cf38a0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/41e4d80961a3c86634e9a85b634a0383c85914ed3fe7f684eb3acac66e2b2fd5.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/44d9a6ebb6d89df01e95a29479d369af45c089fc522a3f94b80df52463a54fc3.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/4333d7a547d833dece21c5fd3f174e2490c0b6342d2f6f1d607457fc339dd81b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/4c829f78d1b8b9734f2e389cadbe7cc9fdd2c277df4d36794873c23f35966293.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/4b6d8194e390a140ec4368355c8a2e8d0e4e4f8983dd0533003b5306992cd6d9.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/51c5c1e2e758ef73b4577f460d31fe1eda7f47fa2b120348175ee9ae3acb9a81.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/48a2ef089f8f50cba2d4a6b94376dd5cd5a18e57584490bc54e7a6afc24af657.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/4f61f1c77d9a01011c179da580c9bb4d2bea45577bc0b084bb043cbc9b0fa1f6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/53491938da0a4a4e8d32033d536b9f018a87774be6d94a02cec2e65718c1da9f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/57cfd2162754217262bc8a01408ff9274de4334e865288949ea465d82f72c31a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/59510bf1f8f6875caeb5103c6caaa39d8a1ed9c4034aa87bd1239ad6debdb473.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/5832893f29ca1c81ea2d655556caa92ecee5920f773074626209761323f2ce0a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/59373f0d0c3a8a0d8274daaca2ffc5742471ad4158e5d49771f285132cabfc01.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/5e254f6616986b14edd2935905263f96bf0f729eee058019032a9a661dcee4b9.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/5f65ad7893df82ce842ca3fd11345743d861724b87ad52fac8d9dd91476db68a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/68c9c4284ea41bff2e59285162bf094e23278d464d4a63bb407d0cba6dae5b54.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/6c5d5341d251d82f9edc2c8b7480f175174b8d8e7c62eeb506c588a2ea537a98.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/599e53b83c7c6337f213cfa9f3bbdfdfa96cdf2ce6b21929e34db158a30fa0f8.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/6de9e64a3c1600b5dfc3560787dbee56b3d316d948ae466c46b48e33b74260fd.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/70c957fc65c277879b61dbf11c2bab508e6a29af7150256dee0ae022c852b9c1.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/717570ef8a4ccec9940c56f48374aa7ba4fed400e10b5e1713e147b99c3667aa.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/73e4191544da35f5a7d270c0100d180343c53a78b8ddbad608cb61f9d0dbefc9.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/728291e1967fa7f9342ab1793ee077078ee11ea6fb7990feb347bdea2f3ebcb9.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/7ed0e298fa62f91e5c8c3fc509e801de410a6659c373d58e510deb58e89f75ce.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/843be903981e8f724c755bf5095aea07dd2a99291ce0c5d52c6d07e6e684c035.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/84dbd51b3d6f9d791297fbc149de62fb04d2338b7ed30953fa80c3cc46c7d68f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/8450c7b0e28194f712d00553bb50fa002cbef3f6aed58879c2e16b0dc9a64b4a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/7d7cfd91a323a79694f8254eda3f518c44afcefb8a753c55c51442c2776c9527.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/861dd6a53b7d7cc65f798792e3f51570af32718da8c1fd7f2b6806a03c7ab0b5.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/87d9ff1faf270f96233ac5e73e4282f13cc9c645cb6cb75aa04029b49a0b1771.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/8940b3434b89c4d86ffe66f8d502a98a8ce8ee04e0231388fa3e852a8177cf6f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/8bafe6694c2516c7de6a151728f97c53106e88247f07a0343235bc2009431873.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/8da84a19f963bf5eb6e01b251fb1a3be095ddcc8fc11bdcae7ae096beb4a4972.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/9733ccb9e45dbb7040d2b9cc6fce2174203e40ba3d75e3f094ecac9944e800b0.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/99f57b630d81ecf1488259041665ce557d56342880836bff93be657a7510dfea.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/9faace5a04962cabf3eab60854bf688868b75a8018b6926c2476efa3d0f9a6a7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a28d65b81977325779c591c5fb0ed5fa7e1632a279529a1aa4654dc4c92d317d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a2dd2860bc49d00e71c3f1656f972cafbbd616e0a4b4b86ed4fcf22518100c4b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a39e30138e08a2a131904a88d605899d6a63eef4ad02e3170143635de8c1d613.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a3ed160b6126e6494427569d026e76884f41928cd3fbd1777000636c96bb90c5.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a660b28e3a48df3f905d2f0ff9251abe9adb9d1e853ac27fbf1f8aac912cb4a6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a4d6350a1f42d2f57ac9f4e493eb72a6e477c633c6660a88808db7e5cafb1426.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/a8a24372165870837df85eabdb8f43bc23ff7a3124d04e8606b64fa404780b85.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/ac3ebe22b1e29428cc3807805c4e1578757aa837f6abb174b55054fefb058a3f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/ae89a947aca32b97e3daecdfae91580f96188b3a5c98474db722439914864d97.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/ac2420172a79c40a5cc29ceaec05635f8189e2c6e9fcd03ee879d712dd6a3c3f.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/b594387d8ee3636384e298b2a0b35ddb1b685c0c0613952a522301bbb28f6921.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/af35747a8f6ebb664256100f9ca84dc76ca952897ead415ffff5c33c73d4171b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/b5bddb2d2123619850dd7657799cd5179f9ac6e0d3b405017acb93377c23d89b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/b8f27eda9a7f21d193dccce3dedd1b23d3b0e1f446feeae4437297ab7364773a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/b9ea93eebc6a48d0301f816fdc21b22232339c22ed42235916c3cb7276f45609.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/be20ea4e5dfef1d9c6226582027bdcba3db697ae63a038de1b51a4d090f8bb6d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/ba2ed9caebe51774a158858c573c00415079c38b155b5142c02815396ea7b8a4.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/bef463816864fb01e17893e79c241eeb62d38a2dd3906967a151f6cbfbd52d65.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/bf979d48dac5a021cec0359ec2b69b75a0fed3986f05bc97af2430bc7a756916.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/bfaa38465cb5f438fd8db9b424874a816e445db2a7161c552fe2238676edbd8e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/bfdd48f881fab0357bfa350acc7609dc1bf5a136a206288ead106e81acba1ccb.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/c0219679c4954b34a4c897077d69561948a228f378f164a26005ff1db601536e.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/cad0511d8f3fbe9a8b6a5e3f4b20c6a7a90f71ef8bf668ce0816806b20b1cc01.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/cad1aedde70032212067657f02ab56a1cf373c8669c3eab1efe6d93577cd8a67.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/d1e63847b271ffff6cebf86d45a994c8dfde9af339002213e4f349b520e3631a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/d654ce634dd0414f71747824ad2363301986c0b0d96e65389b91b411d7db941d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/cb64686c79199e86579cc1d7360cb69a3d2b49a1d898985334f123053b1229e3.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/d789588021705324649c7356ac55e8af69fe82521af05935e5e90f4ac564a4e1.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/da63b2f9a73c7dc9491df93239f1a648b2d4fb7f294b667b8f5e84cac60389eb.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/da81ee2c259581f71c7f83208065cc4fc601cacfc5738c0bed72e9828a63d299.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/d8b3dbed7d5b5dd2f25e01f1a303d7c7eeb3f2f080f475eb6e1a626c64ddf6e6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/db3a6ee50164064bdab82c9af36d9f5475022a4d0493c76da0dba9727c37835b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/db0cf2d8e375eb4f206a8e3390377c4bc889de826ea88b0d9d7c903dafd68b3d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/e2d0c13a6960ffb9774ceba289aa2d21dc91997fdc348710e3acdd3e68521743.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/e702fb96e6e70bba468d7dcedafcd6ae7cf0971536904166af566aeb21fec071.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/e968eea14e4c74eeb05e105060342aa6fdece26fddf2dc6838e2aa2173f978c6.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/e880cfdfcda39ad1e66e7ceb7af5804ec75a8a4d3b0904701647007bd9f2c320.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/edd91bfd3401cd1986b7323eb4a7715f861036e702136f07508fb5adac63abf3.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/f0fc21c5975e0da84cb912e4c8f90eb8d5378dc9bf24c9fa6de2f0eb078dc5df.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/f14eaedba1855ab0b48e2835728d9924a1cb67b6070041314a9c9feaa886193b.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/fb07347a01bbca5bb37d0e1f39084a3549c970c3f6d5e61666329a82e79e7e24.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/ee574dcb3be4898a084e0e95e04580fcfc7d14f5be91ba620fd0a2f4c4d60ae7.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/f816403f2105772aa2404bc193ca398ac662798ba57afaab8d28752fd7ddeff5.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/fc90af685d3fd4a6266a5814a12cfb2249b4af0908ff4784d4dc9f8864c1c256.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_525375a4/parsed/fdb72dc590ff5dae28b3b4bb23a8fe57b1f630b1f76c721300ad1b6592bd9b57.jpg

*2 figure references validated as consistent.*


## Analysis Warnings
- ⚠️ Evaluation section not detected
- ⚠️ Methodology section not detected


---

*Analysis cost: $1.36 (111 images analyzed)*