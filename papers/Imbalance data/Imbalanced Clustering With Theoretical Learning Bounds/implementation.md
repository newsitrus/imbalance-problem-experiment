# Implementation Plan — ICTLB Replication

**Paper:** "Imbalanced Clustering With Theoretical Learning Bounds"
**Authors:** Jing Zhang, Hong Tao, Chenping Hou
**Published:** IEEE TKDE, Vol. 35, No. 9, September 2023

---

## Experiment Subset

Chosen to maximise faithfulness to the paper while remaining feasible to run
and allowing qualitative comparison against the paper's text claims.

| Setting | Paper | This replication |
|---|---|---|
| Datasets | 14 (KEEL/UCI) | 4 — haberman, ecoli3, vowel0, shuttle-2vs5 |
| Methods | 8 (incl. DEC, IDEC, SMCL) | 4 — k-means, k-means++, MACW, ICTLB |
| Runs | 10, mean reported | 10, mean reported ✓ |
| α | −1/2 | −0.5 ✓ |
| W grid | {0.1,0.2,0.3,1,2,3} | {0.1,0.2,0.3,1,2,3} ✓ |
| W strategy | fix one=2, tune other | W_fixed=2, tune W_tuned ✓ |
| W selection | by ACC (oracle) | by ACC with y_true ✓ |
| Feature norm | L2 (‖x‖=1) | sklearn normalize(l2) ✓ |
| Init | density-k-means++ | custom density-k-means++ ✓ |
| Convergence | not stated | max_iter=300, tol=1e-4 |
| Metrics | ACC, F-score, Recall, DCV | all four ✓ |

**Excluded methods and why:**
- `density-k-means++` as baseline: kept as initialiser only; standalone results
  would depend on [31] (Fan et al. 2019) which is a separate paper.
- `SMCL`: only described at high level in the ICTLB paper; exact algorithm requires
  Lu et al. 2021 (IEEE Trans. Cybern.).
- `DEC`, `IDEC`: deep learning methods requiring a separate autoencoder setup not
  described in the ICTLB paper.

---

## File Structure

```
papers/Imbalanced Clustering With Theoretical Learning Bounds/
├── src/
│   ├── __init__.py      Public API
│   ├── data.py          KEEL downloader, .dat parser, L2 normaliser
│   ├── algorithms.py    KMeans, KMeansPP, MACW, ICTLB + density-k-means++ init
│   ├── metrics.py       ACC (Hungarian), F-score, Recall, DCV
│   └── pipeline.py      ExperimentPipeline (run / get_summary / compare_with_paper)
├── config.yaml          All experiment settings (single source of truth)
├── ICTLB_Replication.ipynb   User-facing notebook
├── data/                Downloaded & cached dataset .npz files (auto-created)
├── results/             raw_results.csv + summary.csv (auto-created)
└── implementation.md    This file
```

---

## Algorithm Details

### density-k-means++ (initialiser for all methods)

Used by all four methods as the starting-point centre initialisation.

```
1. Estimate local density for each point:
     density[i] = 1 / (mean_kNN_distance[i] + ε),  k_nn = 5
2. Sample first centre: p(x) ∝ density[x]
3. Sample each subsequent centre:
     p(x) ∝ density[x] × min_c d²(x, c)
```

### MACW (Eq. 13)

```
Init: centres = density-k-means++,  λ_r = 1/k
Repeat (max 300 iters, tol 1e-4):
  Assign:  r_i = argmin_r  ‖x_i − c_r‖² · λ_r^α
  Weights: numer_r = (Σ_{x∈θ_r} ‖x−c_r‖²)^{1/(1−α)}
           λ_r = numer_r / Σ_s numer_s
  Centres: c_r = mean(θ_r)
```

### ICTLB (Eq. 36)

```
For each W_tuned in {0.1, 0.2, 0.3, 1.0, 2.0, 3.0}:
  W = [W_fixed=2.0, W_tuned]
  Init: centres = density-k-means++ (same seed across all W candidates)
  Repeat (max 300 iters, tol 1e-4):
    Assign:  r_i = argmin_r  ‖x_i − c_r‖² · λ_r^α
    Weights: λ_r^α = W_r · |θ_r|^{α/(2(1−α))}
             (inner fixed-point: iterate assign+reweight until stable, max 10 steps)
    Centres: c_r = mean(θ_r)
  Compute ACC(y_true, labels)  ← oracle W selection
Select W_tuned maximising ACC; return its labels.
```

---

## Metrics

| Metric | Formula | Note |
|---|---|---|
| ACC | Hungarian-matched accuracy | `scipy.optimize.linear_sum_assignment` |
| F-score | Macro F1 after optimal label alignment | `sklearn.metrics.f1_score(average='macro')` |
| Recall | Macro recall after optimal label alignment | `sklearn.metrics.recall_score(average='macro')` |
| DCV | `std(cluster_sizes) / mean(cluster_sizes)` | Coefficient of variation of predicted cluster sizes; consistent with all qualitative paper observations (larger → worse) |

**Note on DCV:** The exact formula from Xiong et al. 2009 (ref [44]) is embedded
in the PDF as a vector image and could not be extracted.  The CV-based
operationalisation above is consistent with all paper text statements:
- k-means uniform effect → near-equal sizes → small DCV ✓
- density-k-means++ over-concentrates → very unequal sizes → large DCV ✓
- Small DCV does not guarantee good performance ✓

---

## Reproducibility Notes

1. **Random seed:** Each (dataset, method, run) gets a deterministic seed:
   `seed = base_seed + hash(dataset+method) % 10000 + run_idx`
   This ensures full reproducibility while avoiding seed collisions.

2. **Dataset reconstruction:** KEEL datasets are derived versions of originals
   (subsampled/merged).  The paper does not release the exact splits.  Datasets
   are downloaded directly from KEEL's imbalanced section, which is the closest
   publicly available match.

3. **Convergence criterion:** Not stated in the paper.  Using `max_iter=300,
   tol=1e-4` (standard k-means practice).  This is unlikely to affect results
   for small datasets; may affect shuttle-2vs5 slightly.

4. **W strategy:** Paper says "fix one constant W = 2 and tune another."  This
   replication follows that exactly: `W_fixed=2.0`, `W_tuned ∈ W_grid`.

---

## Comparison With Paper

Exact numeric comparison is not possible — Tables III–VI are vector-image
graphics in the PDF with no extractable text.  Comparison is against the
paper's own text claims (Section IV-D, pp. 9606–9607).

| Dataset | Verifiable claim |
|---|---|
| haberman | ICTLB best ACC + F-score |
| ecoli3 | ICTLB best ACC + F-score |
| vowel0 | ICTLB best ACC + F-score + Recall (all three) |
| shuttle-2vs5 | ICTLB best ACC + F-score |

The notebook's comparison table checks each claim automatically.
