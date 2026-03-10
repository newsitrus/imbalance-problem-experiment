# DeepSMOTE Replication — Implementation Plan

## Scope
Replication of **Experiment 1 (Table II, imbalanced test set)** from:
> Dablain et al., "DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data", IEEE TNNLS, Vol. 34, No. 9, Sep 2023.

**Subset:** MNIST dataset only, DeepSMOTE vs. BAGAN baseline, 5-fold CV, imbalanced test set.

---

## Settings Matrix

### Chosen by User (Subset)
| Setting | Value | Paper Reference |
|---------|-------|-----------------|
| Dataset | MNIST | §V-A-1 |
| Baseline | BAGAN | §V-A-3 |
| Test protocol | Imbalanced test set | §V-A-6 (Table II) |
| Downstream classifier | ResNet-18 | §V-A-4 |

### Matched Exactly to Paper
| Setting | Value | Paper Reference |
|---------|-------|-----------------|
| Imbalance distribution | [4000,2000,1000,750,500,350,200,100,60,40] | §V-A-2 |
| Imbalance ratio | 100:1 | §V-A-2 |
| Encoder conv channels | [64, 128, 256, 512] | §V-A-8 |
| Kernel sizes | [4, 4, 4, 4] | §V-A-8 |
| Strides | [2, 2, 2, 2] | §V-A-8 |
| Latent dimension | 300 | §V-A-8 |
| Encoder activation | LeakyReLU (slope=0.2) | §V-A-8 |
| Decoder activation | ReLU + Tanh (final) | §V-A-8 |
| Batch normalisation | After every conv layer | §V-A-8 |
| Optimizer (encoder/decoder) | Adam, lr=0.0002 | §V-A-8 |
| Cross-validation | 5-fold, stratified | §V-A-6 |
| Metrics | ACSA, GM, FM | §V-A-5 |
| Statistical significance | α=0.05 | §V-A-7 |
| Framework | PyTorch | §V-A-8 |

### Inferred / Defaulted (Not Specified in Paper)
| Setting | Value | Rationale |
|---------|-------|-----------|
| Batch size | 64 | Standard DCGAN default |
| Training epochs (encoder/decoder) | 100 | Midpoint of paper range 50–350; early-stop on plateau |
| SMOTE k-neighbours | 5 | Standard SMOTE default (Chawla et al. 2002) |
| Image input size | 28→32 (padded) | Required for 4 stride-2 conv layers to reach 2×2 spatial |
| ResNet-18 optimizer | Adam, lr=1e-3, wd=1e-4 | Standard for MNIST fine-tuning |
| ResNet-18 epochs | 30 | Sufficient convergence on balanced MNIST |
| ResNet-18 input | 1→3 channels (repeat) | Simplest adaptation; avoids modifying weights |
| BAGAN AE epochs | 50 | Standard pre-training budget |
| BAGAN GAN epochs | 100 | Matching DeepSMOTE encoder/decoder epochs |
| BAGAN class conditioning | One-hot concat (image + label channels) | Simplest conditional GAN formulation |
| Random seed | 42 | Reproducibility |

---

## Architecture Details

### Encoder (paper-matched)
```
Input:  (B, 1, 28, 28) → F.pad → (B, 1, 32, 32)
Conv1:  (B,  64, 16, 16)  k=4, s=2, p=1 + BN + LeakyReLU(0.2)
Conv2:  (B, 128,  8,  8)  k=4, s=2, p=1 + BN + LeakyReLU(0.2)
Conv3:  (B, 256,  4,  4)  k=4, s=2, p=1 + BN + LeakyReLU(0.2)
Conv4:  (B, 512,  2,  2)  k=4, s=2, p=1 + BN + LeakyReLU(0.2)
Flatten → Linear(2048, 300) → z (B, 300)
```

### Decoder (paper-matched, mirrored)
```
z (B, 300) → Linear(300, 2048) → Reshape (B, 512, 2, 2)
ConvT1: (B, 256,  4,  4)  k=4, s=2, p=1 + BN + ReLU
ConvT2: (B, 128,  8,  8)  k=4, s=2, p=1 + BN + ReLU
ConvT3: (B,  64, 16, 16)  k=4, s=2, p=1 + BN + ReLU
ConvT4: (B,   1, 32, 32)  k=4, s=2, p=1 + Tanh
Crop   → (B,   1, 28, 28)  [remove 2-pixel border: x[:,:,2:30,2:30]]
```

### DeepSMOTE Loss
```
RL = MSE(Decoder(Encoder(x_batch)), x_batch)
PL:
  cls_batch = sample_n_from_class(random_class, n=batch_size)
  z_cls     = Encoder(cls_batch)
  z_perm    = z_cls[torch.randperm(n)]   # permute encoded order
  x_perm    = Decoder(z_perm)
  PL        = MSE(x_perm, cls_batch)     # compare to ORIGINAL (not permuted) images
TL = RL + PL
```

### BAGAN
```
Phase 1 — Autoencoder pre-training:
  Encoder + Decoder trained with standard MSE reconstruction loss
  Class conditioning: one-hot(c) appended as extra image channels

Phase 2 — Conditional GAN:
  Generator   = Decoder init from AE (conv-transpose layers copied)
               Input: noise (300-dim) + class one-hot (10-dim) → project → decoder
  Discriminator = Encoder-like conv stack
                  Input: image + one-hot channels → logit → BCEWithLogitsLoss
  Training: DCGAN-style (no gradient penalty)

Phase 3 — Generation:
  For each minority class: sample noise → G(noise, class_label) → synthetic images
```

---

## Pipeline Flow (per fold)
```
For each CV fold (k=5):
  ┌─ DeepSMOTE ────────────────────────────────────────────────────────┐
  │  1. Train Encoder/Decoder on X_train (imbalanced, 100 epochs)      │
  │  2. Encode all X_train → embeddings Z_train                        │
  │  3. SMOTE(Z_train, y_train) → Z_synthetic, y_synthetic             │
  │  4. Decode Z_synthetic → X_synthetic                               │
  │  5. X_balanced = concat(X_train, X_synthetic) [10 × majority_n]   │
  │  6. Train ResNet-18 on X_balanced (30 epochs)                      │
  │  7. Evaluate on X_test (imbalanced) → ACSA, GM, FM                 │
  └────────────────────────────────────────────────────────────────────┘
  ┌─ BAGAN ─────────────────────────────────────────────────────────────┐
  │  1. Train AE on X_train (50 epochs, conditional)                   │
  │  2. Init G from AE decoder; train conditional GAN (100 epochs)     │
  │  3. Generate synthetic minority images → X_balanced                │
  │  4. Train ResNet-18 on X_balanced (30 epochs)                      │
  │  5. Evaluate on X_test (imbalanced) → ACSA, GM, FM                 │
  └────────────────────────────────────────────────────────────────────┘

Report: mean ± std across 5 folds for each metric × each method
```

---

## Project Structure
```
papers/DeepSMOTE.../
├── implementation.md       ← this file
├── requirements.txt        ← dependencies
├── replication.ipynb       ← user-facing Jupyter notebook
└── src/
    ├── __init__.py
    ├── config.py           ← all hyperparameters (paper-matched)
    ├── data.py             ← MNIST loading, imbalancing, CV splits
    ├── architecture.py     ← Encoder, Decoder, Discriminator, ResNet-18
    ├── deepsmote.py        ← DeepSMOTE training + SMOTE generation
    ├── bagan.py            ← BAGAN AE pre-training + conditional GAN
    ├── classifier.py       ← ResNet-18 training + evaluation
    ├── metrics.py          ← ACSA, GM, FM
    └── pipeline.py         ← end-to-end ReplicationPipeline
```

---

## Comparison with Original Paper
- **Rank order** (DeepSMOTE > BAGAN on ACSA and GM): directly verifiable
- **Exact numbers**: Tables II/III in the paper are image-rendered (not text-extractable); visual side-by-side comparison only
- **FID**: Only reported for CelebA (not MNIST) in the paper; not computed here
- **Statistical significance**: Friedman + Bayesian Wilcoxon tests replicated at α=0.05

---

## Known Deviations from Paper
| Deviation | Impact |
|-----------|--------|
| Batch size not reported (using 64) | Minor — affects optimisation trajectory only |
| Epochs fixed at 100 (paper: 50–350) | Minor — may affect final quality slightly |
| SMOTE k not reported (using 5) | Minor — standard default |
| ResNet-18 training details not reported | Minor — standard settings used |
| Random seeds not fixed in paper | Results will have some variance across runs |
