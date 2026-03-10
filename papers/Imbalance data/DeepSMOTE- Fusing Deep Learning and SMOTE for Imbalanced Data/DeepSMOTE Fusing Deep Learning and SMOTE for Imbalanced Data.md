---
paper_id: sha256_ce74b016
exported_at: 2026-02-22T19:44:26
source: /home/doanhtran03/Python/paperanal/warehouse/outputs/sha256_ce74b016/report.md/analysis_report.md
format: markdown
---

# DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data


## Executive Summary

This paper addresses class imbalance in deep learning by proposing DeepSMOTE, an end-to-end oversampling method that trains an encoder/decoder to produce low-dimensional embeddings, uses a penalty-augmented reconstruction loss (via permutation of encoded same-class batches) to introduce variance, and applies SMOTE interpolation in embedding space to generate synthetic minority-class images which are decoded back to the input domain. Experiments on five image benchmarks (MNIST, FMNIST, CIFAR-10, SVHN, CelebA) show DeepSMOTE outperforms pixel-based and GAN-based oversampling baselines across skew-insensitive metrics (ACSA, GM, F1), yields higher visual quality (lower FID), and demonstrates robustness and stability across a wide range of imbalance ratios.


## Metadata
- **Authors:** Damien Dablain, Bartosz Krawczyk, Nitesh V. Chawla
- **Venue:** N/A
- **Year:** N/A
- **Keywords:** Class imbalance, deep learning, machine learning, oversampling, synthetic minority oversampling technique (SMOTE)

### Abstract
Despite over two decades of progress, imbalanced data is still considered a significant challenge for contemporary machine learning models. Modern advances in deep learning have further magnified the importance of the imbalanced data problem, especially when learning from images. Therefore, there is a need for an oversampling method that is specifically tailored to deep learning models, can work on raw images while preserving their properties, and is capable of generating highquality, artificial images that can enhance minority classes and balance the training set. We propose Deep synthetic minority oversampling technique (SMOTE), a novel oversampling algorithm for deep learning models that leverages the properties of the successful SMOTE algorithm. It is simple, yet effective in its design. It consists of three major components: 1) an encoder/decoder framework; 2) SMOTE-based oversampling; and 3) a dedicated loss function that is enhanced with a penalty term. An important advantage of DeepSMOTE over generative adversarial network (GAN)-based oversampling is that DeepSMOTE does not require a discriminator, and it generates high-quality artificial images that are both information-rich and suitable for visual inspection. DeepSMOTE code is publicly available at https://github.com/dd1github/DeepSMOTE.


## Problem Statement
**Problem:** Deep learning models are vulnerable to class-imbalanced training data; there is a need for an oversampling method tailored to deep models that can operate end-to-end on raw images, learn low-dimensional embeddings, and generate high-quality, information-rich artificial images to balance minority classes and reduce bias.

**Motivation:** Imbalanced class distributions bias classifiers toward majority classes, causing poor recognition of minority classes which is unacceptable in real-world applications (e.g., medicine). Existing deep-learning-focused resampling methods (pixel-based and GAN-based) have limitations: pixel-based cannot capture complex image properties and GANs require large data, are hard to tune, and suffer mode collapse. Hence an oversampling method that preserves image properties, works end-to-end with deep architectures, and generates high-quality images is needed.

**Confidence:** high

### Existing Limitations
- Pixel-based oversampling cannot capture complex spatial properties of images and often generates meaningless artificial images.
- GAN-based oversampling requires significant data, is difficult to tune, and is prone to mode collapse.
- Standard SMOTE and many SMOTE variants struggle with multimodal data, high intraclass overlap, noise, and are not directly suited for deep image representations.
- Prior deep oversampling approaches often do not operate end-to-end on raw images while also producing visually inspectable outputs.


## Proposed Method

### DeepSMOTE (Main Method)

**Algorithm Steps:**
1. Train encoder/decoder end-to-end on imbalanced raw images using reconstruction loss plus a penalty term derived by encoding same-class batches, permuting encoded order, decoding, and computing MSE.
2. After training, for each minority class: encode minority examples to embeddings, apply SMOTE in embedding space to create synthetic embeddings, decode synthetic embeddings to generate artificial images.
3. Augment training set with decoded synthetic images (balanced set) and use to train a classifier (ResNet-18 used in experiments).

**Pseudocode:**
```
Algorithm 1 DEEPSMOTE
Data: B: batches of imbalanced training data (D) B = { b1, b2, ..., bn }
Input: Model parameters: Θ = { Θ0, Θ1, ..., Θj }; Learning Rate: α
Output: Balanced training set.
Symbols: RL - Reconstruction loss; PL - Penalty loss; TL - Total loss; C - Set of classes in D; CM - Set of minority classes in D; G - Set of generated and encoded examples; S - Set of generated and decoded data (balanced).
Train the Encoder / Decoder:
for e ← epochs do
  for b ← B do
    Eb ← encode(b)
    Db ← decode(Eb)
    RL = (1/n) Σ_{i=1}^n (Dbi - bi)^2
    CD ← randomly sample a class from C
    Cb ← randomly sample |b| instances from CD
    ES ← encode(Cb)
    PE ← permute_order(ES)
    DP ← decode(PE)
    PL = (1/n) Σ_{i=1}^n (DPi - CDi)^2
    TL = RL + PL
Generate Samples:
foreach m ← minority class (CM) do
  Cmd ← select Cm (imbalanced data)
  Em ← encode(Cmd)
  Gm ← SMOTE(Em)
  Sm ← decode(Gm)
```

**Parameters:**
- `Θ (model parameters)`: learned - Weights of encoder and decoder networks trained end-to-end
- `α (learning rate)`: 0.0002 - Learning rate used with Adam optimizer
- `optimizer`: Adam - Stochastic optimizer used for training encoder/decoder (Adam with lr 0.0002)
- `latent_dimension`: 300 (MNIST/FMNIST), 600 (CIFAR-10/SVHN/CelebA) - Dimension of final dense latent layer produced by encoder
- `encoder_conv_channels`: [64, 128, 256, 512] - Number of channels per convolutional layer in encoder
- `kernel_sizes`: [4, 4, 4, 4] - Kernel sizes for encoder convolutional layers
- `strides`: [2, 2, 2, 2] - Strides for encoder convolutional layers
- `activation_functions`: LeakyReLU in encoder, ReLU in decoder, Tanh final layer - Activations used in encoder/decoder
- `training_epochs`: 50-350 - Number of epochs (varies depending on when training loss plateaus)
- `SMOTE_scaling_factor`: random in [0,1] - Random interpolation factor used by SMOTE to combine embeddings

**Inputs:** Batches of imbalanced training data (raw images) from datasets such as MNIST, FMNIST, CIFAR-10, SVHN, CelebA
**Outputs:** Balanced training set containing decoded artificial images generated by SMOTE in embedding space, Synthetic images for minority classes (decoded from SMOTEd embeddings)

### Encoder/Decoder Framework

**Algorithm Steps:**
1. Feed imbalanced dataset in batches to encoder/decoder
2. Compute reconstruction loss between decoded output and original input
3. Use majority-class examples to help learn general reconstruction patterns

### SMOTE-based Oversampling in Embedding Space

**Algorithm Steps:**
1. Encode minority-class examples to embedding space
2. For each minority example, select a neighbor embedding and compute convex combination via SMOTE (random factor in [0,1])
3. Produce generated embedding points for minority classes

### Dedicated Loss Function with Penalty Term

**Algorithm Steps:**
1. Randomly sample a class and batch of examples from that class
2. Encode them, permute the order of encoded vectors, decode the permuted encodings
3. Compute penalty loss as MSE between decoded permuted outputs and the original (same-class) images and add to reconstruction loss


## Evaluation

### Metrics
- **Average class specific accuracy (ACSA):**  %
- **Macro-averaged geometric mean (GM):** 
- **Macro-averaged F1 measure (FM):** 
- **Frechet Inception Distance (FID):** 

### Datasets
- **MNIST:** 60,000 training images (used splits described in paper)
  - Source: LeCun et al. (original MNIST)
- **Fashion-MNIST (FMNIST):** 60,000 training images
  - Source: Xiao et al. (Fashion-MNIST)
- **CIFAR-10:** 50,000 training images
  - Source: Krizhevsky et al. (CIFAR-10)
- **SVHN:** 73,257 training images
  - Source: Netzer et al. (SVHN)
- **CelebA:** 200,000 images (5 selected attribute classes used, images resized to 3x32x32)
  - Source: Liu et al. (CelebA)

### Baseline Comparisons
- **SMOTE**
- **AMDO**
- **MC-CCR**
- **MC-RBO**
- **BAGAN**
- **GAMO**


## Limitations & Future Work

### Limitations
- [Explicit] Current DeepSMOTE does not incorporate class-level and instance-level difficulty information (authors propose this as future enhancement).
- [Explicit] Performance is lower on datasets where classes do not share similar attributes (e.g., CIFAR-10) due to limited transfer of reconstruction patterns between dissimilar classes.
- [Explicit] Method was evaluated on image datasets and has not yet been extended/tested on other modalities.
- [Implicit] Evaluation uses a single classifier architecture (ResNet-18) for downstream classification; generality to other classifiers is not shown.
- [Implicit] Experiments focus on vision/image datasets; applicability to non-image data (text, graph) is not demonstrated.
- [Implicit] No explicitly reported wall-clock training cost comparisons versus GAN baselines; computational cost of encoder/decoder + SMOTE for large images/classes is not fully explored.

### Future Work
- Enhance DeepSMOTE with class-level and instance-level difficulty information and instance-level penalties to focus on borderline/overlapping instances and discard outliers/noisy examples.
- Adapt DeepSMOTE for continual and lifelong learning scenarios to handle dynamic class ratios and mitigate catastrophic forgetting.
- Extend DeepSMOTE to other data modalities such as graphs and text.


## Cross-Reference Validation

### Unreferenced Figures
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/1bc48e79fbe32a7213d4dcc54e873fb966289d44a933b6287e22d42974865f56.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/825482ac2e6c54a131f332e464635644b93a8222c80d943ac51f235d1e13fd8a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/7b4d05fe7a52c1e79a2a5ae409265db19c8da17b5a83281bc097a151c0bd3790.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/a1de691e0caaedb19b91ffdebb0ab14791eeb7dc59c939be212c39c14feda971.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/cf94b2810efe3c26ce39c35f81174ca8575b17ad75e35f7b8e6617979e3bd65a.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/d5a5908aca5b60c9397440ffde34770c0ff5af40ac3322d44760814670fb377d.jpg
- /home/doanhtran03/Python/paperanal/warehouse/library/sha256_ce74b016/parsed/ecc1b3a41d0eba99fbf26e9487fd4000b1e6c6de66303c3dc6436ccb57df37e0.jpg

*10 figure references validated as consistent.*


## Analysis Warnings
- ⚠️ Evaluation section not detected
- ⚠️ Methodology section not detected


---

*Analysis cost: $1.02 (17 images analyzed)*