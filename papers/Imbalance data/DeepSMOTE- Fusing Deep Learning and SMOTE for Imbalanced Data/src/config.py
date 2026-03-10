"""
Centralised configuration for the DeepSMOTE replication experiment.

Every value is either:
  [PAPER]    — directly stated in the paper (§V-A)
  [INFERRED] — standard default used when the paper is silent
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    # [PAPER §V-A-1] Dataset
    dataset: str = "MNIST"

    # [PAPER §V-A-2] Imbalanced class counts (class 0 = majority, class 9 = rarest)
    # MNIST: 10 classes, sum = 9 000, IR = 4000/40 = 100:1
    imbalance_counts: List[int] = field(
        default_factory=lambda: [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    )

    # [PAPER §V-A-6] 5-fold stratified cross-validation
    n_folds: int = 5

    # [INFERRED] Random seed for reproducibility
    random_seed: int = 42

    # [INFERRED] MNIST is 28×28; padded to 32×32 to fit 4 stride-2 conv layers
    original_img_size: int = 28
    padded_img_size: int = 32
    n_channels: int = 1
    n_classes: int = 10

    # Data directory (torchvision downloads here)
    data_dir: str = "./data"


@dataclass
class EncoderDecoderConfig:
    # [PAPER §V-A-8] Architecture specs
    latent_dim: int = 300                                   # MNIST/FMNIST
    conv_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    leaky_relu_slope: float = 0.2                           # standard DCGAN


@dataclass
class DeepSMOTEConfig:
    enc_dec: EncoderDecoderConfig = field(default_factory=EncoderDecoderConfig)

    # [PAPER §V-A-8] Adam, lr=0.0002
    learning_rate: float = 0.0002
    adam_betas: Tuple[float, float] = (0.5, 0.999)         # standard DCGAN betas

    # [PAPER §V-A-8] 50–350 epochs until loss plateaus; we use 100 as default
    n_epochs: int = 100

    # [INFERRED] standard batch size for DCGAN-based models
    batch_size: int = 64

    # [INFERRED] standard SMOTE default (Chawla et al. 2002)
    smote_k_neighbors: int = 5


@dataclass
class BAGANConfig:
    enc_dec: EncoderDecoderConfig = field(default_factory=EncoderDecoderConfig)

    # [INFERRED] matching DeepSMOTE lr
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002
    adam_betas: Tuple[float, float] = (0.5, 0.999)

    # [INFERRED] standard pre-training + GAN budget
    n_epochs_autoencoder: int = 50
    n_epochs_gan: int = 100

    batch_size: int = 64


@dataclass
class ClassifierConfig:
    # [PAPER §V-A-4] ResNet-18; training hypers not specified → standard settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 30
    batch_size: int = 64


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    deepsmote: DeepSMOTEConfig = field(default_factory=DeepSMOTEConfig)
    bagan: BAGANConfig = field(default_factory=BAGANConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Runtime
    device: str = "auto"          # "auto" selects cuda if available, else cpu
    results_dir: str = "./results"
    verbose: bool = True


def get_config() -> Config:
    """Return default Config matching the paper as closely as possible."""
    return Config()
