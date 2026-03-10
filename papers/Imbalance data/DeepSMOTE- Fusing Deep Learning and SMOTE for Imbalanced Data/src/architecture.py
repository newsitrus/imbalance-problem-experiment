"""
Neural network architectures for the DeepSMOTE replication.

All dimensions match the paper exactly (§V-A-8):
  - Encoder: 4 conv layers, C=[64,128,256,512], k=4, s=2, p=1, BN+LeakyReLU
  - Decoder: mirrored transposed conv, BN+ReLU, Tanh on final
  - Latent dim: 300 (MNIST/FMNIST)
  - Input handling: MNIST 28×28 padded to 32×32 inside Encoder

BAGAN additions:
  - BAGANGenerator  : class-conditional generator (decoder + class embedding)
  - BAGANDiscriminator : conditional discriminator (encoder + class channels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from config import EncoderDecoderConfig


# ─────────────────────────────────────────────────────────────────
#  Shared Encoder / Decoder
# ─────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    DCGAN-based encoder (paper §V-A-8).

    Input : (B, 1, 28, 28)  — MNIST (padded internally to 32×32)
    Output: (B, latent_dim) — embedding vector z
    """

    def __init__(self, cfg: EncoderDecoderConfig, in_channels: int = 1):
        super().__init__()
        channels = cfg.conv_channels          # [64, 128, 256, 512]
        k, s, p  = cfg.kernel_size, cfg.stride, cfg.padding
        slope    = cfg.leaky_relu_slope

        layers, c_in = [], in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(slope, inplace=True),
            ]
            c_in = c_out

        self.conv = nn.Sequential(*layers)
        # After 4 stride-2 convs on 32×32 input: spatial = 32/2^4 = 2
        self._spatial = 2
        self.fc = nn.Linear(channels[-1] * self._spatial * self._spatial,
                            cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad MNIST 28→32 (2 pixels each side)
        if x.shape[-1] == 28:
            x = F.pad(x, (2, 2, 2, 2))
        z = self.conv(x)
        z = z.view(z.size(0), -1)
        return self.fc(z)


class Decoder(nn.Module):
    """
    Mirrored DCGAN-based decoder (paper §V-A-8).

    Input : (B, latent_dim)
    Output: (B, 1, 28, 28) in [-1, 1]  (Tanh activation, then cropped from 32)
    """

    def __init__(self, cfg: EncoderDecoderConfig, out_channels: int = 1,
                 original_img_size: int = 28):
        super().__init__()
        self._spatial = 2
        self._orig_size = original_img_size
        channels = cfg.conv_channels          # [64, 128, 256, 512]
        k, s, p  = cfg.kernel_size, cfg.stride, cfg.padding

        self.fc = nn.Linear(cfg.latent_dim,
                            channels[-1] * self._spatial * self._spatial)

        # Reversed: 512 → 256 → 128 → 64 (BN+ReLU), then 64 → out (Tanh)
        rev = list(reversed(channels))        # [512, 256, 128, 64]
        layers, c_in = [], rev[0]
        for c_out in rev[1:]:
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s,
                                   padding=p, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        layers += [
            nn.ConvTranspose2d(c_in, out_channels, kernel_size=k, stride=s,
                               padding=p, bias=False),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 512, self._spatial, self._spatial)
        x = self.conv(x)                   # → (B, 1, 32, 32)
        if x.shape[-1] != self._orig_size:
            pad = (x.shape[-1] - self._orig_size) // 2
            x = x[:, :, pad: pad + self._orig_size,
                         pad: pad + self._orig_size]  # → (B, 1, 28, 28)
        return x


# ─────────────────────────────────────────────────────────────────
#  BAGAN — Conditional Generator & Discriminator
# ─────────────────────────────────────────────────────────────────

class BAGANGenerator(nn.Module):
    """
    Class-conditional generator for BAGAN.

    Conditioning: class one-hot (n_classes dims) is concatenated with the
    noise vector z before the projection linear layer.
    Conv-transpose layers are initialised from the pre-trained AE decoder.
    """

    def __init__(self, cfg: EncoderDecoderConfig, n_classes: int = 10,
                 out_channels: int = 1, original_img_size: int = 28):
        super().__init__()
        self._spatial = 2
        self._orig_size = original_img_size
        channels = cfg.conv_channels
        k, s, p  = cfg.kernel_size, cfg.stride, cfg.padding

        # Project (latent_dim + n_classes) → 512 × 2 × 2
        self.fc = nn.Linear(cfg.latent_dim + n_classes,
                            channels[-1] * self._spatial * self._spatial)

        rev = list(reversed(channels))
        layers, c_in = [], rev[0]
        for c_out in rev[1:]:
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s,
                                   padding=p, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        layers += [
            nn.ConvTranspose2d(c_in, out_channels, kernel_size=k, stride=s,
                               padding=p, bias=False),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, labels_onehot], dim=1)
        x = self.fc(inp)
        x = x.view(x.size(0), 512, self._spatial, self._spatial)
        x = self.conv(x)                   # → (B, 1, 32, 32)
        if x.shape[-1] != self._orig_size:
            pad = (x.shape[-1] - self._orig_size) // 2
            x = x[:, :, pad: pad + self._orig_size,
                         pad: pad + self._orig_size]
        return x

    def init_from_decoder(self, decoder: Decoder) -> None:
        """Copy conv-transpose weights from a pre-trained AE decoder."""
        for g_layer, ae_layer in zip(self.conv, decoder.conv):
            if hasattr(g_layer, 'weight') and hasattr(ae_layer, 'weight'):
                g_layer.weight.data.copy_(ae_layer.weight.data)
                if g_layer.bias is not None and ae_layer.bias is not None:
                    g_layer.bias.data.copy_(ae_layer.bias.data)


class BAGANDiscriminator(nn.Module):
    """
    Class-conditional discriminator for BAGAN.

    Conditioning: class one-hot is broadcast to spatial size and concatenated
    with the image as extra input channels.
    Input channels = image_channels + n_classes (1 + 10 = 11 for MNIST).
    """

    def __init__(self, cfg: EncoderDecoderConfig, n_classes: int = 10,
                 in_channels: int = 1):
        super().__init__()
        channels = cfg.conv_channels
        k, s, p  = cfg.kernel_size, cfg.stride, cfg.padding
        slope    = cfg.leaky_relu_slope

        # First conv layer takes (img_channels + n_classes) input channels
        total_in = in_channels + n_classes
        layers, c_in = [], total_in
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(slope, inplace=True),
            ]
            c_in = c_out

        self.conv = nn.Sequential(*layers)
        self._spatial = 2
        self.fc = nn.Linear(channels[-1] * self._spatial * self._spatial, 1)

    def forward(self, x: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        # Pad MNIST 28→32
        if x.shape[-1] == 28:
            x = F.pad(x, (2, 2, 2, 2))
        # Broadcast one-hot to spatial: (B, n_classes) → (B, n_classes, H, W)
        H, W = x.shape[2], x.shape[3]
        cond = labels_onehot[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([x, cond], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)                  # raw logit (BCEWithLogitsLoss)


# ─────────────────────────────────────────────────────────────────
#  ResNet-18 Classifier
# ─────────────────────────────────────────────────────────────────

def get_resnet18(n_classes: int = 10, in_channels: int = 1) -> nn.Module:
    """
    ResNet-18 adapted for grayscale input (paper §V-A-4).

    The simplest adaptation is to repeat the single channel 3× before the
    first conv layer, keeping all pre-trained ResNet-18 weights valid if
    we later choose to pre-train on ImageNet.  We do NOT pre-train here.
    """
    model = resnet18(weights=None)
    # Modify first conv to accept 1-channel input
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
