"""
ResNet-18 Encoder with SimCLR Projection Head for contrastive pre-training
and multi-scale feature extraction for segmentation.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Image [B, 3, 256, 256]                             │
    │    │                                                 │
    │    ▼                                                 │
    │  Stem (3x3 stride-2 conv, no maxpool) → [B,64,128]  │
    │    │                                                 │
    │    ├─ layer1 (stride 1) → p1 [B, 64,  128, 128]     │
    │    ├─ layer2 (stride 2) → p2 [B, 128,  64,  64]     │
    │    └─ layer3 (stride 2) → p3 [B, 256,  32,  32]     │
    │                                                      │
    │  For contrastive:  p3 → GAP → proj_head → z (128-d)  │
    │  For segmentation: (p1, p2, p3) → DHBP module        │
    └─────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18Encoder(nn.Module):
    """ResNet-18 backbone with modified stem for 256x256 aerial imagery.

    The standard ResNet stem (7x7 stride-2 conv + maxpool) downsamples 4x,
    giving 64x64 after stem. We replace it with a 3x3 stride-2 conv (2x
    downsample) so features start at 128x128, matching the spatial resolution
    needed by DHBP.

    Output feature pyramid:
        p1: [B, 64,  128, 128]  after layer1
        p2: [B, 128,  64,  64]  after layer2
        p3: [B, 256,  32,  32]  after layer3
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Modified stem: 3x3 stride-2 (2x downsample instead of 4x)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Copy pretrained BN stats from original stem
        if pretrained:
            # Initialize 3x3 conv with center-cropped 7x7 weights
            with torch.no_grad():
                original_weight = resnet.conv1.weight.data  # [64, 3, 7, 7]
                self.stem[0].weight.copy_(original_weight[:, :, 2:5, 2:5])
            self.stem[1].load_state_dict(resnet.bn1.state_dict())

        # ResNet layers (pretrained weights preserved)
        self.layer1 = resnet.layer1  # 64ch,  stride 1 → 128x128
        self.layer2 = resnet.layer2  # 128ch, stride 2 → 64x64
        self.layer3 = resnet.layer3  # 256ch, stride 2 → 32x32
        # layer4 skipped: too coarse for 256x256 input, unnecessary params

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: [B, 3, 256, 256] input images normalized to [0, 1]

        Returns:
            (p1, p2, p3) feature pyramid
        """
        x = self.stem(x)          # [B, 64, 128, 128]
        p1 = self.layer1(x)       # [B, 64, 128, 128]
        p2 = self.layer2(p1)      # [B, 128, 64, 64]
        p3 = self.layer3(p2)      # [B, 256, 32, 32]
        return p1, p2, p3


class SimCLRProjectionHead(nn.Module):
    """2-layer MLP projection head for SimCLR contrastive learning.

    Projects the global representation to a lower-dimensional space where
    the contrastive loss is applied. Standard SimCLR design.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            x: [B, in_dim] global representation

        Returns:
            [B, out_dim] L2-normalized projection
        """
        return F.normalize(self.net(x), dim=1)


class ContrastiveEncoder(nn.Module):
    """Complete encoder for both contrastive pre-training and segmentation.

    Two modes of use:
        - encode(x) → (p1, p2, p3) multi-scale features for DHBP
        - project(x) → z (128-d normalized) for SimCLR contrastive loss
    """

    def __init__(self, pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projection_head = SimCLRProjectionHead(in_dim=256, out_dim=proj_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features for segmentation.

        Args:
            x: [B, 3, 256, 256]

        Returns:
            (p1, p2, p3) where:
                p1: [B, 64,  128, 128]
                p2: [B, 128,  64,  64]
                p3: [B, 256,  32,  32]
        """
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Extract projected features for contrastive learning.

        Args:
            x: [B, 3, 256, 256]

        Returns:
            [B, proj_dim] L2-normalized projection
        """
        _, _, p3 = self.encoder(x)
        h = self.gap(p3).flatten(1)        # [B, 256]
        return self.projection_head(h)     # [B, proj_dim]

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass returning both features and projections.

        Args:
            x: [B, 3, 256, 256]

        Returns:
            dict with keys: p1, p2, p3, representation, projection
        """
        p1, p2, p3 = self.encoder(x)
        h = self.gap(p3).flatten(1)        # [B, 256]
        z = self.projection_head(h)        # [B, proj_dim]
        return {
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'representation': h,
            'projection': z,
        }
