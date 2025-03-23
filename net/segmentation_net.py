import torch
import torch.nn as nn

class SegmentationNet(nn.Module):
    def __init__(self, with_feature_fusion=False, latent_dim=128, num_classes=2):
        super(SegmentationNet, self).__init__()
        self.with_feature_fusion = with_feature_fusion
        self.latent_dim = latent_dim
        
        # Feature extractor: simple conv layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        if self.with_feature_fusion:
            # Project latent vector to spatial feature map. Here we assume input images are 32x32
            # and we want to produce a tensor of shape [B, 64, 32, 32]
            self.latent_proj = nn.Linear(latent_dim, 64 * 32 * 32)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x, latent=None):
        # x: [B, 3, 32, 32]
        features = self.feature_extractor(x)  # [B, 64, 32, 32]
        if self.with_feature_fusion and latent is not None:
            B = latent.size(0)
            latent_feature = self.latent_proj(latent)  # [B, 64*32*32]
            latent_feature = latent_feature.view(B, 64, 32, 32)
            features = features + latent_feature  # simple fusion by addition
        
        seg_out = self.classifier(features)  # [B, num_classes, 32, 32]
        return seg_out
