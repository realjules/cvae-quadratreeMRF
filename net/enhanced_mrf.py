import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMRF(nn.Module):
    """
    Enhanced Markov Random Field model that builds upon SimplifiedMRF,
    incorporating attention mechanisms, boundary refinement, and multi-scale processing.
    """
    def __init__(self, n_classes=6, feature_dim=256, device="cuda"):
        super(EnhancedMRF, self).__init__()
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.device = device
        
        # Feature adaptation with residual blocks
        self.feature_adaptation = nn.Sequential(
            ResidualBlock(feature_dim, 256),
            SelfAttention(256),
            ResidualBlock(256, 256)
        )
        
        # Deeper hierarchical structure with attention
        self.level1 = DeepFeatureExtractor(256, 128)
        self.level2 = DeepFeatureExtractor(128, 128)
        self.level3 = DeepFeatureExtractor(128, 128)
        
        # Context module with dilated convolutions
        self.context = DilatedContextModule(128, 128)
        
        # Decoder with attention gates
        self.decoder1 = AttentionUpBlock(128, 128)
        self.decoder2 = AttentionUpBlock(256, 128)  # 256 = 128 + 128 (skip)
        
        # Boundary refinement module
        self.boundary_refinement = BoundaryRefinementModule(256, 128)
        
        # Final classifier with class-balanced attention
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
    
    def forward(self, features):
        """Forward pass through the enhanced MRF model"""
        # Feature adaptation
        x = self.feature_adaptation(features)
        
        # Multi-scale processing
        f1 = self.level1(x)
        f2 = self.level2(F.max_pool2d(f1, 2))
        f3 = self.level3(F.max_pool2d(f2, 2))
        
        # Apply dilated context module
        f3_ctx = self.context(f3)
        
        # Decoder with attention gates
        up3 = self.decoder1(f3_ctx)
        up2 = self.decoder2(torch.cat([up3, f2], dim=1))
        
        # Boundary refinement using f1 features
        refined = self.boundary_refinement(torch.cat([up2, f1], dim=1))
        
        # Final prediction
        logits = self.classifier(refined)
        
        return logits


class ResidualBlock(nn.Module):
    """Residual block with bottleneck structure"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies"""
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Project to query, key, value
        proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # (B, HW, C/8)
        proj_key = self.key(x).view(batch_size, -1, H*W)  # (B, C/8, HW)
        proj_value = self.value(x).view(batch_size, -1, H*W)  # (B, C, HW)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # (B, HW, HW)
        attention = F.softmax(energy, dim=2)  # (B, HW, HW)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, C, H, W)  # (B, C, H, W)
        
        # Apply learnable weight
        out = self.gamma * out + x
        
        return out


class DeepFeatureExtractor(nn.Module):
    """Deep feature extractor with stacked convolutions and SE block"""
    def __init__(self, in_channels, out_channels):
        super(DeepFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation block
        self.se = SqueezeExcitation(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, C, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, C)
        y = self.excitation(y).view(batch_size, C, 1, 1)
        return x * y


class DilatedContextModule(nn.Module):
    """Dilated convolution context module for large receptive field"""
    def __init__(self, in_channels, out_channels):
        super(DilatedContextModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=8, dilation=8)
        
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        # Bottleneck to reduce channels
        self.bottleneck = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_cat = self.relu(self.bn(x_cat))
        
        out = self.bottleneck(x_cat)
        out = self.relu(self.bn_out(out))
        
        return out


class AttentionUpBlock(nn.Module):
    """Attention-guided upsampling block"""
    def __init__(self, in_channels, out_channels):
        super(AttentionUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Attention gate
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Upsample
        x_up = self.up(x)
        
        # Apply attention
        attention = self.attention(x_up)
        x_attended = x_up * attention
        
        # Conv block
        out = self.conv(x_attended)
        
        return out


class BoundaryRefinementModule(nn.Module):
    """Boundary refinement module for precise segmentation edges"""
    def __init__(self, in_channels, out_channels):
        super(BoundaryRefinementModule, self).__init__()
        # Initial projection
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Edge detection branch
        self.edge_detect = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial projection
        features = self.relu(self.bn1(self.conv1(x)))
        
        # Edge detection
        edge_features = self.edge_detect(features)
        
        # Fusion
        out = self.fusion(torch.cat([features, edge_features], dim=1))
        
        return out
