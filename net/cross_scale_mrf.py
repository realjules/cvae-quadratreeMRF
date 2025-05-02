import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention module for relating features at different scales.
    Applies self-attention across different scale levels to enhance feature integration.
    """
    def __init__(self, feature_dims, embed_dim=64):
        super(CrossScaleAttention, self).__init__()
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        
        # Projection layers for each scale
        self.query_projections = nn.ModuleList([
            nn.Conv2d(dim, embed_dim, kernel_size=1)
            for dim in feature_dims
        ])
        
        self.key_projections = nn.ModuleList([
            nn.Conv2d(dim, embed_dim, kernel_size=1)
            for dim in feature_dims
        ])
        
        self.value_projections = nn.ModuleList([
            nn.Conv2d(dim, embed_dim, kernel_size=1)
            for dim in feature_dims
        ])
        
        # Output projections to return to original dimensions
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
            for dim in feature_dims
        ])
        
        # Scaling factor for attention
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        
        # Learnable weights for attention fusion
        self.alpha = nn.Parameter(torch.ones(len(feature_dims)))
        
    def forward(self, features):
        """
        Apply cross-scale attention to a list of features at different scales
        
        Args:
            features: List of feature tensors [f1, f2, ..., fn] at different scales
                      f1 is highest resolution, fn is lowest resolution
        
        Returns:
            List of enhanced features with same dimensions as input
        """
        batch_size = features[0].size(0)
        num_scales = len(features)
        
        # Target size is the second smallest feature map size
        # This is a balance between computational efficiency and detail preservation
        target_size = features[-2].shape[2:] if num_scales > 1 else features[0].shape[2:]
        
        # Project and resize all features to common dimensions
        queries, keys, values = [], [], []
        
        for i, feature in enumerate(features):
            # Project to embedding space
            q = self.query_projections[i](feature)
            k = self.key_projections[i](feature)
            v = self.value_projections[i](feature)
            
            # Resize to target dimensions
            if q.shape[2:] != target_size:
                q = F.interpolate(q, size=target_size, mode='bilinear', align_corners=False)
                k = F.interpolate(k, size=target_size, mode='bilinear', align_corners=False)
                v = F.interpolate(v, size=target_size, mode='bilinear', align_corners=False)
            
            # Reshape for attention computation
            # (B, C, H, W) -> (B, C, H*W)
            queries.append(q.view(batch_size, self.embed_dim, -1))
            # (B, C, H*W) -> (B, H*W, C)
            keys.append(k.view(batch_size, self.embed_dim, -1).permute(0, 2, 1))
            values.append(v.view(batch_size, self.embed_dim, -1))
        
        # Calculate cross-scale attention for each scale
        enhanced_features = []
        
        for i in range(num_scales):
            # Current scale's query
            q = queries[i]  # (B, C, H*W)
            
            # Initialize weighted sum of attended values
            attended_sum = torch.zeros_like(q)
            weights_sum = 0
            
            # Calculate attention with every other scale
            for j in range(num_scales):
                # Get key and value from scale j
                k = keys[j]  # (B, H*W, C)
                v = values[j]  # (B, C, H*W)
                
                # Calculate scaled dot-product attention
                # (B, C, H*W) x (B, H*W, C) -> (B, C, C)
                attn = torch.bmm(q, k) / self.scale
                attn = F.softmax(attn, dim=2)
                
                # Apply attention weights to values
                # (B, C, C) x (B, C, H*W) -> (B, C, H*W)
                attended = torch.bmm(attn, v.permute(0, 2, 1)).permute(0, 2, 1)
                
                # Use learned scale weights
                weight = torch.sigmoid(self.alpha[j])
                attended_sum += weight * attended
                weights_sum += weight
            
            # Normalize by sum of weights
            attended_normalized = attended_sum / (weights_sum + 1e-8)
            
            # Reshape back to spatial dimensions
            attended_spatial = attended_normalized.view(
                batch_size, self.embed_dim, target_size[0], target_size[1]
            )
            
            # Resize back to original dimensions
            if attended_spatial.shape[2:] != features[i].shape[2:]:
                attended_spatial = F.interpolate(
                    attended_spatial,
                    size=features[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Project back to original feature dimension
            enhanced = self.output_projections[i](attended_spatial)
            
            # Add residual connection
            enhanced = enhanced + features[i]
            
            enhanced_features.append(enhanced)
        
        return enhanced_features


class FeaturePyramidAttention(nn.Module):
    """
    Feature Pyramid Attention module that integrates multi-scale context
    within a single feature map using pyramid pooling and attention.
    """
    def __init__(self, in_channels, reduction=4):
        super(FeaturePyramidAttention, self).__init__()
        self.in_channels = in_channels
        
        # Pyramid pooling at different scales
        self.pyramid_levels = [1, 2, 4, 8]
        
        # Convolutions for each pyramid level
        self.pyramid_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(level),
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True)
            ) for level in self.pyramid_levels
        ])
        
        # Merge convolution
        self.merge_conv = nn.Conv2d(
            in_channels // reduction * len(self.pyramid_levels),
            in_channels,
            kernel_size=1
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Apply feature pyramid attention
        
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            Enhanced feature map (B, C, H, W)
        """
        h, w = x.size(2), x.size(3)
        
        # Process each pyramid level
        pyramid_features = []
        
        for i, conv in enumerate(self.pyramid_convs):
            # Apply pooling and processing
            feat = conv(x)
            # Upsample back to original size
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(feat)
        
        # Concatenate pyramid features
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        # Merge pyramid features
        context = self.merge_conv(pyramid_concat)
        
        # Apply channel attention
        attention = self.channel_attention(context)
        
        # Apply attention to input features
        enhanced = x * attention
        
        # Add residual connection
        output = enhanced + x
        
        return output


class CrossScaleMRF(nn.Module):
    """
    Enhanced SimplifiedMRF with cross-scale attention for better multi-scale feature integration.
    This model implements cross-scale interactions through multiple attention mechanisms.
    """
    def __init__(self, n_classes=6, feature_dim=256, device="cuda"):
        super(CrossScaleMRF, self).__init__()
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.device = device
        
        # Feature adaptation layer
        self.feature_adaptation = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Multi-level feature extraction (hierarchical encoder)
        # Level 1: Full resolution
        self.level1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: Half resolution
        self.level2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Level 3: Quarter resolution
        self.level3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature Pyramid Attention for context enhancement
        self.context1 = FeaturePyramidAttention(64)
        self.context2 = FeaturePyramidAttention(64)
        self.context3 = FeaturePyramidAttention(64)
        
        # Cross-scale attention mechanism to relate features across scales
        self.cross_scale_attention = CrossScaleAttention([64, 64, 64], embed_dim=32)
        
        # Hierarchical fusion with upsampling
        # Upsample level 3 to level 2 size
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fuse upsampled level 3 with level 2
        self.fuse3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Upsample fused features to level 1 size
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fuse upsampled features with level 1
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # CRF-like smoothing with pairwise potentials
        self.crf_smoothing = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier for segmentation
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Scale-Specific classifiers for deep supervision
        self.classifier_s1 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.classifier_s2 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.classifier_s3 = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, features):
        """Forward pass through the CrossScaleMRF"""
        # Feature adaptation
        x = self.feature_adaptation(features)
        
        # Multi-level feature extraction
        f1 = self.level1(x)  # Full resolution
        
        # Downsample for level 2
        x_down1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        f2 = self.level2(x_down1)  # Half resolution
        
        # Downsample for level 3
        x_down2 = F.avg_pool2d(x_down1, kernel_size=2, stride=2)
        f3 = self.level3(x_down2)  # Quarter resolution
        
        # Apply Feature Pyramid Attention at each level
        f1_ctx = self.context1(f1)
        f2_ctx = self.context2(f2)
        f3_ctx = self.context3(f3)
        
        # Apply cross-scale attention to relate features across different scales
        f1_enh, f2_enh, f3_enh = self.cross_scale_attention([f1_ctx, f2_ctx, f3_ctx])
        
        # Get scale-specific predictions for deep supervision
        pred_s3 = self.classifier_s3(f3_enh)
        pred_s2 = self.classifier_s2(f2_enh)
        pred_s1 = self.classifier_s1(f1_enh)
        
        # Hierarchical fusion with skip connections
        # Upsample level 3 to level 2
        f3_up = self.upsample3(f3_enh)
        
        # Fuse with level 2
        f2_cat = torch.cat([f2_enh, f3_up], dim=1)
        f2_fused = self.fuse3(f2_cat)
        
        # Upsample to level 1
        f2_up = self.upsample2(f2_fused)
        
        # Fuse with level 1
        f1_cat = torch.cat([f1_enh, f2_up], dim=1)
        f1_fused = self.fuse2(f1_cat)
        
        # Apply CRF-like smoothing
        f_smooth = self.crf_smoothing(f1_fused)
        
        # Final prediction
        logits = self.classifier(f_smooth)
        
        # Return final prediction and scale-specific predictions for deep supervision
        return {
            'final_segmentation': logits,
            'hierarchical_segmentations': [pred_s1, pred_s2, pred_s3]
        }


class ScaleAwareAttention(nn.Module):
    """
    Scale-aware attention module that dynamically weights features from different scales.
    """
    def __init__(self, feature_dims, reduction=4):
        super(ScaleAwareAttention, self).__init__()
        self.feature_dims = feature_dims
        self.num_scales = len(feature_dims)
        
        # Projection for each scale to a common dimension
        self.projection_dim = min(feature_dims) // reduction if reduction > 0 else min(feature_dims)
        
        # Scale-specific projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, self.projection_dim, kernel_size=1),
                nn.BatchNorm2d(self.projection_dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.projection_dim * self.num_scales, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Output projection back to original dimensions
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.projection_dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
    
    def forward(self, features):
        """
        Apply scale-aware attention to weight different scales
        
        Args:
            features: List of feature tensors at different scales
        
        Returns:
            List of enhanced feature tensors with original dimensions
        """
        batch_size = features[0].size(0)
        
        # Project each scale to common dimension
        projected = []
        for i, feat in enumerate(features):
            proj = self.projections[i](feat)
            # Resize all to the size of the highest resolution
            if i > 0:  # If not the highest resolution
                proj = F.interpolate(
                    proj, 
                    size=features[0].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            projected.append(proj)
        
        # Concatenate projected features along channel dimension
        concat_features = torch.cat(projected, dim=1)
        
        # Calculate scale attention weights
        attention = self.scale_attention(concat_features)  # B, num_scales, 1, 1
        
        # Apply attention to each scale and project back
        enhanced_features = []
        
        for i, feat in enumerate(features):
            # Get attention weight for this scale
            weight = attention[:, i:i+1]
            
            # Resize weight if needed
            if weight.shape[2:] != projected[i].shape[2:]:
                weight = F.interpolate(
                    weight,
                    size=projected[i].shape[2:],
                    mode='nearest'
                )
            
            # Apply weight to projected features
            weighted = projected[i] * weight
            
            # Project back to original dimension
            enhanced = self.output_projections[i](weighted)
            
            # Resize back to original size if needed
            if enhanced.shape[2:] != feat.shape[2:]:
                enhanced = F.interpolate(
                    enhanced,
                    size=feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Add residual connection
            enhanced = enhanced + feat
            
            enhanced_features.append(enhanced)
        
        return enhanced_features