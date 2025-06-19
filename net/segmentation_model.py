import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSegmentationModel(nn.Module):
    """
    Multi-scale segmentation model that works with CVAE features.
    Replaces the SimplifiedMRF with proper spatial reasoning capabilities.
    """
    def __init__(self, n_classes=6, device="cuda"):
        super(MultiScaleSegmentationModel, self).__init__()
        self.n_classes = n_classes
        self.device = device
        
        # Multi-scale feature processing modules
        # These process features from CVAE encoder at different resolutions
        self.process_p1 = SpatialReasoningBlock(64, 128, scale='fine')    # 128x128 → fine details
        self.process_p2 = SpatialReasoningBlock(128, 256, scale='medium') # 64x64 → medium features
        self.process_p3 = SpatialReasoningBlock(256, 512, scale='coarse') # 32x32 → semantic features
        
        # Cross-scale attention for information flow between resolutions
        self.cross_attention_32_64 = CrossScaleAttention(512, 256)
        self.cross_attention_64_128 = CrossScaleAttention(256, 128)
        
        # Multi-scale segmentation heads - different complexity for different scales
        self.seg_head_p1 = SegmentationHead(128, n_classes, 'fine')     # High resolution, simple head
        self.seg_head_p2 = SegmentationHead(256, n_classes, 'medium')   # Medium resolution, medium head
        self.seg_head_p3 = SegmentationHead(512, n_classes, 'coarse')   # Low resolution, complex head
        
        # Final fusion module to combine multi-scale predictions
        self.final_fusion = FinalFusionModule(n_classes * 3, n_classes)
        
        # Boundary refinement for crisp edges
        self.boundary_refiner = BoundaryRefinementModule(n_classes)
        
    def forward(self, multi_scale_features):
        """
        Forward pass with multi-scale spatial reasoning
        
        Args:
            multi_scale_features: Dict with keys ['p1', 'p2', 'p3', 'global_context']
        """
        p1 = multi_scale_features['p1']  # [B, 64, 128, 128] - fine details
        p2 = multi_scale_features['p2']  # [B, 128, 64, 64] - medium features
        p3 = multi_scale_features['p3']  # [B, 256, 32, 32] - semantic features
        
        # Apply spatial reasoning at each scale
        p1_processed = self.process_p1(p1)  # [B, 128, 128, 128]
        p2_processed = self.process_p2(p2)  # [B, 256, 64, 64]
        p3_processed = self.process_p3(p3)  # [B, 512, 32, 32]
        
        # Cross-scale attention for information flow from coarse to fine
        # This allows semantic information to flow to finer resolutions
        p2_enhanced = self.cross_attention_32_64(p3_processed, p2_processed)  # [B, 256, 64, 64]
        p1_enhanced = self.cross_attention_64_128(p2_enhanced, p1_processed)  # [B, 128, 128, 128]
        
        # Generate predictions at each scale
        seg_p1 = self.seg_head_p1(p1_enhanced)  # [B, n_classes, 128, 128]
        seg_p2 = self.seg_head_p2(p2_enhanced)  # [B, n_classes, 64, 64]
        seg_p3 = self.seg_head_p3(p3_processed) # [B, n_classes, 32, 32]
        
        # Upsample all predictions to highest resolution (128x128) for fusion
        seg_p2_up = F.interpolate(seg_p2, size=(128, 128), mode='bilinear', align_corners=False)
        seg_p3_up = F.interpolate(seg_p3, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Fuse multi-scale predictions
        multi_scale_seg = torch.cat([seg_p1, seg_p2_up, seg_p3_up], dim=1)  # [B, n_classes*3, 128, 128]
        fused_seg = self.final_fusion(multi_scale_seg)  # [B, n_classes, 128, 128]
        
        # Apply boundary refinement for crisp edges
        refined_seg = self.boundary_refiner(fused_seg)  # [B, n_classes, 128, 128]
        
        return {
            'final_segmentation': refined_seg,
            'multi_scale_predictions': [seg_p3, seg_p2, seg_p1],  # Coarse to fine order
            'intermediate_features': [p3_processed, p2_enhanced, p1_enhanced]
        }


class SpatialReasoningBlock(nn.Module):
    """
    Spatial reasoning block with dilated convolutions and attention mechanisms.
    Captures spatial context at different scales using multi-scale dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, scale='medium'):
        super(SpatialReasoningBlock, self).__init__()
        
        # Scale-adaptive dilation rates for different receptive fields
        if scale == 'fine':
            dilations = [1, 2, 4]  # Small receptive field for fine details
        elif scale == 'medium':  
            dilations = [1, 2, 4, 8]  # Medium receptive field for balanced features
        else:  # coarse
            dilations = [1, 2, 4, 8, 16]  # Large receptive field for global context
            
        # Multi-scale dilated convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels//len(dilations), 3, 
                         padding=d, dilation=d),
                nn.BatchNorm2d(out_channels//len(dilations)),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        
        # Channel attention for adaptive feature weighting
        self.channel_attention = ChannelAttentionModule(out_channels)
        
        # Spatial attention for spatial focus
        self.spatial_attention = SpatialAttentionModule(out_channels)
        
        # Residual connection with projection if needed
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        # Apply multi-scale dilated convolutions
        dilated_features = []
        for conv in self.dilated_convs:
            dilated_features.append(conv(x))
        
        # Concatenate multi-scale features
        concat_features = torch.cat(dilated_features, dim=1)
        
        # Apply attention mechanisms
        channel_weighted = self.channel_attention(concat_features)
        spatial_weighted = self.spatial_attention(channel_weighted)
        
        # Residual connection
        residual = self.residual_conv(x)
        output = spatial_weighted + residual
        
        return output


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention to transfer semantic information between different resolutions.
    Allows coarse semantic features to guide fine-scale predictions.
    """
    def __init__(self, coarse_channels, fine_channels):
        super(CrossScaleAttention, self).__init__()
        
        self.coarse_channels = coarse_channels
        self.fine_channels = fine_channels
        
        # Attention computation: Query from fine, Key/Value from coarse
        self.query_conv = nn.Conv2d(fine_channels, fine_channels//8, 1)
        self.key_conv = nn.Conv2d(coarse_channels, fine_channels//8, 1)  
        self.value_conv = nn.Conv2d(coarse_channels, fine_channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable gate for attention strength
        
    def forward(self, coarse_features, fine_features):
        """
        Args:
            coarse_features: [B, coarse_channels, H_c, W_c] - semantic features
            fine_features: [B, fine_channels, H_f, W_f] - spatial features
        """
        B, C_f, H_f, W_f = fine_features.size()
        
        # Upsample coarse features to fine resolution
        coarse_up = F.interpolate(coarse_features, size=(H_f, W_f), 
                                 mode='bilinear', align_corners=False)
        
        # Compute attention weights
        query = self.query_conv(fine_features).view(B, -1, H_f * W_f).permute(0, 2, 1)  # [B, H_f*W_f, C//8]
        key = self.key_conv(coarse_up).view(B, -1, H_f * W_f)  # [B, C//8, H_f*W_f]
        value = self.value_conv(coarse_up).view(B, -1, H_f * W_f)  # [B, C_f, H_f*W_f]
        
        # Attention matrix: how much each fine location attends to coarse semantics
        attention = torch.bmm(query, key)  # [B, H_f*W_f, H_f*W_f]
        attention = self.softmax(attention)
        
        # Apply attention to values
        attended = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C_f, H_f*W_f]
        attended = attended.view(B, C_f, H_f, W_f)
        
        # Residual connection with learnable gate
        output = fine_features + self.gamma * attended
        
        return output


class ChannelAttentionModule(nn.Module):
    """Channel attention for adaptive feature channel weighting"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Global pooling
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        
        # Channel attention weights
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


class SpatialAttentionModule(nn.Module):
    """Spatial attention for focusing on relevant spatial locations"""
    def __init__(self, channels):
        super(SpatialAttentionModule, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Spatial attention map
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention_input)
        
        return x * attention


class SegmentationHead(nn.Module):
    """Scale-specific segmentation head with adaptive complexity"""
    def __init__(self, in_channels, n_classes, scale):
        super(SegmentationHead, self).__init__()
        
        # Scale-adaptive head complexity
        if scale == 'fine':
            # Simple head for fine details - preserve spatial information
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, n_classes, 1)
            )
        elif scale == 'medium':
            # Medium complexity for balanced features
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.BatchNorm2d(in_channels//2), 
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//4, n_classes, 1)
            )
        else:  # coarse
            # Complex head for semantic reasoning - more capacity for classification
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//4, in_channels//8, 3, padding=1),
                nn.BatchNorm2d(in_channels//8),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//8, n_classes, 1)
            )
    
    def forward(self, x):
        return self.head(x)


class FinalFusionModule(nn.Module):
    """Final fusion of multi-scale predictions"""
    def __init__(self, in_channels, out_channels):
        super(FinalFusionModule, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//4, 3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, out_channels, 1)
        )
        
    def forward(self, x):
        return self.fusion(x)


class BoundaryRefinementModule(nn.Module):
    """Boundary refinement for crisp segmentation edges"""
    def __init__(self, n_classes):
        super(BoundaryRefinementModule, self).__init__()
        
        # Edge detection branch
        self.edge_detector = nn.Sequential(
            nn.Conv2d(n_classes, n_classes//2, 3, padding=1),
            nn.BatchNorm2d(n_classes//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes//2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Boundary-aware refinement
        self.refiner = nn.Sequential(
            nn.Conv2d(n_classes + 1, n_classes, 3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes, n_classes, 3, padding=1)
        )
        
    def forward(self, x):
        # Detect potential edges/boundaries
        edges = self.edge_detector(x)
        
        # Boundary-aware refinement
        boundary_aware = torch.cat([x, edges], dim=1)
        refined = self.refiner(boundary_aware)
        
        # Residual connection
        return x + refined