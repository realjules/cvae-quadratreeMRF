import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFeatureExtractor(nn.Module):
    """
    Enhanced feature extractor that leverages multiple features from the CVAE
    to create richer representations for the MRF model.
    """
    def __init__(self, latent_dim=512, output_dim=256):
        super(EnhancedFeatureExtractor, self).__init__()
        
        # Calculate expected input dimension after concatenating all encoder features
        # e1(64), e2(128), e3(256), z(latent_dim)
        combined_channels = 64 + 128 + 256 + latent_dim
        
        # Initial projection to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Conv2d(combined_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention to focus on important features
        self.attention = SpatialAttention(output_dim)
        
        # Final refinement layer
        self.refine = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_features, z):
        """
        Create enhanced features from CVAE outputs
        
        Args:
            encoder_features: List of encoder features [e1, e2, e3]
            z: Latent code (batch_size, latent_dim)
            
        Returns:
            Enhanced features tensor (batch_size, output_dim, H, W)
        """
        # Get individual encoder features
        e1, e2, e3 = encoder_features  # Shapes: e1(B,64,H/2,W/2), e2(B,128,H/4,W/4), e3(B,256,H/8,W/8)
        
        # Get sizes
        batch_size = e2.size(0)
        target_size = e2.size(2), e2.size(3)  # Use e2's dimensions as reference (H/4, W/4)
        
        # Resize all features to the same dimensions
        e1_resized = F.interpolate(e1, size=target_size, mode='bilinear', align_corners=False)
        e3_resized = F.interpolate(e3, size=target_size, mode='bilinear', align_corners=False)
        
        # Convert latent vector to spatial dimensions
        z_spatial = z.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
        z_spatial = F.interpolate(z_spatial, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        combined = torch.cat([e1_resized, e2, e3_resized, z_spatial], dim=1)
        
        # Project to lower dimension
        x = self.projection(combined)
        
        # Apply spatial attention
        x = self.attention(x)
        
        # Final refinement
        x = self.refine(x)
        
        return x
    
    def extract_from_cvae_output(self, cvae_output):
        """
        Helper method to extract features directly from CVAE output dict
        
        Args:
            cvae_output: Dictionary containing CVAE outputs
            
        Returns:
            Enhanced features tensor
        """
        encoder_features = cvae_output['encoder_features']
        z = cvae_output['z']
        
        return self.forward(encoder_features, z)


class SpatialAttention(nn.Module):
    """
    Spatial attention module that helps the model focus on relevant features.
    """
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        
        # Channel attention first
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention next
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply channel attention
        channel_attention = self.channel_gate(x)
        x = x * channel_attention
        
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_gate(spatial_map)
        
        # Apply spatial attention
        return x * spatial_attention
