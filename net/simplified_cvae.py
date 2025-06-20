"""
Simplified CVAE for Phase 2 optimization (optional)

This provides a streamlined version of the CVAE that focuses on:
1. Efficient encoder for feature extraction
2. Simplified decoder (since we mainly need encoder features)
3. Better memory efficiency
4. Faster training

Use this only if the original CVAE becomes a bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedCVAE(nn.Module):
    """
    Simplified CVAE with focus on encoder features for segmentation.
    
    Key optimizations:
    - Lighter decoder (we mainly need encoder features)
    - More efficient skip connections
    - Better gradient flow
    - Reduced memory usage
    """
    
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=None):
        super(SimplifiedCVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims  
        self.input_channels = input_channels
        
        # ====== OPTIMIZED ENCODER ======
        # More efficient encoder blocks with residual connections
        self.encoder_block1 = EncoderBlock(input_channels, hidden_dims[0], stride=2)
        self.encoder_block2 = EncoderBlock(hidden_dims[0], hidden_dims[1], stride=2)  
        self.encoder_block3 = EncoderBlock(hidden_dims[1], hidden_dims[2], stride=2)
        
        # Efficient adaptive pooling instead of fixed size assumptions
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Variational components
        self.fc_mu = nn.Linear(hidden_dims[2], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[2], latent_dim)
        
        # Optimized projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # ====== SIMPLIFIED DECODER ======
        # Lightweight decoder since we mainly need encoder features
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[2] * 4 * 4)  # Smaller spatial size
        
        self.decoder_block1 = DecoderBlock(hidden_dims[2], hidden_dims[1])
        self.decoder_block2 = DecoderBlock(hidden_dims[1], hidden_dims[0])
        self.decoder_block3 = DecoderBlock(hidden_dims[0], input_channels, final=True)
        
        # Contrastive learning memory bank (smaller for efficiency)
        self.register_buffer("queue", torch.randn(2048, latent_dim // 4))  # Reduced from 4096
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def encode(self, x):
        """Optimized encoding with efficient feature extraction"""
        # Extract multi-scale features
        x1 = self.encoder_block1(x)      # [B, 64, H/2, W/2]
        x2 = self.encoder_block2(x1)     # [B, 128, H/4, W/4]
        x3 = self.encoder_block3(x2)     # [B, 256, H/8, W/8]
        
        encoder_features = [x1, x2, x3]
        
        # Global features for latent space
        global_feat = self.adaptive_pool(x3).flatten(1)  # [B, 256]
        
        # Get latent parameters
        mu = self.fc_mu(global_feat)
        log_var = self.fc_log_var(global_feat)
        
        return mu, log_var, encoder_features
    
    def reparameterize(self, mu, log_var):
        """Standard reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, target_size=(256, 256)):
        """
        Simplified decoder that reconstructs to target size
        
        Args:
            z: latent vector [B, latent_dim]
            target_size: output image size (H, W)
        """
        batch_size = z.size(0)
        
        # Start from small spatial size
        x = self.decoder_input(z)
        x = x.view(batch_size, self.hidden_dims[2], 4, 4)  # [B, 256, 4, 4]
        
        # Progressive upsampling
        x = self.decoder_block1(x)  # [B, 128, 8, 8]
        x = self.decoder_block2(x)  # [B, 64, 16, 16]  
        x = self.decoder_block3(x)  # [B, 3, 32, 32]
        
        # Final upsampling to target size
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x)
    
    def forward(self, x):
        """Forward pass through simplified CVAE"""
        mu, log_var, encoder_features = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        # Lightweight reconstruction (can be disabled during inference)
        x_recon = self.decode(z, target_size=(x.size(2), x.size(3)))
        
        # Contrastive projection
        z_proj = self.project(z)
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Update memory bank during training
        if self.training:
            with torch.no_grad():
                self.update_queue(z_proj_norm.detach())
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'z_proj': z_proj_norm,
            'original_input': x,
            'queue': self.queue,
            'encoder_features': encoder_features
        }
    
    def project(self, z):
        """Project latent representations for contrastive learning"""
        return self.projection_head(z)
    
    @torch.no_grad()
    def update_queue(self, z_proj):
        """Update memory bank queue (same as original)"""
        batch_size = z_proj.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.queue.shape[0]:
            first_part = self.queue.shape[0] - ptr
            self.queue[ptr:] = z_proj[:first_part]
            self.queue[:batch_size - first_part] = z_proj[first_part:]
            ptr = batch_size - first_part
        else:
            self.queue[ptr:ptr + batch_size] = z_proj
            ptr = (ptr + batch_size) % self.queue.shape[0]
        
        self.queue_ptr[0] = ptr


class EncoderBlock(nn.Module):
    """Efficient encoder block with residual connections"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(EncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        
        return F.leaky_relu(out + identity, 0.2)


class DecoderBlock(nn.Module):
    """Efficient decoder block"""
    
    def __init__(self, in_channels, out_channels, final=False):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=4, stride=2, padding=1
        )
        
        if not final:
            self.bn = nn.BatchNorm2d(out_channels)
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.bn = None
            self.activation = None  # Sigmoid applied in decode()
    
    def forward(self, x):
        x = self.upsample(x)
        
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
            
        return x


class FeatureOnlyCVAE(nn.Module):
    """
    Ultra-lightweight version that only extracts features (no decoder)
    Use this if you only need encoder features for segmentation
    """
    
    def __init__(self, input_channels=3, hidden_dims=None):
        super(FeatureOnlyCVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        self.hidden_dims = hidden_dims
        self.input_channels = input_channels
        
        # Only encoder blocks
        self.encoder_block1 = EncoderBlock(input_channels, hidden_dims[0], stride=2)
        self.encoder_block2 = EncoderBlock(hidden_dims[0], hidden_dims[1], stride=2)  
        self.encoder_block3 = EncoderBlock(hidden_dims[1], hidden_dims[2], stride=2)
    
    def forward(self, x):
        """Extract only multi-scale features"""
        x1 = self.encoder_block1(x)      # [B, 64, H/2, W/2]
        x2 = self.encoder_block2(x1)     # [B, 128, H/4, W/4]
        x3 = self.encoder_block3(x2)     # [B, 256, H/8, W/8]
        
        return {
            'encoder_features': [x1, x2, x3]
        }
    
    def encode(self, x):
        """Compatibility with original CVAE interface"""
        outputs = self.forward(x)
        encoder_features = outputs['encoder_features']
        
        # Return dummy mu, log_var for compatibility
        batch_size = x.size(0)
        device = x.device
        dummy_mu = torch.zeros(batch_size, 256, device=device)
        dummy_log_var = torch.zeros(batch_size, 256, device=device)
        
        return dummy_mu, dummy_log_var, encoder_features


# Backward compatibility
SimplifiedCVAE_v2 = SimplifiedCVAE
CVAE_FeatureOnly = FeatureOnlyCVAE


def compare_cvae_efficiency():
    """
    Compare efficiency of different CVAE variants
    """
    from net.cvae import EnhancedCVAE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 256, 256, device=device)
    
    models = {
        'Original CVAE': EnhancedCVAE(),
        'Simplified CVAE': SimplifiedCVAE(), 
        'Feature Only': FeatureOnlyCVAE()
    }
    
    print("CVAE Efficiency Comparison")
    print("=" * 50)
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # Measure memory and speed
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            if name == 'Feature Only':
                outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_mb = 0
        
        print(f"{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Output keys: {list(outputs.keys())}")
        print()


if __name__ == "__main__":
    compare_cvae_efficiency()