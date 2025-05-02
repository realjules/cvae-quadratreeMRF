import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for better feature integration.
    Implements a transformer-style cross-attention mechanism to fuse features.
    """
    def __init__(self, query_dim, key_dim, embed_dim=64):
        super(CrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # Projection layers
        self.query_proj = nn.Conv2d(query_dim, embed_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(key_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(key_dim, embed_dim, kernel_size=1)
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, query_dim, kernel_size=1)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        
        # Layer normalization for better training stability
        self.norm1 = nn.LayerNorm([embed_dim])
        self.norm2 = nn.LayerNorm([embed_dim])
        
    def forward(self, query, key_value):
        """
        Forward pass for cross-attention fusion
        Args:
            query: Main features to be enhanced (B, C1, H, W)
            key_value: Supporting features providing context (B, C2, H, W)
        Returns:
            Enhanced features (B, C1, H, W)
        """
        batch_size = query.size(0)
        
        # Project inputs to embedding space
        q = self.query_proj(query)  # (B, E, H, W)
        k = self.key_proj(key_value)  # (B, E, H, W)
        v = self.value_proj(key_value)  # (B, E, H, W)
        
        # Reshape for attention computation
        q_shape = q.shape
        # (B, E, H*W) -> (B, H*W, E)
        q = q.view(batch_size, self.embed_dim, -1).permute(0, 2, 1)
        # (B, E, H*W)
        k = k.view(batch_size, self.embed_dim, -1)
        # (B, E, H*W) -> (B, H*W, E)
        v = v.view(batch_size, self.embed_dim, -1).permute(0, 2, 1)
        
        # Compute scaled dot-product attention
        # (B, H*W, E) @ (B, E, H*W) -> (B, H*W, H*W)
        attn = torch.bmm(q, k) / self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention weights to values
        # (B, H*W, H*W) @ (B, H*W, E) -> (B, H*W, E)
        out = torch.bmm(attn, v)
        
        # Reshape back to spatial dimensions
        # (B, H*W, E) -> (B, E, H, W)
        out = out.permute(0, 2, 1).view(batch_size, self.embed_dim, q_shape[2], q_shape[3])
        
        # Output projection with residual connection
        enhanced = self.output_proj(out) + query
        
        return enhanced


class AttentionFusionCVAE(nn.Module):
    """
    Enhanced Contrastive Variational Autoencoder with:
    - Cross-attention fusion for skip connections
    - U-Net style skip connections
    - Spatial attention modules
    - Improved decoder with residual blocks
    - Enhanced projection head
    - Perceptual loss capability
    - Structural consistency
    - Increased network capacity
    - Spatial latent representation
    - Improved contrastive learning
    """
    def __init__(self, input_channels=3, latent_dim=512, hidden_dims=[64, 128, 256, 512]):
        super(AttentionFusionCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()  # Create a copy to avoid in-place modification
        self.input_channels = input_channels
        self.pos_encoding_channels = 5  # Number of positional encoding channels
        
        # Initial projection layer to handle positional encoding
        # This adapts the input (RGB + positional encoding) to the proper channel count
        # We'll use only 3 positional encoding channels to keep things simpler
        self.input_projection = nn.Conv2d(
            input_channels + 3,  # 3 (RGB) + 3 (reduced positional encoding) = 6
            input_channels,  # Project back to 3 channels
            kernel_size=1,  # 1x1 convolution for channel projection
            stride=1,
            padding=0
        )
        
        # Encoder blocks with access to intermediate features
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_channels
        
        # Build encoder with residual blocks
        for h_dim in hidden_dims:
            self.encoder_blocks.append(
                nn.Sequential(
                    ResidualConvBlock(in_channels, h_dim, stride=2),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                    SpatialAttention(h_dim)
                )
            )
            in_channels = h_dim
            
        # Calculate output size for a 256x256 input
        # For the enhanced model, we use 3 downsampling layers (not 4)
        # This gives us 256/(2^3) = 32x32 resolution at the bottleneck
        # We use 32x32 instead of 8x8 to maintain more spatial information
        self.output_height = 32  # Using 32x32 spatial dimensions at bottleneck
        self.output_width = 32
        self.output_features = hidden_dims[-2]  # Using the 3rd feature map (256), not the 4th (512)
        
        # Total flattened size (256 * 32 * 32 = 262,144)
        flatten_size = self.output_features * self.output_height * self.output_width
        
        # Latent space with variational info bottleneck
        # Increased latent dimension for richer representation
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(flatten_size, latent_dim)
        
        # Enhanced projection head for contrastive learning (3-layer MLP)
        # Using larger dimensions for projection to maintain expressivity
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Decoder input - increased spatial resolution at bottleneck
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-2] * 32 * 32)
        
        # Reverse hidden dims for decoder
        hidden_dims_reversed = hidden_dims[::-1]
        
        # Decoder blocks with skip connections - completely rebuilt for new structure
        self.decoder_blocks = nn.ModuleList()
        
        # We're using a different decoder structure that matches our 3-level encoder
        # Instead of using all 4 levels of hidden_dims, we use only 3 (like in the encoder)
        decoder_dims = hidden_dims[:3][::-1]  # Reverse and take first 3: [256, 128, 64]
        
        # First decoder block: 256->128, upsampling from 32x32 to 64x64
        self.decoder_blocks.append(
            nn.Sequential(
                ResidualConvTransposeBlock(decoder_dims[0], decoder_dims[1]),  # 256->128
                nn.BatchNorm2d(decoder_dims[1]),
                nn.LeakyReLU(0.2)
            )
        )
        
        # Cross-attention fusion modules for enhanced skip connections
        # These will be used to fuse decoder features with encoder features
        self.fusion1 = CrossAttentionFusion(decoder_dims[1], hidden_dims[1], embed_dim=64)  # 128, 128
        self.fusion2 = CrossAttentionFusion(decoder_dims[2], hidden_dims[0], embed_dim=64)  # 64, 64
        
        # Second decoder block: 128->64, upsampling from 64x64 to 128x128
        # Input channels are now just 128 since we use attention fusion instead of concatenation
        self.decoder_blocks.append(
            nn.Sequential(
                ResidualConvTransposeBlock(decoder_dims[1], decoder_dims[2]),  # 128->64
                nn.BatchNorm2d(decoder_dims[2]),
                nn.LeakyReLU(0.2)
            )
        )
        
        # Final layer to output image, upsampling from 128x128 to 256x256
        # Input channels are now just 64 since we use attention fusion instead of concatenation
        self.final_layer = nn.Sequential(
            ResidualConvTransposeBlock(decoder_dims[2], decoder_dims[2]),  # 64->64
            nn.BatchNorm2d(decoder_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(decoder_dims[2], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Greatly increased memory bank size for contrastive learning
        # 4096 items instead of 512 for better representation of dataset distribution
        self.register_buffer("queue", torch.randn(4096, latent_dim // 4))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Initialize VGG model for perceptual loss
        self.init_perceptual_network()
        
        # Positional encoding for spatial awareness
        # Creating higher-resolution positional encoding for better spatial detail
        self.register_buffer("positional_encoding", self.create_positional_encoding(256, 256))
        
    def init_perceptual_network(self):
        """Initialize VGG network for perceptual loss"""
        vgg = models.vgg16(pretrained=True)
        self.perceptual_network = nn.Sequential(*list(vgg.features.children())[:16])
        # Freeze the network
        for param in self.perceptual_network.parameters():
            param.requires_grad = False
            
    def create_positional_encoding(self, height, width):
        """Create enhanced positional encoding for spatial awareness"""
        # Create coordinate grid (basic x,y coordinates)
        y_coords = torch.linspace(-1, 1, height).view(-1, 1).expand(-1, width)
        x_coords = torch.linspace(-1, 1, width).expand(height, -1)
        
        # Add radial distance from center
        center_y, center_x = height // 2, width // 2
        y_grid = torch.arange(height).view(-1, 1).expand(-1, width).float() - center_y
        x_grid = torch.arange(width).expand(height, -1).float() - center_x
        
        radius = torch.sqrt((y_grid / center_y) ** 2 + (x_grid / center_x) ** 2)
        
        # Add periodic functions for better spatial representation
        # Scale these based on image size to create the right frequency
        freq_factor = 2.0 * 3.14159 / max(height, width)
        sin_x = torch.sin(x_grid * freq_factor * 4)  # Higher frequency for finer details
        sin_y = torch.sin(y_grid * freq_factor * 4)
        
        # Stack all positional encodings (now 5 channels)
        coords = torch.stack([y_coords, x_coords, radius, sin_y, sin_x], dim=0)
        return coords
        
    def encode(self, x):
        """Encode input images to latent representations with skip connections"""
        # Process through encoder blocks and store intermediate activations
        intermediate_features = []
        
        # If input has positional encoding channels, apply initial projection
        if x.size(1) > self.input_channels:  # If input has RGB + positional encoding
            # Project input with positional encoding to original channel count
            current_x = self.input_projection(x)
        else:
            current_x = x
        
        # Process through only the first 3 encoder blocks (not all 4)
        # This provides higher resolution feature maps (32x32 instead of 16x16)
        for i, block in enumerate(self.encoder_blocks):
            # Stop after the third encoder block (indexed as 2)
            if i >= 3:  # Only process first 3 blocks (0, 1, 2)
                break
                
            current_x = block(current_x)
            intermediate_features.append(current_x)
            
        # Ensure we have the expected feature size
        assert current_x.size(2) == self.output_height and current_x.size(3) == self.output_width, \
            f"Expected features of size ({self.output_height}, {self.output_width}), got ({current_x.size(2)}, {current_x.size(3)})"
            
        # Flatten final features
        flattened = torch.flatten(current_x, start_dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(flattened)
        log_var = self.fc_log_var(flattened)
        
        return mu, log_var, intermediate_features
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick with improved numerical stability"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, encoder_features=None):
        """Decode latent representations with cross-attention fusion for skip connections"""
        # Map from latent space to initial decoder volume
        result = self.decoder_input(z)
        
        # Reshape to spatial volume - now using 32x32 spatial starting dimensions
        batch_size = z.size(0)
        result = result.view(batch_size, self.hidden_dims[2], 32, 32)  # Using the 3rd dim (256)
        
        # Process through decoder blocks with cross-attention fusion for skip connections
        x = result
        
        # If we don't have encoder features, create dummy ones for inference
        if encoder_features is None or len(encoder_features) < 3:
            # Create encoder features with the right shapes and dimensions for testing
            dummy_features = []
            # First level feature: 64 channels, 64x64
            dummy_features.append(torch.zeros(batch_size, self.hidden_dims[0], 64, 64, device=z.device))
            # Second level feature: 128 channels, 32x32
            dummy_features.append(torch.zeros(batch_size, self.hidden_dims[1], 32, 32, device=z.device))
            # Third level feature: 256 channels, 16x16
            dummy_features.append(torch.zeros(batch_size, self.hidden_dims[2], 16, 16, device=z.device))
            
            encoder_features = dummy_features
        
        # Now we have a consistent simplified decoder path with cross-attention fusion:
        
        # 1. First block: 256->128, 32x32 -> 64x64 (no skip connection yet)
        x = self.decoder_blocks[0](x)  # Now x is [B, 128, 64, 64]
        
        # 2. Apply cross-attention fusion between decoder features and encoder features
        # Ensure dimensions match for encoder feature
        if x.size(2) != encoder_features[1].size(2) or x.size(3) != encoder_features[1].size(3):
            encoder_feat = F.interpolate(
                encoder_features[1],
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
        else:
            encoder_feat = encoder_features[1]
        
        # Apply cross-attention fusion instead of concatenation
        x = self.fusion1(x, encoder_feat)  # Now x is still [B, 128, 64, 64] but enhanced with encoder features
        
        # 3. Second decoder block: 128->64, 64x64 -> 128x128
        x = self.decoder_blocks[1](x)  # Now x is [B, 64, 128, 128]
        
        # 4. Apply cross-attention fusion again
        # Ensure dimensions match for encoder feature
        if x.size(2) != encoder_features[0].size(2) or x.size(3) != encoder_features[0].size(3):
            encoder_feat = F.interpolate(
                encoder_features[0],
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
        else:
            encoder_feat = encoder_features[0]
        
        # Apply cross-attention fusion instead of concatenation
        x = self.fusion2(x, encoder_feat)  # Now x is still [B, 64, 128, 128] but enhanced with encoder features
        
        # 5. Final layer: 64->3, 128x128 -> 256x256
        x = self.final_layer(x)  # Now x is [B, 3, 256, 256]
        
        return x
    
    def project(self, z):
        """Project latent representations for contrastive learning"""
        return self.projection_head(z)
    
    @torch.no_grad()
    def update_queue(self, z_proj):
        """Update memory bank queue for contrastive learning"""
        batch_size = z_proj.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the features at ptr (dequeue and enqueue)
        if ptr + batch_size > self.queue.shape[0]:
            # Handle wrapping
            first_part = self.queue.shape[0] - ptr
            self.queue[ptr:] = z_proj[:first_part]
            self.queue[:batch_size - first_part] = z_proj[first_part:]
            ptr = batch_size - first_part
        else:
            # Normal case
            self.queue[ptr:ptr + batch_size] = z_proj
            ptr = (ptr + batch_size) % self.queue.shape[0]
        
        self.queue_ptr[0] = ptr
    
    def perceptual_loss(self, x_recon, x_orig):
        """Calculate perceptual loss using VGG16 features with improved stability"""
        # Only compute if we have a GPU available (VGG is large)
        if torch.cuda.is_available():
            try:
                # Ensure input is in valid range for VGG (0-1)
                x_recon_safe = torch.clamp(x_recon, 0, 1)
                x_orig_safe = torch.clamp(x_orig, 0, 1)
                
                # Apply the perceptual network with gradient handling
                x_recon_vgg = self.perceptual_network(x_recon_safe)
                x_orig_vgg = self.perceptual_network(x_orig_safe)
                
                # Check for NaN values in features
                if torch.isnan(x_recon_vgg).any() or torch.isnan(x_orig_vgg).any():
                    return torch.tensor(0.0, device=x_recon.device)
                
                # L2 loss between feature representations with clamping
                loss = F.mse_loss(x_recon_vgg, x_orig_vgg)
                return torch.clamp(loss, 0, 10.0)  # Prevent extremely large values
            except Exception as e:
                print(f"Warning in perceptual loss: {e}")
                return torch.tensor(0.0, device=x_recon.device)
        else:
            # Return a dummy tensor if no GPU
            return torch.tensor(0.0, device=x_recon.device)
    
    def ssim_loss(self, x_recon, x_orig, window_size=11, sigma=1.5):
        """Calculate structural similarity loss with improved numerical stability"""
        # Add small constants for numerical stability
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        try:
            # Ensure inputs are properly clipped to valid range
            x_recon = torch.clamp(x_recon, 0, 1)
            x_orig = torch.clamp(x_orig, 0, 1)
            
            # Create Gaussian kernel
            kernel_size = window_size
            kernel = self.create_gaussian_kernel(kernel_size, sigma).to(x_recon.device)
            
            # Use a smaller window_size for better performance and stability
            if window_size > 7:
                kernel_size = 7
                kernel = self.create_gaussian_kernel(kernel_size, sigma).to(x_recon.device)
            
            # Convert to grayscale if RGB
            if x_recon.size(1) == 3:
                # Convert to grayscale using RGB weights
                # Red: 0.299, Green: 0.587, Blue: 0.114
                weights = torch.FloatTensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(x_recon.device)
                x_recon_gray = (x_recon * weights).sum(dim=1, keepdim=True)
                x_orig_gray = (x_orig * weights).sum(dim=1, keepdim=True)
            else:
                x_recon_gray = x_recon
                x_orig_gray = x_orig
            
            # Compute means
            mu1 = F.conv2d(x_recon_gray, kernel, padding=kernel_size//2, groups=1)
            mu2 = F.conv2d(x_orig_gray, kernel, padding=kernel_size//2, groups=1)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            # Compute variances with epsilon for stability
            eps = 1e-6
            sigma1_sq = F.conv2d(x_recon_gray**2, kernel, padding=kernel_size//2, groups=1) - mu1_sq + eps
            sigma2_sq = F.conv2d(x_orig_gray**2, kernel, padding=kernel_size//2, groups=1) - mu2_sq + eps
            sigma12 = F.conv2d(x_recon_gray * x_orig_gray, kernel, padding=kernel_size//2, groups=1) - mu1_mu2 + eps
            
            # Ensure positive variances (shouldn't be negative, but just in case)
            sigma1_sq = torch.clamp(sigma1_sq, min=eps)
            sigma2_sq = torch.clamp(sigma2_sq, min=eps)
            
            # SSIM formula with careful handling of divisions
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            ssim_map = numerator / (denominator + eps)
            
            # Clamp to avoid extreme values
            ssim_map = torch.clamp(ssim_map, 0, 1)
            
            # Convert to loss (1 - SSIM)
            loss = 1 - ssim_map.mean()
            
            # Final safety check 
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, device=x_recon.device)
                
            return loss
            
        except Exception as e:
            print(f"Warning in SSIM calculation: {e}")
            return torch.tensor(0.0, device=x_recon.device)
    
    def create_gaussian_kernel(self, kernel_size, sigma):
        """Create a Gaussian kernel for SSIM calculation"""
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        
        # Create 1D Gaussian kernel
        kernel_1d = torch.exp(-0.5 * (coords / sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Reshape to conv filter shape [1, 1, kernel_size, kernel_size]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        return kernel_2d
    
    def forward(self, x):
        """Forward pass through the Enhanced CVAE with attention fusion"""
        # Add positional encoding for additional spatial awareness
        # Resize positional encoding if input size doesn't match
        if x.size(2) != self.positional_encoding.size(1) or x.size(3) != self.positional_encoding.size(2):
            pos_enc = F.interpolate(
                self.positional_encoding.unsqueeze(0), 
                size=(x.size(2), x.size(3)), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            pos_enc = self.positional_encoding
        
        # Add positional encoding to input for improved spatial sensitivity
        # Use only the first 3 channels to keep input size reasonable
        pos_enc_reduced = pos_enc[:3]  # Use only x, y coordinates and radius
        x_with_pos = torch.cat([x, pos_enc_reduced.expand(x.size(0), -1, -1, -1)], dim=1)
        
        # Encode (using input with positional encoding)
        mu, log_var, encoder_features = self.encode(x_with_pos)
        
        # Sample latent
        z = self.reparameterize(mu, log_var)
        
        # Decode with cross-attention fusion for skip connections
        x_recon = self.decode(z, encoder_features)
        
        # Project for contrastive learning
        z_proj = self.project(z)
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Update memory bank (only during training)
        if self.training:
            with torch.no_grad():
                self.update_queue(z_proj_norm.detach())
        
        # Calculate perceptual loss if available
        perceptual = self.perceptual_loss(x_recon, x) if torch.cuda.is_available() else torch.tensor(0.0, device=x.device)
        
        # Calculate SSIM loss
        ssim = self.ssim_loss(x_recon, x)
        
        # Return all intermediate results and losses
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'z_proj': z_proj_norm,
            'original_input': x,
            'queue': self.queue,
            'perceptual_loss': perceptual,
            'ssim_loss': ssim,
            'encoder_features': encoder_features
        }
    
    def contrastive_loss(self, z_proj, labels=None, temperature=0.05):
        """Enhanced contrastive loss with improved memory bank support"""
        batch_size = z_proj.size(0)
        device = z_proj.device
        
        # Normalize projections (should already be normalized, but just to be safe)
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Compute cosine similarity matrix with lower temperature for sharper contrasts
        # Reduced temperature from 0.07 to 0.05 for sharper distinctions
        sim_matrix = torch.mm(z_proj_norm, z_proj_norm.t()) / temperature
        
        # Include enhanced memory bank
        queue = self.queue.clone().detach()
        
        # Calculate similarities with larger queue (now 4096 items)
        queue_sim = torch.mm(z_proj_norm, queue.t()) / temperature
        
        # Apply hard negative mining - identify difficult negatives
        # These are samples that are close in embedding space but should be different
        with torch.no_grad():
            # Find hardest negatives in current batch (highest similarity excluding self)
            hard_indices = torch.argsort(sim_matrix, dim=1, descending=True)
            # Remove self-similarity
            hard_indices = hard_indices[:, 1:int(batch_size * 0.25) + 1]  # Top 25% hardest
        
        # Weight harder negatives more in the loss calculation
        hard_weights = torch.ones(batch_size, batch_size + queue.shape[0], device=device)
        for i in range(batch_size):
            # Increase weight for hard negatives (in batch)
            hard_weights[i, hard_indices[i]] = 2.0
            
        # Combine for full similarity matrix
        sim_matrix_expanded = torch.cat([sim_matrix, queue_sim], dim=1)
        
        if labels is not None:
            # Enhanced supervised contrastive loss
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask = pos_mask.float()
            
            # Remove self-comparisons
            eye_mask = torch.eye(batch_size, device=device)
            pos_mask = pos_mask - eye_mask
            pos_mask = torch.clamp(pos_mask, 0, 1)
            
            # Count positive pairs
            num_pos = pos_mask.sum(dim=1)
            
            # Handle samples with no positives
            valid_samples = num_pos > 0
            if valid_samples.sum() == 0:
                return torch.tensor(0.0, device=device)
            
            # Enhanced log probabilities with hard negative weighting
            weighted_sim = sim_matrix_expanded * hard_weights
            log_prob = F.log_softmax(weighted_sim, dim=1)[:, :batch_size]
            
            # Apply focal loss modification to emphasize harder examples
            focal_weight = (1 - torch.exp(log_prob)) ** 2  # squared focal term
            weighted_log_prob = focal_weight * log_prob
            
            # Compute final supervised loss
            mean_log_prob_pos = (pos_mask * weighted_log_prob).sum(1) / num_pos.clamp(min=1)
            loss = -mean_log_prob_pos[valid_samples].mean()
        else:
            # Enhanced unsupervised contrastive loss (InfoNCE)
            labels = torch.arange(batch_size, device=device)
            
            # Mask out self-comparisons
            sim_matrix_no_diag = sim_matrix - torch.eye(batch_size, device=device) * 1e9
            
            # Apply weighted cross entropy with hard negative emphasis
            weighted_sim_no_diag = sim_matrix_no_diag * hard_weights[:, :batch_size]
            
            # Standard cross entropy loss with improved weighting
            loss = F.cross_entropy(weighted_sim_no_diag, labels)
        
        return loss


class ResidualConvBlock(nn.Module):
    """Residual convolutional block for the encoder"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualConvBlock, self).__init__()
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResidualConvTransposeBlock(nn.Module):
    """Residual transposed convolutional block for the decoder"""
    def __init__(self, in_channels, out_channels):
        super(ResidualConvTransposeBlock, self).__init__()
        
        # Main branch - transposed convolution for upsampling
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Additional convolution for feature refinement
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with upsampling - creating fresh, not reusing existing one
        # This is a crucial fix - creating a new skip connection for each instance
        self.skip_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.skip_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # Process skip connection - manually to avoid reusing old objects
        # This ensures the skip connection uses the correct input channels
        skip = self.skip_upsample(x)
        skip = self.skip_conv(skip)
        residual = self.skip_bn(skip)
        
        # Main path
        out = self.conv_transpose(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += residual
        out = self.relu(out)
        
        return out


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on relevant features"""
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Create attention map using avg and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_cat)
        
        # Apply attention
        return x * attention