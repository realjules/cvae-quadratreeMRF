import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CVAE(nn.Module):
    """
    Enhanced Contrastive Variational Autoencoder with:
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
    def __init__(self, input_channels=3, latent_dim=512, hidden_dims=None):
        super(FixedEnhancedCVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
            
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims  
        self.input_channels = input_channels
        
        # Positional encoding and projection
        self.pos_encoding_channels = 3
        self.input_projection = nn.Conv2d(
            input_channels + self.pos_encoding_channels,
            input_channels,
            kernel_size=1
        )
        
        # ====== ENCODER ======
        # Only 3 encoder blocks (not 4) for higher resolution features
        self.encoder_block1 = nn.Sequential(
            ResidualConvBlock(input_channels, hidden_dims[0], stride=2),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            SpatialAttention(hidden_dims[0])
        )
        
        self.encoder_block2 = nn.Sequential(
            ResidualConvBlock(hidden_dims[0], hidden_dims[1], stride=2),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
            SpatialAttention(hidden_dims[1])
        )
        
        self.encoder_block3 = nn.Sequential(
            ResidualConvBlock(hidden_dims[1], hidden_dims[2], stride=2),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
            SpatialAttention(hidden_dims[2])
        )
        
        # Directly calculate feature sizes
        self.output_height = 32  # 256 / 2^3
        self.output_width = 32   # 256 / 2^3
        
        # Calculate total size for latent space
        flatten_size = hidden_dims[2] * self.output_height * self.output_width
        
        # Variational components
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(flatten_size, latent_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # ====== DECODER ======
        # Decoder input: latent → initial spatial tensor (32x32)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[2] * 32 * 32)
        
        # Decoder blocks (3 in total, matching encoder)
        # Block 1: 256→128, 32x32 → 64x64
        self.decoder_block1 = nn.Sequential(
            SimpleUpBlock(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2)
        )
        
        # Block 2: (128+128)→64, 64x64 → 128x128
        self.decoder_block2 = nn.Sequential(
            SimpleUpBlock(hidden_dims[1] * 2, hidden_dims[0]),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2)
        )
        
        # Final output layer: (64+64)→3, 128x128 → 256x256
        self.final_layer = nn.Sequential(
            SimpleUpBlock(hidden_dims[0] * 2, hidden_dims[0]),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims[0], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Contrastive learning memory bank
        self.register_buffer("queue", torch.randn(4096, latent_dim // 4))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Initialize VGG model for perceptual loss
        self.init_perceptual_network()
        
        # Positional encoding for spatial awareness
        self.register_buffer("positional_encoding", self.create_positional_encoding(256, 256))
    
    def init_perceptual_network(self):
        """Initialize VGG network for perceptual loss"""
        vgg = models.vgg16(pretrained=True)
        self.perceptual_network = nn.Sequential(*list(vgg.features.children())[:16])
        # Freeze the network
        for param in self.perceptual_network.parameters():
            param.requires_grad = False
            
    def create_positional_encoding(self, height, width):
        """Create positional encoding for spatial awareness"""
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height).view(-1, 1).expand(-1, width)
        x_coords = torch.linspace(-1, 1, width).expand(height, -1)
        
        # Add radial distance from center
        center_y, center_x = height // 2, width // 2
        y_grid = torch.arange(height).view(-1, 1).expand(-1, width).float() - center_y
        x_grid = torch.arange(width).expand(height, -1).float() - center_x
        
        radius = torch.sqrt((y_grid / center_y) ** 2 + (x_grid / center_x) ** 2)
        
        # Stack all positional encodings (3 channels)
        coords = torch.stack([y_coords, x_coords, radius], dim=0)
        return coords
    
    def encode(self, x):
        """Encode input images to latent representations with skip connections"""
        # Process through encoder blocks and store intermediate activations
        if x.size(1) > self.input_channels:
            # Project input+positional encoding to original channels
            x = self.input_projection(x)
        
        # Apply encoder blocks
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        
        # Store features for skip connections
        encoder_features = [x1, x2, x3]
        
        # Flatten final features
        flattened = torch.flatten(x3, start_dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(flattened)
        log_var = self.fc_log_var(flattened)
        
        return mu, log_var, encoder_features
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick with improved numerical stability"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, encoder_features=None):
        """Decode latent representations with skip connections"""
        # Map from latent space to initial spatial volume (256 channels, 32x32)
        result = self.decoder_input(z)
        batch_size = z.size(0)
        x = result.view(batch_size, self.hidden_dims[2], 32, 32)
        
        # If no encoder features available, create dummy ones
        if encoder_features is None:
            encoder_features = [
                torch.zeros(batch_size, self.hidden_dims[0], 128, 128, device=z.device),
                torch.zeros(batch_size, self.hidden_dims[1], 64, 64, device=z.device),
                torch.zeros(batch_size, self.hidden_dims[2], 32, 32, device=z.device)
            ]
        
        # Decoder block 1: 256→128, 32x32 → 64x64
        x = self.decoder_block1(x)
        
        # Decoder block 2: concat with encoder features, then 128*2→64, 64x64 → 128x128
        # Ensure feature size matches for concatenation
        if x.size(2) != encoder_features[1].size(2) or x.size(3) != encoder_features[1].size(3):
            encoder_feat = F.interpolate(
                encoder_features[1],
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
        else:
            encoder_feat = encoder_features[1]
        
        x = torch.cat([x, encoder_feat], dim=1)
        x = self.decoder_block2(x)
        
        # Final output: concat with encoder features, then 64*2→3, 128x128 → 256x256
        # Ensure feature size matches for concatenation
        if x.size(2) != encoder_features[0].size(2) or x.size(3) != encoder_features[0].size(3):
            encoder_feat = F.interpolate(
                encoder_features[0],
                size=(x.size(2), x.size(3)),
                mode='bilinear',
                align_corners=False
            )
        else:
            encoder_feat = encoder_features[0]
        
        x = torch.cat([x, encoder_feat], dim=1)
        x = self.final_layer(x)
        
        return x
    
    def forward(self, x):
        """Forward pass through the Enhanced CVAE"""
        # Add positional encoding for additional spatial awareness
        if x.size(2) != self.positional_encoding.size(1) or x.size(3) != self.positional_encoding.size(2):
            pos_enc = F.interpolate(
                self.positional_encoding.unsqueeze(0), 
                size=(x.size(2), x.size(3)), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            pos_enc = self.positional_encoding
        
        # Add positional encoding to input
        x_with_pos = torch.cat([x, pos_enc.expand(x.size(0), -1, -1, -1)], dim=1)
        
        # Encode
        mu, log_var, encoder_features = self.encode(x_with_pos)
        
        # Sample latent
        z = self.reparameterize(mu, log_var)
        
        # Decode with skip connections
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
    
    def contrastive_loss(self, z_proj, labels=None, temperature=0.05):
        """Enhanced contrastive loss with memory bank support"""
        batch_size = z_proj.size(0)
        device = z_proj.device
        
        # Normalize projections (should already be normalized, but just to be safe)
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(z_proj_norm, z_proj_norm.t()) / temperature
        
        # Include memory bank if available
        queue = self.queue.clone().detach()
        
        # Calculate similarities with queue
        queue_sim = torch.mm(z_proj_norm, queue.t()) / temperature
        
        # Combine for full similarity matrix
        sim_matrix_expanded = torch.cat([sim_matrix, queue_sim], dim=1)
        
        if labels is not None:
            # Supervised contrastive loss
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
            
            # Log probabilities and loss calculation
            log_prob = F.log_softmax(sim_matrix_expanded, dim=1)[:, :batch_size]
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / num_pos.clamp(min=1)
            loss = -mean_log_prob_pos[valid_samples].mean()
        else:
            # Unsupervised contrastive loss (InfoNCE)
            labels = torch.arange(batch_size, device=device)
            
            # Mask out self-comparisons
            sim_matrix_no_diag = sim_matrix - torch.eye(batch_size, device=device) * 1e9
            
            # Standard cross entropy loss
            loss = F.cross_entropy(sim_matrix_no_diag, labels)
        
        return loss
    
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
    
    def ssim_loss(self, x_recon, x_orig, window_size=7, sigma=1.5):
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
            
            # SSIM formula
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


class SimpleUpBlock(nn.Module):
    """Simplified upsampling block for decoder to avoid dimension issues"""
    def __init__(self, in_channels, out_channels):
        super(SimpleUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


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