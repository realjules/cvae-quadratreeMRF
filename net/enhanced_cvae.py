import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EnhancedCVAE(nn.Module):
    """
    Enhanced Contrastive Variational Autoencoder with:
    - U-Net style skip connections
    - Spatial attention modules
    - Improved decoder with residual blocks
    - Enhanced projection head
    - Perceptual loss capability
    - Structural consistency
    """
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=[32, 64, 128, 256]):
        super(EnhancedCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()  # Create a copy to avoid in-place modification
        
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
            
        # Calculate output size for a 256x256 input after stride-2 convolutions
        self.output_height = 256 // (2 ** len(hidden_dims))  # 16 for default hidden_dims
        self.output_width = 256 // (2 ** len(hidden_dims))   # 16 for default hidden_dims
        self.output_features = hidden_dims[-1]
        
        # Total flattened size
        flatten_size = self.output_features * self.output_height * self.output_width
        
        # Latent space with variational info bottleneck
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(flatten_size, latent_dim)
        
        # Enhanced projection head for contrastive learning (3-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)
        
        # Reverse hidden dims for decoder
        hidden_dims_reversed = hidden_dims[::-1]
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList()
        
        # First decoder block (from bottleneck)
        self.decoder_blocks.append(
            nn.Sequential(
                ResidualConvTransposeBlock(hidden_dims_reversed[0], hidden_dims_reversed[1 % len(hidden_dims_reversed)]),
                nn.BatchNorm2d(hidden_dims_reversed[1 % len(hidden_dims_reversed)]),
                nn.LeakyReLU(0.2)
            )
        )
        
        # Additional decoder blocks with skip connections
        for i in range(1, len(hidden_dims_reversed) - 1):
            # Input channels are doubled due to skip connection
            in_channels = hidden_dims_reversed[i]
            out_channels = hidden_dims_reversed[i+1]
            
            self.decoder_blocks.append(
                nn.Sequential(
                    ResidualConvTransposeBlock(in_channels * 2, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Final layer to output image
        self.final_layer = nn.Sequential(
            ResidualConvTransposeBlock(hidden_dims_reversed[-1] * 2, hidden_dims_reversed[-1]),
            nn.BatchNorm2d(hidden_dims_reversed[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims_reversed[-1], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Memory bank for contrastive learning (larger bank)
        self.register_buffer("queue", torch.randn(512, latent_dim // 4))
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
        
        # Stack coordinates
        coords = torch.stack([y_coords, x_coords], dim=0)
        return coords
        
    def encode(self, x):
        """Encode input images to latent representations with skip connections"""
        # Process through encoder blocks and store intermediate activations
        intermediate_features = []
        current_x = x
        
        for block in self.encoder_blocks:
            current_x = block(current_x)
            intermediate_features.append(current_x)
            
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
        """Decode latent representations with skip connections"""
        # Map from latent space to initial decoder volume
        result = self.decoder_input(z)
        
        # Reshape to spatial volume - fixed 4x4 spatial starting dimensions
        batch_size = z.size(0)
        result = result.view(batch_size, self.hidden_dims[-1], 4, 4)
        
        # Process through decoder blocks with skip connections
        x = result
        
        # If we don't have encoder features, create dummy ones for inference
        if encoder_features is None:
            encoder_features = [None] * len(self.decoder_blocks)
        
        # Apply first decoder block (no skip connection for this one)
        x = self.decoder_blocks[0](x)
        
        # Apply remaining decoder blocks with skip connections
        for i in range(1, len(self.decoder_blocks)):
            # Upsample to match encoder feature size if needed
            if x.size(2) != encoder_features[-(i+1)].size(2) or x.size(3) != encoder_features[-(i+1)].size(3):
                x = F.interpolate(
                    x, 
                    size=(encoder_features[-(i+1)].size(2), encoder_features[-(i+1)].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Concatenate with corresponding encoder features
            x = torch.cat([x, encoder_features[-(i+1)]], dim=1)
            x = self.decoder_blocks[i](x)
        
        # Apply final layer with skip connection to first encoder layer
        if x.size(2) != encoder_features[0].size(2) or x.size(3) != encoder_features[0].size(3):
            x = F.interpolate(
                x,
                size=(encoder_features[0].size(2), encoder_features[0].size(3)),
                mode='bilinear',
                align_corners=False
            )
        
        # Final concatenation and output layer
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.final_layer(x)
        
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
        """Calculate perceptual loss using VGG16 features"""
        # Only compute if we have a GPU available (VGG is large)
        if torch.cuda.is_available():
            # Make sure both inputs are in correct format (may need to denormalize)
            x_recon_vgg = self.perceptual_network(x_recon)
            x_orig_vgg = self.perceptual_network(x_orig)
            
            # L2 loss between feature representations
            return F.mse_loss(x_recon_vgg, x_orig_vgg)
        else:
            # Return a dummy tensor if no GPU
            return torch.tensor(0.0, device=x_recon.device)
    
    def ssim_loss(self, x_recon, x_orig, window_size=11, sigma=1.5):
        """Calculate structural similarity loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
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
        
        # Compute variances
        sigma1_sq = F.conv2d(x_recon_gray**2, kernel, padding=kernel_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(x_orig_gray**2, kernel, padding=kernel_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(x_recon_gray * x_orig_gray, kernel, padding=kernel_size//2, groups=1) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Convert to loss (1 - SSIM)
        return 1 - ssim_map.mean()
    
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
        """Forward pass through the Enhanced CVAE"""
        # Add positional encoding if needed for additional spatial awareness
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
            
        # Optional: add positional encoding to input for improved spatial sensitivity
        # x_with_pos = torch.cat([x, pos_enc.expand(x.size(0), -1, -1, -1)], dim=1)
        
        # Encode
        mu, log_var, encoder_features = self.encode(x)
        
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
    
    def contrastive_loss(self, z_proj, labels=None, temperature=0.07):
        """Enhanced contrastive loss with memory bank support"""
        batch_size = z_proj.size(0)
        device = z_proj.device
        
        # Normalize projections (should already be normalized, but just to be safe)
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Compute cosine similarity matrix with lower temperature for sharper contrasts
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
            
            # Log probabilities and loss calculation with temperature scaling
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
        
        # Skip connection with upsampling
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Process skip connection
        residual = self.skip(x)
        
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