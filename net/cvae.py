import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Enhanced Contrastive Variational Autoencoder for learning discriminative latent representations
    from unlabeled remote sensing imagery. This implementation includes both the standard
    VAE components and contrastive learning mechanisms, with improved architecture for
    better performance.
    """
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=[32, 64, 128, 256]):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()  # Create a copy to avoid in-place modification
        
        # Store encoder feature maps for skip connections
        self.encoder_features = []
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_channels
        
        # Build encoder with spatial attention
        for i, h_dim in enumerate(hidden_dims):
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(h_dim),  # Instance norm for better stability
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1),  # Additional conv for more capacity
                    nn.InstanceNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            # Add spatial attention after each encoder block except the first
            if i > 0:
                self.encoder_blocks[-1].add_module(
                    "attention", 
                    SpatialAttention(h_dim)
                )
            in_channels = h_dim
        
        # Adaptive pooling for flexible input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Latent space
        pooled_dim = hidden_dims[-1] * 8 * 8
        self.fc_mu = nn.Linear(pooled_dim, latent_dim)
        self.fc_var = nn.Linear(pooled_dim, latent_dim)
        
        # Enhanced projection head for contrastive learning (3-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Positional encoding for latent space
        self.positional_encoding = PositionalEncoding(latent_dim)
        
        # Decoder with skip connections
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 8 * 8)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Input channels: If using skip connections, double the channel count
            in_ch = hidden_dims[i] * 2 if i > 0 else hidden_dims[i]
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, hidden_dims[i+1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[i+1], hidden_dims[i+1], kernel_size=3, padding=1),  # Additional conv
                    nn.InstanceNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Memory bank for contrastive learning
        self.register_buffer("queue", torch.randn(128, latent_dim // 4))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.temperature = 0.5  # Initial temperature for contrastive loss
        
    def encode(self, x):
        """Encode input images to latent representations with feature storage for skip connections."""
        # Reset encoder features
        self.encoder_features = []
        
        # Process through encoder blocks
        result = x
        for block in self.encoder_blocks:
            result = block(result)
            self.encoder_features.append(result)
        
        # Adaptive pooling for flexible input sizes
        result = self.adaptive_pool(result)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        # Add positional encoding to latent space
        mu = self.positional_encoding(mu)
        
        return mu, log_var
    
    def decode(self, z):
        """Decode latent representations back to images using skip connections."""
        batch_size = z.size(0)
        result = self.decoder_input(z)
        
        # Reshape to spatial volume
        result = result.view(batch_size, self.hidden_dims[0], 8, 8)
        
        # Reverse encoder features for skip connections
        encoder_features = list(reversed(self.encoder_features))
        
        # Process through decoder blocks with skip connections
        for i, block in enumerate(self.decoder_blocks):
            # Add skip connection if we're past the first block
            if i > 0:
                # Resize encoder feature to match current size if needed
                encoder_feature = encoder_features[i]
                if encoder_feature.shape[2:] != result.shape[2:]:
                    encoder_feature = F.interpolate(
                        encoder_feature, 
                        size=result.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                # Concatenate along channel dimension
                result = torch.cat([result, encoder_feature], dim=1)
            
            result = block(result)
        
        # Apply final layer
        result = self.final_layer(result)
        
        return result
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def project(self, z):
        """Project latent representations for contrastive learning."""
        return self.projection_head(z)
    
    @torch.no_grad()
    def update_queue(self, z_proj):
        """Update memory bank queue for contrastive learning."""
        batch_size = z_proj.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the features at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = z_proj
        ptr = (ptr + batch_size) % 128  # Cycle back when queue is full
        
        self.queue_ptr[0] = ptr
    
    def forward(self, x):
        """Forward pass through the enhanced CVAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        z_proj = self.project(z)  # For contrastive learning
        
        # Update memory bank (only during training)
        if self.training:
            with torch.no_grad():
                self.update_queue(F.normalize(z_proj.detach(), dim=1))
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'z_proj': z_proj,
            'original_input': x,  # Store original input for loss calculation
            'queue': self.queue,  # Provide queue for expanded contrastive loss
        }
    
    def contrastive_loss(self, z_proj, labels=None, temperature=None):
        """
        Enhanced contrastive loss with memory bank and temperature annealing.
        
        Args:
            z_proj: Projected latent representations
            labels: Optional class labels for supervised contrastive loss
            temperature: Optional temperature parameter (uses self.temperature if None)
        """
        if temperature is None:
            temperature = self.temperature
            
        batch_size = z_proj.size(0)
        device = z_proj.device
        
        # Normalize projections
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Compute logits with both batch samples and queue
        queue = self.queue.clone().detach()
        
        # Compute similarities within batch
        sim_batch = torch.mm(z_proj_norm, z_proj_norm.t()) / temperature
        
        # Compute similarities with queue
        sim_queue = torch.mm(z_proj_norm, queue.t()) / temperature
        
        # Combine for full similarity matrix
        logits = torch.cat([sim_batch, sim_queue], dim=1)
        
        if labels is not None:
            # Supervised contrastive loss
            # Create a mask for positive pairs (same class)
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask = pos_mask.float()
            
            # Remove self-comparisons
            self_mask = torch.eye(batch_size, device=device)
            pos_mask = pos_mask - self_mask
            pos_mask = torch.clamp(pos_mask, 0, 1)
            
            # Count positive pairs for each anchor
            num_pos = pos_mask.sum(dim=1)
            
            # Handle anchors with no positives (use self as positive)
            no_pos_mask = (num_pos == 0).float()
            pos_mask = pos_mask + self_mask * no_pos_mask.unsqueeze(1)
            num_pos = pos_mask.sum(dim=1)
            
            # Apply log-softmax and compute loss
            log_prob = F.log_softmax(logits, dim=1)[:, :batch_size]
            
            # Calculate loss as negative mean of positive log-likelihood
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / num_pos
            loss = -mean_log_prob_pos.mean()
        else:
            # Unsupervised contrastive loss (InfoNCE)
            # For each query, the positive key is itself (diagonal of sim_batch)
            labels = torch.arange(batch_size, device=device)
            
            # Mask out self-comparisons for loss calculation
            mask = torch.eye(batch_size, device=device)
            sim_batch_masked = sim_batch - mask * 1e9
            
            # Only use masked batch similarity for loss
            loss = F.cross_entropy(sim_batch_masked, labels)
            
        return loss


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on relevant features."""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Generate attention map
        attention = self.conv(x)
        # Apply attention
        return x * attention


class PositionalEncoding(nn.Module):
    """Positional encoding for latent space to preserve spatial information."""
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Create simplified positional encoding for latent vector
        self.position_embedding = nn.Embedding(1, d_model)
        
        # Linear projection to combine with input
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # For latent vectors without sequence dimension
        batch_size = x.size(0)
        
        # Generate position embedding (same for all samples in batch)
        pos_emb = self.position_embedding(torch.zeros(1, dtype=torch.long, device=x.device))
        pos_emb = pos_emb.expand(batch_size, self.d_model)
        
        # Add position embedding and apply projection
        x = x + 0.1 * pos_emb  # Scale down positional contribution
        return self.dropout(self.fc(x))