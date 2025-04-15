import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Simplified Contrastive Variational Autoencoder with predictable dimensions
    """
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=[32, 64, 128, 256]):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims.copy()  # Create a copy to avoid in-place modification
        
        # Encoder
        modules = []
        in_channels = input_channels
        
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Calculate output size for a 256x256 input after four stride-2 convolutions: 16x16
        self.output_height = 16
        self.output_width = 16
        self.output_features = hidden_dims[-1]
        
        # Total flattened size
        flatten_size = self.output_features * self.output_height * self.output_width
        
        # Latent space
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_var = nn.Linear(flatten_size, latent_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Decoder - Start with latent dim to initial volume
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)
        
        # Reverse hidden dims for decoder
        hidden_dims_reversed = hidden_dims[::-1]
        
        # Decoder layers
        decoder_modules = []
        
        # Add transposed convolutions to increase spatial dimensions
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i], hidden_dims_reversed[i+1],
                                    kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims_reversed[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer to output image
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_reversed[-1], hidden_dims_reversed[-1],
                                kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims_reversed[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims_reversed[-1], input_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*decoder_modules)
        
        # Memory bank for contrastive learning
        self.register_buffer("queue", torch.randn(128, latent_dim // 4))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def encode(self, x):
        """Encode input images to latent representations"""
        # Process through encoder
        features = self.encoder(x)
        
        # Flatten
        flattened = torch.flatten(features, start_dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from the latent space"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representations back to images"""
        # Map from latent space to initial decoder volume
        result = self.decoder_input(z)
        
        # Reshape to spatial volume - fixed 4x4 spatial starting dimensions
        batch_size = z.size(0)
        result = result.view(batch_size, self.hidden_dims[-1], 4, 4)
        
        # Process through decoder
        return self.decoder(result)
    
    def project(self, z):
        """Project latent representations for contrastive learning"""
        return self.projection_head(z)
    
    @torch.no_grad()
    def update_queue(self, z_proj):
        """Update memory bank queue for contrastive learning"""
        batch_size = z_proj.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the features at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = z_proj
        ptr = (ptr + batch_size) % 128  # Cycle back when queue is full
        
        self.queue_ptr[0] = ptr
    
    def forward(self, x):
        """Forward pass through the CVAE"""
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        # Project for contrastive learning
        z_proj = self.project(z)
        
        # Update memory bank (only during training)
        if self.training:
            with torch.no_grad():
                self.update_queue(F.normalize(z_proj.detach(), dim=1))
        
        # Return all intermediate results
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'z_proj': z_proj,
            'original_input': x,  # Store original input for loss calculation
            'queue': self.queue,  # Provide queue for expanded contrastive loss
        }
    
    def contrastive_loss(self, z_proj, labels=None, temperature=0.5):
        """Calculate contrastive loss with memory bank support"""
        batch_size = z_proj.size(0)
        device = z_proj.device
        
        # Normalize projections
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
            # Unsupervised contrastive loss
            labels = torch.arange(batch_size, device=device)
            
            # Mask out self-comparisons
            sim_matrix_no_diag = sim_matrix - torch.eye(batch_size, device=device) * 1e9
            
            # Standard cross entropy loss
            loss = F.cross_entropy(sim_matrix_no_diag, labels)
        
        return loss


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