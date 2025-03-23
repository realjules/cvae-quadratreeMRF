import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Contrastive Variational Autoencoder for learning discriminative latent representations
    from unlabeled remote sensing imagery. This implementation includes both the standard
    VAE components and contrastive learning mechanisms.
    """
    def __init__(self, input_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256]):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
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
        
        # Calculate output size of encoder for fully connected layers
        # For an input of size (256, 256), after 4 strides of 2, the output is (16, 16)
        self.encoder_output_size = hidden_dims[-1] * (256 // (2**len(hidden_dims))) * (256 // (2**len(hidden_dims)))
        
        # Latent space
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder
        modules = []
        
        # Projection from latent space to initial decoder volume
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_size)
        
        # Build decoder
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
    def encode(self, x):
        """Encode input images to latent representations."""
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    
    def decode(self, z):
        """Decode latent representations back to images."""
        result = self.decoder_input(z)
        result = result.view(-1, 256, 16, 16)  # Reshape to match encoder output shape
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick for sampling from the latent space."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def project(self, z):
        """Project latent representations for contrastive learning."""
        return self.projection_head(z)
    
    def forward(self, x):
        """Forward pass through the CVAE."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        z_proj = self.project(z)  # For contrastive learning
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'z_proj': z_proj,
            'original_input': x  # Store original input for loss calculation
        }
    
    def contrastive_loss(self, z_proj, labels=None, temperature=0.5):
        """
        Calculate contrastive loss based on projected latent representations.
        If labels are provided, use supervised contrastive loss, otherwise use unsupervised contrastive loss.
        """
        batch_size = z_proj.size(0)
        
        # Normalize projections
        z_proj_norm = F.normalize(z_proj, dim=1)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(z_proj_norm, z_proj_norm.t()) / temperature
        
        # Mask out self-comparisons
        mask = torch.eye(batch_size, device=z_proj.device)
        sim_matrix = sim_matrix - mask * 1e9
        
        if labels is not None:
            # Supervised contrastive loss
            # Create a mask for positive pairs (same class)
            pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask = pos_mask - mask  # Remove self-pairs
            
            # Count positive pairs for each anchor
            num_pos = pos_mask.sum(dim=1)
            
            # Handle anchors with no positives
            num_pos = torch.clamp(num_pos, min=1)
            
            # For numerical stability
            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - logits_max.detach()
            
            # Compute log probabilities
            exp_logits = torch.exp(sim_matrix)
            log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
            
            # Compute mean of positive log-likelihood
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / num_pos
            
            # Supervised contrastive loss
            loss = -mean_log_prob_pos.mean()
        else:
            # Unsupervised contrastive loss (InfoNCE)
            logits = sim_matrix
            labels = torch.arange(batch_size, device=z_proj.device)
            
            loss = F.cross_entropy(logits, labels)
            
        return loss