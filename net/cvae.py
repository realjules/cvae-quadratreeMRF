import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EnhancedCVAE(nn.Module):
    """
    Enhanced Contrastive Variational Autoencoder matching the report specifications
    """
    def __init__(self, input_channels=3, latent_dim=256, hidden_dims=None):
        super(EnhancedCVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims  
        self.input_channels = input_channels
        
        # ====== ENCODER ======
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate feature sizes (256/8 = 32)
        self.output_height = 32
        self.output_width = 32
        
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
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[2] * 32 * 32)
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[1] * 2, hidden_dims[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2)
        )
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0] * 2, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # Contrastive learning memory bank
        self.register_buffer("queue", torch.randn(4096, latent_dim // 4))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def encode(self, x):
        """Encode input images to latent representations with skip connections"""
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        
        encoder_features = [x1, x2, x3]
        
        # Flatten final features
        flattened = torch.flatten(x3, start_dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(flattened)
        log_var = self.fc_log_var(flattened)
        
        return mu, log_var, encoder_features
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, encoder_features=None):
        """Decode latent representations with skip connections"""
        result = self.decoder_input(z)
        batch_size = z.size(0)
        x = result.view(batch_size, self.hidden_dims[2], 32, 32)
        
        if encoder_features is None:
            encoder_features = [
                torch.zeros(batch_size, self.hidden_dims[0], 128, 128, device=z.device),
                torch.zeros(batch_size, self.hidden_dims[1], 64, 64, device=z.device),
                torch.zeros(batch_size, self.hidden_dims[2], 32, 32, device=z.device)
            ]
        
        # Decoder block 1: 256→128, 32x32 → 64x64
        x = self.decoder_block1(x)
        
        # Decoder block 2: concat with encoder features
        if x.size(2) != encoder_features[1].size(2):
            encoder_feat = F.interpolate(encoder_features[1], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            encoder_feat = encoder_features[1]
        
        x = torch.cat([x, encoder_feat], dim=1)
        x = self.decoder_block2(x)
        
        # Final output: concat with encoder features
        if x.size(2) != encoder_features[0].size(2):
            encoder_feat = F.interpolate(encoder_features[0], size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        else:
            encoder_feat = encoder_features[0]
        
        x = torch.cat([x, encoder_feat], dim=1)
        x = self.final_layer(x)
        
        return x
    
    def forward(self, x):
        """Forward pass through the Enhanced CVAE"""
        mu, log_var, encoder_features = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, encoder_features)
        z_proj = self.project(z)
        z_proj_norm = F.normalize(z_proj, dim=1)
        
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
        """Update memory bank queue for contrastive learning"""
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

# Backward compatibility
CVAE = EnhancedCVAE