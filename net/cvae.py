import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EnhancedCVAE(nn.Module):
    """
    Enhanced Contrastive Variational Autoencoder matching the report specifications
    This file remains unchanged as the CVAE implementation is already optimal
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
        
        # Momentum coefficient for MoCo
        self.m = 0.999 # As per MoCo paper

        # ====== ENCODER (Query Encoder) ======
        self.base_encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
        )
        
        self.base_encoder_block2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
        )
        
        self.base_encoder_block3 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
        )
        
        # Projection head for contrastive learning (Query Projection Head)
        self.base_projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )

        # ====== KEY ENCODER (Momentum Encoder) ======
        self.key_encoder_block1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
        )
        
        self.key_encoder_block2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.LeakyReLU(0.2),
        )
        
        self.key_encoder_block3 = nn.Sequential(
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2),
        )
        
        # Key Projection Head
        self.key_projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )

        # Initialize key encoder and key projection head with base encoder weights
        for param_q, param_k in zip(self.base_encoder_block1.parameters(), self.key_encoder_block1.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.base_encoder_block2.parameters(), self.key_encoder_block2.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.base_encoder_block3.parameters(), self.key_encoder_block3.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.base_projection_head.parameters(), self.key_projection_head.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Contrastive learning memory bank
        self.register_buffer("queue", torch.randn(4096, latent_dim // 4))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _encode_base(self, x):
        x1 = self.base_encoder_block1(x)
        x2 = self.base_encoder_block2(x1)
        x3 = self.base_encoder_block3(x2)
        return x1, x2, x3

    @torch.no_grad()
    def _encode_key(self, x):
        x1 = self.key_encoder_block1(x)
        x2 = self.key_encoder_block2(x1)
        x3 = self.key_encoder_block3(x2)
        return x1, x2, x3

    def encode(self, x):
        """Encode input images to latent representations with skip connections (using base encoder)"""
        x1, x2, x3 = self._encode_base(x)
        
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
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.base_encoder_block1.parameters(), self.key_encoder_block1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.base_encoder_block2.parameters(), self.key_encoder_block2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.base_encoder_block3.parameters(), self.key_encoder_block3.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.base_projection_head.parameters(), self.key_projection_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x_query, x_key=None):
        """Forward pass through the Enhanced CVAE for MoCo-style training"""
        # Query branch
        mu_q, log_var_q, encoder_features_q = self.encode(x_query)
        z_q = self.reparameterize(mu_q, log_var_q)
        z_proj_q = F.normalize(self.base_projection_head(z_q), dim=1)

        # Key branch (no grad)
        with torch.no_grad():
            self._momentum_update_key_encoder() # Update key encoder
            if x_key is None: # If x_key is not provided, use x_query
                x_key = x_query
            mu_k, log_var_k, _ = self._encode_key(x_key)
            z_k = self.reparameterize(mu_k, log_var_k)
            z_proj_k = F.normalize(self.key_projection_head(z_k), dim=1)

        # Reconstruction (from query branch)
        x_recon = self.decode(z_q, encoder_features_q)
        
        return {
            'reconstruction': x_recon,
            'mu': mu_q,
            'log_var': log_var_q,
            'z_q': z_q,
            'z_proj_q': z_proj_q,
            'z_proj_k': z_proj_k,
            'original_input': x_query,
            'queue': self.queue,
            'encoder_features': encoder_features_q
        }
    
    def project(self, z):
        """Project latent representations for contrastive learning (using base projection head)"""
        return self.base_projection_head(z)
    
    @torch.no_grad()
    def update_queue(self, z_proj):
        """Update memory bank queue for contrastive learning"""
        # The input z_proj should be detached before calling this method
        batch_size = z_proj.shape[0]
        ptr = int(self.queue_ptr)
        
        # Ensure z_proj is on the same device as the queue
        if z_proj.device != self.queue.device:
            z_proj = z_proj.to(self.queue.device)

        if ptr + batch_size > self.queue.shape[0]:
            # Handle wrap-around
            first_part_size = self.queue.shape[0] - ptr
            second_part_size = batch_size - first_part_size
            
            self.queue[ptr:] = z_proj[:first_part_size]
            self.queue[:second_part_size] = z_proj[first_part_size:]
            ptr = second_part_size
        else:
            self.queue[ptr:ptr + batch_size] = z_proj
            ptr = (ptr + batch_size) % self.queue.shape[0]
        
        self.queue_ptr[0] = ptr

# Backward compatibility
CVAE = EnhancedCVAE