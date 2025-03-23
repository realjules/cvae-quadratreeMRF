import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(CVAE, self).__init__()
        # Encoder: two conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Fully-connected layers for latent variables
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder: project latent to feature map and then deconvolve
        self.fc_dec = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        enc = self.encoder(x)            # [B, 128, 8, 8] for input 32x32
        enc_flat = enc.view(batch_size, -1)  
        mu = self.fc_mu(enc_flat)
        logvar = self.fc_logvar(enc_flat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        dec_input = self.fc_dec(z).view(batch_size, 128, 8, 8)
        rec = self.decoder(dec_input)
        return z, rec