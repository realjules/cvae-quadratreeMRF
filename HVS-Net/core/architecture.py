# HVS-Net/core/architecture.py

"""
This file defines the core architecture of the HVS-Net.

It contains:
1.  The Shared Encoder (borrowing from the original CVAE's encoder).
2.  The Generative Decoder (for image reconstruction).
3.  The Segmentation Decoder (for pixel-wise classification).
4.  The novel Cross-Decoder Attention mechanism that allows the two decoders to communicate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building Blocks ---

class SharedEncoder(nn.Module):
    """The shared encoder for both generative and segmentation tasks."""
    def __init__(self, n_channels=3, latent_dim=256, hidden_dims=[64, 128, 256], input_size=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder blocks
        self.enc_block1 = self._make_block(n_channels, hidden_dims[0])
        self.enc_block2 = self._make_block(hidden_dims[0], hidden_dims[1])
        self.enc_block3 = self._make_block(hidden_dims[1], hidden_dims[2])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dims[2], hidden_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[2]),
            nn.LeakyReLU(0.2)
        )

        # Calculate the size of the flattened features after the bottleneck
        final_size = input_size // (2**4) # 4 downsampling layers
        self.flattened_size = hidden_dims[2] * final_size * final_size

        # Latent space mapping
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_size, latent_dim)

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Encode
        e1 = self.enc_block1(x)
        e2 = self.enc_block2(e1)
        e3 = self.enc_block3(e2)
        bottleneck = self.bottleneck(e3)

        # Flatten for latent space
        flattened = torch.flatten(bottleneck, start_dim=1)
        assert flattened.shape[1] == self.flattened_size, f"Flattened size mismatch: expected {self.flattened_size}, got {flattened.shape[1]}"
        mu = self.fc_mu(flattened)
        log_var = self.fc_log_var(flattened)

        return mu, log_var, [e1, e2, e3, bottleneck]

class CrossDecoderAttention(nn.Module):
    """The novel cross-decoder attention mechanism."""
    def __init__(self, seg_channels, gen_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(seg_channels, seg_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(gen_channels, seg_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(gen_channels, gen_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, seg_features, gen_features):
        B, C_s, H_s, W_s = seg_features.size()
        B, C_g, H_g, W_g = gen_features.size()
        assert H_s == H_g and W_s == W_g, "Feature maps must have the same dimensions for attention."

        query = self.query_conv(seg_features).view(B, -1, H_s * W_s).permute(0, 2, 1)
        key = self.key_conv(gen_features).view(B, -1, H_s * W_s)
        value = self.value_conv(gen_features).view(B, -1, H_s * W_s)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C_g, H_s, W_s)

        return seg_features + self.gamma * out

class BaseDecoder(nn.Module):
    """A base decoder to ensure symmetric architecture."""
    def __init__(self, latent_dim=256, hidden_dims=[256, 128, 64, 32], output_size=256):
        super().__init__()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * (output_size // 16) * (output_size // 16))
        self.output_size = output_size
        self.hidden_dims = hidden_dims

        self.dec_block1 = self._make_block(hidden_dims[0], hidden_dims[1])
        self.dec_block2 = self._make_block(hidden_dims[1], hidden_dims[2])
        self.dec_block3 = self._make_block(hidden_dims[2], hidden_dims[3])
        self.dec_block4 = self._make_block(hidden_dims[3], hidden_dims[3])


    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

class GenerativeDecoder(BaseDecoder):
    """The generative decoder for image reconstruction."""
    def __init__(self, n_channels=3, latent_dim=256, hidden_dims=[256, 128, 64, 32], output_size=256):
        super().__init__(latent_dim, hidden_dims, output_size)
        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[3], n_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder_input(z).view(-1, self.hidden_dims[0], self.output_size // 16, self.output_size // 16)
        d1 = self.dec_block1(x)
        d2 = self.dec_block2(d1)
        d3 = self.dec_block3(d2)
        d4 = self.dec_block4(d3)
        recon = self.final_layer(d4)
        return recon, [d1, d2, d3, d4]

class SegmentationDecoder(BaseDecoder):
    """The segmentation decoder for pixel-wise classification."""
    def __init__(self, n_classes=6, latent_dim=256, hidden_dims=[256, 128, 64, 32], output_size=256):
        super().__init__(latent_dim, hidden_dims, output_size)
        self.final_layer = nn.Conv2d(hidden_dims[3], n_classes, 1)

        # Attention modules
        self.attn1 = CrossDecoderAttention(hidden_dims[1], hidden_dims[1])
        self.attn2 = CrossDecoderAttention(hidden_dims[2], hidden_dims[2])
        self.attn3 = CrossDecoderAttention(hidden_dims[3], hidden_dims[3])


    def forward(self, z, gen_features):
        x = self.decoder_input(z).view(-1, self.hidden_dims[0], self.output_size // 16, self.output_size // 16)
        d1 = self.dec_block1(x)
        d1_attn = self.attn1(d1, gen_features[0])
        d2 = self.dec_block2(d1_attn)
        d2_attn = self.attn2(d2, gen_features[1])
        d3 = self.dec_block3(d2_attn)
        d3_attn = self.attn3(d3, gen_features[2])
        d4 = self.dec_block4(d3_attn)
        
        seg = self.final_layer(d4)
        return seg

# --- Main Model ---

class HVSNet(nn.Module):
    """The complete Hierarchical VAE-Segmenter network."""
    def __init__(self, n_channels=3, n_classes=6, latent_dim=256, input_size=256, 
                 encoder_hidden_dims=[64, 128, 256], decoder_hidden_dims=[256, 128, 64, 32]):
        super(HVSNet, self).__init__()
        self.encoder = SharedEncoder(n_channels=n_channels, latent_dim=latent_dim, hidden_dims=encoder_hidden_dims, input_size=input_size)
        self.gen_decoder = GenerativeDecoder(n_channels=n_channels, latent_dim=latent_dim, hidden_dims=decoder_hidden_dims, output_size=input_size)
        self.seg_decoder = SegmentationDecoder(n_classes=n_classes, latent_dim=latent_dim, hidden_dims=decoder_hidden_dims, output_size=input_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var, encoder_features = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        recon, gen_features = self.gen_decoder(z)
        seg = self.seg_decoder(z, gen_features)

        return {
            'segmentation': seg,
            'reconstruction': recon,
            'mu': mu,
            'log_var': log_var
        }
