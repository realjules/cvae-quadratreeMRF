"""
CVAE Training Integration for Semi-Supervised Learning

This module provides proper training integration for the CVAE with:
1. Contrastive learning on unlabeled data
2. Feature extraction for segmentation
3. Proper loss combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from net.cvae import EnhancedCVAE
from utils.losses import contrastive_loss, simclr_loss_simple
from utils.contrastive_augmentations import ContrastiveAugmentation, create_contrastive_pair


class CVAETrainer:
    """
    Trainer for CVAE with contrastive learning and semi-supervised integration
    """
    
    def __init__(self, 
                 input_channels=3, 
                 latent_dim=256, 
                 hidden_dims=None,
                 learning_rate=1e-4,
                 device="cuda",
                 temperature=0.07,
                 kl_weight=0.1,
                 contrastive_weight=1.0,
                 kl_warmup_epochs=0,
                 use_memory_bank=True):
        
        self.device = device
        self.temperature = temperature
        self.use_memory_bank = use_memory_bank
        
        # Initialize CVAE
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        
        self.cvae = EnhancedCVAE(
            input_channels=input_channels,
            latent_dim=latent_dim, 
            hidden_dims=hidden_dims
        ).to(device)
        
        # Optimizer for CVAE
        self.optimizer = torch.optim.AdamW(
            self.cvae.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Augmentation pipeline
        self.contrastive_aug = ContrastiveAugmentation(size=256, strength=0.8)
        
        # Loss weights
        self.recon_weight = 1.0
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.beta = 0.0 if kl_warmup_epochs > 0 else 1.0
        
        # Metrics tracking
        self.metrics = {
            'recon_loss': [],
            'kl_loss': [],
            'contrastive_loss': [],
            'total_loss': []
        }
        
        # Best model tracking
        self.best_contrastive_loss = float('inf')
        self.best_model_path = "./output/cvae_best.pth"
        
    def train_step_contrastive(self, batch_images):
        """
        Training step focused on contrastive learning (unsupervised)
        
        Args:
            batch_images: [B, C, H, W] batch of images
            
        Returns:
            dict: Loss components and metrics
        """
        self.cvae.train()
        batch_size = batch_images.size(0)
        
        # Generate augmented pairs for contrastive learning
        view1_batch = []
        view2_batch = []
        
        for i in range(batch_size):
            # Get single image
            image = batch_images[i]  # [C, H, W]
            
            # Generate two augmented views
            view1, view2 = self.contrastive_aug(image)
            view1_batch.append(view1)
            view2_batch.append(view2)
        
        # Stack into batches
        view1_batch = torch.stack(view1_batch).to(self.device)  # [B, C, H, W]
        view2_batch = torch.stack(view2_batch).to(self.device)  # [B, C, H, W]
        
        # Forward pass through CVAE for both views
        outputs1 = self.cvae(view1_batch)
        outputs2 = self.cvae(view2_batch)
        
        # Extract components
        recon1, mu1, log_var1 = outputs1['reconstruction'], outputs1['mu'], outputs1['log_var']
        recon2, mu2, log_var2 = outputs2['reconstruction'], outputs2['mu'], outputs2['log_var']
        z_proj1, z_proj2 = outputs1['z_proj'], outputs2['z_proj']
        
        # 1. Reconstruction loss (for both views)
        recon_loss1 = F.mse_loss(recon1, view1_batch)
        recon_loss2 = F.mse_loss(recon2, view2_batch)
        recon_loss = (recon_loss1 + recon_loss2) / 2
        
        # 2. KL divergence loss (for both views)
        kl_loss1 = -0.5 * torch.sum(1 + log_var1 - mu1.pow(2) - log_var1.exp(), dim=1).mean()
        kl_loss2 = -0.5 * torch.sum(1 + log_var2 - mu2.pow(2) - log_var2.exp(), dim=1).mean()
        kl_loss = (kl_loss1 + kl_loss2) / 2
        
        # 3. Contrastive loss between projected features
        if self.use_memory_bank and hasattr(self.cvae, 'queue'):
            # Use MoCo-style contrastive loss with memory bank
            queue = self.cvae.queue
            # Check queue dimensions before using
            if queue.size(1) == z_proj1.size(1):
                contrast_loss = contrastive_loss(z_proj1, z_proj2, self.temperature, queue)
            else:
                print(f"⚠️  Queue dimension mismatch: using simple contrastive loss")
                contrast_loss = simclr_loss_simple(z_proj1, z_proj2, self.temperature)
        else:
            # Use simple SimCLR-style loss
            contrast_loss = simclr_loss_simple(z_proj1, z_proj2, self.temperature)
        
        # Total loss with KL warm-up
        total_loss = (self.recon_weight * recon_loss + 
                     self.beta * self.kl_weight * kl_loss + 
                     self.contrastive_weight * contrast_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'contrastive_loss': contrast_loss.item(),
            'batch_size': batch_size
        }
    
    def train_step_reconstruction(self, batch_images):
        """
        Training step focused on reconstruction (can be used with limited labeled data)
        
        Args:
            batch_images: [B, C, H, W] batch of images
            
        Returns:
            dict: Loss components and metrics  
        """
        self.cvae.train()
        
        # Forward pass
        outputs = self.cvae(batch_images)
        recon, mu, log_var = outputs['reconstruction'], outputs['mu'], outputs['log_var']
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(recon, batch_images)
        
        # 2. KL divergence loss  
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # Total loss (no contrastive component)
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(), 
            'kl_loss': kl_loss.item(),
            'contrastive_loss': 0.0,
            'batch_size': batch_images.size(0)
        }
    
    def extract_features(self, images):
        """
        Extract multi-scale features for segmentation
        
        Args:
            images: [B, C, H, W] input images
            
        Returns:
            dict: Multi-scale features for segmentation model
        """
        self.cvae.eval()
        with torch.no_grad():
            # Get encoder features (this is what we need for segmentation)
            mu, log_var, encoder_features = self.cvae.encode(images)
            
            # encoder_features contains [feat_l1, feat_l2, feat_l3]
            # feat_l1: [B, 64, 128, 128] - fine details
            # feat_l2: [B, 128, 64, 64] - medium features  
            # feat_l3: [B, 256, 32, 32] - semantic features
            
            return {
                'p1': encoder_features[0],  # Fine level
                'p2': encoder_features[1],  # Medium level
                'p3': encoder_features[2],  # Coarse level
                'global_context': encoder_features[2],  # Use coarse as global context
                'latent_mu': mu,
                'latent_log_var': log_var
            }
    
    def train_epoch_contrastive(self, dataloader, epoch):
        """Train one epoch with contrastive learning"""
        self.cvae.train()
        epoch_metrics = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'contrastive_loss': 0, 'count': 0}
        
        progress_bar = tqdm(dataloader, desc=f"CVAE Contrastive Epoch {epoch}")
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # Contrastive training step
            metrics = self.train_step_contrastive(images)
            
            # Update epoch metrics
            for key in ['total_loss', 'recon_loss', 'kl_loss', 'contrastive_loss']:
                epoch_metrics[key] += metrics[key] * metrics['batch_size']
            epoch_metrics['count'] += metrics['batch_size']
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                'Recon': f"{metrics['recon_loss']:.4f}", 
                'KL': f"{metrics['kl_loss']:.4f}",
                'Contrast': f"{metrics['contrastive_loss']:.4f}"
            })
        
        # Calculate average metrics
        for key in ['total_loss', 'recon_loss', 'kl_loss', 'contrastive_loss']:
            epoch_metrics[key] /= epoch_metrics['count']
            self.metrics[key].append(epoch_metrics[key])
        
        # Update KL warm-up beta
        if self.kl_warmup_epochs > 0:
            self.beta = min(1.0, epoch / self.kl_warmup_epochs)

        # Step scheduler
        self.scheduler.step()
        
        return epoch_metrics
    
    def save_model(self, path, epoch, metrics=None):
        """Save CVAE model and training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.cvae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': {
                'latent_dim': self.cvae.latent_dim,
                'hidden_dims': self.cvae.hidden_dims,
                'input_channels': self.cvae.input_channels,
                'temperature': self.temperature,
                'use_memory_bank': self.use_memory_bank
            }
        }
        
        if metrics is not None:
            checkpoint['epoch_metrics'] = metrics
            
        torch.save(checkpoint, path)
        print(f"✅ CVAE model saved to {path}")
    
    def save_best_if_improved(self, current_metrics, epoch):
        """Save model if contrastive loss improved"""
        current_contrastive_loss = current_metrics['contrastive_loss']
        
        print(f"   📊 Contrastive loss: {current_contrastive_loss:.4f} (best: {self.best_contrastive_loss:.4f})")
        
        if current_contrastive_loss < self.best_contrastive_loss:
            improvement = self.best_contrastive_loss - current_contrastive_loss
            self.best_contrastive_loss = current_contrastive_loss
            self.save_model(self.best_model_path, epoch, current_metrics)
            print(f"🏆 NEW BEST CVAE MODEL! Epoch {epoch}")
            print(f"   ✅ Contrastive loss improved by {improvement:.4f}")
            print(f"   💾 Saved to: {self.best_model_path}")
            return True
        else:
            print(f"   📈 No improvement in contrastive loss")
            return False
    
    def load_model(self, path):
        """Load CVAE model and training state"""
        if not os.path.exists(path):
            print(f"❌ Model file {path} does not exist")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.cvae.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.metrics = checkpoint.get('metrics', self.metrics)
            
            epoch = checkpoint.get('epoch', 0)
            print(f"✅ CVAE model loaded from {path} (epoch {epoch})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def test_cvae_trainer():
    """Test function to verify CVAE trainer works correctly"""
    print("Testing CVAE Trainer...")
    
    # Create trainer
    trainer = CVAETrainer(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 256, 256)
    
    if torch.cuda.is_available():
        dummy_images = dummy_images.cuda()
    
    # Test contrastive training step
    print("Testing contrastive training step...")
    metrics = trainer.train_step_contrastive(dummy_images)
    print(f"✅ Contrastive step metrics: {metrics}")
    
    # Test feature extraction
    print("Testing feature extraction...")
    features = trainer.extract_features(dummy_images)
    print(f"✅ Extracted features shapes:")
    for key, feat in features.items():
        if isinstance(feat, torch.Tensor):
            print(f"  {key}: {feat.shape}")
    
    print("✅ CVAE Trainer test passed!")


if __name__ == "__main__":
    test_cvae_trainer()