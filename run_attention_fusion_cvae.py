#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train the Attention Fusion CVAE component with cross-attention skip connections
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from skimage import io
import torch.amp

from net.attention_fusion_cvae import AttentionFusionCVAE
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import AerialImageDataset
from dataset.unsupervised_dataset import UnsupervisedAerialDataset

class EnhancedCVAELoss(torch.nn.Module):
    """
    Enhanced CVAE loss function with:
    - Reconstruction loss (MSE + SSIM)
    - KL divergence
    - Contrastive loss
    - Perceptual loss (if available)
    """
    def __init__(self, kld_weight=0.0001, contrastive_weight=0.1, ssim_weight=0.05, perceptual_weight=0.01):
        super(EnhancedCVAELoss, self).__init__()
        # Using much smaller initial weights to prevent instability
        self.kld_weight = kld_weight
        self.contrastive_weight = contrastive_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        
        # For adaptive loss weights
        self.epoch = 0
        self.total_epochs = 100  # Will be updated in update_epoch
        
        # Add warm-up factor for KL divergence
        self.kld_warmup_epochs = 10  # Warm up KLD over first 10 epochs
            
    def forward(self, outputs, targets=None, mode='full'):
        # Extract values from outputs dictionary
        x_recon = outputs['reconstruction']
        mu = outputs['mu']
        log_var = outputs['log_var']
        z_proj = outputs['z_proj']
        original_input = outputs['original_input']
        
        # Queue for contrastive learning
        queue = outputs['queue']
        
        # Perceptual and SSIM losses if available
        perceptual = outputs.get('perceptual_loss', torch.tensor(0.0, device=x_recon.device))
        ssim = outputs.get('ssim_loss', torch.tensor(0.0, device=x_recon.device))
        
        # Reconstruction loss (MSE)
        recon_loss = self.mse_loss(x_recon, original_input)
        
        # KL Divergence with warm-up
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Normalize by batch size and latent dimension
        kld_loss = kld_loss / (mu.size(0) * mu.size(1))
        
        # Get KL warmup factor
        warmup_factor = min(1.0, self.epoch / self.kld_warmup_epochs) if self.kld_warmup_epochs > 0 else 1.0
        
        # Apply warmup to KLD weight
        kld_weight = self.kld_weight * warmup_factor
        
        # Calculate adaptive contrastive weight based on training progress
        # Gradually increase contrastive weight as training progresses
        progress = min(1.0, self.epoch / (self.total_epochs * 0.7))
        adaptive_contrastive = self.contrastive_weight * (0.8 + 0.2 * progress)
        
        # Contrastive loss
        model = outputs.get('model', None)
        if model is not None:
            contrastive_loss = model.contrastive_loss(z_proj)
        else:
            # Fallback implementation if model isn't provided
            contrastive_loss = torch.tensor(0.0, device=x_recon.device)
        
        # Combine all losses
        total_loss = recon_loss + \
                    kld_weight * kld_loss + \
                    adaptive_contrastive * contrastive_loss + \
                    self.ssim_weight * ssim + \
                    self.perceptual_weight * perceptual
        
        # Return all loss components for monitoring
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'contrastive_loss': contrastive_loss,
            'ssim_loss': ssim,
            'perceptual_loss': perceptual
        }
    
    def update_epoch(self, epoch, total_epochs=None):
        """Update current epoch for adaptive weighting"""
        self.epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs


def save_checkpoint(model, optimizer, epoch, scheduler, scaler, loss, path):
    """Save model checkpoint with all training state"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, scaler, path):
    """Load model checkpoint with all training state"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def train_model(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/attention_fusion_cvae_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    if args.unsupervised:
        dataset = UnsupervisedAerialDataset(
            args.data_path,
            size=args.image_size,
            transform_prob=0.5
        )
    else:
        dataset = AerialImageDataset(
            args.data_path,
            args.unsupervised,
            size=args.image_size,
            transform_prob=0.5
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = AttentionFusionCVAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=[64, 128, 256, 512]
    ).to(device)
    
    # Create loss function
    criterion = EnhancedCVAELoss(
        kld_weight=args.kld_weight,
        contrastive_weight=args.contrastive_weight,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader),
        eta_min=1e-6
    )
    
    # Create grad scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Starting epoch and best loss
    start_epoch = 0
    best_loss = float('inf')
    
    # Load checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, scaler, args.resume)
        print(f"Resumed from epoch {start_epoch}")
        start_epoch += 1  # Start from the next epoch
    
    # Update criterion with total epochs
    criterion.update_epoch(start_epoch, args.epochs)
    
    # Loss history
    loss_history = {
        'total': [],
        'recon': [],
        'kld': [],
        'contrastive': [],
        'ssim': [],
        'perceptual': []
    }
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Update criterion epoch
        criterion.update_epoch(epoch, args.epochs)
        
        # Set model to training mode
        model.train()
        
        # Epoch losses
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_perceptual_loss = 0.0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            if args.unsupervised:
                images = batch.to(device)
                targets = None
            else:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = model(images)
                    
                    # Add model to outputs for contrastive loss
                    outputs['model'] = model
                    
                    # Calculate loss
                    loss_dict = criterion(outputs, targets)
                    
                # Backward pass with scaler
                scaler.scale(loss_dict['loss']).backward()
                
                # Clip gradients
                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = model(images)
                
                # Add model to outputs for contrastive loss
                outputs['model'] = model
                
                # Calculate loss
                loss_dict = criterion(outputs, targets)
                
                # Backward pass
                loss_dict['loss'].backward()
                
                # Clip gradients
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
                # Update weights
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update epoch losses
            epoch_loss += loss_dict['loss'].item()
            epoch_recon_loss += loss_dict['recon_loss'].item()
            epoch_kld_loss += loss_dict['kld_loss'].item()
            epoch_contrastive_loss += loss_dict['contrastive_loss'].item()
            epoch_ssim_loss += loss_dict['ssim_loss'].item()
            epoch_perceptual_loss += loss_dict['perceptual_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': epoch_loss / (batch_idx + 1),
                'recon': epoch_recon_loss / (batch_idx + 1),
                'kld': epoch_kld_loss / (batch_idx + 1),
                'contrastive': epoch_contrastive_loss / (batch_idx + 1)
            })
            
            # Generate samples periodically
            if args.sample_interval > 0 and batch_idx % args.sample_interval == 0:
                generate_samples(model, images, outputs, os.path.join(output_dir, f"samples_epoch{epoch+1}_batch{batch_idx}.png"))
        
        # Calculate average epoch losses
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kld_loss = epoch_kld_loss / len(dataloader)
        avg_contrastive_loss = epoch_contrastive_loss / len(dataloader)
        avg_ssim_loss = epoch_ssim_loss / len(dataloader)
        avg_perceptual_loss = epoch_perceptual_loss / len(dataloader)
        
        # Update loss history
        loss_history['total'].append(avg_loss)
        loss_history['recon'].append(avg_recon_loss)
        loss_history['kld'].append(avg_kld_loss)
        loss_history['contrastive'].append(avg_contrastive_loss)
        loss_history['ssim'].append(avg_ssim_loss)
        loss_history['perceptual'].append(avg_perceptual_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
              f"KLD: {avg_kld_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f}, "
              f"SSIM: {avg_ssim_loss:.4f}, Perceptual: {avg_perceptual_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, scheduler, scaler, avg_loss,
                os.path.join(output_dir, "best_model.pth")
            )
            print(f"Saved best model with loss {best_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, scheduler, scaler, avg_loss,
                os.path.join(output_dir, f"model_epoch{epoch+1}.pth")
            )
            
            # Plot and save loss curves
            plot_loss_curves(loss_history, os.path.join(output_dir, "loss_curves.png"))
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1, scheduler, scaler, avg_loss,
        os.path.join(output_dir, "final_model.pth")
    )
    
    # Plot final loss curves
    plot_loss_curves(loss_history, os.path.join(output_dir, "loss_curves.png"))
    
    return model, loss_history


def generate_samples(model, images, outputs, save_path):
    """Generate reconstructions and save them as a grid"""
    model.eval()
    with torch.no_grad():
        # Get original images and reconstructions
        originals = images[:8].cpu()  # Take up to 8 images
        recons = outputs['reconstruction'][:8].cpu()
        
        # Create a grid of images
        n_samples = min(8, originals.shape[0])
        fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))
        
        for i in range(n_samples):
            # Original image
            img = originals[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')
            
            # Reconstruction
            recon = recons[i].permute(1, 2, 0).numpy()
            recon = np.clip(recon, 0, 1)
            axes[1, i].imshow(recon)
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def plot_loss_curves(loss_history, save_path):
    """Plot loss curves and save the figure"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(loss_history['total'], label='Total')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(loss_history['recon'], label='Reconstruction')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # KLD loss
    axes[1, 0].plot(loss_history['kld'], label='KLD')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Contrastive loss
    axes[1, 1].plot(loss_history['contrastive'], label='Contrastive')
    axes[1, 1].set_title('Contrastive Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    # SSIM loss
    axes[2, 0].plot(loss_history['ssim'], label='SSIM')
    axes[2, 0].set_title('SSIM Loss')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].grid(True)
    
    # Perceptual loss
    axes[2, 1].plot(loss_history['perceptual'], label='Perceptual')
    axes[2, 1].set_title('Perceptual Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention Fusion CVAE Model")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='input/dataset', help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--unsupervised', action='store_true', help='Train in unsupervised mode (no labels)')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of the latent space')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--kld_weight', type=float, default=0.0001, help='Weight for KL divergence loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='Weight for contrastive loss')
    parser.add_argument('--ssim_weight', type=float, default=0.05, help='Weight for SSIM loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.01, help='Weight for perceptual loss')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Misc parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for dataloader')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--save_interval', type=int, default=10, help='Epochs between saving model checkpoints')
    parser.add_argument('--sample_interval', type=int, default=100, help='Batches between generating samples (0 to disable)')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Train the model
    model, loss_history = train_model(args)