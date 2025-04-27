#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train only the Enhanced CVAE component with improved losses
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from skimage import io
import torch.amp  # Updated imports for torch.amp

from net.enhanced_cvae import EnhancedCVAE
from torch.utils.data import Dataset, DataLoader

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
        
        # Reconstruction loss (pixel-wise MSE) with value clipping
        mse_loss = torch.nn.functional.mse_loss(x_recon, original_input)
        mse_loss = torch.clamp(mse_loss, 0, 100)  # Prevent extremely large values
        
        # SSIM loss with safety checks
        ssim_loss = outputs.get('ssim_loss', torch.tensor(0.0, device=mse_loss.device))
        ssim_loss = torch.clamp(ssim_loss, 0, 10)  # Prevent extremely large values
        
        # KL Divergence loss with improved numerical stability
        # Clamp log_var to prevent extreme values
        log_var_clamped = torch.clamp(log_var, -20, 20)  # Prevent extreme values
        
        # Add epsilon inside the calculation for numerical stability
        kld_loss = -0.5 * torch.sum(1 + log_var_clamped - mu.pow(2).clamp(max=10) - log_var_clamped.exp().clamp(max=10) + 1e-8)
        kld_loss = kld_loss / (original_input.size(0) * original_input.size(1) * original_input.size(2) * original_input.size(3))  # Normalize by entire input size
        kld_loss = torch.clamp(kld_loss, 0, 1.0)  # Stricter clamping to prevent exploding gradients
        
        # Use a much smaller SSIM weight to prevent NaN
        ssim_weight_adjusted = self.ssim_weight * 0.01
        
        # Combined reconstruction loss with smaller weights
        recon_loss = mse_loss + ssim_weight_adjusted * ssim_loss
        
        # Use adaptive KL weight based on epoch
        if hasattr(self, 'current_kld_weight'):
            kld_weight_to_use = self.current_kld_weight
        else:
            # Fallback if update_epoch wasn't called yet
            kld_weight_to_use = self.kld_weight * 0.1
            
        # Total loss with adaptive KLD weight
        total_loss = recon_loss + kld_weight_to_use * kld_loss
        
        # Safety check for NaN or Inf values - IMPROVED CRITICAL FIX
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN or Inf detected in loss calculation")
            # Creating a more stable fallback loss
            # Use a small constant loss instead of trying to maintain gradient connection
            # This helps training continue without propagating bad gradients
            total_loss = torch.tensor(0.01, device=total_loss.device, requires_grad=True)
        
        # Return loss components for monitoring
        loss_components = {
            'recon_loss': recon_loss.detach() if not torch.isnan(recon_loss) and not torch.isinf(recon_loss) else torch.tensor(0.0, device=recon_loss.device),
            'mse_loss': mse_loss.detach() if not torch.isnan(mse_loss) and not torch.isinf(mse_loss) else torch.tensor(0.0, device=mse_loss.device),
            'ssim_loss': ssim_loss.detach() if not torch.isnan(ssim_loss) and not torch.isinf(ssim_loss) else torch.tensor(0.0, device=ssim_loss.device), 
            'kld_loss': kld_loss.detach() if not torch.isnan(kld_loss) and not torch.isinf(kld_loss) else torch.tensor(0.0, device=kld_loss.device),
            'total_loss': total_loss.detach() if not torch.isnan(total_loss) and not torch.isinf(total_loss) else torch.tensor(0.0, device=total_loss.device),
        }
        
        return total_loss, loss_components
    def update_epoch(self, current_epoch, total_epochs):
        """Update internal epoch counter for adaptive loss weights"""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Implement KL annealing - gradually increase KL weight
        if current_epoch < self.kld_warmup_epochs:
            # Linear warmup of KL weight
            self.current_kld_weight = self.kld_weight * (current_epoch / self.kld_warmup_epochs)
        else:
            self.current_kld_weight = self.kld_weight


class UnsupervisedDataset(Dataset):
    """Dataset for unsupervised CVAE training with data augmentation"""
    def __init__(self, ids, data_files, window_size=(256, 256), augment=True):
        self.ids = ids
        self.data_files = data_files
        self.window_size = window_size
        self.augment = augment
        
        # Load data
        self.data = []
        for id in self.ids:
            try:
                img = 1/255 * np.asarray(io.imread(self.data_files.format(id)), dtype='float32')
                self.data.append(img)
            except Exception as e:
                print(f"Error loading image {id}: {e}")
        
        # Create windows with overlap
        self.windows = []
        for img_idx, img in enumerate(self.data):
            height, width, _ = img.shape
            step_h = window_size[0] // 2  # 50% overlap
            step_w = window_size[1] // 2  # 50% overlap
            
            for i in range(0, height - window_size[0] + 1, step_h):
                for j in range(0, width - window_size[1] + 1, step_w):
                    self.windows.append((img_idx, i, j))
        
        print(f"Dataset created with {len(self.windows)} patches from {len(self.data)} images")
    
    def __len__(self):
        return len(self.windows)
    
    def augment_patch(self, patch):
        """Apply random augmentations to the patch"""
        # Convert back to HWC format for augmentation
        patch_np = patch.numpy().transpose(1, 2, 0)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            patch_np = np.flip(patch_np, axis=1).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            patch_np = np.flip(patch_np, axis=0).copy()
        
        # Random brightness/contrast adjustment
        if np.random.random() > 0.5:
            alpha = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
            beta = -0.1 + 0.2 * np.random.random()  # -0.1 to 0.1
            patch_np = np.clip(alpha * patch_np + beta, 0, 1)
        
        # Convert back to CHW format
        return torch.from_numpy(patch_np.transpose(2, 0, 1)).float()
    
    def __getitem__(self, idx):
        img_idx, i, j = self.windows[idx]
        img = self.data[img_idx]
        
        # Extract window
        patch = img[i:i+self.window_size[0], j:j+self.window_size[1], :]
        
        # Convert to torch format
        patch = np.transpose(patch, (2, 0, 1))  # HWC -> CHW
        patch = torch.from_numpy(patch).float()
        
        # Apply augmentations during training
        if self.augment:
            patch = self.augment_patch(patch)
        
        # Return patch and a dummy target (not used)
        return patch, torch.zeros(1)


def train_enhanced_cvae(net, criterion, optimizer, scheduler, train_loader, val_loader, epochs, save_epoch, output_path):
    """Train function specialized for the Enhanced CVAE"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize mixed precision training with updated API
    scaler = torch.amp.GradScaler('cuda')  # Fix for FutureWarning
    
    # Loss tracking
    train_losses = []
    val_losses = []
    component_losses = {
        'mse_loss': [],
        'ssim_loss': [],
        'perceptual_loss': [],
        'kld_loss': [],
        'contrastive_loss': [],
        'total_loss': []
    }
    
    # Gradient norm tracking for debugging
    grad_norms = []
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Update criterion's epoch counter for adaptive weights
        criterion.update_epoch(epoch, epochs)
        
        # Training mode
        net.train()
        running_loss = 0.0
        epoch_component_losses = {k: 0.0 for k in component_losses.keys()}
        
        # Progress bar
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for batch_idx, (data, _) in enumerate(progress):
            if torch.cuda.is_available():
                data = data.cuda()
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass with updated API
            with torch.amp.autocast('cuda'):  # Fix for FutureWarning
                # Add model to outputs for loss calculation
                outputs = net(data)
                outputs['model'] = net
                
                # Calculate loss
                loss, loss_components = criterion(outputs)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Calculate gradient norm for debugging
            total_norm = 0
            parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            # Enhanced gradient clipping with better handling
            scaler.unscale_(optimizer)
            
            # Check for NaN or Inf gradients and reset them
            for param in net.parameters():
                if param.grad is not None:
                    # Replace NaN/Inf gradients with zeros
                    param.grad.data.masked_fill_(torch.isnan(param.grad.data) | torch.isinf(param.grad.data), 0.0)
            
            # Stricter gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)  # Reduced from 1.0
            
            # Update with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update component losses
            for k, v in loss_components.items():
                if k in epoch_component_losses:
                    epoch_component_losses[k] += v.item() if torch.is_tensor(v) else v
            
            # Update progress bar
            progress.set_postfix({"loss": running_loss / (batch_idx + 1)})
        
        # Calculate average losses
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Average component losses
        for k in component_losses.keys():
            component_losses[k].append(epoch_component_losses[k] / len(train_loader))
        
        # Validation
        val_loss = validate_enhanced_cvae(net, criterion, val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), f"{output_path}/model_best.pth")
            print(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint
        if epoch % save_epoch == 0:
            torch.save(net.state_dict(), f"{output_path}/model_epoch{epoch}.pth")
            
        # Visualize a sample
        if epoch % save_epoch == 0 or epoch == epochs:
            visualize_reconstructions(net, val_loader, epoch, output_path)
        
        # Plot losses
        plot_training_curves(train_losses, val_losses, component_losses, grad_norms, output_path)
        
        # Print epoch summary
        print(f"Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save final model
    torch.save(net.state_dict(), f"{output_path}/model_final.pth")
    print("Training completed!")


def validate_enhanced_cvae(net, criterion, val_loader):
    """Validation function for Enhanced CVAE"""
    net.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data, _ in val_loader:
            if torch.cuda.is_available():
                data = data.cuda()
            
            outputs = net(data)
            outputs['model'] = net
            
            loss, _ = criterion(outputs)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def visualize_reconstructions(net, val_loader, epoch, output_path):
    """Visualize reconstructions from the Enhanced CVAE"""
    net.eval()
    
    with torch.no_grad():
        # Get a sample batch
        data, _ = next(iter(val_loader))
        if torch.cuda.is_available():
            data = data.cuda()
        
        # Get reconstructions
        outputs = net(data)
        recons = outputs['reconstruction']
        
        # Select up to 8 samples
        n_samples = min(8, data.size(0))
        
        # Create figure
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
        
        for i in range(n_samples):
            # Original
            orig = data[i].cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(np.clip(orig, 0, 1))
            axes[0, i].set_title("Original" if i == 0 else "")
            axes[0, i].axis('off')
            
            # Reconstruction
            recon = recons[i].cpu().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(np.clip(recon, 0, 1))
            axes[1, i].set_title("Reconstruction" if i == 0 else "")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/reconstructions_epoch{epoch}.png")
        plt.close()


def plot_training_curves(train_losses, val_losses, component_losses, grad_norms, output_path):
    """Plot training curves for the Enhanced CVAE"""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall losses
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Component losses plot 1
    axes[0, 1].plot(component_losses['mse_loss'], label='MSE')
    axes[0, 1].plot(component_losses['ssim_loss'], label='SSIM')
    axes[0, 1].plot(component_losses['perceptual_loss'], label='Perceptual')
    axes[0, 1].set_title('Reconstruction Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Component losses plot 2
    axes[1, 0].plot(component_losses['kld_loss'], label='KLD')
    axes[1, 0].plot(component_losses['contrastive_loss'], label='Contrastive')
    axes[1, 0].set_title('Regularization Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Gradient norms
    if grad_norms:
        axes[1, 1].plot(grad_norms)
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Norm')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/training_curves.png")
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Enhanced CVAE with improved losses')
    parser.add_argument('-i', '--input', help='Path of input directory', 
                        default="./input/")
    parser.add_argument('-o', '--output', help='Path of output directory',
                        default="./output/Enhanced-CVAE/")
    parser.add_argument('-w', '--window', nargs=2, type=int, default=[256, 256],
                        help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-lr', '--base_lr', default=0.001, type=float, help='Base learning rate')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('-se', '--save_epoch', default=5, type=int, help='Save model every N epochs')
    parser.add_argument('-ld', '--latent_dim', default=256, type=int, help='Latent dimension size')
    parser.add_argument('-kw', '--kld_weight', default=0.001, type=float, help='KL divergence weight')
    parser.add_argument('-cw', '--contrastive_weight', default=0.5, type=float, help='Contrastive loss weight')
    parser.add_argument('-sw', '--ssim_weight', default=0.3, type=float, help='SSIM loss weight')
    parser.add_argument('-pw', '--perceptual_weight', default=0.1, type=float, help='Perceptual loss weight')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Parameters
    WINDOW_SIZE = tuple(args.window)
    IN_CHANNELS = 3
    LATENT_DIM = args.latent_dim
    FOLDER = args.input
    OUTPUT_FOLDER = args.output
    batch_size = args.batch_size
    epochs = args.epochs
    save_epoch = args.save_epoch
    
    # Data paths
    DATA_FOLDER = f"{FOLDER}/top/top_mosaic_09cm_area{{}}.tif"
    
    # Create CVAE model
    cvae = EnhancedCVAE(input_channels=IN_CHANNELS, latent_dim=LATENT_DIM)
    
    # Initialize loss function
    criterion = EnhancedCVAELoss(
        kld_weight=args.kld_weight,
        contrastive_weight=args.contrastive_weight,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight
    )
    
    # Set up optimizer with improved stability parameters
    optimizer = torch.optim.AdamW(
        cvae.parameters(), 
        lr=args.base_lr,
        weight_decay=0.0001,
        eps=1e-5,  # Increase epsilon for more stability
        amsgrad=True  # Use AMSGrad variant for better stability
    )
    
    # Use GPU if available
    if torch.cuda.is_available():
        cvae.cuda()
        criterion.cuda()
    
    # Define train and validation data
    all_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37', '5', '15', '21', '30']
    
    # Split data into train and validation (80/20 split)
    train_val_split = int(len(all_ids) * 0.8)
    train_ids = all_ids[:train_val_split]
    val_ids = all_ids[train_val_split:]
    
    print(f"Training IDs: {train_ids}")
    print(f"Validation IDs: {val_ids}")
    
    # Create datasets with data augmentation
    train_set = UnsupervisedDataset(train_ids, data_files=DATA_FOLDER, window_size=WINDOW_SIZE, augment=True)
    val_set = UnsupervisedDataset(val_ids, data_files=DATA_FOLDER, window_size=WINDOW_SIZE, augment=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    # Set up OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    
    # Use more conservative learning rate scheduling
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.base_lr * 0.5,  # Lower max learning rate for stability
        total_steps=total_steps,
        pct_start=0.4,  # Longer warm-up period (40% of training)
        anneal_strategy='cos',
        div_factor=20.0,  # Less aggressive initial LR reduction
        final_div_factor=500.0,  # Less aggressive final LR reduction
        cycle_momentum=False  # Disable momentum cycling for stability
    )
    
    # Train the model
    train_enhanced_cvae(cvae, criterion, optimizer, scheduler, train_loader, val_loader, epochs, save_epoch, OUTPUT_FOLDER)
    
    print("Enhanced CVAE training completed!")


if __name__ == "__main__":
    main()