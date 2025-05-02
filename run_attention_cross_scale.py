#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train the integrated model combining AttentionFusionCVAE with CrossScaleMRF
This combines the attention fusion mechanism for decoder skip connections with
cross-scale attention for MRF feature integration.
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.amp
import torch.nn.functional as F

from net.cross_scale_mrf import CrossScaleMRF
from net.attention_fusion_cvae import AttentionFusionCVAE
from net.loss import EnhancedHierarchicalPGMLoss
from dataset.dataset import AerialImageDataset
from utils.utils import colorize_segmentation

class IntegratedSegmentationTrainer:
    """
    Trainer class for integrated segmentation model.
    This class handles:
    - Training with AttentionFusionCVAE + CrossScaleMRF
    - Loss calculation and optimization
    - Metrics tracking
    - Checkpointing and visualization
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else "cpu")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"output/attention_cross_scale_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models and datasets
        self._init_dataset()
        self._init_models()
        self._init_optimizer()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': []
        }
        
        self.best_val_iou = 0.0
        self.start_epoch = 0
        
        # Load checkpoint if provided
        if args.resume and os.path.exists(args.resume):
            self._load_checkpoint(args.resume)
    
    def _init_dataset(self):
        """Initialize training and validation datasets"""
        # Training dataset
        self.train_dataset = AerialImageDataset(
            os.path.join(self.args.data_path, 'train'),
            supervised=True,
            size=self.args.image_size,
            transform_prob=0.5
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Validation dataset
        self.val_dataset = AerialImageDataset(
            os.path.join(self.args.data_path, 'val'),
            supervised=True,
            size=self.args.image_size,
            transform_prob=0.0  # No augmentation for validation
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"Datasets initialized - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def _init_models(self):
        """Initialize CVAE and MRF models"""
        # AttentionFusionCVAE model (fixed, pre-trained encoder)
        self.cvae = AttentionFusionCVAE(
            input_channels=3,
            latent_dim=self.args.latent_dim,
            hidden_dims=[64, 128, 256, 512]
        ).to(self.device)
        
        # Load pre-trained CVAE weights
        if self.args.cvae_weights and os.path.exists(self.args.cvae_weights):
            checkpoint = torch.load(self.args.cvae_weights, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.cvae.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.cvae.load_state_dict(checkpoint)
            print(f"Loaded pre-trained AttentionFusionCVAE weights from {self.args.cvae_weights}")
        else:
            print("Warning: No pre-trained CVAE weights provided or file not found.")
        
        # Set CVAE to evaluation mode (fixed encoder)
        self.cvae.eval()
        for param in self.cvae.parameters():
            param.requires_grad = False
        
        # CrossScaleMRF model
        self.mrf = CrossScaleMRF(
            n_classes=self.train_dataset.num_classes,
            feature_dim=self.args.latent_dim,
            device=self.device
        ).to(self.device)
        
        # Loss function
        class_weights = torch.ones(self.train_dataset.num_classes).to(self.device)
        if self.args.class_weights:
            # Calculate inverse frequency class weights
            label_counts = self.train_dataset.get_class_distribution()
            for i, count in enumerate(label_counts):
                if count > 0:
                    class_weights[i] = 1.0 / (count / sum(label_counts))
            
            # Normalize weights to have mean of 1
            class_weights = class_weights / class_weights.mean()
            print(f"Class weights: {class_weights}")
        
        self.criterion = EnhancedHierarchicalPGMLoss(
            n_classes=self.train_dataset.num_classes,
            weights=class_weights,
            kld_weight=0.001,
            contrastive_weight=0.5,
            consistency_weight=0.2
        )
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.mrf.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Cosine annealing LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.learning_rate,
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.args.epochs,
            pct_start=0.1,  # Warmup for 10% of training
            div_factor=25,  # Initial LR is max_lr/25
            final_div_factor=1000  # Final LR is max_lr/1000
        )
    
    def _extract_cvae_features(self, images):
        """Extract features from AttentionFusionCVAE encoder"""
        with torch.no_grad():
            # Add positional encoding
            pos_enc = self.cvae.positional_encoding
            # Resize positional encoding if needed
            if images.size(2) != pos_enc.size(1) or images.size(3) != pos_enc.size(2):
                pos_enc = F.interpolate(
                    pos_enc.unsqueeze(0), 
                    size=(images.size(2), images.size(3)), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Add positional encoding to input (use only first 3 channels)
            pos_enc_reduced = pos_enc[:3]
            images_with_pos = torch.cat([images, pos_enc_reduced.expand(images.size(0), -1, -1, -1)], dim=1)
            
            # Encode with CVAE
            mu, log_var, encoder_features = self.cvae.encode(images_with_pos)
            
            # Return latent features and encoder features
            return mu, encoder_features
    
    def _save_checkpoint(self, epoch, val_iou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.mrf.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_iou': val_iou,
            'metrics': self.metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best one so far
        if is_best:
            best_model_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with IoU: {val_iou:.4f}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.mrf.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_iou = checkpoint['val_iou']
        
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val IoU: {self.best_val_iou:.4f}")
    
    def _compute_iou(self, preds, targets, n_classes=6, ignore_index=6):
        """Compute mean IoU across classes"""
        # Convert predictions to class indices
        preds = torch.argmax(preds, dim=1)
        
        # Initialize IoU for each class
        iou_per_class = []
        
        # Compute IoU for each class
        for cls in range(n_classes):
            # Create binary masks
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            # Skip ignored index
            if cls == ignore_index:
                continue
            
            # Compute intersection and union
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            # Avoid division by zero
            if union > 0:
                iou = intersection / union
                iou_per_class.append(iou.item())
        
        # Return mean IoU if there are valid classes
        if len(iou_per_class) > 0:
            return np.mean(iou_per_class)
        else:
            return 0.0
    
    def train_epoch(self, epoch):
        """Train model for one epoch"""
        self.mrf.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        
        # Update criterion epoch
        self.criterion.update_epoch(epoch, self.args.epochs)
        
        # Progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Extract features from CVAE
            features, _ = self._extract_cvae_features(images)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.args.mixed_precision:
                with torch.cuda.amp.autocast():
                    # MRF forward pass
                    outputs = self.mrf(features)
                    
                    # Calculate loss
                    loss, loss_components = self.criterion(outputs, targets, mode='supervised')
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                
                # Clip gradients
                if self.args.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.mrf.parameters(), self.args.gradient_clip)
                
                # Step optimizer and scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # MRF forward pass
                outputs = self.mrf(features)
                
                # Calculate loss
                loss, loss_components = self.criterion(outputs, targets, mode='supervised')
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                if self.args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.mrf.parameters(), self.args.gradient_clip)
                
                # Step optimizer
                self.optimizer.step()
            
            # Step scheduler
            self.scheduler.step()
            
            # Calculate IoU
            batch_iou = self._compute_iou(
                outputs['final_segmentation'],
                targets,
                n_classes=self.train_dataset.num_classes
            )
            
            # Update epoch metrics
            epoch_loss += loss.item()
            epoch_iou += batch_iou
            
            # Update progress bar
            pbar.set_postfix({
                'loss': epoch_loss / (batch_idx + 1),
                'iou': epoch_iou / (batch_idx + 1),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Generate visualizations periodically
            if self.args.vis_interval > 0 and batch_idx % self.args.vis_interval == 0:
                self._visualize_predictions(
                    images, targets, outputs,
                    os.path.join(self.output_dir, f"train_vis_epoch{epoch+1}_batch{batch_idx}.png")
                )
        
        # Calculate average epoch metrics
        avg_loss = epoch_loss / len(self.train_dataloader)
        avg_iou = epoch_iou / len(self.train_dataloader)
        
        # Update metrics
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['train_iou'].append(avg_iou)
        self.metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        return avg_loss, avg_iou
    
    def validate(self, epoch):
        """Validate model on validation set"""
        self.mrf.eval()
        epoch_loss = 0.0
        epoch_iou = 0.0
        
        with torch.no_grad():
            # Progress bar
            pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Extract features from CVAE
                features, _ = self._extract_cvae_features(images)
                
                # MRF forward pass
                outputs = self.mrf(features)
                
                # Calculate loss
                loss, loss_components = self.criterion(outputs, targets, mode='supervised')
                
                # Calculate IoU
                batch_iou = self._compute_iou(
                    outputs['final_segmentation'],
                    targets,
                    n_classes=self.train_dataset.num_classes
                )
                
                # Update epoch metrics
                epoch_loss += loss.item()
                epoch_iou += batch_iou
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': epoch_loss / (batch_idx + 1),
                    'iou': epoch_iou / (batch_idx + 1)
                })
                
                # Generate visualizations periodically
                if batch_idx == 0:
                    self._visualize_predictions(
                        images, targets, outputs,
                        os.path.join(self.output_dir, f"val_vis_epoch{epoch+1}.png")
                    )
        
        # Calculate average epoch metrics
        avg_loss = epoch_loss / len(self.val_dataloader)
        avg_iou = epoch_iou / len(self.val_dataloader)
        
        # Update metrics
        self.metrics['val_loss'].append(avg_loss)
        self.metrics['val_iou'].append(avg_iou)
        
        return avg_loss, avg_iou
    
    def _visualize_predictions(self, images, targets, outputs, save_path):
        """Generate visualization of predictions"""
        with torch.no_grad():
            # Get predictions
            preds = torch.argmax(outputs['final_segmentation'], dim=1)
            
            # Take only first few images
            n_samples = min(4, images.shape[0])
            images = images[:n_samples].cpu()
            targets = targets[:n_samples].cpu()
            preds = preds[:n_samples].cpu()
            
            # Create visualization grid
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
            
            # Ensure axes is 2D even for n_samples=1
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_samples):
                # Original image
                img = images[i].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title("Image")
                axes[i, 0].axis('off')
                
                # Ground truth segmentation
                target_vis = colorize_segmentation(targets[i].numpy(), self.train_dataset.num_classes)
                axes[i, 1].imshow(target_vis)
                axes[i, 1].set_title("Ground Truth")
                axes[i, 1].axis('off')
                
                # Predicted segmentation
                pred_vis = colorize_segmentation(preds[i].numpy(), self.train_dataset.num_classes)
                axes[i, 2].imshow(pred_vis)
                axes[i, 2].set_title("Prediction")
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            plt.close()
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        axes[0].plot(self.metrics['train_loss'], label='Train')
        axes[0].plot(self.metrics['val_loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot IoU
        axes[1].plot(self.metrics['train_iou'], label='Train')
        axes[1].plot(self.metrics['val_iou'], label='Validation')
        axes[1].set_title('Mean IoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=200)
        
        plt.close()
    
    def train(self):
        """Train model for specified number of epochs"""
        print(f"Starting training for {self.args.epochs} epochs")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train for one epoch
            train_loss, train_iou = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_iou = self.validate(epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Check if this is the best model so far
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self._save_checkpoint(epoch, val_iou, is_best)
            
            # Plot metrics
            if (epoch + 1) % self.args.plot_interval == 0:
                self.plot_metrics(os.path.join(self.output_dir, f"metrics_epoch{epoch+1}.png"))
        
        # Save final model
        self._save_checkpoint(self.args.epochs - 1, val_iou)
        
        # Plot final metrics
        self.plot_metrics(os.path.join(self.output_dir, "metrics_final.png"))
        
        print(f"Training completed. Best val IoU: {self.best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Integrated AttentionFusionCVAE + CrossScaleMRF Model")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='input/dataset', help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of the latent space')
    parser.add_argument('--cvae_weights', type=str, default=None, help='Path to pre-trained AttentionFusionCVAE weights')
    parser.add_argument('--class_weights', action='store_true', help='Use class weights for loss calculation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Misc parameters
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for dataloader')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--save_interval', type=int, default=10, help='Epochs between saving model checkpoints')
    parser.add_argument('--plot_interval', type=int, default=10, help='Epochs between plotting metrics')
    parser.add_argument('--vis_interval', type=int, default=100, help='Batches between generating visualizations (0 to disable)')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = IntegratedSegmentationTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()