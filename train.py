#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training script with multi-scale features and improved architecture.
Key improvements:
1. Multi-scale CVAE feature extraction
2. Enhanced segmentation model with spatial reasoning
3. Multi-scale loss function with focal loss for class imbalance
4. Better training monitoring and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from skimage import io
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import cv2

# Import the enhanced components
from net.cvae import EnhancedCVAE
from net.segmentation_model import MultiScaleSegmentationModel
from net.loss import MultiScaleLoss, SimpleCrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

# Import augmentation functions
from utils.utils_dataset import elastic_transform, get_augmentation_transforms, cutmix_augmentation

# Configure PyTorch for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with enhanced augmentation"""
    def __init__(self, ids, image_files, label_files, window_size=(256, 256), augment=True, 
                 stride=None, enable_cutmix=True, cutmix_prob=0.3):
        self.ids = ids
        self.image_files = image_files
        self.label_files = label_files
        self.window_size = window_size
        self.augment = augment
        self.stride = stride if stride is not None else window_size[0] // 2
        self.enable_cutmix = enable_cutmix
        self.cutmix_prob = cutmix_prob
        
        # Load data
        self.images = []
        self.labels = []
        for id in self.ids:
            try:
                # Load image
                img_path = self.image_files.format(id)
                img = np.asarray(io.imread(img_path), dtype='float32') / 255.0
                self.images.append(img)
                
                # Load label
                label_path = self.label_files.format(id)
                lbl = np.asarray(io.imread(label_path), dtype='int64')
                
                # Check if label has 3 channels (RGB), convert to single channel
                if len(lbl.shape) == 3 and lbl.shape[2] == 3:
                    lbl_mapped = np.zeros((lbl.shape[0], lbl.shape[1]), dtype='int64')
                    
                    # ISPRS color mapping
                    lbl_mapped[(lbl[:,:,0] > 200) & (lbl[:,:,1] > 200) & (lbl[:,:,2] > 200)] = 0  # Impervious
                    lbl_mapped[(lbl[:,:,0] < 50) & (lbl[:,:,1] < 50) & (lbl[:,:,2] > 200)] = 1   # Building  
                    lbl_mapped[(lbl[:,:,0] < 50) & (lbl[:,:,1] > 200) & (lbl[:,:,2] > 200)] = 2  # Low veg
                    lbl_mapped[(lbl[:,:,0] < 50) & (lbl[:,:,1] > 200) & (lbl[:,:,2] < 50)] = 3   # Tree
                    lbl_mapped[(lbl[:,:,0] > 200) & (lbl[:,:,1] > 200) & (lbl[:,:,2] < 50)] = 4  # Car
                    lbl_mapped[(lbl[:,:,0] > 200) & (lbl[:,:,1] < 50) & (lbl[:,:,2] < 50)] = 5   # Clutter
                    
                    lbl = lbl_mapped
                
                self.labels.append(lbl)
                print(f"Loaded image and label for ID {id}, shape: {img.shape}, {lbl.shape}")
                
            except Exception as e:
                print(f"Error loading image/label {id}: {e}")
        
        # Create windows with overlap
        self.windows = []
        for img_idx, (img, lbl) in enumerate(zip(self.images, self.labels)):
            height, width = img.shape[:2]
            
            for i in range(0, height - window_size[0] + 1, self.stride):
                for j in range(0, width - window_size[1] + 1, self.stride):
                    self.windows.append((img_idx, i, j))
        
        print(f"Dataset created with {len(self.windows)} patches from {len(self.images)} images")
    
    def augment_data(self, image, label):
        """Apply enhanced random augmentations"""
        if torch.is_tensor(image):
            image_np = image.cpu().numpy().transpose(1, 2, 0)
        else:
            image_np = image
            
        if torch.is_tensor(label):
            label_np = label.cpu().numpy()
        else:
            label_np = label
        
        # Basic augmentations
        if np.random.random() > 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            label_np = np.flip(label_np, axis=1).copy()
        
        if np.random.random() > 0.5:
            image_np = np.flip(image_np, axis=0).copy()
            label_np = np.flip(label_np, axis=0).copy()
        
        if np.random.random() > 0.5:
            alpha = 0.8 + 0.4 * np.random.random()
            beta = -0.1 + 0.2 * np.random.random()
            image_np = np.clip(alpha * image_np + beta, 0, 1)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        label_tensor = torch.from_numpy(label_np).long()
        
        # Advanced augmentations
        if np.random.random() > 0.7:
            image_tensor, label_tensor = elastic_transform(
                image_tensor, label_tensor, 
                alpha=50 + np.random.random() * 50,
                sigma=4 + np.random.random() * 2
            )
        
        if np.random.random() > 0.6:
            color_transform = get_augmentation_transforms(p=0.8)
            image_tensor = color_transform(image_tensor)
        
        return image_tensor, label_tensor
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        img_idx, i, j = self.windows[idx]
        image = self.images[img_idx]
        label = self.labels[img_idx]
        
        # Extract window
        image_patch = image[i:i+self.window_size[0], j:j+self.window_size[1], :]
        label_patch = label[i:i+self.window_size[0], j:j+self.window_size[1]]
        
        # Convert to torch format
        image_patch = np.transpose(image_patch, (2, 0, 1))
        image_patch = torch.from_numpy(image_patch).float()
        label_patch = torch.from_numpy(label_patch).long()
        
        # Apply augmentations
        if self.augment:
            image_patch, label_patch = self.augment_data(image_patch, label_patch)
            
            # CutMix augmentation
            if self.enable_cutmix and len(self.images) > 1 and random.random() < self.cutmix_prob:
                second_idx = random.randint(0, len(self.windows) - 1)
                if second_idx != idx:
                    second_img_idx, second_i, second_j = self.windows[second_idx]
                    second_image = self.images[second_img_idx]
                    second_label = self.labels[second_img_idx]
                    
                    second_image_patch = second_image[second_i:second_i+self.window_size[0], 
                                                   second_j:second_j+self.window_size[1], :]
                    second_label_patch = second_label[second_i:second_i+self.window_size[0], 
                                                   second_j:second_j+self.window_size[1]]
                    
                    second_image_patch = np.transpose(second_image_patch, (2, 0, 1))
                    second_image_patch = torch.from_numpy(second_image_patch).float()
                    second_label_patch = torch.from_numpy(second_label_patch).long()
                    
                    second_image_patch, second_label_patch = self.augment_data(second_image_patch, second_label_patch)
                    
                    try:
                        image_patch, label_patch = cutmix_augmentation(
                            image_patch, label_patch, 
                            second_image_patch, second_label_patch, 
                            alpha=0.5
                        )
                    except ValueError:
                        pass
        
        return image_patch, label_patch


class EnhancedSegmentationTrainer:
    """Enhanced training class with multi-scale features and improved architecture"""
    def __init__(self, cvae_path, n_classes=6, learning_rate=0.001, device="cuda", use_multi_scale_loss=True):
        self.device = device
        self.n_classes = n_classes
        self.use_multi_scale_loss = use_multi_scale_loss
        
        # Load pre-trained CVAE
        self.cvae = self._load_cvae(cvae_path)
        
        # Use enhanced segmentation model
        self.model = MultiScaleSegmentationModel(
            n_classes=n_classes,
            device=device
        ).to(device)
        
        # Enhanced optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0005,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Choose loss function
        if use_multi_scale_loss:
            self.criterion = MultiScaleLoss(n_classes=n_classes, device=device)
        else:
            # Fallback to simple loss for compatibility
            self.criterion = SimpleCrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': [],
            'val_mean_iou': [], 'val_f1': [], 'train_loss_components': []
        }
    
    def _load_cvae(self, model_path):
        """Load pre-trained CVAE model"""
        cvae = EnhancedCVAE(input_channels=3, latent_dim=256, hidden_dims=[64, 128, 256])
        
        try:
            cvae.load_state_dict(torch.load(
                model_path, 
                map_location=torch.device(self.device),
                weights_only=True
            ), strict=False)
            print("CVAE model loaded successfully")
        except Exception as e:
            print(f"Warning when loading CVAE weights: {e}")
            print("Proceeding with newly initialized model - performance may be affected")
        
        cvae = cvae.to(self.device)
        cvae.eval()
        return cvae
    
    def extract_cvae_features(self, images):
        """Extract multi-scale features from the pre-trained CVAE"""
        with torch.no_grad():
            try:
                # Get outputs from CVAE
                outputs = self.cvae(images)
                
                # Extract ALL encoder features, not just the deepest
                z = outputs['z']  # latent code (batch_size, 256)
                encoder_features = outputs['encoder_features']  # [e1, e2, e3]
                
                # Multi-scale features with preserved spatial information
                feat_l1 = encoder_features[0]  # [B, 64, 128, 128]
                feat_l2 = encoder_features[1]  # [B, 128, 64, 64]
                feat_l3 = encoder_features[2]  # [B, 256, 32, 32]
                
                # Global context from latent vector
                z_global = z.unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
                
                # Create feature pyramid with proper channel alignment
                target_channels = 256
                
                # Project each level to target channels using learnable projections
                if not hasattr(self, 'level_projections'):
                    self.level_projections = nn.ModuleDict({
                        'proj_l1': nn.Sequential(
                            nn.Conv2d(64, target_channels//4, 1),
                            nn.BatchNorm2d(target_channels//4),
                            nn.ReLU()
                        ),
                        'proj_l2': nn.Sequential(
                            nn.Conv2d(128, target_channels//2, 1), 
                            nn.BatchNorm2d(target_channels//2),
                            nn.ReLU()
                        ),
                        'proj_l3': nn.Sequential(
                            nn.Conv2d(256, target_channels, 1),
                            nn.BatchNorm2d(target_channels), 
                            nn.ReLU()
                        ),
                        'proj_global': nn.Sequential(
                            nn.Conv2d(256, target_channels, 1),
                            nn.BatchNorm2d(target_channels),
                            nn.ReLU()
                        )
                    }).to(self.device)
                
                # Project features to aligned channels
                feat_l1_proj = self.level_projections['proj_l1'](feat_l1)  # [B, 64, 128, 128]
                feat_l2_proj = self.level_projections['proj_l2'](feat_l2)  # [B, 128, 64, 64]  
                feat_l3_proj = self.level_projections['proj_l3'](feat_l3)  # [B, 256, 32, 32]
                
                # Expand global context to match l3 spatial size
                z_spatial = F.interpolate(z_global, size=(32, 32), mode='bilinear', align_corners=False)
                z_proj = self.level_projections['proj_global'](z_spatial)  # [B, 256, 32, 32]
                
                # Create Feature Pyramid Network
                # Start from the most semantic level (l3 + global)
                p3 = feat_l3_proj + z_proj  # [B, 256, 32, 32]
                
                # Propagate semantic info to level 2
                p3_up = F.interpolate(p3, size=(64, 64), mode='bilinear', align_corners=False)
                p2 = feat_l2_proj + p3_up  # [B, 128, 64, 64]
                
                # Propagate semantic info to level 1  
                p2_up = F.interpolate(p2, size=(128, 128), mode='bilinear', align_corners=False)
                p1 = feat_l1_proj + p2_up  # [B, 64, 128, 128]
                
                # Return multi-scale features for the segmentation model
                return {
                    'p1': p1,  # High resolution, low semantics
                    'p2': p2,  # Medium resolution, medium semantics  
                    'p3': p3,  # Low resolution, high semantics
                    'global_context': z_proj  # Global context
                }
                
            except Exception as e:
                print(f"Error in multi-scale feature extraction: {e}")
                # Fallback to single-scale random features
                return {
                    'p1': torch.randn(images.size(0), 64, 128, 128, device=self.device) * 0.1,
                    'p2': torch.randn(images.size(0), 128, 64, 64, device=self.device) * 0.1,
                    'p3': torch.randn(images.size(0), 256, 32, 32, device=self.device) * 0.1,
                    'global_context': torch.randn(images.size(0), 256, 32, 32, device=self.device) * 0.1
                }
    
    def train_step(self, images, labels):
        """Enhanced training step with multi-scale supervision"""
        # Extract multi-scale features
        multi_scale_features = self.extract_cvae_features(images)
        
        # Forward pass through enhanced model
        outputs = self.model(multi_scale_features)
        
        # Multi-scale loss computation
        if self.use_multi_scale_loss:
            loss, loss_components = self.criterion(
                outputs, labels, original_size=(images.size(2), images.size(3))
            )
        else:
            loss, loss_components = self.criterion(outputs, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item(), loss_components, outputs
    
    def compute_metrics(self, predictions, targets):
        """Compute detailed metrics including per-class performance"""
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Remove ignored pixels
        valid_idx = target_np != 255
        pred_np = pred_np[valid_idx] 
        target_np = target_np[valid_idx]
        
        # Overall metrics
        acc = accuracy_score(target_np, pred_np)
        mean_iou = jaccard_score(target_np, pred_np, average='macro', labels=range(self.n_classes), zero_division=0)
        f1 = f1_score(target_np, pred_np, average='macro', labels=range(self.n_classes), zero_division=0)
        
        # Per-class metrics for detailed analysis
        per_class_iou = jaccard_score(target_np, pred_np, average=None, labels=range(self.n_classes), zero_division=0)
        per_class_f1 = f1_score(target_np, pred_np, average=None, labels=range(self.n_classes), zero_division=0)
        
        return {
            'accuracy': acc,
            'mean_iou': mean_iou,
            'f1_score': f1,
            'per_class_iou': per_class_iou,
            'per_class_f1': per_class_f1
        }
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        """Enhanced training loop with improved monitoring"""
        os.makedirs(save_dir, exist_ok=True)
        best_iou = 0.0
        
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            epoch_loss_components = {}
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            
            for batch_idx, (images, labels) in enumerate(train_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Enhanced training step
                loss, loss_components, outputs = self.train_step(images, labels)
                
                # Accumulate loss components
                for key, value in loss_components.items():
                    if key not in epoch_loss_components:
                        epoch_loss_components[key] = 0.0
                    epoch_loss_components[key] += value
                
                epoch_loss += loss
                train_bar.set_postfix({
                    "loss": epoch_loss / (batch_idx + 1),
                    "lr": self.optimizer.param_groups[0]['lr']
                })
            
            # Update learning rate
            self.scheduler.step()
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Save metrics
            avg_train_loss = epoch_loss / len(train_loader)
            self.metrics['train_loss'].append(avg_train_loss)
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['val_accuracy'].append(val_metrics['accuracy'])
            self.metrics['val_mean_iou'].append(val_metrics['mean_iou'])
            self.metrics['val_f1'].append(val_metrics['f1_score'])
            self.metrics['train_loss_components'].append(epoch_loss_components)
            
            # Enhanced logging
            print(f"Epoch {epoch}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Mean IoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Val F1 Score: {val_metrics['f1_score']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # Print per-class F1 scores for debugging
            if 'per_class_f1' in val_metrics:
                class_names = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
                print("  Per-class F1 scores:")
                for i, (name, f1_score) in enumerate(zip(class_names, val_metrics['per_class_f1'])):
                    print(f"    {name}: {f1_score:.3f}")
            
            # Print loss components for debugging
            if self.use_multi_scale_loss and epoch_loss_components:
                print("  Loss Components:")
                for key, value in epoch_loss_components.items():
                    if key != 'total_loss':
                        print(f"    {key}: {value/len(train_loader):.6f}")
            
            # Save best model
            if val_metrics['mean_iou'] > best_iou:
                best_iou = val_metrics['mean_iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_iou': best_iou,
                    'metrics': self.metrics
                }, f"{save_dir}/model_best.pth")
                print(f"  New best model saved with Mean IoU: {best_iou:.4f}")
            
            # Enhanced visualizations every 5 epochs
            if epoch % 5 == 0:
                self.visualize_results(images[:4], labels[:4], outputs, epoch, save_dir)
            
            # Plot training curves
            self.plot_training_curves(save_dir)
    
    def validate(self, val_loader):
        """Enhanced validation with detailed metrics"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Multi-scale features and prediction
                multi_scale_features = self.extract_cvae_features(images)
                outputs = self.model(multi_scale_features)
                
                # Loss computation
                if self.use_multi_scale_loss:
                    loss, _ = self.criterion(outputs, labels, (images.size(2), images.size(3)))
                else:
                    loss, _ = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get final predictions
                if isinstance(outputs, dict) and 'final_segmentation' in outputs:
                    final_seg = outputs['final_segmentation']
                else:
                    final_seg = outputs
                    
                if final_seg.size(2) != labels.size(1):
                    final_seg = F.interpolate(final_seg, size=labels.shape[1:], mode='bilinear', align_corners=False)
                
                preds = torch.argmax(final_seg, dim=1)
                all_preds.append(preds)
                all_targets.append(labels)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute enhanced metrics
        metrics = self.compute_metrics(all_preds, all_targets)
        metrics['loss'] = val_loss / len(val_loader)
        
        return metrics
    
    def visualize_results(self, images, labels, outputs, epoch, save_dir):
        """Enhanced visualization showing multi-scale predictions"""
        colors = np.array([
            [255, 255, 255],  # Impervious surfaces (white)
            [0, 0, 255],      # Building (blue)
            [0, 255, 255],    # Low vegetation (cyan)
            [0, 255, 0],      # Tree (green)
            [255, 255, 0],    # Car (yellow)
            [255, 0, 0]       # Clutter (red)
        ])
        
        n_samples = min(4, images.size(0))
        
        # Check if we have multi-scale predictions
        if isinstance(outputs, dict) and 'multi_scale_predictions' in outputs:
            # Multi-scale visualization
            fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
            
            for i in range(n_samples):
                # Original image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                axes[i, 0].imshow(np.clip(img, 0, 1))
                axes[i, 0].set_title("Input" if i == 0 else "")
                axes[i, 0].axis("off")
                
                # Ground truth
                gt = labels[i].cpu().numpy()
                gt_colored = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                for c in range(self.n_classes):
                    gt_colored[gt == c] = colors[c]
                axes[i, 1].imshow(gt_colored)
                axes[i, 1].set_title("Ground Truth" if i == 0 else "")
                axes[i, 1].axis("off")
                
                # Multi-scale predictions
                multi_scale_preds = outputs['multi_scale_predictions']
                for j, (pred, title) in enumerate(zip(multi_scale_preds, ["Coarse", "Medium", "Fine"])):
                    pred_resized = F.interpolate(pred[i:i+1], size=gt.shape, mode='bilinear', align_corners=False)
                    pred_class = torch.argmax(pred_resized, dim=1)[0].cpu().numpy()
                    
                    pred_colored = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
                    for c in range(self.n_classes):
                        pred_colored[pred_class == c] = colors[c]
                    
                    axes[i, j + 2].imshow(pred_colored)
                    axes[i, j + 2].set_title(title if i == 0 else "")
                    axes[i, j + 2].axis("off")
        else:
            # Simple visualization for non-multi-scale outputs
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
            
            for i in range(n_samples):
                # Original image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                if n_samples == 1:
                    axes[0].imshow(np.clip(img, 0, 1))
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")
                else:
                    axes[i, 0].imshow(np.clip(img, 0, 1))
                    axes[i, 0].set_title("Input Image" if i == 0 else "")
                    axes[i, 0].axis("off")
                
                # Ground truth
                gt = labels[i].cpu().numpy()
                gt_colored = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                for c in range(self.n_classes):
                    gt_colored[gt == c] = colors[c]
                
                if n_samples == 1:
                    axes[1].imshow(gt_colored)
                    axes[1].set_title("Ground Truth")
                    axes[1].axis("off")
                else:
                    axes[i, 1].imshow(gt_colored)
                    axes[i, 1].set_title("Ground Truth" if i == 0 else "")
                    axes[i, 1].axis("off")
                
                # Prediction
                if isinstance(outputs, dict) and 'final_segmentation' in outputs:
                    pred_tensor = outputs['final_segmentation'][i]
                else:
                    pred_tensor = outputs[i]
                
                if pred_tensor.dim() == 3:
                    pred_class = torch.argmax(pred_tensor, dim=0).cpu().numpy()
                else:
                    pred_class = pred_tensor.cpu().numpy()
                
                pred_colored = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
                for c in range(self.n_classes):
                    pred_colored[pred_class == c] = colors[c]
                
                if n_samples == 1:
                    axes[2].imshow(pred_colored)
                    axes[2].set_title("Prediction")
                    axes[2].axis("off")
                else:
                    axes[i, 2].imshow(pred_colored)
                    axes[i, 2].set_title("Prediction" if i == 0 else "")
                    axes[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/results_epoch{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, save_dir):
        """Enhanced training curve visualization"""
        if len(self.metrics['train_loss']) == 0:
            return
            
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.metrics['val_accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean IoU
        axes[1, 0].plot(epochs, self.metrics['val_mean_iou'], 'm-', linewidth=2)
        axes[1, 0].set_title('Validation Mean IoU', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mean IoU')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 1].plot(epochs, self.metrics['val_f1'], 'c-', linewidth=2)
        axes[1, 1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced training with multi-scale features')
    parser.add_argument('-i', '--input', help='Path of input directory', 
                      default="./input/")
    parser.add_argument('-o', '--output', help='Path of output directory',
                      default="./output/EnhancedMRF/")
    parser.add_argument('-c', '--cvae', help='Path to pre-trained CVAE model',
                      default="./output/Enhanced-CVAE/model_best.pth")
    parser.add_argument('-w', '--window', nargs=2, type=int, default=[256, 256],
                      help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('-nc', '--n_classes', default=6, type=int, help='Number of classes')
    parser.add_argument('-m', '--mode', choices=['train', 'validate'], default='train',
                      help='Mode: train or validate')
    parser.add_argument('-cp', '--checkpoint', help='Path to model checkpoint for validation',
                      default=None)
    parser.add_argument('-lp', '--labeled_percentage', default=100, type=int, 
                      help='Percentage of labeled data to use (10, 30, 75, 100)')
    parser.add_argument('-s', '--seed', default=42, type=int,
                      help='Random seed for reproducibility')
    parser.add_argument('--simple_loss', action='store_true',
                      help='Use simple cross entropy loss instead of multi-scale loss')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    if args.labeled_percentage < 100:
        OUTPUT_FOLDER = f"{args.output}_{args.labeled_percentage}pct_labeled"
    else:
        OUTPUT_FOLDER = args.output
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Parameters
    WINDOW_SIZE = tuple(args.window)
    FOLDER = args.input
    CVAE_PATH = args.cvae
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    N_CLASSES = args.n_classes
    
    # Data paths
    IMAGE_FILES = f"{FOLDER}/top/top_mosaic_09cm_area{{}}.tif"
    LABEL_FILES = f"{FOLDER}/gt/top_mosaic_09cm_area{{}}.tif"
    
    # Define train and test IDs
    all_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37', '5', '15', '21', '30']
    
    # Split data
    train_val_split = int(len(all_ids) * 0.8)
    train_ids_full = all_ids[:train_val_split]
    val_ids = all_ids[train_val_split:]
    
    # Handle labeled percentage
    labeled_percentage = args.labeled_percentage
    if labeled_percentage < 100:
        num_labeled = max(1, int(len(train_ids_full) * labeled_percentage / 100))
        rng = np.random.RandomState(args.seed)
        shuffled_indices = rng.permutation(len(train_ids_full))
        labeled_indices = shuffled_indices[:num_labeled]
        train_ids = [train_ids_full[i] for i in labeled_indices]
        print(f"Using {len(train_ids)} images ({labeled_percentage}%) as labeled data: {train_ids}")
    else:
        train_ids = train_ids_full
        print(f"Using all {len(train_ids)} images (100%) as labeled data")
    
    print(f"Validation IDs: {val_ids}")
    
    # Create datasets
    if args.mode == 'train':
        train_set = SegmentationDataset(
            train_ids, IMAGE_FILES, LABEL_FILES, WINDOW_SIZE, 
            augment=True, enable_cutmix=True, cutmix_prob=0.3
        )
    
    val_set = SegmentationDataset(
        val_ids, IMAGE_FILES, LABEL_FILES, WINDOW_SIZE, 
        augment=False, enable_cutmix=False
    )
    
    # Create data loaders
    if args.mode == 'train':
        train_loader = DataLoader(
            train_set, BATCH_SIZE, shuffle=True, 
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
    
    val_loader = DataLoader(
        val_set, BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    
    # Create trainer
    trainer = EnhancedSegmentationTrainer(
        cvae_path=CVAE_PATH,
        n_classes=N_CLASSES,
        learning_rate=LEARNING_RATE,
        device=device,
        use_multi_scale_loss=not args.simple_loss
    )
    
    if args.mode == 'train':
        # Train model
        trainer.train(train_loader, val_loader, EPOCHS, OUTPUT_FOLDER)
        
        # Save experiment metadata
        with open(f"{OUTPUT_FOLDER}/experiment_info.txt", "w") as f:
            f.write(f"Labeled percentage: {labeled_percentage}%\n")
            f.write(f"Labeled images: {train_ids}\n")
            f.write(f"Validation images: {val_ids}\n")
            f.write(f"Total labeled patches: {len(train_set)}\n")
            f.write(f"Total validation patches: {len(val_set)}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Learning rate: {LEARNING_RATE}\n")
            f.write(f"Epochs: {EPOCHS}\n")
            f.write(f"Multi-scale loss: {not args.simple_loss}\n")
            f.write(f"Random seed: {args.seed}\n")
    else:  # Validate mode
        # Load model
        checkpoint_path = args.checkpoint or f"{OUTPUT_FOLDER}/model_best.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run validation
        metrics = trainer.validate(val_loader)
        
        print("Final Validation Results:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Mean IoU: {metrics['mean_iou']*100:.2f}%")
        print(f"  F1 Score: {metrics['f1_score']*100:.2f}%")
        
        # Save results to file
        with open(f"{OUTPUT_FOLDER}/final_results.txt", "w") as f:
            f.write(f"Labeled percentage: {labeled_percentage}%\n")
            f.write(f"Accuracy -> {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Mean IoU -> {metrics['mean_iou']*100:.2f}%\n")
            f.write(f"F1 Score -> {metrics['f1_score']*100:.2f}%\n")
            
            if 'per_class_f1' in metrics:
                class_names = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
                f.write("\nPer-class F1 scores:\n")
                for name, score in zip(class_names, metrics['per_class_f1']):
                    f.write(f"  {name}: {score:.3f}\n")


if __name__ == "__main__":
    main()