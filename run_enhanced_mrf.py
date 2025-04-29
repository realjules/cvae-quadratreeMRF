#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced implementation of the SimplifiedMRF model with improved feature extraction
and advanced architectural components for better performance.

This builds upon the SimplifiedMRF approach but adds:
1. Enhanced feature extraction from the CVAE
2. Attention mechanisms
3. Boundary refinement
4. Multi-scale processing
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

from net.fixed_enhanced_cvae import FixedEnhancedCVAE as EnhancedCVAE
from net.enhanced_feature_extractor import EnhancedFeatureExtractor
from net.enhanced_mrf import EnhancedMRF
from torch.utils.data import Dataset, DataLoader
# Import advanced augmentation functions
from utils.utils_dataset import elastic_transform, get_augmentation_transforms, cutmix_augmentation

# Configure PyTorch for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)


# Reuse the SegmentationDataset from the SimplifiedMRF implementation
class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with ground truth labels and enhanced augmentation"""
    def __init__(self, ids, image_files, label_files, window_size=(256, 256), augment=True, 
                 stride=None, enable_cutmix=True, cutmix_prob=0.3):
        self.ids = ids
        self.image_files = image_files
        self.label_files = label_files
        self.window_size = window_size
        self.augment = augment
        self.stride = stride if stride is not None else window_size[0] // 2  # Default to 50% overlap
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
                    # Map colors to class indices (adjust according to your dataset)
                    lbl_mapped = np.zeros((lbl.shape[0], lbl.shape[1]), dtype='int64')
                    
                    # Common color mapping for ISPRS dataset - adjust these for your dataset
                    # Impervious surfaces (RGB: 255, 255, 255) -> 0
                    lbl_mapped[(lbl[:,:,0] > 200) & (lbl[:,:,1] > 200) & (lbl[:,:,2] > 200)] = 0
                    # Building (RGB: 0, 0, 255) -> 1
                    lbl_mapped[(lbl[:,:,0] < 50) & (lbl[:,:,1] < 50) & (lbl[:,:,2] > 200)] = 1
                    # Low vegetation (RGB: 0, 255, 255) -> 2
                    lbl_mapped[(lbl[:,:,0] < 50) & (lbl[:,:,1] > 200) & (lbl[:,:,2] > 200)] = 2
                    # Tree (RGB: 0, 255, 0) -> 3
                    lbl_mapped[(lbl[:,:,0] < 50) & (lbl[:,:,1] > 200) & (lbl[:,:,2] < 50)] = 3
                    # Car (RGB: 255, 255, 0) -> 4
                    lbl_mapped[(lbl[:,:,0] > 200) & (lbl[:,:,1] > 200) & (lbl[:,:,2] < 50)] = 4
                    # Clutter/background (RGB: 255, 0, 0) -> 5
                    lbl_mapped[(lbl[:,:,0] > 200) & (lbl[:,:,1] < 50) & (lbl[:,:,2] < 50)] = 5
                    
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
        
        print(f"Dataset created with {len(self.windows)} patches from {len(self.images)} images with enhanced augmentation")
    
    def augment_data(self, image, label):
        """Apply enhanced random augmentations to the image and label"""
        # Convert to HWC format for augmentation if image is a tensor
        if torch.is_tensor(image):
            image_np = image.cpu().numpy().transpose(1, 2, 0)
        else:
            image_np = image
            
        # Convert label to numpy if it's a tensor
        if torch.is_tensor(label):
            label_np = label.cpu().numpy()
        else:
            label_np = label
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image_np = np.flip(image_np, axis=1).copy()
            label_np = np.flip(label_np, axis=1).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image_np = np.flip(image_np, axis=0).copy()
            label_np = np.flip(label_np, axis=0).copy()
        
        # Random brightness/contrast adjustment (image only)
        if np.random.random() > 0.5:
            alpha = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
            beta = -0.1 + 0.2 * np.random.random()  # -0.1 to 0.1
            image_np = np.clip(alpha * image_np + beta, 0, 1)
        
        # Convert to tensors for intermediate processing
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        label_tensor = torch.from_numpy(label_np).long()
        
        # ENHANCED AUGMENTATIONS:
        
        # 1. Apply elastic transform with 30% probability
        if np.random.random() > 0.7:
            image_tensor, label_tensor = elastic_transform(
                image_tensor, label_tensor, 
                alpha=50 + np.random.random() * 50,  # 50-100 range for alpha
                sigma=4 + np.random.random() * 2     # 4-6 range for sigma
            )
        
        # 2. Apply color augmentations with 40% probability
        if np.random.random() > 0.6:
            # Get color transform
            color_transform = get_augmentation_transforms(p=0.8)
            # Apply only to the image (not the label)
            image_tensor = color_transform(image_tensor)
        
        # 3. Random rotation with 20% probability
        if np.random.random() > 0.8:
            angle = np.random.randint(-30, 30)  # -30 to +30 degrees
            
            # Convert tensors back to numpy for rotation
            if torch.is_tensor(image_tensor):
                rot_img = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                rot_img = image_tensor.transpose(1, 2, 0)
                
            if torch.is_tensor(label_tensor):
                rot_label = label_tensor.cpu().numpy()
            else:
                rot_label = label_tensor
            
            # Apply rotation
            center = (rot_img.shape[1] // 2, rot_img.shape[0] // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rot_img = cv2.warpAffine(rot_img, rot_matrix, (rot_img.shape[1], rot_img.shape[0]), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            rot_label = cv2.warpAffine(rot_label, rot_matrix, (rot_label.shape[1], rot_label.shape[0]), 
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            
            # Convert back to tensors
            image_tensor = torch.from_numpy(rot_img.transpose(2, 0, 1)).float()
            label_tensor = torch.from_numpy(rot_label).long()
        
        # Return the final augmented tensors
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
        image_patch = np.transpose(image_patch, (2, 0, 1))  # HWC -> CHW
        image_patch = torch.from_numpy(image_patch).float()
        label_patch = torch.from_numpy(label_patch).long()
        
        # Apply standard augmentations during training
        if self.augment:
            image_patch, label_patch = self.augment_data(image_patch, label_patch)
            
            # Apply CutMix augmentation if enabled and we have more than one image
            # This is especially important for low-data regimes
            if self.enable_cutmix and len(self.images) > 1 and random.random() < self.cutmix_prob:
                # Get a random second sample from the dataset
                second_idx = random.randint(0, len(self.windows) - 1)
                if second_idx != idx:  # Make sure it's different from current
                    second_img_idx, second_i, second_j = self.windows[second_idx]
                    second_image = self.images[second_img_idx]
                    second_label = self.labels[second_img_idx]
                    
                    # Extract window
                    second_image_patch = second_image[second_i:second_i+self.window_size[0], 
                                               second_j:second_j+self.window_size[1], :]
                    second_label_patch = second_label[second_i:second_i+self.window_size[0], 
                                               second_j:second_j+self.window_size[1]]
                    
                    # Convert to torch format
                    second_image_patch = np.transpose(second_image_patch, (2, 0, 1))
                    second_image_patch = torch.from_numpy(second_image_patch).float()
                    second_label_patch = torch.from_numpy(second_label_patch).long()
                    
                    # Apply standard augmentation to second patch
                    second_image_patch, second_label_patch = self.augment_data(second_image_patch, second_label_patch)
                    
                    # Apply CutMix between the two patches
                    try:
                        image_patch, label_patch = cutmix_augmentation(
                            image_patch, label_patch, 
                            second_image_patch, second_label_patch, 
                            alpha=0.5
                        )
                    except ValueError:
                        # Fall back to original patches if CutMix fails
                        pass  # Keep original patches if CutMix fails
        
        return image_patch, label_patch


class EnhancedSegmentationTrainer:
    """Training class for the EnhancedMRF model with improved CVAE feature extraction"""
    def __init__(self, cvae_path, n_classes=6, feature_dim=256, learning_rate=0.001, device="cuda"):
        """
        Initialize the trainer with a pre-trained CVAE for feature extraction
        """
        self.device = device
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        
        # Load pre-trained CVAE
        self.cvae = self._load_cvae(cvae_path, feature_dim)
        
        # Create enhanced feature extractor
        self.feature_extractor = EnhancedFeatureExtractor(
            latent_dim=feature_dim,
            output_dim=feature_dim
        ).to(device)
        
        # Create the enhanced MRF model
        self.model = EnhancedMRF(
            n_classes=n_classes,
            feature_dim=feature_dim,
            device=device
        ).to(device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.feature_extractor.parameters()),
            lr=learning_rate,
            weight_decay=0.0001
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Setup loss function with class weighting for imbalanced data
        self.class_weights = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        
        # Create metrics tracker
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_mean_iou': [],
            'val_f1': []
        }
    
    def _load_cvae(self, model_path, feature_dim):
        """Load pre-trained CVAE model"""
        # Use the Enhanced CVAE with appropriate dimensions
        cvae = EnhancedCVAE(
            input_channels=3, 
            latent_dim=feature_dim,
            hidden_dims=[64, 128, 256, 512]
        )
        
        # Load weights
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
        
        # Move to device and set to evaluation mode
        cvae = cvae.to(self.device)
        cvae.eval()
        
        return cvae
    
    def extract_enhanced_features(self, images):
        """Extract enhanced features using both CVAE and our feature extractor"""
        with torch.no_grad():
            try:
                # Process through CVAE
                outputs = self.cvae(images)
                
                # Use our enhanced feature extractor
                features = self.feature_extractor.extract_from_cvae_output(outputs)
                
                return features
                
            except RuntimeError as e:
                print(f"Warning in feature extraction: {e}")
                # Create random features if extraction fails
                return torch.randn(
                    images.size(0), 
                    self.feature_dim, 
                    images.size(2) // 4,  # 1/4 of input size 
                    images.size(3) // 4,  # 1/4 of input size
                    device=self.device
                ) * 0.1
    
    def update_class_weights(self, labels):
        """Update class weights based on label distribution"""
        # Count class frequencies
        counts = torch.zeros(self.n_classes, device=self.device)
        for c in range(self.n_classes):
            counts[c] = (labels == c).sum().float()
        
        # Add small constant to avoid division by zero
        counts = counts + 1.0
        
        # Compute inverse frequency weights
        weights = 1.0 / counts
        
        # Normalize weights
        weights = weights / weights.sum() * self.n_classes
        
        # Update class weights
        self.class_weights = weights
        
        # Update criterion
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=255,
            reduction='mean'
        )
    
    def compute_metrics(self, predictions, targets):
        """Compute segmentation metrics"""
        # Convert to numpy arrays
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Remove ignored pixels (255)
        valid_idx = target_np != 255
        pred_np = pred_np[valid_idx]
        target_np = target_np[valid_idx]
        
        # Compute metrics
        acc = accuracy_score(target_np, pred_np)
        iou = jaccard_score(target_np, pred_np, average='macro', labels=range(self.n_classes), zero_division=0)
        f1 = f1_score(target_np, pred_np, average='macro', labels=range(self.n_classes), zero_division=0)
        
        return {
            'accuracy': acc,
            'mean_iou': iou,
            'f1_score': f1
        }
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        """Train the EnhancedMRF model"""
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Track best validation performance
        best_iou = 0.0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Clear GPU cache before each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Training phase
            self.model.train()
            self.feature_extractor.train()
            train_loss = 0.0
            
            # Progress bar
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            
            for batch_idx, (images, labels) in enumerate(train_bar):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Update class weights on first batch
                if epoch == 1 and batch_idx == 0 and self.class_weights is None:
                    self.update_class_weights(labels)
                
                # Extract features from CVAE with reduced memory footprint
                with torch.no_grad():
                    # Process in smaller batches if needed
                    batch_size = images.size(0)
                    if batch_size > 4:  # Use smaller batches for feature extraction
                        cvae_features = []
                        for i in range(0, batch_size, 4):
                            mini_batch = images[i:i+4]
                            mini_features = self.extract_enhanced_features(mini_batch)
                            cvae_features.append(mini_features)
                        cvae_features = torch.cat(cvae_features, dim=0)
                    else:
                        cvae_features = self.extract_enhanced_features(images)
                
                # Forward pass through the MRF model
                logits = self.model(cvae_features)
                
                # Resize logits to match label size if needed
                if logits.shape[2:] != labels.shape[1:]:
                    logits = F.interpolate(
                        logits,
                        size=(labels.shape[1], labels.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                train_bar.set_postfix({"loss": train_loss / (batch_idx + 1)})
            
            # Compute average training loss
            avg_train_loss = train_loss / len(train_loader)
            self.metrics['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            self.feature_extractor.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            # Progress bar
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_bar):
                    # Move to device
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Extract features from CVAE with reduced memory footprint
                    # Process in smaller batches if needed
                    batch_size = images.size(0)
                    if batch_size > 4:  # Use smaller batches for feature extraction
                        cvae_features = []
                        for i in range(0, batch_size, 4):
                            mini_batch = images[i:i+4]
                            mini_features = self.extract_enhanced_features(mini_batch)
                            cvae_features.append(mini_features)
                        cvae_features = torch.cat(cvae_features, dim=0)
                    else:
                        cvae_features = self.extract_enhanced_features(images)
                    
                    # Forward pass
                    logits = self.model(cvae_features)
                    
                    # Resize logits to match label size if needed
                    if logits.shape[2:] != labels.shape[1:]:
                        logits = F.interpolate(
                            logits,
                            size=(labels.shape[1], labels.shape[2]),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Compute loss
                    loss = self.criterion(logits, labels)
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=1)
                    
                    # Update statistics
                    val_loss += loss.item()
                    val_bar.set_postfix({"loss": val_loss / (batch_idx + 1)})
                    
                    # Store predictions and targets for metric computation
                    all_preds.append(preds)
                    all_targets.append(labels)
            
            # Compute average validation loss
            avg_val_loss = val_loss / len(val_loader)
            self.metrics['val_loss'].append(avg_val_loss)
            
            # Update learning rate based on validation loss
            self.scheduler.step(avg_val_loss)
            
            # Concatenate predictions and targets
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Compute validation metrics
            metrics = self.compute_metrics(all_preds, all_targets)
            self.metrics['val_accuracy'].append(metrics['accuracy'])
            self.metrics['val_mean_iou'].append(metrics['mean_iou'])
            self.metrics['val_f1'].append(metrics['f1_score'])
            
            # Visualize some segmentation results
            if epoch % 5 == 0 or epoch == epochs:
                self.visualize_segmentations(images[:8], labels[:8], all_preds[:8], epoch, save_dir)
            
            # Print epoch summary
            print(f"Epoch {epoch}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Val Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Val F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if metrics['mean_iou'] > best_iou:
                best_iou = metrics['mean_iou']
                # Save both models
                torch.save(self.model.state_dict(), f"{save_dir}/model_best.pth")
                torch.save(self.feature_extractor.state_dict(), f"{save_dir}/feature_extractor_best.pth")
                print(f"  New best model saved with Mean IoU: {best_iou:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': self.metrics
                }, f"{save_dir}/checkpoint_epoch{epoch}.pth")
            
            # Plot and save training curves
            self.plot_training_curves(save_dir)
        
        print("Training completed!")
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Loss curves
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curve
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.metrics['val_accuracy'], 'g-')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Mean IoU curve
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.metrics['val_mean_iou'], 'm-')
        plt.title('Validation Mean IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.grid(True)
        
        # F1 Score curve
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.metrics['val_f1'], 'c-')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curves.png")
        plt.close()
    
    def visualize_segmentations(self, images, labels, predictions, epoch, save_dir):
        """Visualize segmentation results"""
        # Define color map for segmentation visualization
        colors = [
            [255, 255, 255],  # Impervious surfaces (white)
            [0, 0, 255],      # Building (blue)
            [0, 255, 255],    # Low vegetation (cyan)
            [0, 255, 0],      # Tree (green)
            [255, 255, 0],    # Car (yellow)
            [255, 0, 0]       # Clutter (red)
        ]
        colors = np.array(colors)
        
        # Number of samples to visualize
        n_samples = min(8, images.size(0))
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        
        for i in range(n_samples):
            # Get data for this sample
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            lbl = labels[i].cpu().numpy()
            pred = predictions[i].cpu().numpy()
            
            # Create colored segmentation maps
            lbl_colored = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
            pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            
            for c in range(self.n_classes):
                lbl_colored[lbl == c] = colors[c]
                pred_colored[pred == c] = colors[c]
            
            # Display images
            if n_samples == 1:
                axes[0].imshow(np.clip(img, 0, 1))
                axes[0].set_title("Input Image")
                axes[0].axis("off")
                
                axes[1].imshow(lbl_colored)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                
                axes[2].imshow(pred_colored)
                axes[2].set_title("Prediction")
                axes[2].axis("off")
            else:
                axes[i, 0].imshow(np.clip(img, 0, 1))
                axes[i, 0].set_title("Input Image" if i == 0 else "")
                axes[i, 0].axis("off")
                
                axes[i, 1].imshow(lbl_colored)
                axes[i, 1].set_title("Ground Truth" if i == 0 else "")
                axes[i, 1].axis("off")
                
                axes[i, 2].imshow(pred_colored)
                axes[i, 2].set_title("Prediction" if i == 0 else "")
                axes[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/segmentation_epoch{epoch}.png")
        plt.close()
    
    def validate(self, val_loader, output_dir):
        """Validate the model on the validation set"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set models to evaluation mode
        self.model.eval()
        self.feature_extractor.eval()
        
        # Validation variables
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Progress bar
        val_bar = tqdm(val_loader, desc=f"Validating")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_bar):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features from CVAE
                cvae_features = self.extract_enhanced_features(images)
                
                # Forward pass
                logits = self.model(cvae_features)
                
                # Resize logits to match label size if needed
                if logits.shape[2:] != labels.shape[1:]:
                    logits = F.interpolate(
                        logits,
                        size=(labels.shape[1], labels.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Update statistics
                val_loss += loss.item()
                val_bar.set_postfix({"loss": val_loss / (batch_idx + 1)})
                
                # Store predictions and targets for metric computation
                all_preds.append(preds)
                all_targets.append(labels)
                
                # Visualize first few batches
                if batch_idx < 4:
                    self.visualize_segmentations(
                        images, labels, preds, f"val_batch{batch_idx}", output_dir
                    )
        
        # Compute average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Concatenate predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute validation metrics
        metrics = self.compute_metrics(all_preds, all_targets)
        
        # Print results
        print("Validation Results:")
        print(f"  Loss: {avg_val_loss:.6f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Save results to file
        with open(f"{output_dir}/validation_metrics.txt", "w") as f:
            f.write(f"Loss: {avg_val_loss:.6f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        
        return metrics


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and evaluate EnhancedMRF with pre-trained CVAE')
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
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-ld', '--latent_dim', default=512, type=int, help='CVAE latent dimension')
    parser.add_argument('-nc', '--n_classes', default=6, type=int, help='Number of classes')
    parser.add_argument('-m', '--mode', choices=['train', 'validate'], default='train',
                      help='Mode: train or validate')
    parser.add_argument('-cp', '--checkpoint', help='Path to model checkpoint for validation',
                      default=None)
    parser.add_argument('-lp', '--labeled_percentage', default=100, type=int, 
                      help='Percentage of labeled data to use (10, 30, 75, 100)')
    parser.add_argument('-s', '--seed', default=42, type=int,
                      help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory with percentage in name for experiments
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
    LATENT_DIM = args.latent_dim
    N_CLASSES = args.n_classes
    
    # Data paths
    IMAGE_FILES = f"{FOLDER}/top/top_mosaic_09cm_area{{}}.tif"
    LABEL_FILES = f"{FOLDER}/gt/top_mosaic_09cm_area{{}}.tif"
    
    # Define train and test IDs
    all_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37', '5', '15', '21', '30']
    
    # Split data (same as for CVAE)
    train_val_split = int(len(all_ids) * 0.8)
    train_ids_full = all_ids[:train_val_split]
    val_ids = all_ids[train_val_split:]
    
    # For semi-supervised experiments, use only a subset of labeled data
    labeled_percentage = args.labeled_percentage
    if labeled_percentage < 100:
        # Use only a subset of training data as labeled
        num_labeled = max(1, int(len(train_ids_full) * labeled_percentage / 100))
        # Instead of sequential selection, we can shuffle for better class representation
        # But first set the seed for reproducibility
        rng = np.random.RandomState(args.seed)
        shuffled_indices = rng.permutation(len(train_ids_full))
        labeled_indices = shuffled_indices[:num_labeled]
        
        # Get the actual labeled IDs
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
            augment=True, 
            enable_cutmix=True,
            cutmix_prob=0.3  # 30% chance of applying CutMix augmentation
        )
    
    val_set = SegmentationDataset(
        val_ids, IMAGE_FILES, LABEL_FILES, WINDOW_SIZE, 
        augment=False,
        enable_cutmix=False
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
        feature_dim=LATENT_DIM,  # This should match the CVAE latent_dim
        learning_rate=LEARNING_RATE,
        device=device
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
            f.write(f"Random seed: {args.seed}\n")
    else:  # Validate mode
        # Load model from checkpoint
        checkpoint_path = args.checkpoint or f"{OUTPUT_FOLDER}/checkpoint_latest.pth"
        
        try:
            # Try to load the complete checkpoint with both models
            checkpoint = torch.load(checkpoint_path, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
            print(f"Loaded complete checkpoint from {checkpoint_path}")
        except:
            # If that fails, try to load individual model files
            try:
                trainer.model.load_state_dict(torch.load(
                    f"{OUTPUT_FOLDER}/model_best.pth",
                    map_location=device, 
                    weights_only=True
                ))
                trainer.feature_extractor.load_state_dict(torch.load(
                    f"{OUTPUT_FOLDER}/feature_extractor_best.pth",
                    map_location=device,
                    weights_only=True
                ))
                print(f"Loaded individual model files from {OUTPUT_FOLDER}")
            except Exception as e:
                print(f"Warning when loading models: {e}")
                print("Proceeding with newly initialized models - results will be meaningless")
        
        # Run validation
        validate_output_dir = f"{OUTPUT_FOLDER}/validation"
        metrics = trainer.validate(val_loader, validate_output_dir)
        
        # Save results to file for easier experiment comparison
        with open(f"{OUTPUT_FOLDER}/result.txt", "w") as f:
            f.write(f"Labeled percentage: {labeled_percentage}%\n")
            f.write(f"Accuracy -> {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Mean IoU -> {metrics['mean_iou']*100:.2f}%\n")
            f.write(f"F1 Score -> {metrics['f1_score']*100:.2f}%\n")


if __name__ == "__main__":
    main()
