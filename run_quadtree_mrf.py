#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train and validate the optimized QuadtreeMRF using a pre-trained Enhanced CVAE
as a feature extractor for aerial image segmentation.
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage import io
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import cv2

from net.enhanced_cvae import EnhancedCVAE
from net.optimized_quadtree_mrf import OptimizedQuadtreeMRF
from torch.utils.data import Dataset, DataLoader

# Configure PyTorch for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with ground truth labels"""
    def __init__(self, ids, image_files, label_files, window_size=(256, 256), augment=True, stride=None):
        self.ids = ids
        self.image_files = image_files
        self.label_files = label_files
        self.window_size = window_size
        self.augment = augment
        self.stride = stride if stride is not None else window_size[0] // 2  # Default to 50% overlap
        
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
        
        print(f"Dataset created with {len(self.windows)} patches from {len(self.images)} images")
    
    def __len__(self):
        return len(self.windows)
    
    def augment_data(self, image, label):
        """Apply random augmentations to the image and label"""
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
        
        # Convert back to tensors
        if torch.is_tensor(image):
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            label = torch.from_numpy(label_np).long()
        else:
            image = image_np
            label = label_np
            
        return image, label
    
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
        
        # Apply augmentations during training
        if self.augment:
            image_patch, label_patch = self.augment_data(image_patch, label_patch)
        
        return image_patch, label_patch


class QuadtreeMRFTrainer:
    """Trainer class for the QuadtreeMRF model with pre-trained CVAE features"""
    def __init__(self, cvae_path, n_classes=6, feature_dim=256, quadtree_depth=3, 
                 learning_rate=0.001, device="cuda"):
        self.device = device
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        
        # Load pre-trained CVAE
        self.cvae = self._load_cvae(cvae_path, feature_dim)
        
        # Create QuadtreeMRF model
        self.quadtree_mrf = OptimizedQuadtreeMRF(
            n_classes=n_classes,
            quadtree_depth=quadtree_depth,
            feature_dim=feature_dim,
            device=device
        ).to(device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.quadtree_mrf.parameters(),
            lr=learning_rate,
            weight_decay=0.0001,
            eps=1e-5
        )
        
        # Setup loss function with class weighting for imbalanced data
        # Will compute weights based on first batch
        self.class_weights = None
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        
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
        cvae = EnhancedCVAE(input_channels=3, latent_dim=feature_dim)
        
        # Load weights
        cvae.load_state_dict(torch.load(
            model_path, 
            map_location=torch.device(self.device),
            weights_only=True
        ))
        
        # Move to device and set to evaluation mode
        cvae = cvae.to(self.device)
        cvae.eval()
        
        return cvae
    
    def extract_cvae_features(self, images):
        """Extract features from the pre-trained CVAE with memory optimization"""
        with torch.no_grad():
            try:
                # Get outputs from CVAE
                outputs = self.cvae(images)
                
                # Extract relevant features
                # 1. Get just the latent representation (most important)
                z = outputs['z']
                
                # If memory is a concern, we can return just the latent code
                # expanded to a spatial dimension for the QuadtreeMRF
                z_spatial = z.unsqueeze(-1).unsqueeze(-1)
                z_spatial = z_spatial.expand(-1, -1, 16, 16)  # Expand to modest spatial size
                
                # Simple projection to desired feature dimension
                if z_spatial.shape[1] != self.feature_dim:
                    # Use a more memory-efficient 1x1 convolution
                    if not hasattr(self, 'projection') or self.projection.in_channels != z_spatial.shape[1]:
                        self.projection = torch.nn.Conv2d(
                            z_spatial.shape[1], 
                            self.feature_dim, 
                            kernel_size=1
                        ).to(self.device)
                    
                    combined_features = self.projection(z_spatial)
                else:
                    combined_features = z_spatial
                
                return combined_features
                
            except RuntimeError as e:
                # Memory optimization: if we run out of memory, fall back to a simpler approach
                if "out of memory" in str(e):
                    print("Warning: Memory issue detected. Using simplified feature extraction.")
                    # Create a simple feature tensor from the latent code
                    z = outputs.get('z', None)
                    if z is None:
                        # Last resort: create random features of the right shape
                        return torch.randn(images.size(0), self.feature_dim, 16, 16, device=self.device) * 0.1
                    
                    # Just use the latent code expanded to spatial dimensions
                    z_spatial = z.unsqueeze(-1).unsqueeze(-1)
                    z_spatial = z_spatial.expand(-1, -1, 16, 16)
                    
                    # Simple projection if needed
                    if z_spatial.shape[1] != self.feature_dim:
                        # Memory-efficient linear projection
                        z_flat = z_spatial.permute(0, 2, 3, 1).reshape(-1, z_spatial.shape[1])
                        if not hasattr(self, 'linear_proj'):
                            self.linear_proj = torch.nn.Linear(
                                z_spatial.shape[1], 
                                self.feature_dim
                            ).to(self.device)
                        
                        projected = self.linear_proj(z_flat)
                        return projected.reshape(z_spatial.shape[0], 16, 16, self.feature_dim).permute(0, 3, 1, 2)
                    
                    return z_spatial
                else:
                    # Re-raise other errors
                    raise
    
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
        self.criterion = torch.nn.CrossEntropyLoss(
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
        """Train the QuadtreeMRF model"""
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
            self.quadtree_mrf.train()
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
                            mini_features = self.extract_cvae_features(mini_batch)
                            cvae_features.append(mini_features)
                        cvae_features = torch.cat(cvae_features, dim=0)
                    else:
                        cvae_features = self.extract_cvae_features(images)
                
                # Forward pass
                segmentation = self.quadtree_mrf(cvae_features, cvae_features)
                
                # Convert segmentation (Long tensor with class indices) to one-hot encoded tensor 
                # as expected by CrossEntropyLoss
                batch_size, height, width = segmentation.size()
                
                # Create empty logits tensor [B, C, H, W]
                logits = torch.zeros(batch_size, self.n_classes, height, width, device=self.device)
                
                # For each class, set a high value (10.0) where the segmentation equals that class
                for c in range(self.n_classes):
                    logits[:, c] = (segmentation == c).float() * 10.0
                
                # Now we have proper logits that can be used with CrossEntropyLoss
                loss = self.criterion(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.quadtree_mrf.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                train_bar.set_postfix({"loss": train_loss / (batch_idx + 1)})
            
            # Compute average training loss
            avg_train_loss = train_loss / len(train_loader)
            self.metrics['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.quadtree_mrf.eval()
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
                            mini_features = self.extract_cvae_features(mini_batch)
                            cvae_features.append(mini_features)
                        cvae_features = torch.cat(cvae_features, dim=0)
                    else:
                        cvae_features = self.extract_cvae_features(images)
                    
                    # Forward pass
                    segmentation = self.quadtree_mrf(cvae_features, cvae_features)
                    
                    # Convert segmentation (Long tensor with class indices) to one-hot encoded tensor 
                    # as expected by CrossEntropyLoss
                    batch_size, height, width = segmentation.size()
                    
                    # Create empty logits tensor [B, C, H, W]
                    logits = torch.zeros(batch_size, self.n_classes, height, width, device=self.device)
                    
                    # For each class, set a high value (10.0) where the segmentation equals that class
                    for c in range(self.n_classes):
                        logits[:, c] = (segmentation == c).float() * 10.0
                    
                    # Now we have proper logits that can be used with CrossEntropyLoss
                    loss = self.criterion(logits, labels)
                    
                    # Update statistics
                    val_loss += loss.item()
                    val_bar.set_postfix({"loss": val_loss / (batch_idx + 1)})
                    
                    # Store predictions and targets for metric computation
                    all_preds.append(segmentation)
                    all_targets.append(labels)
            
            # Compute average validation loss
            avg_val_loss = val_loss / len(val_loader)
            self.metrics['val_loss'].append(avg_val_loss)
            
            # Concatenate predictions and targets
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Compute validation metrics
            metrics = self.compute_metrics(all_preds, all_targets)
            self.metrics['val_accuracy'].append(metrics['accuracy'])
            self.metrics['val_mean_iou'].append(metrics['mean_iou'])
            self.metrics['val_f1'].append(metrics['f1_score'])
            
            # Print epoch summary
            print(f"Epoch {epoch}/{epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Val Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Val F1 Score: {metrics['f1_score']:.4f}")
            
            # Save best model
            if metrics['mean_iou'] > best_iou:
                best_iou = metrics['mean_iou']
                torch.save(self.quadtree_mrf.state_dict(), f"{save_dir}/model_best.pth")
                print(f"  New best model saved with Mean IoU: {best_iou:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.quadtree_mrf.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': self.metrics
                }, f"{save_dir}/checkpoint_epoch{epoch}.pth")
            
            # Plot and save training curves
            self.plot_training_curves(save_dir)
            
            # Visualize some segmentation results
            if epoch % 5 == 0 or epoch == epochs:
                self.visualize_segmentations(images, labels, all_preds[:8], epoch, save_dir)
        
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
        # Number of samples to visualize
        n_samples = min(8, images.size(0))
        
        # Define color map for segmentation visualization
        # Each class gets a unique color - adjust for your dataset
        colors = [
            [255, 255, 255],  # Impervious surfaces (white)
            [0, 0, 255],      # Building (blue)
            [0, 255, 255],    # Low vegetation (cyan)
            [0, 255, 0],      # Tree (green)
            [255, 255, 0],    # Car (yellow)
            [255, 0, 0]       # Clutter (red)
        ]
        colors = np.array(colors)
        
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
            
            # Plot original image
            if n_samples == 1:
                axes[0].imshow(np.clip(img, 0, 1))
                axes[0].set_title("Input Image")
                axes[0].axis("off")
                
                # Plot ground truth
                axes[1].imshow(lbl_colored)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                
                # Plot prediction
                axes[2].imshow(pred_colored)
                axes[2].set_title("Prediction")
                axes[2].axis("off")
            else:
                axes[i, 0].imshow(np.clip(img, 0, 1))
                axes[i, 0].set_title("Input Image" if i == 0 else "")
                axes[i, 0].axis("off")
                
                # Plot ground truth
                axes[i, 1].imshow(lbl_colored)
                axes[i, 1].set_title("Ground Truth" if i == 0 else "")
                axes[i, 1].axis("off")
                
                # Plot prediction
                axes[i, 2].imshow(pred_colored)
                axes[i, 2].set_title("Prediction" if i == 0 else "")
                axes[i, 2].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/segmentation_epoch{epoch}.png")
        plt.close()


def validate_quadtree_mrf(model, cvae, val_loader, n_classes, device, output_dir):
    """Validate the QuadtreeMRF model on the validation set"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set models to evaluation mode
    model.eval()
    cvae.eval()
    
    # Track metrics
    all_preds = []
    all_targets = []
    
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
    
    # Process all validation batches
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract features from CVAE
            outputs = cvae(images)
            z = outputs['z']
            encoder_features = outputs['encoder_features']
            deep_features = encoder_features[-1]
            
            # Reshape z to match spatial dimensions
            z_spatial = z.unsqueeze(-1).unsqueeze(-1)
            z_spatial = z_spatial.expand(-1, -1, 2, 2)
            z_spatial = torch.nn.functional.interpolate(
                z_spatial, 
                size=deep_features.shape[2:], 
                mode='bilinear',
                align_corners=False
            )
            
            # Combine features
            combined_features = torch.cat([z_spatial, deep_features], dim=1)
            
            # Forward pass
            segmentation = model(combined_features, combined_features)
            
            # Store predictions and targets
            all_preds.append(segmentation)
            all_targets.append(labels)
            
            # Visualize first few batches
            if batch_idx < 4:
                # Number of samples to visualize
                n_samples = min(4, images.size(0))
                
                # Create figure
                fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
                
                for i in range(n_samples):
                    # Get data for this sample
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    lbl = labels[i].cpu().numpy()
                    pred = segmentation[i].cpu().numpy()
                    
                    # Create colored segmentation maps
                    lbl_colored = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
                    pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    
                    for c in range(n_classes):
                        lbl_colored[lbl == c] = colors[c]
                        pred_colored[pred == c] = colors[c]
                    
                    # Plot original image
                    if n_samples == 1:
                        axes[0].imshow(np.clip(img, 0, 1))
                        axes[0].set_title("Input Image")
                        axes[0].axis("off")
                        
                        # Plot ground truth
                        axes[1].imshow(lbl_colored)
                        axes[1].set_title("Ground Truth")
                        axes[1].axis("off")
                        
                        # Plot prediction
                        axes[2].imshow(pred_colored)
                        axes[2].set_title("Prediction")
                        axes[2].axis("off")
                    else:
                        axes[i, 0].imshow(np.clip(img, 0, 1))
                        axes[i, 0].set_title("Input Image" if i == 0 else "")
                        axes[i, 0].axis("off")
                        
                        # Plot ground truth
                        axes[i, 1].imshow(lbl_colored)
                        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
                        axes[i, 1].axis("off")
                        
                        # Plot prediction
                        axes[i, 2].imshow(pred_colored)
                        axes[i, 2].set_title("Prediction" if i == 0 else "")
                        axes[i, 2].axis("off")
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/validation_batch{batch_idx}.png")
                plt.close()
    
    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_segmentation_metrics(all_preds, all_targets, n_classes)
    
    # Print and save metrics
    print("Validation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Save metrics to file
    with open(f"{output_dir}/validation_metrics.txt", "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
        
        # Per-class metrics
        f.write("Per-class IoU:\n")
        for c in range(n_classes):
            f.write(f"  Class {c}: {metrics['class_iou'][c]:.4f}\n")
        
        f.write("\nPer-class F1 Score:\n")
        for c in range(n_classes):
            f.write(f"  Class {c}: {metrics['class_f1'][c]:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))
    
    # Create visualization of per-class metrics
    plot_class_metrics(metrics, n_classes, output_dir)
    
    return metrics


def compute_segmentation_metrics(predictions, targets, n_classes):
    """Compute detailed segmentation metrics"""
    # Convert to numpy arrays
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()
    
    # Remove ignored pixels (255)
    valid_idx = target_np != 255
    pred_np = pred_np[valid_idx]
    target_np = target_np[valid_idx]
    
    # Compute overall metrics
    acc = accuracy_score(target_np, pred_np)
    iou = jaccard_score(target_np, pred_np, average='macro', labels=range(n_classes), zero_division=0)
    f1 = f1_score(target_np, pred_np, average='macro', labels=range(n_classes), zero_division=0)
    
    # Compute per-class metrics
    class_iou = jaccard_score(target_np, pred_np, average=None, labels=range(n_classes), zero_division=0)
    class_f1 = f1_score(target_np, pred_np, average=None, labels=range(n_classes), zero_division=0)
    
    # Compute confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(target_np, pred_np):
        conf_matrix[t, p] += 1
    
    return {
        'accuracy': acc,
        'mean_iou': iou,
        'f1_score': f1,
        'class_iou': class_iou,
        'class_f1': class_f1,
        'confusion_matrix': conf_matrix
    }


def plot_class_metrics(metrics, n_classes, output_dir):
    """Plot per-class metrics"""
    # Define class names (adjust for your dataset)
    class_names = [
        "Imp Surface",
        "Building",
        "Low Veg",
        "Tree",
        "Car",
        "Clutter"
    ]
    
    # Ensure we have names for all classes
    if len(class_names) < n_classes:
        class_names.extend([f"Class {i}" for i in range(len(class_names), n_classes)])
    
    # Create bar plot for IoU and F1 score
    plt.figure(figsize=(12, 6))
    
    # IoU plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_names, metrics['class_iou'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.title("Per-class IoU")
    plt.ylabel("IoU")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # F1 score plot
    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names, metrics['class_f1'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.title("Per-class F1 Score")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_metrics.png")
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add values in each cell
    thresh = metrics['confusion_matrix'].max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(metrics['confusion_matrix'][i, j], 'd'),
                    ha="center", va="center",
                    color="white" if metrics['confusion_matrix'][i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and evaluate QuadtreeMRF with pre-trained CVAE')
    parser.add_argument('-i', '--input', help='Path of input directory', 
                      default="./input/")
    parser.add_argument('-o', '--output', help='Path of output directory',
                      default="./output/QuadtreeMRF/")
    parser.add_argument('-c', '--cvae', help='Path to pre-trained CVAE model',
                      default="./output/Enhanced-CVAE/model_best.pth")
    parser.add_argument('-w', '--window', nargs=2, type=int, default=[256, 256],
                      help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('-qd', '--quadtree_depth', default=2, type=int, help='Quadtree depth')
    parser.add_argument('-ld', '--latent_dim', default=256, type=int, help='CVAE latent dimension')
    parser.add_argument('-nc', '--n_classes', default=6, type=int, help='Number of classes')
    parser.add_argument('-m', '--mode', choices=['train', 'validate'], default='train',
                      help='Mode: train or validate')
    parser.add_argument('-cp', '--checkpoint', help='Path to model checkpoint for validation',
                      default=None)
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Parameters
    WINDOW_SIZE = tuple(args.window)
    FOLDER = args.input
    OUTPUT_FOLDER = args.output
    CVAE_PATH = args.cvae
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    QUADTREE_DEPTH = args.quadtree_depth
    LATENT_DIM = args.latent_dim
    N_CLASSES = args.n_classes
    
    # Data paths
    IMAGE_FILES = f"{FOLDER}/top/top_mosaic_09cm_area{{}}.tif"
    LABEL_FILES = f"{FOLDER}/gt/top_mosaic_09cm_area{{}}.tif"
    
    # Define train and test IDs
    all_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37', '5', '15', '21', '30']
    
    # Split data (same as for CVAE)
    train_val_split = int(len(all_ids) * 0.8)
    train_ids = all_ids[:train_val_split]
    val_ids = all_ids[train_val_split:]
    
    print(f"Training IDs: {train_ids}")
    print(f"Validation IDs: {val_ids}")
    
    # Create datasets
    if args.mode == 'train':
        train_set = SegmentationDataset(
            train_ids, IMAGE_FILES, LABEL_FILES, WINDOW_SIZE, augment=True
        )
    
    val_set = SegmentationDataset(
        val_ids, IMAGE_FILES, LABEL_FILES, WINDOW_SIZE, augment=False
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
    
    # Load pre-trained CVAE
    cvae = EnhancedCVAE(input_channels=3, latent_dim=LATENT_DIM)
    cvae.load_state_dict(torch.load(
        CVAE_PATH, 
        map_location=device,
        weights_only=True
    ))
    cvae = cvae.to(device)
    cvae.eval()
    
    if args.mode == 'train':
        # Create trainer
        trainer = QuadtreeMRFTrainer(
            cvae_path=CVAE_PATH,
            n_classes=N_CLASSES,
            feature_dim=LATENT_DIM,
            quadtree_depth=QUADTREE_DEPTH,
            learning_rate=LEARNING_RATE,
            device=device
        )
        
        # Train model
        trainer.train(train_loader, val_loader, EPOCHS, OUTPUT_FOLDER)
        
    else:  # Validate mode
        # Load model
        checkpoint_path = args.checkpoint or f"{OUTPUT_FOLDER}/model_best.pth"
        
        # Create model
        model = OptimizedQuadtreeMRF(
            n_classes=N_CLASSES,
            quadtree_depth=QUADTREE_DEPTH,
            feature_dim=LATENT_DIM,
            device=device
        ).to(device)
        
        # Load weights
        model.load_state_dict(torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        ))
        
        # Run validation
        validate_output_dir = f"{OUTPUT_FOLDER}/validation"
        metrics = validate_quadtree_mrf(
            model, cvae, val_loader, N_CLASSES, device, validate_output_dir
        )
        
        # Print results
        print("\nValidation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()