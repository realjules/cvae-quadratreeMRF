#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train only the CVAE component of the model
"""

import torch
import argparse
import os
from datetime import datetime
import numpy as np

from net.net import HierarchicalPGM
from net.cvae import CVAE
from dataset.dataset import ISPRS_dataset
from utils.utils_dataset import *
from train import train

class CVAELoss(torch.nn.Module):
    """Simple CVAE loss function with reconstruction, KLD, and contrastive loss"""
    def __init__(self, kld_weight=0.001, contrastive_weight=0.5):
        super(CVAELoss, self).__init__()
        self.kld_weight = kld_weight
        self.contrastive_weight = contrastive_weight
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        
    def forward(self, outputs, targets=None, mode='full'):
        # Extract values from outputs dictionary
        x_recon = outputs['reconstruction']
        mu = outputs['mu']
        log_var = outputs['log_var']
        z_proj = outputs['z_proj']
        original_input = outputs['original_input']
        
        # Reconstruction loss (pixel-wise MSE)
        recon_loss = self.mse_loss(x_recon, original_input) / original_input.size(0)
        
        # KL Divergence loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / original_input.size(0)
        
        # Contrastive loss
        contrastive_loss = 0.0
        if 'z_proj' in outputs:
            contrastive_module = CVAE(input_channels=3, latent_dim=256)
            if hasattr(contrastive_module, 'contrastive_loss'):
                contrastive_loss = contrastive_module.contrastive_loss(
                    outputs['z_proj'], 
                    labels=None,  # Unsupervised mode
                    temperature=0.5
                )
        
        # Total loss
        total_loss = recon_loss + self.kld_weight * kld_loss + self.contrastive_weight * contrastive_loss
        
        # Return loss components for monitoring
        loss_components = {
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'contrastive_loss': contrastive_loss,
            'seg_loss': torch.tensor(0.0, device=total_loss.device),  # Placeholder for compatibility
            'hier_loss': torch.tensor(0.0, device=total_loss.device),  # Placeholder for compatibility
            'hier_consistency_loss': torch.tensor(0.0, device=total_loss.device)  # Placeholder for compatibility
        }
        
        return total_loss, loss_components
    
    def update_epoch(self, current_epoch, total_epochs):
        # Adjust weights over time if needed
        pass


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train only the CVAE component')
    parser.add_argument('-i', '--input', help='Path of input directory', 
                        metavar='INPUT_DIR_PATH', default="./input/")
    parser.add_argument('-o', '--output', help='Path of output directory',
                        metavar='OUTPUT_DIR_PATH', default="./output/CVAE-only/")
    parser.add_argument('-w', '--window', nargs=2, type=int, default=[256, 256],
                        help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-lr', '--base_lr', default=0.001, type=float, help='Base learning rate')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('-se', '--save_epoch', default=5, type=int, help='Save model every N epochs')
    parser.add_argument('-ld', '--latent_dim', default=256, type=int, help='Latent dimension size')
    parser.add_argument('-kw', '--kld_weight', default=0.001, type=float, help='KL divergence weight')
    parser.add_argument('-cw', '--contrastive_weight', default=0.5, type=float, help='Contrastive loss weight')
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
    cvae = CVAE(input_channels=IN_CHANNELS, latent_dim=LATENT_DIM)
    
    # Create wrapper class that matches the expected interface
    class CVAEWrapper(torch.nn.Module):
        def __init__(self, cvae):
            super(CVAEWrapper, self).__init__()
            self.cvae = cvae
        
        def forward(self, x, mode=None):
            return self.cvae(x)
    
    # Wrap the CVAE model
    net = CVAEWrapper(cvae)
    
    # Initialize loss function
    criterion = CVAELoss(
        kld_weight=args.kld_weight,
        contrastive_weight=args.contrastive_weight
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=0.0001)
    
    # Use GPU if available
    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()
    
    # Define train and validation data
    all_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37', '5', '15', '21', '30']
    
    # Split data into train and validation (80/20 split)
    train_val_split = int(len(all_ids) * 0.8)
    train_ids = all_ids[:train_val_split]
    val_ids = all_ids[train_val_split:]
    
    print(f"Training IDs: {train_ids}")
    print(f"Validation IDs: {val_ids}")
    
    # Create datasets (without labels, using only images)
    train_set = UnsupervisedDataset(train_ids, data_files=DATA_FOLDER, window_size=WINDOW_SIZE)
    val_set = UnsupervisedDataset(val_ids, data_files=DATA_FOLDER, window_size=WINDOW_SIZE)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size)
    
    # Set up OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.base_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3
    )
    
    # Placeholder weights (not used for CVAE)
    weights = torch.ones(1)
    
    # Train the model
    train(net, criterion, optimizer, scheduler, train_loader, None,
        val_loader, epochs, save_epoch, weights, batch_size, WINDOW_SIZE, OUTPUT_FOLDER)
    
    print("CVAE training completed!")


# Simple unsupervised dataset class that returns only images
class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files, window_size=(256, 256), cache=True):
        self.ids = ids
        self.data_files = data_files
        self.window_size = window_size
        self.cache = cache
        
        # Load data
        self.data = []
        for id in self.ids:
            img = 1/255 * np.asarray(io.imread(self.data_files.format(id)), dtype='float32')
            self.data.append(img)
        
        # Create windows
        self.windows = []
        for img_idx, img in enumerate(self.data):
            height, width, _ = img.shape
            for i in range(0, height - window_size[0], window_size[0] // 2):
                for j in range(0, width - window_size[1], window_size[1] // 2):
                    self.windows.append((img_idx, i, j))
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        img_idx, i, j = self.windows[idx]
        img = self.data[img_idx]
        
        # Extract window
        patch = img[i:i+self.window_size[0], j:j+self.window_size[1], :]
        
        # Convert to torch format
        patch = np.transpose(patch, (2, 0, 1))  # HWC -> CHW
        patch = torch.from_numpy(patch)
        
        # Return patch and a dummy target (not used)
        return patch, torch.zeros(1)


if __name__ == "__main__":
    main()