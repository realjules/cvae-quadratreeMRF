# HVS-Net/core/trainer.py

"""
This file contains the HVSTrainer class, which orchestrates the entire training process.

It handles:
1.  Initializing the HVS-Net model, optimizer, and learning rate scheduler.
2.  The main training loop.
3.  The logic for computing the multi-component loss (Supervised, Generative, Consistency).
4.  Validation loop and evaluation metric calculation.
5.  Saving model checkpoints.
"""

import torch
from .architecture import HVSNet
from .losses import CombinedLoss
from tqdm import tqdm
import os

class HVSTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = HVSNet(
            n_channels=config['model']['n_channels'],
            n_classes=config['model']['n_classes'],
            latent_dim=config['model']['latent_dim'],
            encoder_hidden_dims=config['model']['encoder_hidden_dims'],
            decoder_hidden_dims=config['model']['decoder_hidden_dims']
        ).to(self.device)

        # Initialize losses
        self.criterion = CombinedLoss(config).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )

        self.best_val_loss = float('inf')

    def train(self, train_loader, val_loader):
        print("Starting HVS-Net training...")
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self._train_epoch(epoch, train_loader)
            val_loss = self._validate_epoch(epoch, val_loader)
            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best_model.pth')
                print(f"Epoch {epoch}: New best model saved with validation loss: {val_loss:.4f}")

    def _train_epoch(self, epoch, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in progress_bar:
            # Move data to device
            labeled_image = batch['labeled_image'].to(self.device)
            labeled_mask = batch['labeled_mask'].to(self.device)
            unlabeled_image1 = batch['unlabeled_image1'].to(self.device)
            unlabeled_image2 = batch['unlabeled_image2'].to(self.device)

            # Concatenate all images for a single forward pass
            all_images = torch.cat([labeled_image, unlabeled_image1, unlabeled_image2], dim=0)

            # Forward pass
            self.optimizer.zero_grad()
            model_outputs = self.model(all_images)

            # Separate the outputs
            num_labeled = labeled_image.size(0)
            num_unlabeled = unlabeled_image1.size(0)
            
            outputs = {
                'labeled_seg': model_outputs['segmentation'][:num_labeled],
                'unlabeled_seg1': model_outputs['segmentation'][num_labeled:num_labeled + num_unlabeled],
                'unlabeled_seg2': model_outputs['segmentation'][num_labeled + num_unlabeled:],
                'reconstruction': model_outputs['reconstruction'],
                'mu': model_outputs['mu'],
                'log_var': model_outputs['log_var']
            }

            inputs = {
                'labeled': (labeled_image, labeled_mask),
                'unlabeled': (unlabeled_image1, unlabeled_image2)
            }

            # Compute loss
            loss_dict = self.criterion(outputs, inputs)
            loss = loss_dict['total_loss']

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch} [Train] - Average Loss: {total_loss / len(train_loader):.4f}")

    def _validate_epoch(self, epoch, val_loader):
        self.model.eval()
        total_val_loss = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for batch in progress_bar:
                image = batch['image'].to(self.device)
                mask = batch['mask'].to(self.device)

                # Forward pass (only segmentation is needed for validation)
                outputs = self.model(image)
                seg_pred = outputs['segmentation']

                # For validation, we only need a simple segmentation loss
                val_loss = F.cross_entropy(seg_pred, mask)
                total_val_loss += val_loss.item()
                progress_bar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch} [Val] - Average Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _save_checkpoint(self, filename):
        path = os.path.join(self.config['training']['checkpoint_dir'], filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)