"""
SimCLR Contrastive Trainer for pre-training the ResNet-18 encoder
on unlabeled ISPRS aerial imagery.

Pipeline:
    raw images → augment (exactly once) → encoder.project() → SimCLR loss

No MoCo, no reconstruction, no VAE. Simple and proven.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from net.cvae import ContrastiveEncoder
from utils.losses import simclr_loss_simple
from utils.contrastive_augmentations import ContrastiveAugmentation


class ContrastiveTrainer:
    """SimCLR contrastive pre-training for the ResNet-18 encoder.

    Key design decisions:
        - Augmentation applied exactly once inside train_step() (fixes double-aug bug)
        - Uses simclr_loss_simple from utils/losses.py (proven correct)
        - Consistent save/load format: {encoder_state_dict, optimizer_state_dict, epoch}
    """

    def __init__(
        self,
        encoder: ContrastiveEncoder | None = None,
        learning_rate: float = 1e-3,
        temperature: float = 0.1,
        device: str = 'cuda',
    ):
        self.device = device
        self.temperature = temperature

        if encoder is None:
            encoder = ContrastiveEncoder(pretrained=True)
        self.encoder = encoder.to(device)

        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.augmentation = ContrastiveAugmentation(size=256, strength=0.8)
        self.best_loss = float('inf')

    def train_step(self, images: torch.Tensor) -> dict:
        """Single SimCLR training step.

        Args:
            images: [B, 3, 256, 256] raw images (NOT pre-augmented).
                    Augmentation is applied exactly once here.

        Returns:
            dict with 'total_loss' and 'contrastive_loss'
        """
        self.encoder.train()
        B = images.size(0)

        # Create two augmented views (augmentation applied ONCE, here)
        view1_list, view2_list = [], []
        for i in range(B):
            v1, v2 = self.augmentation(images[i])
            view1_list.append(v1)
            view2_list.append(v2)

        view1 = torch.stack(view1_list).to(self.device)
        view2 = torch.stack(view2_list).to(self.device)

        # Project both views
        z1 = self.encoder.project(view1)  # [B, proj_dim]
        z2 = self.encoder.project(view2)  # [B, proj_dim]

        # SimCLR loss
        loss = simclr_loss_simple(z1, z2, self.temperature)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'contrastive_loss': loss.item(),
        }

    def save(self, path: str, epoch: int):
        """Save encoder and optimizer state.

        Format: {epoch, encoder_state_dict, optimizer_state_dict}
        This is the ONLY save format — no ambiguity, no mismatch.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> int:
        """Load encoder and optimizer state.

        Args:
            path: checkpoint file path

        Returns:
            epoch number from checkpoint

        Raises:
            FileNotFoundError: if path doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('epoch', 0)
