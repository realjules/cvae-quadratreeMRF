# HVS-Net/core/losses.py

"""
This file defines the custom loss functions used in our framework.

It contains:
1.  FocalLoss: For the supervised segmentation task.
2.  DiceLoss: To be combined with FocalLoss for better segmentation performance.
3.  ConsistencyLoss: To enforce that the segmentation of two augmented versions of an unlabeled image are similar.
4.  A wrapper function to combine these into the final multi-component loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss, adapted for multi-class cases."""
    def __init__(self, smooth=1e-6, n_classes=6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.n_classes = n_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()

        intersect = torch.sum(pred * target_one_hot, dim=[2, 3])
        union = torch.sum(pred, dim=[2, 3]) + torch.sum(target_one_hot, dim=[2, 3])

        dice = (2 * intersect + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """Focal Loss, for addressing class imbalance."""
    def __init__(self, alpha=0.8, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """The main loss function for HVS-Net, combining all components."""
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss(n_classes=config['model']['n_classes'])
        self.recon_loss = nn.L1Loss()
        self.consistency_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, inputs):
        # Unpack inputs and outputs
        labeled_images, labeled_masks = inputs['labeled']
        unlabeled_images_aug1, unlabeled_images_aug2 = inputs['unlabeled']
        
        pred_labeled = outputs['labeled_seg']
        pred_unlabeled_aug1 = outputs['unlabeled_seg1']
        pred_unlabeled_aug2 = outputs['unlabeled_seg2']
        recon_images = outputs['reconstruction']
        mu, log_var = outputs['mu'], outputs['log_var']

        # 1. Supervised Segmentation Loss (on labeled data)
        focal = self.focal_loss(pred_labeled, labeled_masks)
        dice = self.dice_loss(pred_labeled, labeled_masks)
        supervised_loss = focal + dice

        # 2. Generative Reconstruction Loss (on all images)
        all_images = torch.cat([labeled_images, unlabeled_images_aug1, unlabeled_images_aug2], dim=0)
        reconstruction_loss = self.recon_loss(recon_images, all_images)

        # 3. KL Divergence Loss (for VAE regularization)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 4. Unsupervised Consistency Loss (on unlabeled data)
        log_p1 = F.log_softmax(pred_unlabeled_aug1, dim=1)
        p2 = F.softmax(pred_unlabeled_aug2, dim=1)
        consistency_loss = self.consistency_loss(log_p1, p2)

        # Combine losses with weights from config
        total_loss = (
            self.config['loss']['supervised_weight'] * supervised_loss +
            self.config['loss']['reconstruction_weight'] * reconstruction_loss +
            self.config['loss']['kl_weight'] * kl_loss +
            self.config['loss']['consistency_weight'] * consistency_loss
        )

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'consistency_loss': consistency_loss.item()
        }