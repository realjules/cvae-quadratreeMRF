"""
Neural network models for Semi-Supervised Hierarchical PGM with Contrastive Learning

This package contains:
- EnhancedCVAE: Contrastive Variational Autoencoder for feature learning
- MultiScaleSegmentationModel: Enhanced segmentation model with spatial reasoning
- MultiScaleLoss: Multi-scale loss function with focal loss and boundary awareness
"""

from .cvae import EnhancedCVAE, CVAE
from .segmentation_model import MultiScaleSegmentationModel
from .loss import MultiScaleLoss, SimpleCrossEntropyLoss, FocalLoss, BoundaryLoss

__all__ = [
    'EnhancedCVAE',
    'CVAE',
    'MultiScaleSegmentationModel', 
    'MultiScaleLoss',
    'SimpleCrossEntropyLoss',
    'FocalLoss',
    'BoundaryLoss'
]