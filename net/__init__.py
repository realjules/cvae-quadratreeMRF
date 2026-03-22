"""
Neural network modules for DHBP semi-supervised segmentation.

- ContrastiveEncoder: ResNet-18 backbone + SimCLR projection head
- DHBPModule: Differentiable Hierarchical Belief Propagation
- SegmentationLoss: Focal + differentiable boundary loss
"""

from .cvae import ResNet18Encoder, SimCLRProjectionHead, ContrastiveEncoder
from .dhbp import DHBPModule, UnaryPotentialHead, PairwisePotentialHead
from .loss import FocalLoss, DifferentiableBoundaryLoss, SegmentationLoss

__all__ = [
    'ResNet18Encoder', 'SimCLRProjectionHead', 'ContrastiveEncoder',
    'DHBPModule', 'UnaryPotentialHead', 'PairwisePotentialHead',
    'FocalLoss', 'DifferentiableBoundaryLoss', 'SegmentationLoss',
]
