"""
Loss functions for DHBP segmentation.

FocalLoss: handles class imbalance (cars F1=0.39 baseline).
DifferentiableBoundaryLoss: Sobel on softmax probabilities (NOT argmax).
SegmentationLoss: fixed-weight combination (no learnable loss weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    With per-class weights passed to cross_entropy for additional
    control over class imbalance (e.g., cars get 3x weight).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        class_weights: torch.Tensor | None = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        # Store class_weights as buffer so they move with .to(device)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits (pre-softmax)
            targets: [B, H, W] integer class labels
        """
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.class_weights, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DifferentiableBoundaryLoss(nn.Module):
    """Differentiable boundary loss using Sobel filters on soft predictions.

    Unlike the previous BoundaryLoss which used argmax (non-differentiable),
    this operates on softmax probabilities and one-hot targets with fixed
    Sobel kernels, producing gradients that flow all the way back.
    """

    def __init__(self):
        super().__init__()
        # Fixed Sobel kernels (not learned)
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _compute_boundary(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute boundary magnitude for each class channel.

        Args:
            tensor: [B, C, H, W] per-class probability or one-hot map

        Returns:
            [B, C, H, W] boundary magnitude
        """
        B, C, H, W = tensor.shape
        # Reshape to apply same kernel to all classes in one conv call
        flat = tensor.reshape(B * C, 1, H, W)
        gx = F.conv2d(flat, self.kernel_x, padding=1)
        gy = F.conv2d(flat, self.kernel_y, padding=1)
        boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return boundary.reshape(B, C, H, W)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits (pre-softmax)
            targets: [B, H, W] integer class labels

        Returns:
            scalar boundary loss
        """
        n_classes = predictions.size(1)

        # Resize predictions to match target spatial dims if needed
        if predictions.shape[2:] != targets.shape[1:]:
            predictions = F.interpolate(
                predictions, size=targets.shape[1:],
                mode='bilinear', align_corners=False,
            )

        # Soft predictions (differentiable)
        soft_pred = F.softmax(predictions, dim=1)  # [B, C, H, W]

        # One-hot targets
        target_onehot = F.one_hot(targets.long(), num_classes=n_classes)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()       # [B, C, H, W]

        # Compute boundary magnitude per class
        pred_boundary = self._compute_boundary(soft_pred)
        target_boundary = self._compute_boundary(target_onehot)

        return F.l1_loss(pred_boundary, target_boundary)


class SegmentationLoss(nn.Module):
    """Combined segmentation loss with fixed weights.

    loss = 1.0 * FocalLoss + 0.1 * DifferentiableBoundaryLoss

    No learnable loss weights — fixed and explicit.
    """

    def __init__(self, n_classes: int = 6, focal_weight: float = 1.0,
                 boundary_weight: float = 0.1):
        super().__init__()
        class_weights = torch.tensor([1.0, 1.5, 1.0, 1.0, 3.0, 1.2])
        self.focal = FocalLoss(gamma=2.0, class_weights=class_weights)
        self.boundary = DifferentiableBoundaryLoss()
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits: [B, n_classes, H, W]
            targets: [B, H, W] integer class labels

        Returns:
            (total_loss, loss_components_dict)
        """
        focal_loss = self.focal(logits, targets)
        boundary_loss = self.boundary(logits, targets)
        total = self.focal_weight * focal_loss + self.boundary_weight * boundary_loss
        return total, {
            'total_loss': total.item(),
            'focal_loss': focal_loss.item(),
            'boundary_loss': boundary_loss.item(),
        }
