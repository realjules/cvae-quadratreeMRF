import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss function designed specifically for our segmentation model.
    Replaces the overly complex EnhancedHierarchicalPGMLoss with focused functionality.
    """
    def __init__(self, n_classes, device="cuda"):
        super(MultiScaleLoss, self).__init__()
        self.n_classes = n_classes
        self.device = device
        
        # Focal loss for handling class imbalance (especially cars with F1=0.39)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        
        # Boundary loss for crisp edges
        self.boundary_loss = BoundaryLoss()
        
        # Learnable loss weights that adapt during training
        self.register_parameter('scale_weights', nn.Parameter(torch.ones(3)))  # For 3 scales
        self.register_parameter('loss_type_weights', nn.Parameter(torch.tensor([1.0, 0.1, 0.05])))  # [focal, boundary, consistency]
        
    def forward(self, outputs, targets, original_size):
        """
        Compute multi-scale loss
        
        Args:
            outputs: Dict with 'final_segmentation' and 'multi_scale_predictions'
            targets: Ground truth labels [B, H, W]
            original_size: (H, W) tuple for final output size
        """
        total_loss = 0.0
        loss_components = {}
        
        # Ensure targets are at correct size
        if targets.size(1) != original_size[0] or targets.size(2) != original_size[1]:
            targets = F.interpolate(targets.float().unsqueeze(1), size=original_size, mode='nearest').squeeze(1).long()
        
        # 1. Multi-scale supervision loss
        multi_scale_preds = outputs['multi_scale_predictions']
        scale_losses = []
        
        for i, pred in enumerate(multi_scale_preds):
            # Resize prediction to target size
            pred_resized = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
            
            # Focal loss for this scale
            scale_loss = self.focal_loss(pred_resized, targets)
            scale_losses.append(scale_loss)
            loss_components[f'scale_{i}_loss'] = scale_loss.item()
        
        # Weighted combination of scale losses
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        multi_scale_loss = sum(w * loss for w, loss in zip(scale_weights_norm, scale_losses))
        loss_components['multi_scale_loss'] = multi_scale_loss.item()
        
        # 2. Final segmentation loss
        final_seg = outputs['final_segmentation']
        if final_seg.size(2) != original_size[0] or final_seg.size(3) != original_size[1]:
            final_seg = F.interpolate(final_seg, size=original_size, mode='bilinear', align_corners=False)
        
        final_focal_loss = self.focal_loss(final_seg, targets)
        loss_components['final_focal_loss'] = final_focal_loss.item()
        
        # 3. Boundary loss for crisp edges
        boundary_loss = self.boundary_loss(final_seg, targets)
        loss_components['boundary_loss'] = boundary_loss.item()
        
        # 4. Multi-scale consistency loss
        consistency_loss = self.compute_consistency_loss(multi_scale_preds, original_size)
        loss_components['consistency_loss'] = consistency_loss.item()
        
        # Combine all losses with learned weights
        loss_weights_norm = F.softmax(self.loss_type_weights, dim=0)
        total_loss = (loss_weights_norm[0] * (multi_scale_loss + final_focal_loss) + 
                     loss_weights_norm[1] * boundary_loss + 
                     loss_weights_norm[2] * consistency_loss)
        
        loss_components['total_loss'] = total_loss.item()
        loss_components['loss_weights'] = loss_weights_norm.detach().cpu().numpy().tolist()
        loss_components['scale_weights'] = scale_weights_norm.detach().cpu().numpy().tolist()
        
        return total_loss, loss_components
    
    def compute_consistency_loss(self, multi_scale_preds, target_size):
        """Compute consistency loss between different scales"""
        if len(multi_scale_preds) < 2:
            return torch.tensor(0.0, device=self.device)
        
        consistency_loss = 0.0
        n_pairs = 0
        
        # Resize all predictions to target size
        resized_preds = []
        for pred in multi_scale_preds:
            resized = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
            resized_preds.append(F.softmax(resized, dim=1))
        
        # Compute pairwise KL divergence between scales
        for i in range(len(resized_preds)):
            for j in range(i + 1, len(resized_preds)):
                # KL divergence between scale i and scale j
                kl_loss = F.kl_div(
                    F.log_softmax(resized_preds[i], dim=1),
                    resized_preds[j],
                    reduction='batchmean'
                )
                consistency_loss += kl_loss
                n_pairs += 1
        
        return consistency_loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=self.device)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Critical for improving cars detection (currently F1=0.39).
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula: α(1-pt)^γ * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BoundaryLoss(nn.Module):
    """
    Boundary loss for crisp segmentation edges.
    Helps improve segmentation quality at object boundaries.
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        
    def forward(self, predictions, targets):
        # Ensure predictions and targets have same spatial size
        if predictions.size(2) != targets.size(1) or predictions.size(3) != targets.size(2):
            predictions = F.interpolate(
                predictions, 
                size=(targets.size(1), targets.size(2)), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Convert predictions to class predictions
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Compute boundaries using simple gradient
        def compute_boundary(tensor):
            # Gradient in x and y directions
            grad_x = torch.abs(tensor[:, :, 1:] - tensor[:, :, :-1])
            grad_y = torch.abs(tensor[:, 1:, :] - tensor[:, :-1, :])
            
            # Pad to maintain size
            grad_x = F.pad(grad_x, (0, 1), mode='constant', value=0)
            grad_y = F.pad(grad_y, (1, 0), mode='constant', value=0)
            
            return grad_x + grad_y
        
        # Compute boundaries
        pred_boundary = compute_boundary(pred_classes.float())
        target_boundary = compute_boundary(targets.float())
        
        # Boundary loss (L1 difference)
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)
        
        return boundary_loss


# Simple Cross Entropy Loss for compatibility
class SimpleCrossEntropyLoss(nn.Module):
    """Simple cross entropy loss with class weights"""
    def __init__(self, class_weights=None, ignore_index=255):
        super(SimpleCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        
    def forward(self, outputs, targets, original_size=None):
        """
        Simplified loss function for compatibility
        
        Args:
            outputs: Can be dict with 'final_segmentation' or direct tensor
            targets: Ground truth labels
            original_size: Ignored for compatibility
        """
        # Handle both dict and tensor inputs
        if isinstance(outputs, dict):
            if 'final_segmentation' in outputs:
                predictions = outputs['final_segmentation']
            else:
                # Fallback to first available output
                predictions = list(outputs.values())[0]
        else:
            predictions = outputs
            
        # Resize if needed
        if predictions.size(2) != targets.size(1) or predictions.size(3) != targets.size(2):
            predictions = F.interpolate(
                predictions, 
                size=(targets.size(1), targets.size(2)), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Compute loss
        loss = F.cross_entropy(
            predictions, 
            targets, 
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )
        
        # Return in expected format
        loss_components = {'total_loss': loss.item()}
        return loss, loss_components