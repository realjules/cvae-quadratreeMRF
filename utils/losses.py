import torch
import torch.nn.functional as F

def segmentation_loss(pred, target):
    # pred: [B, num_classes, H, W], target: [B, H, W] with class indices
    loss = F.cross_entropy(pred, target)
    return loss

def contrastive_loss(z1, z2, temperature=0.5, queue=None):
    """
    Fixed SimCLR-style contrastive loss for proper self-supervised learning
    
    Args:
        z1, z2: [B, latent_dim] - normalized projected features from two augmented views
        temperature: scaling factor for similarities
        queue: [queue_size, latent_dim] - memory bank for additional negatives (MoCo style)
    """
    batch_size = z1.size(0)
    device = z1.device
    
    # Ensure features are normalized
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Simple approach: compute loss for z1->z2 and z2->z1 separately
    def single_direction_loss(anchor, positive, negatives_queue=None):
        # anchor: [B, latent_dim], positive: [B, latent_dim]
        B = anchor.size(0)
        
        # Positive similarities
        pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True) / temperature  # [B, 1]
        
        # Negative similarities (all other samples in batch)
        neg_sim = torch.matmul(anchor, positive.T) / temperature  # [B, B]
        
        # Remove self-similarities (diagonal)
        mask = torch.eye(B, device=device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # Add queue negatives if available and compatible
        if negatives_queue is not None and negatives_queue.size(0) > 0:
            if negatives_queue.size(1) == anchor.size(1):
                # Compute similarities with queue
                queue_sim = torch.matmul(anchor, negatives_queue.T) / temperature  # [B, queue_size]
                # Combine with batch negatives
                logits = torch.cat([pos_sim, neg_sim, queue_sim], dim=1)  # [B, 1 + B + queue_size]
            else:
                # Skip queue if dimensions don't match
                logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1 + B]
        else:
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1 + B]
        
        # Labels: positive is always at index 0
        labels = torch.zeros(B, device=device, dtype=torch.long)
        
        return F.cross_entropy(logits, labels)
    
    # Compute loss in both directions and average
    loss_12 = single_direction_loss(z1, z2, queue)
    loss_21 = single_direction_loss(z2, z1, queue)
    
    return (loss_12 + loss_21) / 2


def simclr_loss_simple(z1, z2, temperature=0.1):
    """
    Simplified SimCLR loss for easier debugging
    
    Args:
        z1, z2: [B, latent_dim] - normalized projected features from augmented views
        temperature: scaling factor
    """
    batch_size = z1.size(0)
    device = z1.device
    
    # Normalize features
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Positive similarities
    pos_sim = torch.sum(z1 * z2, dim=1) / temperature  # [B]
    
    # Negative similarities
    neg_sim = torch.matmul(z1, z2.T) / temperature  # [B, B]
    
    # Remove diagonal (self-similarities)
    mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    neg_sim = neg_sim.masked_fill(mask, float('-inf'))
    
    # Compute loss for each sample
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1 + B]
    labels = torch.zeros(batch_size, device=device, dtype=torch.long)  # Positive is always first
    
    loss = F.cross_entropy(logits, labels)
    return loss
