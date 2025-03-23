import torch
import torch.nn.functional as F

def segmentation_loss(pred, target):
    # pred: [B, num_classes, H, W], target: [B, H, W] with class indices
    loss = F.cross_entropy(pred, target)
    return loss

def contrastive_loss(z, temperature=0.5):
    # z: [B, latent_dim]
    # Normalize latent vectors
    z = F.normalize(z, dim=1)
    batch_size = z.size(0)
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(z, z.T)  # [B, B]
    
    # Scale similarities by temperature
    logits = similarity_matrix / temperature

    # Create labels: assume that for each sample the positive sample is itself
    # Note: this is a dummy implementation; in a real contrastive setting, positive pairs should come from augmented views
    labels = torch.arange(batch_size, device=z.device)
    loss = F.cross_entropy(logits, labels)
    return loss
