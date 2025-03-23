import torch
import torch.nn as nn
import torch.nn.functional as F

def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, reduction, ignore_index=6)
    elif dim == 4:
        output = input.reshape(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.reshape(-1)  
        return F.cross_entropy(output, target, weight, reduction, ignore_index=6)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

class HierarchicalPGMLoss(nn.Module):
    """
    Combined loss function for the Hierarchical PGM with Contrastive Learning.
    
    This loss function combines:
    1. Supervised segmentation loss (cross entropy)
    2. CVAE reconstruction and KL divergence losses
    3. Contrastive learning loss
    4. Hierarchical consistency loss
    """
    def __init__(self, n_classes, weights=None, kld_weight=0.005, contrastive_weight=0.1, 
                 consistency_weight=0.05, temperature=0.5, ignore_index=6):
        super(HierarchicalPGMLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights  # Class weights for segmentation loss
        self.kld_weight = kld_weight  # Weight for KL divergence loss
        self.contrastive_weight = contrastive_weight  # Weight for contrastive loss
        self.consistency_weight = consistency_weight  # Weight for hierarchical consistency loss
        self.temperature = temperature  # Temperature for contrastive loss
        self.ignore_index = ignore_index  # Ignore index for segmentation loss
        
    def forward(self, outputs, targets=None, mode='full'):
        """
        Calculate the combined loss.
        
        Args:
            outputs: Dictionary of model outputs from HierarchicalPGM
            targets: Ground truth segmentation (optional, required for supervised components)
            mode: Training mode ('full', 'supervised', 'unsupervised', or 'inference')
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        loss_components = {}
        
        # Supervised segmentation loss
        if targets is not None and mode in ['supervised', 'full']:
            # Primary segmentation loss (final output)
            if 'final_segmentation' in outputs:
                seg_loss = CrossEntropy2d(
                    outputs['final_segmentation'], 
                    targets,
                    weight=self.weights
                )
                loss_components['seg_loss'] = seg_loss
            
            # Hierarchical segmentation losses
            if 'hierarchical_segmentations' in outputs:
                hier_losses = []
                for i, seg in enumerate(outputs['hierarchical_segmentations']):
                    # Resize target to match hierarchical output size
                    target_size = seg.size()[2:]
                    scaled_target = F.interpolate(
                        targets.float().unsqueeze(1), 
                        size=target_size, 
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    hier_loss = CrossEntropy2d(
                        seg, 
                        scaled_target,
                        weight=self.weights
                    )
                    hier_losses.append(hier_loss)
                    loss_components[f'hier_loss_{i}'] = hier_loss
                
                if hier_losses:
                    # Average hierarchical losses
                    loss_components['hier_loss'] = sum(hier_losses) / len(hier_losses)
        
        # CVAE unsupervised losses
        if mode in ['unsupervised', 'full']:
            # Reconstruction loss
            if 'reconstruction' in outputs:
                if 'original_input' in outputs:
                    original_input = outputs['original_input']
                else:
                    # Use the actual input provided to the model if original_input is not stored
                    original_input = targets  # This assumes targets is the input in unsupervised mode
                
                recon_loss = F.mse_loss(outputs['reconstruction'], original_input)
                loss_components['recon_loss'] = recon_loss
            
            # KL divergence loss
            if 'mu' in outputs and 'logvar' in outputs:
                mu = outputs['mu']
                logvar = outputs['logvar']
                
                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # Normalize by batch size
                kld_loss = kld_loss / mu.size(0)
                loss_components['kld_loss'] = kld_loss * self.kld_weight
            
            # Contrastive loss
            if 'z_proj' in outputs:
                z_proj = outputs['z_proj']
                
                if targets is not None and mode == 'full':
                    # Supervised contrastive loss (use targets for positive pairs)
                    # Extract class labels for each sample (use dominant class)
                    batch_size = targets.size(0)
                    labels = []
                    
                    for i in range(batch_size):
                        target = targets[i].flatten()
                        # Remove ignore index
                        valid_pixels = target[target != self.ignore_index]
                        if len(valid_pixels) > 0:
                            # Get most common class
                            unique, counts = torch.unique(valid_pixels, return_counts=True)
                            dominant_class = unique[counts.argmax()]
                            labels.append(dominant_class.item())
                        else:
                            labels.append(-1)  # No valid pixels
                    
                    labels_tensor = torch.tensor(labels, device=z_proj.device)
                    
                    # Compute supervised contrastive loss
                    contrastive_loss = self._supervised_contrastive_loss(
                        z_proj, 
                        labels_tensor, 
                        temperature=self.temperature
                    )
                else:
                    # Unsupervised contrastive loss
                    contrastive_loss = self._unsupervised_contrastive_loss(
                        z_proj, 
                        temperature=self.temperature
                    )
                
                loss_components['contrastive_loss'] = contrastive_loss * self.contrastive_weight
        
        # Hierarchical consistency loss
        if 'hierarchical_segmentations' in outputs and len(outputs['hierarchical_segmentations']) > 1:
            hier_consistency_loss = self._hierarchical_consistency_loss(
                outputs['hierarchical_segmentations']
            )
            loss_components['hier_consistency_loss'] = hier_consistency_loss * self.consistency_weight
        
        # Total loss
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components
    
    def _supervised_contrastive_loss(self, features, labels, temperature=0.5):
        """
        Supervised contrastive loss as defined in SupCon paper.
        
        Args:
            features: Feature vectors (B, D)
            labels: Class labels (B)
            temperature: Temperature parameter
        """
        device = features.device
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        # Create mask for positive pairs (same class)
        pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Remove self-comparisons
        eye_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask - eye_mask
        pos_mask = torch.clamp(pos_mask, min=0)
        
        # Count number of positives for each sample
        num_positives = pos_mask.sum(dim=1)
        
        # Handle samples with no positives
        valid_samples = num_positives > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute log probability
        exp_sim = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0])
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True) - exp_sim * eye_mask
        log_prob = sim_matrix - torch.log(exp_sim_sum)
        
        # Compute mean log-likelihood of positive pairs
        pos_log_prob = (pos_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
        
        # Only compute loss for valid samples (with positives)
        loss = -pos_log_prob[valid_samples].mean()
        
        return loss
    
    def _unsupervised_contrastive_loss(self, features, temperature=0.5):
        """
        Unsupervised contrastive loss (InfoNCE).
        
        Args:
            features: Feature vectors (B, D)
            temperature: Temperature parameter
        """
        device = features.device
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        # InfoNCE loss
        # Use each sample as its own class (diagonal is positive)
        eye_mask = torch.eye(batch_size, device=device)
        
        # Remove self-comparison from sim_matrix
        sim_matrix_no_diag = sim_matrix - eye_mask * 1e9
        
        # Create labels (each sample should be most similar to itself)
        labels = torch.arange(batch_size, device=device)
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix_no_diag, labels)
        
        return loss
    
    def _hierarchical_consistency_loss(self, hierarchical_segmentations):
        """
        Consistency loss between adjacent levels in the hierarchy.
        
        Args:
            hierarchical_segmentations: List of segmentation outputs at different levels
        """
        consistency_loss = 0.0
        n_levels = len(hierarchical_segmentations)
        
        for i in range(n_levels - 1):
            # Get adjacent levels
            coarse = hierarchical_segmentations[i]
            fine = hierarchical_segmentations[i + 1]
            
            # Resize coarse to match fine resolution
            coarse_resized = F.interpolate(
                coarse, 
                size=fine.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # KL divergence loss (coarse to fine)
            loss_c2f = F.kl_div(
                F.log_softmax(coarse_resized, dim=1),
                F.softmax(fine, dim=1),
                reduction='batchmean'
            )
            
            # Resize fine to match coarse resolution
            fine_resized = F.interpolate(
                fine, 
                size=coarse.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # KL divergence loss (fine to coarse)
            loss_f2c = F.kl_div(
                F.log_softmax(fine_resized, dim=1),
                F.softmax(coarse, dim=1),
                reduction='batchmean'
            )
            
            # Average bidirectional consistency
            level_consistency = (loss_c2f + loss_f2c) / 2
            consistency_loss += level_consistency
        
        # Average across all level pairs
        consistency_loss = consistency_loss / (n_levels - 1) if n_levels > 1 else 0.0
        
        return consistency_loss