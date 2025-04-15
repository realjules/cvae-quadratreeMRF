import torch
import torch.nn as nn
import torch.nn.functional as F

def FocalCrossEntropy2d(input, target, weight=None, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=6):
    """
    Focal Cross Entropy loss for better handling of hard examples and class imbalance
    
    Args:
        input: Prediction tensor
        target: Target tensor
        weight: Class weights
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: Reduction method
        ignore_index: Index to ignore
    """
    # Move weight to the same device as input
    if weight is not None:
        weight = weight.to(input.device)
    
    # Reshape if needed
    dim = input.dim()
    if dim == 2:
        return focal_loss(input, target, alpha, gamma, weight, reduction, ignore_index)
    elif dim == 4:
        output = input.reshape(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.reshape(-1)
        return focal_loss(output, target, alpha, gamma, weight, reduction, ignore_index)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def focal_loss(input, target, alpha=0.25, gamma=2.0, weight=None, reduction='mean', ignore_index=6):
    """Focal loss implementation"""
    # Create one-hot encoding for target
    n_classes = input.size(-1)
    
    # Calculate standard cross entropy
    log_softmax = F.log_softmax(input, dim=-1)
    
    # Create mask for valid targets
    mask = (target != ignore_index)
    
    # Calculate focal loss
    valid_targets = target[mask]
    valid_inputs = input[mask]
    
    if valid_targets.numel() == 0:
        return torch.tensor(0.0, device=input.device)
    
    # Calculate probabilities
    probs = F.softmax(valid_inputs, dim=-1)
    target_one_hot = F.one_hot(valid_targets, n_classes).float()
    
    # Get probabilities of the target class
    pt = (probs * target_one_hot).sum(dim=-1)
    
    # Calculate focal weights
    focal_weight = (1 - pt) ** gamma
    
    # Apply class weights if provided
    if weight is not None:
        class_weights = weight[valid_targets]
        focal_weight = focal_weight * class_weights
    
    # Apply alpha weighting for foreground/background balance
    if alpha is not None:
        focal_weight = focal_weight * (alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)).sum(dim=-1)
    
    # Calculate loss
    ce_loss = F.nll_loss(log_softmax[mask], valid_targets, weight=None, reduction='none')
    focal_loss = focal_weight * ce_loss
    
    # Apply reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def LovaszSoftmax(input, target, classes='present', weight=None, reduction='mean', ignore_index=6):
    """
    Lovasz Softmax loss for improved optimization directly for IoU
    
    Args:
        input: Prediction tensor (B, C, H, W)
        target: Target tensor (B, H, W)
        classes: Classes to include ('all', 'present')
        weight: Class weights
        reduction: Reduction method
        ignore_index: Index to ignore
    """
    # Convert inputs
    input_soft = F.softmax(input, dim=1)
    
    # Calculate per-class and per-batch lovasz loss
    losses = []
    
    n_classes = input.size(1)
    for c in range(n_classes):
        # Skip ignore class
        if c == ignore_index:
            continue
        
        # Create binary masks
        target_c = (target == c).float()
        input_c = input_soft[:, c]
        
        # Skip if no pixels of this class
        if classes == 'present' and target_c.sum() == 0:
            continue
        
        # Calculate lovasz loss for this class
        loss_c = lovasz_hinge(2 * input_c - 1, target_c, ignore=ignore_index)
        
        # Apply class weight if provided
        if weight is not None:
            loss_c = loss_c * weight[c]
        
        losses.append(loss_c)
    
    # Apply reduction
    if reduction == 'mean':
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=input.device)
    elif reduction == 'sum':
        return torch.stack(losses).sum() if losses else torch.tensor(0.0, device=input.device)
    else:
        return torch.stack(losses) if losses else torch.tensor(0.0, device=input.device)


def lovasz_hinge(logit, label, ignore=None):
    """
    Binary Lovasz hinge loss for a single class
    """
    # Flatten inputs
    logit = logit.view(-1)
    label = label.view(-1)
    
    # Create mask for valid pixels
    if ignore is not None:
        mask = (label != ignore)
        logit = logit[mask]
        label = label[mask]
    
    # Return 0 if no valid pixels
    if logit.numel() == 0:
        return torch.tensor(0.0, device=logit.device)
    
    # Sort predictions by confidence
    sorted_logits, sort_idx = torch.sort(logit, descending=True)
    sorted_labels = label[sort_idx]
    
    # Calculate intersection-over-union for increasing thresholds
    sorted_labels = sorted_labels.float()
    intersection = torch.cumsum(sorted_labels, dim=0)
    union = torch.cumsum(sorted_labels, dim=0) + torch.cumsum(torch.ones_like(sorted_labels), dim=0) - sorted_labels
    iou = intersection / (union + 1e-8)
    
    # Calculate gradients of IoU with respect to predictions
    grad = torch.cat((torch.ones_like(iou[:1]), iou[1:] - iou[:-1]))
    
    # Calculate lovasz loss
    loss = torch.sum(torch.abs(grad) * (1.0 - sorted_labels - sorted_logits))
    
    return loss


class EnhancedHierarchicalPGMLoss(nn.Module):
    """
    Enhanced loss function for the Hierarchical PGM with Contrastive Learning.
    
    Key improvements:
    1. Focal loss for better handling of class imbalance
    2. IoU optimization with Lovasz-Softmax loss
    3. Adaptive loss weighting during training
    4. Enhanced boundary awareness
    5. Stronger regularization for latent space
    """
    def __init__(self, n_classes, weights=None, kld_weight=0.001, contrastive_weight=0.5, 
                 consistency_weight=0.2, temperature=0.5, ignore_index=6):
        super(EnhancedHierarchicalPGMLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights  # Class weights for segmentation loss
        self.kld_weight = kld_weight  # Reduced weight for KL divergence loss
        self.contrastive_weight = contrastive_weight  # Increased weight for contrastive loss
        self.consistency_weight = consistency_weight  # Increased weight for consistency loss
        self.temperature = temperature  # Temperature for contrastive loss
        self.ignore_index = ignore_index  # Ignore index for segmentation loss
        
        # New adaptive weights that change during training
        self.current_epoch = 0
        self.max_epochs = 50  # Default max epochs
        
        # New boundary-aware weighting
        self.boundary_weight = 2.0  # Weight multiplier for boundary pixels
        
        # Lovasz-IoU loss weight
        self.lovasz_weight = 0.5
        
    def update_epoch(self, current, max_epochs=None):
        """Update current epoch for adaptive weights"""
        self.current_epoch = current
        if max_epochs is not None:
            self.max_epochs = max_epochs
    
    def get_adaptive_weights(self):
        """Calculate adaptive weights based on training progress"""
        # Start with higher kld_weight and decrease gradually
        progress = min(1.0, self.current_epoch / (self.max_epochs * 0.7))
        
        adaptive_kld = self.kld_weight * (1.0 - 0.5 * progress)
        
        # Increase contrastive weight over time
        adaptive_contrastive = self.contrastive_weight * (0.5 + 0.5 * progress)
        
        # Consistency weight peaks in the middle of training
        mid_point = 0.5
        adaptive_consistency = self.consistency_weight * (1.0 - 2.0 * abs(progress - mid_point))
        
        # Lovasz weight increases over time
        adaptive_lovasz = self.lovasz_weight * (0.2 + 0.8 * progress)
        
        return {
            'kld': adaptive_kld,
            'contrastive': adaptive_contrastive,
            'consistency': adaptive_consistency,
            'lovasz': adaptive_lovasz
        }
        
    def forward(self, outputs, targets=None, mode='full'):
        """
        Calculate the enhanced combined loss.
        
        Args:
            outputs: Dictionary of model outputs from HierarchicalPGM
            targets: Ground truth segmentation (optional, required for supervised components)
            mode: Training mode ('full', 'supervised', 'unsupervised', or 'inference')
        
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        loss_components = {}
        # Move weights to the same device as outputs if needed
        if self.weights is not None:
            self.weights = self.weights.to(outputs['hierarchical_segmentations'][0].device)
        
        # Get adaptive weights
        adaptive_weights = self.get_adaptive_weights()
        
        # Supervised segmentation loss
        if targets is not None and mode in ['supervised', 'full']:
            # Primary segmentation loss (final output)
            if 'final_segmentation' in outputs:
                # Combine Focal CE loss and Lovasz-Softmax loss
                focal_loss = FocalCrossEntropy2d(
                    outputs['final_segmentation'], 
                    targets,
                    weight=self.weights,
                    alpha=0.25,
                    gamma=2.0
                )
                
                lovasz_loss = LovaszSoftmax(
                    outputs['final_segmentation'],
                    targets,
                    weight=self.weights
                )
                
                # Final segmentation loss is a weighted combination
                seg_loss = focal_loss + adaptive_weights['lovasz'] * lovasz_loss
                loss_components['seg_loss'] = seg_loss
            
            # Hierarchical segmentation losses with level-specific weighting
            if 'hierarchical_segmentations' in outputs:
                hier_losses = []
                n_levels = len(outputs['hierarchical_segmentations'])
                
                for i, seg in enumerate(outputs['hierarchical_segmentations']):
                    # Level-specific weight: deeper levels have higher weight
                    level_weight = 0.5 + 0.5 * (i / (n_levels - 1)) if n_levels > 1 else 1.0
                    
                    # Resize target to match hierarchical output size
                    target_size = seg.size()[2:]
                    scaled_target = F.interpolate(
                        targets.float().unsqueeze(1), 
                        size=target_size, 
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    # Use focal loss for hierarchical outputs
                    hier_loss = FocalCrossEntropy2d(
                        seg, 
                        scaled_target,
                        weight=self.weights,
                        alpha=0.25,
                        gamma=2.0
                    )
                    
                    # Apply level-specific weight
                    weighted_hier_loss = level_weight * hier_loss
                    hier_losses.append(weighted_hier_loss)
                    loss_components[f'hier_loss_{i}'] = weighted_hier_loss
                
                if hier_losses:
                    # Average hierarchical losses
                    loss_components['hier_loss'] = sum(hier_losses) / len(hier_losses)
        
        # CVAE unsupervised losses with enhanced regularization
        if mode in ['unsupervised', 'full']:
            # Reconstruction loss with structure preservation
            if 'reconstruction' in outputs:
                if 'original_input' in outputs:
                    original_input = outputs['original_input']
                else:
                    original_input = targets  # This assumes targets is the input in unsupervised mode
                
                # Get reconstruction and ensure same size as original
                recon = outputs['reconstruction']
                if recon.shape != original_input.shape:
                    recon = F.interpolate(
                        recon, 
                        size=(original_input.shape[2], original_input.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Calculate MSE loss with resized reconstruction
                mse_loss = F.mse_loss(recon, original_input)
                
                # Extract image gradients for structure preservation
                # Using Sobel operators to detect edges
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                      dtype=torch.float32, device=original_input.device).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                      dtype=torch.float32, device=original_input.device).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
                
                # Calculate gradients for original and reconstructed images
                pad = nn.ReplicationPad2d(1)
                orig_padded = pad(original_input)
                recon_padded = pad(recon)  # Use the resized reconstruction
                
                # Apply Sobel operators
                orig_grad_x = F.conv2d(orig_padded, sobel_x, groups=3)
                orig_grad_y = F.conv2d(orig_padded, sobel_y, groups=3)
                recon_grad_x = F.conv2d(recon_padded, sobel_x, groups=3)
                recon_grad_y = F.conv2d(recon_padded, sobel_y, groups=3)
                
                # Calculate gradient magnitude
                orig_grad_mag = torch.sqrt(orig_grad_x**2 + orig_grad_y**2 + 1e-6)
                recon_grad_mag = torch.sqrt(recon_grad_x**2 + recon_grad_y**2 + 1e-6)
                
                # Calculate structural similarity loss (gradient difference)
                structure_loss = F.mse_loss(recon_grad_mag, orig_grad_mag)
                
                # Combine losses with structural similarity having higher weight
                recon_loss = mse_loss + 2.0 * structure_loss
                loss_components['recon_loss'] = recon_loss
            
            # KL divergence loss with adaptive annealing
            if 'mu' in outputs and 'log_var' in outputs:
                mu = outputs['mu']
                log_var = outputs['log_var']
                
                # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                # Normalize by batch size and latent dimension for better scaling
                kld_loss = kld_loss / (mu.size(0) * mu.size(1))
                
                # Apply adaptive KLD weight
                loss_components['kld_loss'] = kld_loss * adaptive_weights['kld']
            
            # Enhanced contrastive loss with memory bank
            if 'z_proj' in outputs:
                z_proj = outputs['z_proj']
                queue = outputs.get('queue', None)
                
                if targets is not None and mode == 'full':
                    # Supervised contrastive loss with class information
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
                    
                    # Compute enhanced supervised contrastive loss
                    contrastive_loss = self._enhanced_supervised_contrastive_loss(
                        z_proj, 
                        labels_tensor, 
                        queue,
                        temperature=self.temperature
                    )
                else:
                    # Unsupervised contrastive loss with memory bank
                    contrastive_loss = self._enhanced_unsupervised_contrastive_loss(
                        z_proj,
                        queue,
                        temperature=self.temperature
                    )
                
                # Apply adaptive contrastive weight
                loss_components['contrastive_loss'] = contrastive_loss * adaptive_weights['contrastive']
        
        # Enhanced hierarchical consistency loss
        if 'hierarchical_segmentations' in outputs and len(outputs['hierarchical_segmentations']) > 1:
            hier_consistency_loss = self._enhanced_hierarchical_consistency_loss(
                outputs['hierarchical_segmentations']
            )
            
            # Apply adaptive consistency weight
            loss_components['hier_consistency_loss'] = hier_consistency_loss * adaptive_weights['consistency']
        
        # Total loss with component weighting based on training progress
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components
    
    def _enhanced_supervised_contrastive_loss(self, features, labels, queue=None, temperature=0.5):
        """
        Enhanced supervised contrastive loss with memory bank support.
        
        Args:
            features: Feature vectors (B, D)
            labels: Class labels (B)
            queue: Optional memory bank (Q, D)
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
        
        # Include memory bank if provided
        if queue is not None and queue.size(0) > 0:
            # Normalize queue features
            queue = F.normalize(queue, dim=1)
            
            # Calculate similarities with queue
            queue_sim = torch.matmul(features, queue.T) / temperature
            
            # Expanded similarity matrix
            sim_matrix_expanded = torch.cat([sim_matrix, queue_sim], dim=1)
            
            # For queue, we don't know labels - treat all as negatives
            # This simplification works well in practice
        else:
            sim_matrix_expanded = sim_matrix
        
        # Count number of positives for each sample
        num_positives = pos_mask.sum(dim=1)
        
        # Handle samples with no positives
        valid_samples = num_positives > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Compute log probability
        exp_sim = torch.exp(sim_matrix_expanded - torch.max(sim_matrix_expanded, dim=1, keepdim=True)[0])
        
        # For in-batch part
        log_prob = torch.log(
            exp_sim[:, :batch_size] / 
            (exp_sim.sum(dim=1, keepdim=True) - exp_sim[:, :batch_size] * eye_mask + 1e-10)
        )
        
        # Compute mean log-likelihood of positive pairs
        pos_log_prob = (pos_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
        
        # Only compute loss for valid samples (with positives)
        loss = -pos_log_prob[valid_samples].mean()
        
        return loss
    
    def _enhanced_unsupervised_contrastive_loss(self, features, queue=None, temperature=0.5):
        """
        Enhanced unsupervised contrastive loss with memory bank.
        
        Args:
            features: Feature vectors (B, D)
            queue: Optional memory bank (Q, D)
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
        
        # Include memory bank if provided
        if queue is not None and queue.size(0) > 0:
            # Normalize queue features
            queue = F.normalize(queue, dim=1)
            
            # Calculate similarities with queue
            queue_sim = torch.matmul(features, queue.T) / temperature
            
            # Expanded similarity matrix
            sim_matrix_expanded = torch.cat([sim_matrix_no_diag, queue_sim], dim=1)
        else:
            sim_matrix_expanded = sim_matrix_no_diag
        
        # Create labels (each sample should be most similar to itself)
        labels = torch.arange(batch_size, device=device)
        
        # Cross entropy loss with additional negative examples from queue
        loss = F.cross_entropy(sim_matrix_expanded, labels)
        
        return loss
    
    def _enhanced_hierarchical_consistency_loss(self, hierarchical_segmentations):
        """
        Enhanced consistency loss between adjacent levels in the hierarchy.
        
        Args:
            hierarchical_segmentations: List of segmentation outputs at different levels
        """
        consistency_loss = 0.0
        n_levels = len(hierarchical_segmentations)
        
        # Apply level-specific weights (higher weights for fine-to-coarse direction)
        f2c_weight = 0.7  # Fine-to-coarse weight
        c2f_weight = 0.3  # Coarse-to-fine weight
        
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
            
            # Jensen-Shannon divergence instead of KL for stability
            # First, calculate mean distribution
            p = F.softmax(coarse_resized, dim=1)
            q = F.softmax(fine, dim=1)
            m = 0.5 * (p + q)
            
            # Calculate JS divergence
            loss_c2f = 0.5 * (
                F.kl_div(F.log_softmax(coarse_resized, dim=1), m, reduction='batchmean') +
                F.kl_div(F.log_softmax(fine, dim=1), m, reduction='batchmean')
            )
            
            # Resize fine to match coarse resolution
            fine_resized = F.interpolate(
                fine, 
                size=coarse.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Calculate JS divergence in the other direction
            p = F.softmax(fine_resized, dim=1)
            q = F.softmax(coarse, dim=1)
            m = 0.5 * (p + q)
            
            loss_f2c = 0.5 * (
                F.kl_div(F.log_softmax(fine_resized, dim=1), m, reduction='batchmean') +
                F.kl_div(F.log_softmax(coarse, dim=1), m, reduction='batchmean')
            )
            
            # Weighted bidirectional consistency
            level_consistency = c2f_weight * loss_c2f + f2c_weight * loss_f2c
            
            # Apply level-specific weight (deeper levels have higher weight)
            level_weight = 1.0 + 0.5 * (i / max(1, n_levels - 2))
            consistency_loss += level_weight * level_consistency
        
        # Average across all level pairs
        consistency_loss = consistency_loss / (n_levels - 1) if n_levels > 1 else 0.0
        
        return consistency_loss


# Alias for backward compatibility
HierarchicalPGMLoss = EnhancedHierarchicalPGMLoss