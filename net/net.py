import torch
import torch.nn as nn
import torch.nn.functional as F
from net.cvae import CVAE
from net.quadtree_mrf import QuadtreeMRF

class HierarchicalPGM(nn.Module):
    """
    Semi-Supervised Hierarchical PGM with Contrastive Learning
    
    This model implements the approach described in the research proposal,
    integrating a CVAE for contrastive learning with a quadtree-based MRF
    for hierarchical spatial modeling.
    """
    def __init__(self, n_channels=3, n_classes=6, latent_dim=128, max_depth=4):
        super(HierarchicalPGM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.max_depth = max_depth
        
        # CVAE module for contrastive learning
        self.cvae = CVAE(input_channels=n_channels, latent_dim=latent_dim)
        
        # Feature extractor (encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Segmentation heads at different levels
        self.segmentation_heads = nn.ModuleList([
            nn.Conv2d(512, n_classes, kernel_size=1),
            nn.Conv2d(256, n_classes, kernel_size=1),
            nn.Conv2d(128, n_classes, kernel_size=1),
            nn.Conv2d(64, n_classes, kernel_size=1)
        ])
        
        # QuadtreeMRF for hierarchical spatial modeling
        self.quadtree_mrf = QuadtreeMRF(num_classes=n_classes, max_depth=max_depth, input_dim=latent_dim)
        
        # Fusion module to combine CVAE latent and encoder features
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_dim + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement layer
        self.refinement = nn.Sequential(
            nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_classes, n_classes, kernel_size=1)
        )
        
        # Decoder upsampling path for multi-level features
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, x, mode='full'):
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor
            mode: Operation mode ('full', 'supervised', 'unsupervised', or 'inference')
        """
        batch_size, _, height, width = x.shape
        results = {}
        
        # Get CVAE outputs for contrastive learning
        if mode in ['full', 'unsupervised']:
            cvae_outputs = self.cvae(x)
            results.update(cvae_outputs)
        
        # Encode input using the main encoder
        encoded_features = self.encoder(x)
        
        # Multi-scale feature extraction for hierarchical segmentation
        features = [encoded_features]
        
        # Generate multi-scale features through decoder blocks
        current_features = encoded_features
        for decoder_block in self.decoder_blocks:
            current_features = decoder_block(current_features)
            features.append(current_features)
        
        # Apply segmentation heads to each level
        hierarchical_segmentations = []
        for i, feat in enumerate(features):
            seg = self.segmentation_heads[i](feat)
            hierarchical_segmentations.append(seg)
        
        results['hierarchical_segmentations'] = hierarchical_segmentations
        
        # For supervised and full modes, apply the QuadtreeMRF
        if mode in ['full', 'supervised', 'inference']:
            # Prepare features for QuadtreeMRF
            if mode == 'full' and 'z' in results:
                # Reshape latent vector to spatial features
                z = results['z']
                z_spatial = z.view(batch_size, -1, 1, 1).expand(-1, -1, encoded_features.size(2), encoded_features.size(3))
                
                # Fuse latent features with encoded features
                fused_features = torch.cat([encoded_features, z_spatial], dim=1)
                fused_features = self.fusion(fused_features)
            else:
                fused_features = encoded_features
            
            # Apply QuadtreeMRF
            quadtree_segmentation = self.quadtree_mrf(fused_features, (height, width))
            results['quadtree_segmentation'] = quadtree_segmentation
            
            # Final refinement integrating hierarchical segmentations and quadtree output
            # Upsample the final hierarchical segmentation to original image size
            final_hier_seg = F.interpolate(
                hierarchical_segmentations[-1],
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            
            # Combine with quadtree output for final prediction
            final_segmentation = self.refinement(
                0.5 * final_hier_seg + 0.5 * quadtree_segmentation
            )
            
            results['final_segmentation'] = final_segmentation
        
        return results