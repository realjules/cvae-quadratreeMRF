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
        self.quadtree_mrf = QuadtreeMRF(n_classes=n_classes, quadtree_depth=max_depth, device="cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # Store original input for reconstruction loss
        results['original_input'] = x
        
        # Get CVAE outputs for contrastive learning if needed
        cvae_latent = None
        if mode in ['full', 'unsupervised']:
            cvae_outputs = self.cvae(x)
            results.update(cvae_outputs)
            cvae_latent = cvae_outputs['z']  # Extract latent representation
        
        # Encode input using the main encoder
        encoded_features = self.encoder(x)
        
        # Fuse CVAE latent with encoder features if available
        if cvae_latent is not None:
            # Reshape latent to match spatial dimensions
            latent_spatial = cvae_latent.unsqueeze(-1).unsqueeze(-1)
            latent_spatial = latent_spatial.expand(-1, -1, encoded_features.size(2), encoded_features.size(3))
            
            # Concatenate along channel dimension
            fused_features = torch.cat([encoded_features, latent_spatial], dim=1)
            
            # Apply fusion module
            encoded_features = self.fusion(fused_features)
        
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
        
        # Use QuadtreeMRF for final refinement
        if mode in ['full', 'supervised', 'inference']:
            # Get initial segmentation from hierarchical output
            initial_seg = hierarchical_segmentations[-1]
            initial_seg_upsampled = F.interpolate(
                initial_seg,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            
            # Process through QuadtreeMRF
            if cvae_latent is not None:
                # Apply QuadtreeMRF with latent features
                quadtree_output = self.quadtree_mrf(
                    features=features[-1],  # Use the finest level features
                    cvae_latent=cvae_latent,
                    initial_segmentation=torch.argmax(initial_seg, dim=1)
                )
                
                # Convert to one-hot for fusion with hierarchical output
                quadtree_one_hot = F.one_hot(quadtree_output, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
                
                # Combine QuadtreeMRF output with hierarchical segmentation
                combined_seg = (initial_seg_upsampled + quadtree_one_hot) / 2.0
                
                # Apply refinement
                final_segmentation = self.refinement(combined_seg)
            else:
                # If no CVAE latent, just use hierarchical segmentation with refinement
                final_segmentation = self.refinement(initial_seg_upsampled)
            
            # Store the result
            results['final_segmentation'] = final_segmentation
        
        return results