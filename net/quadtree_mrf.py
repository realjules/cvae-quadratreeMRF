import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuadtreeNode:
    """
    A node in the Quadtree structure representing a region of the image.
    Each node can be split into four children (quadrants).
    """
    def __init__(self, x, y, size, depth, max_depth):
        self.x = x          # Top-left x coordinate
        self.y = y          # Top-left y coordinate
        self.size = size    # Width/height of the region
        self.depth = depth  # Current depth in the tree
        self.max_depth = max_depth
        self.children = []  # Child nodes (quadrants)
        self.leaf = True    # Whether this is a leaf node
        self.label = None   # Class label for this region
        self.confidence = None  # Confidence score for the label
        self.features = None  # Feature vector for this region
        
    def split(self):
        """Split the current node into four quadrants"""
        if self.depth >= self.max_depth or not self.leaf:
            return False
        
        # Calculate half size for children
        half_size = self.size // 2
        
        # Create four child nodes (quadrants)
        # Top-left
        tl = QuadtreeNode(self.x, self.y, half_size, self.depth + 1, self.max_depth)
        # Top-right
        tr = QuadtreeNode(self.x + half_size, self.y, half_size, self.depth + 1, self.max_depth)
        # Bottom-left
        bl = QuadtreeNode(self.x, self.y + half_size, half_size, self.depth + 1, self.max_depth)
        # Bottom-right
        br = QuadtreeNode(self.x + half_size, self.y + half_size, half_size, self.depth + 1, self.max_depth)
        
        self.children = [tl, tr, bl, br]
        self.leaf = False
        return True
    
    def get_leaf_nodes(self):
        """Return all leaf nodes in the subtree rooted at this node"""
        if self.leaf:
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves


class QuadtreeMRF(nn.Module):
    """
    Quadtree-based Markov Random Field for hierarchical segmentation
    of remote sensing images.
    
    This implementation includes:
    - Quadtree construction based on image content
    - Belief propagation for label inference
    - Integration with latent features from CVAE
    """
    def __init__(self, n_classes=6, quadtree_depth=4, device="cuda"):
        super(QuadtreeMRF, self).__init__()
        self.n_classes = n_classes
        self.max_depth = quadtree_depth
        self.device = device
        
        # Pairwise potential parameters (learned)
        self.pairwise_weights = nn.Parameter(torch.ones(n_classes, n_classes))
        
        # Unary potential parameters (for feature integration)
        self.unary_projection = nn.Sequential(
            nn.Linear(256, 128),  # Adjust input size based on feature dimensions
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        
        # Edge potential parameters for parent-child relationships
        self.vertical_weights = nn.Parameter(torch.ones(n_classes, n_classes))
        
        # Number of belief propagation iterations
        self.bp_iterations = 5
        
    def build_quadtree(self, features, initial_segmentation=None):
        """
        Build a quadtree from input features and optional initial segmentation
        
        Args:
            features: Feature maps [B, C, H, W]
            initial_segmentation: Initial segmentation map [B, H, W]
        
        Returns:
            List of root nodes for each sample in the batch
        """
        batch_size, _, height, width = features.size()
        trees = []
        
        for b in range(batch_size):
            # Create root node covering the entire image
            root = QuadtreeNode(0, 0, width, 0, self.max_depth)
            
            # Build the tree recursively
            self._build_recursive(root, features[b], initial_segmentation[b] if initial_segmentation is not None else None)
            
            trees.append(root)
            
        return trees
    
    def _build_recursive(self, node, features, segmentation=None):
        """
        Recursively build the quadtree by splitting nodes based on feature homogeneity
        
        Args:
            node: Current QuadtreeNode
            features: Feature tensor [C, H, W]
            segmentation: Optional segmentation map [H, W]
        """
        # Extract features for the current node region
        region_features = features[:, node.y:node.y+node.size, node.x:node.x+node.size]
        
        # Calculate feature variance to determine if splitting is needed
        variance = torch.var(region_features.reshape(region_features.size(0), -1), dim=1).mean()
        
        # Set node features (mean pooled)
        node.features = torch.mean(region_features.reshape(region_features.size(0), -1), dim=1)
        
        # If segmentation is provided, calculate majority label
        if segmentation is not None:
            region_seg = segmentation[node.y:node.y+node.size, node.x:node.x+node.size]
            # Find most common label (excluding ignore index)
            labels, counts = torch.unique(region_seg[region_seg != 255], return_counts=True)
            if labels.size(0) > 0:
                node.label = labels[torch.argmax(counts)]
                node.confidence = torch.max(counts).float() / (region_seg != 255).sum().float()
            else:
                node.label = 0
                node.confidence = 0.0
        
        # Determine if the node should be split
        # Split if: Not at max depth, high variance, and minimum size
        should_split = (node.depth < node.max_depth and 
                       variance > 0.01 and 
                       node.size > 8)
        
        if should_split:
            node.split()
            # Recursively build for children
            for child in node.children:
                self._build_recursive(child, features, segmentation)
    
    def compute_unary_potentials(self, trees, latent_features):
        """
        Compute unary potentials for all nodes in the quadtree
        
        Args:
            trees: List of quadtree root nodes
            latent_features: Features from CVAE [B, C, H, W]
        """
        batch_size = len(trees)
        
        for b in range(batch_size):
            root = trees[b]
            features = latent_features[b]
            
            # Process all leaf nodes
            leaves = root.get_leaf_nodes()
            
            for leaf in leaves:
                # Extract features for this region
                region_features = features[:, leaf.y:leaf.y+leaf.size, leaf.x:leaf.x+leaf.size]
                # Pool features to get a single vector
                pooled_features = F.adaptive_avg_pool2d(
                    region_features.unsqueeze(0), 
                    (1, 1)
                ).squeeze()
                
                # Project features to class scores
                leaf.unary_potentials = self.unary_projection(pooled_features).squeeze()
    
    def compute_pairwise_potentials(self, trees):
        """
        Compute pairwise potentials between neighboring nodes
        
        Args:
            trees: List of quadtree root nodes
        """
        batch_size = len(trees)
        
        for b in range(batch_size):
            root = trees[b]
            
            # Get all leaf nodes
            leaves = root.get_leaf_nodes()
            
            # For each leaf, compute potentials with its spatial neighbors
            for i, leaf in enumerate(leaves):
                leaf.neighbors = []
                leaf.neighbor_potentials = []
                
                # Check all other leaves for adjacency
                for j, other in enumerate(leaves):
                    if i == j:
                        continue
                    
                    # Check if leaves are adjacent
                    is_adjacent = (
                        (leaf.x + leaf.size == other.x and 
                         (other.y < leaf.y + leaf.size and leaf.y < other.y + other.size)) or
                        (other.x + other.size == leaf.x and 
                         (other.y < leaf.y + leaf.size and leaf.y < other.y + other.size)) or
                        (leaf.y + leaf.size == other.y and 
                         (other.x < leaf.x + leaf.size and leaf.x < other.x + other.size)) or
                        (other.y + other.size == leaf.y and 
                         (other.x < leaf.x + leaf.size and leaf.x < other.x + other.size))
                    )
                    
                    if is_adjacent:
                        leaf.neighbors.append(other)
                        # Use the learned pairwise potential matrix
                        leaf.neighbor_potentials.append(self.pairwise_weights)
    
    def belief_propagation(self, trees, n_iterations=5):
        """
        Run belief propagation to infer optimal labels
        
        Args:
            trees: List of quadtree root nodes
            n_iterations: Number of belief propagation iterations
            
        Returns:
            Predicted segmentation maps
        """
        batch_size = len(trees)
        height = width = trees[0].size  # Assuming square images
        
        # Initialize beliefs for all leaf nodes
        for tree in trees:
            leaves = tree.get_leaf_nodes()
            for leaf in leaves:
                if not hasattr(leaf, 'unary_potentials'):
                    # Initialize with uniform distribution if no unary potentials
                    leaf.beliefs = torch.ones(self.n_classes, device=self.device) / self.n_classes
                else:
                    # Initialize with unary potentials
                    leaf.beliefs = F.softmax(leaf.unary_potentials, dim=0)
        
        # Run belief propagation for n_iterations
        for _ in range(n_iterations):
            # For each tree
            for tree in trees:
                leaves = tree.get_leaf_nodes()
                
                # Compute messages from each node to its neighbors
                for leaf in leaves:
                    leaf.messages = {}
                    
                    # For each neighbor
                    for neighbor in leaf.neighbors:
                        # Compute message from leaf to neighbor
                        msg = leaf.beliefs.clone()
                        
                        # Multiply by messages from other neighbors
                        for other in leaf.neighbors:
                            if other != neighbor and other in leaf.messages:
                                msg *= leaf.messages[other]
                        
                        # Apply pairwise potential
                        msg = torch.matmul(self.pairwise_weights, msg)
                        
                        # Normalize message
                        msg = msg / msg.sum()
                        
                        # Store message
                        leaf.messages[neighbor] = msg
                
                # Update beliefs
                for leaf in leaves:
                    # Start with unary potentials
                    if hasattr(leaf, 'unary_potentials'):
                        log_belief = leaf.unary_potentials.clone()
                    else:
                        log_belief = torch.zeros(self.n_classes, device=self.device)
                    
                    # Add messages from neighbors
                    for neighbor in leaf.neighbors:
                        if neighbor in leaf.messages:
                            log_belief += torch.log(leaf.messages[neighbor] + 1e-10)
                    
                    # Normalize beliefs
                    leaf.beliefs = F.softmax(log_belief, dim=0)
        
        # Create output segmentation maps
        segmentations = torch.zeros(batch_size, height, width, device=self.device, dtype=torch.long)
        
        # Fill segmentation maps based on leaf node predictions
        for b, tree in enumerate(trees):
            leaves = tree.get_leaf_nodes()
            for leaf in leaves:
                # Get predicted class (maximum belief)
                pred_class = torch.argmax(leaf.beliefs)
                
                # Fill the region with the predicted class
                segmentations[b, leaf.y:leaf.y+leaf.size, leaf.x:leaf.x+leaf.size] = pred_class
        
        return segmentations
    
    def forward(self, features, cvae_latent, initial_segmentation=None):
        """
        Forward pass through the Quadtree MRF
        
        Args:
            features: Feature maps from base network [B, C, H, W]
            cvae_latent: Latent features from CVAE [B, C, H, W]
            initial_segmentation: Optional initial segmentation [B, H, W]
            
        Returns:
            Refined segmentation maps [B, H, W]
        """
        # Build the quadtree structure
        trees = self.build_quadtree(features, initial_segmentation)
        
        # Compute unary potentials for leaf nodes
        self.compute_unary_potentials(trees, cvae_latent)
        
        # Compute pairwise potentials between neighboring nodes
        self.compute_pairwise_potentials(trees)
        
        # Run belief propagation to infer final labels
        refined_segmentation = self.belief_propagation(trees, self.bp_iterations)
        
        return refined_segmentation