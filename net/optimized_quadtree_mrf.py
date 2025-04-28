import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class VectorizedQuadtreeNode:
    """
    A node in the vectorized Quadtree structure representing a region of the image.
    Optimized for batch operations and reduced Python overhead.
    """
    def __init__(self, batch_idx, x, y, size, depth, max_depth, node_idx=None):
        self.batch_idx = batch_idx  # Batch index
        self.x = x                  # Top-left x coordinate
        self.y = y                  # Top-left y coordinate
        self.size = size            # Width/height of the region
        self.depth = depth          # Current depth in the tree
        self.max_depth = max_depth  # Maximum allowed depth
        self.node_idx = node_idx    # Unique node index within the tree
        self.children = []          # Child nodes (quadrants)
        self.leaf = True            # Whether this is a leaf node
        self.label = None           # Class label for this region
        self.confidence = None      # Confidence score for the label
        self.features = None        # Feature tensor for this region
        self.neighbors = []         # Neighboring nodes
        self.region_bounds = (x, y, x + size, y + size)  # Region bounds for fast overlap checking


class OptimizedQuadtreeMRF(nn.Module):
    """
    Highly optimized Quadtree-based Markov Random Field for hierarchical segmentation
    of remote sensing images.
    
    Key improvements:
    - Vectorized tree construction and operations
    - Batched feature processing
    - Optimized belief propagation
    - GPU-accelerated adjacency computation
    - Memory-efficient message passing
    - Early stopping and pruning strategies
    """
    def __init__(self, n_classes=6, quadtree_depth=3, feature_dim=256, max_nodes=2000, 
                 device="cuda", batch_processing=True):
        super(OptimizedQuadtreeMRF, self).__init__()
        self.n_classes = n_classes
        self.max_depth = quadtree_depth
        self.device = device
        self.max_nodes = max_nodes  # Increased limit for larger trees
        self.batch_processing = batch_processing  # Enable batched operations
        
        # Pairwise potential parameters (learned)
        # Initialize with smoother values encouraging spatial consistency
        self.pairwise_weights = nn.Parameter(
            torch.eye(n_classes) * 0.8 + 0.2 / n_classes
        )
        
        # Feature dimensionality reduction
        self.feature_dim = feature_dim
        
        # Improved encoder with residual connections
        self.feature_encoder = nn.Sequential(
            # First block with residual connection
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.BatchNorm1d(feature_dim // 2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(feature_dim // 2, feature_dim // 2),
                nn.BatchNorm1d(feature_dim // 2),
                nn.LeakyReLU(inplace=True)
            ),
            # Second block with classifier head
            nn.Sequential(
                nn.Linear(feature_dim // 2, feature_dim // 4),
                nn.BatchNorm1d(feature_dim // 4),
                nn.LeakyReLU(inplace=True),
                nn.Linear(feature_dim // 4, n_classes)
            )
        )
        
        # Pre-defined feature projections for common dimensions
        projection_dims = [32, 64, 128, 256, 512, 1024]
        self.projections = nn.ModuleDict({
            str(dim): nn.Sequential(
                nn.Linear(dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.LeakyReLU(inplace=True)
            ) for dim in projection_dims if dim != feature_dim
        })
        
        # Edge potentials for hierarchical relationships
        self.vertical_weights = nn.Parameter(torch.eye(n_classes) * 0.9 + 0.1 / n_classes)
        
        # Belief propagation parameters
        self.bp_iterations = 5  # Increased for better convergence
        self.bp_convergence_threshold = 0.001  # Early stopping criterion
        
        # Fallback CNN for robust degradation
        self.fallback_cnn = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )
        
        # Initialize node counter
        self.node_counter = 0
    
    def _reset_node_counter(self):
        """Reset node counter between batches"""
        self.node_counter = 0
    
    def vectorized_quadtree_construction(self, features, initial_segmentation=None):
        """
        Efficiently construct a vectorized quadtree using batched operations
        and tensor-based computations to minimize Python overhead.
        
        Args:
            features: Feature maps [B, C, H, W]
            initial_segmentation: Initial segmentation map [B, H, W]
            
        Returns:
            List of root nodes and a dictionary of all nodes
        """
        batch_size, n_features, height, width = features.shape
        all_trees = []
        all_nodes_dict = {}  # Global node lookup by (batch_idx, node_idx)
        
        # Process each batch item
        for b in range(batch_size):
            self._reset_node_counter()
            
            # Create root node covering the entire image
            root = VectorizedQuadtreeNode(b, 0, 0, width, 0, self.max_depth, node_idx=self.node_counter)
            self.node_counter += 1
            
            # Extract features for the root (entire image)
            root_features = F.adaptive_max_pool2d(
                features[b:b+1], (1, 1)
            ).squeeze()
            
            # Store features in root node
            root.features = root_features
            
            # Queue for nodes to process
            nodes_to_process = [root]
            all_nodes = [root]
            node_count = 1
            
            # Process nodes in breadth-first order (more GPU-friendly)
            while nodes_to_process and node_count < self.max_nodes:
                current_node = nodes_to_process.pop(0)
                
                # Determine whether to split based on optimized criteria
                should_split = self._should_split(
                    current_node, 
                    features[b], 
                    initial_segmentation[b] if initial_segmentation is not None else None
                )
                
                if should_split and node_count + 4 <= self.max_nodes:
                    # Split the node and create children
                    children = self._create_children(current_node, features[b])
                    current_node.children = children
                    current_node.leaf = False
                    
                    # Add children to processing queue and node list
                    nodes_to_process.extend(children)
                    all_nodes.extend(children)
                    node_count += 4
            
            # Store the tree and all its nodes
            all_trees.append(root)
            for node in all_nodes:
                all_nodes_dict[(b, node.node_idx)] = node
            
            # Extract ground truth labels for leaf nodes if segmentation is provided
            if initial_segmentation is not None:
                self._extract_labels_batch(all_nodes, initial_segmentation[b])
        
        return all_trees, all_nodes_dict
    
    def _create_children(self, parent, features):
        """
        Create the four children of a node with optimized feature extraction
        
        Args:
            parent: Parent node
            features: Feature tensor for this batch item [C, H, W]
            
        Returns:
            List of four child nodes
        """
        half_size = parent.size // 2
        children = []
        
        # Child positions (top-left, top-right, bottom-left, bottom-right)
        child_coords = [
            (parent.x, parent.y),
            (parent.x + half_size, parent.y),
            (parent.x, parent.y + half_size),
            (parent.x + half_size, parent.y + half_size)
        ]
        
        # Create all four children and extract their features in one operation
        for i, (x, y) in enumerate(child_coords):
            # Create child node
            child = VectorizedQuadtreeNode(
                parent.batch_idx, x, y, half_size, parent.depth + 1, 
                parent.max_depth, node_idx=self.node_counter
            )
            self.node_counter += 1
            
            # Extract region features efficiently
            region_features = features[:, y:y+half_size, x:x+half_size]
            child.features = F.adaptive_max_pool2d(
                region_features.unsqueeze(0), (1, 1)
            ).squeeze()
            
            children.append(child)
        
        return children
    
    def _should_split(self, node, features, segmentation=None):
        """
        Determine whether a node should be split using optimized criteria
        that balance feature variance, boundary complexity, and depth.
        
        Args:
            node: Current node
            features: Feature tensor [C, H, W]
            segmentation: Optional segmentation map [H, W]
            
        Returns:
            Boolean indicating whether to split
        """
        # Don't split if we've reached max depth or node is too small
        if node.depth >= node.max_depth or node.size < 8:
            return False
        
        # Extract region features
        region = features[:, node.y:node.y+node.size, node.x:node.x+node.size]
        
        # Calculate feature variance (faster implementation)
        # Reshape to (C, pixels) and compute variance along pixel dimension
        region_flat = region.reshape(region.size(0), -1)
        if region_flat.size(1) > 0:  # Ensure there are pixels to calculate variance
            variance = torch.var(region_flat, dim=1).mean().item()
        else:
            variance = 0
        
        # Class heterogeneity check if segmentation is available
        class_diversity = 0
        if segmentation is not None:
            seg_region = segmentation[node.y:node.y+node.size, node.x:node.x+node.size]
            # Count unique classes (excluding ignore index)
            mask = (seg_region != 255)
            if mask.sum() > 0:
                unique_classes = torch.unique(seg_region[mask])
                class_diversity = len(unique_classes) / self.n_classes
        
        # Compute splitting score with improved weighting
        # - More emphasis on feature variance at lower depths
        # - More emphasis on class diversity at higher depths
        variance_weight = max(0.3, 0.7 - (0.1 * node.depth))
        depth_factor = 1.0 - (node.depth / node.max_depth)
        size_factor = min(1.0, node.size / 64)
        
        # Combined score
        splitting_score = (
            variance * variance_weight +
            class_diversity * (1 - variance_weight) * 0.7 +
            depth_factor * 0.2 +
            size_factor * 0.1
        )
        
        # Adaptive threshold based on depth
        threshold = 0.2 + (0.1 * node.depth / node.max_depth)
        
        return splitting_score > threshold
    
    def _extract_labels_batch(self, nodes, segmentation):
        """
        Extract ground truth labels for all leaf nodes in a batch-efficient manner
        
        Args:
            nodes: List of all nodes in the tree
            segmentation: Segmentation map [H, W]
        """
        # Only process leaf nodes
        leaf_nodes = [node for node in nodes if node.leaf]
        
        for node in leaf_nodes:
            # Extract region from segmentation
            region = segmentation[node.y:node.y+node.size, node.x:node.x+node.size]
            
            # Find most common label (excluding ignore index)
            mask = (region != 255)
            if mask.sum() > 0:
                labels, counts = torch.unique(region[mask], return_counts=True)
                if labels.size(0) > 0:
                    node.label = labels[torch.argmax(counts)].item()
                    node.confidence = torch.max(counts).float() / mask.sum().float()
                else:
                    node.label = 0
                    node.confidence = 0.0
            else:
                node.label = 0
                node.confidence = 0.0
    
    def batch_compute_adjacency(self, trees, all_nodes_dict):
        """
        Efficiently compute adjacency relationships between all nodes
        using GPU-accelerated intersection tests
        
        Args:
            trees: List of tree root nodes
            all_nodes_dict: Dictionary of all nodes by (batch_idx, node_idx)
            
        Returns:
            Updated trees with neighbor relationships
        """
        # Get all leaf nodes across all trees
        all_leaves = []
        for tree in trees:
            all_leaves.extend(self._get_leaf_nodes(tree))
        
        # Process in batches to stay within memory limits
        batch_size = 5000  # Adjust based on GPU memory
        
        for i in range(0, len(all_leaves), batch_size):
            batch_leaves = all_leaves[i:i+batch_size]
            
            # Extract batch index and boundaries for each node
            batch_indices = torch.tensor([n.batch_idx for n in batch_leaves], device=self.device)
            left = torch.tensor([n.x for n in batch_leaves], device=self.device)
            top = torch.tensor([n.y for n in batch_leaves], device=self.device)
            right = torch.tensor([n.x + n.size for n in batch_leaves], device=self.device)
            bottom = torch.tensor([n.y + n.size for n in batch_leaves], device=self.device)
            
            # Compute all pairs of intersections in a batched manner
            for j in range(0, len(all_leaves), batch_size):
                other_batch = all_leaves[j:j+batch_size]
                
                # Skip self-comparison if it's the same batch
                if i == j:
                    continue
                
                # Extract boundaries for other batch
                other_batch_indices = torch.tensor([n.batch_idx for n in other_batch], device=self.device)
                other_left = torch.tensor([n.x for n in other_batch], device=self.device)
                other_top = torch.tensor([n.y for n in other_batch], device=self.device)
                other_right = torch.tensor([n.x + n.size for n in other_batch], device=self.device)
                other_bottom = torch.tensor([n.y + n.size for n in other_batch], device=self.device)
                
                # Expand dimensions for broadcasting
                batch_eq = (batch_indices.unsqueeze(1) == other_batch_indices.unsqueeze(0))
                
                # Compute horizontal adjacency
                horiz_adjacent = torch.logical_or(
                    torch.logical_and(
                        (right.unsqueeze(1) == other_left.unsqueeze(0)),
                        torch.logical_not(
                            torch.logical_or(
                                (bottom.unsqueeze(1) <= other_top.unsqueeze(0)),
                                (other_bottom.unsqueeze(0) <= top.unsqueeze(1))
                            )
                        )
                    ),
                    torch.logical_and(
                        (other_right.unsqueeze(0) == left.unsqueeze(1)),
                        torch.logical_not(
                            torch.logical_or(
                                (bottom.unsqueeze(1) <= other_top.unsqueeze(0)),
                                (other_bottom.unsqueeze(0) <= top.unsqueeze(1))
                            )
                        )
                    )
                )
                
                # Compute vertical adjacency
                vert_adjacent = torch.logical_or(
                    torch.logical_and(
                        (bottom.unsqueeze(1) == other_top.unsqueeze(0)),
                        torch.logical_not(
                            torch.logical_or(
                                (right.unsqueeze(1) <= other_left.unsqueeze(0)),
                                (other_right.unsqueeze(0) <= left.unsqueeze(1))
                            )
                        )
                    ),
                    torch.logical_and(
                        (other_bottom.unsqueeze(0) == top.unsqueeze(1)),
                        torch.logical_not(
                            torch.logical_or(
                                (right.unsqueeze(1) <= other_left.unsqueeze(0)),
                                (other_right.unsqueeze(0) <= left.unsqueeze(1))
                            )
                        )
                    )
                )
                
                # Final adjacency: must be in same batch and either horizontally or vertically adjacent
                is_adjacent = torch.logical_and(
                    batch_eq,
                    torch.logical_or(horiz_adjacent, vert_adjacent)
                )
                
                # Get all pairs of adjacent nodes
                adjacent_pairs = torch.nonzero(is_adjacent, as_tuple=True)
                
                # Update neighbor lists
                for idx1, idx2 in zip(adjacent_pairs[0].tolist(), adjacent_pairs[1].tolist()):
                    node1 = batch_leaves[idx1]
                    node2 = other_batch[idx2]
                    
                    # Add to neighbor lists if not already present
                    if node2 not in node1.neighbors:
                        node1.neighbors.append(node2)
                    if node1 not in node2.neighbors:
                        node2.neighbors.append(node1)
        
        return trees
    
    def _get_leaf_nodes(self, node):
        """Recursively get all leaf nodes in a tree"""
        if node.leaf:
            return [node]
        
        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaf_nodes(child))
        return leaves
    
    def batch_unary_potentials(self, trees, latent_features):
        """
        Compute unary potentials for all leaf nodes in a highly optimized batch operation
        
        Args:
            trees: List of tree root nodes
            latent_features: Features from CVAE [B, C, H, W]
            
        Returns:
            Updated trees with unary potentials
        """
        # Get all leaf nodes from all trees
        all_leaves = []
        for tree in trees:
            all_leaves.extend(self._get_leaf_nodes(tree))
        
        # Early exit if no leaves
        if not all_leaves:
            return trees
        
        # Process in batches to avoid memory issues
        batch_size = 500  # Adjust based on GPU memory
        
        for i in range(0, len(all_leaves), batch_size):
            batch_leaves = all_leaves[i:i+batch_size]
            
            # Collect features for all leaf nodes in this batch
            batch_features = []
            for node in batch_leaves:
                # Use pre-computed features from node
                if node.features is not None:
                    batch_features.append(node.features)
                else:
                    # Fallback if features not available
                    b = node.batch_idx
                    features = latent_features[b]
                    region = features[:, node.y:node.y+node.size, node.x:node.x+node.size]
                    pooled = F.adaptive_max_pool2d(region.unsqueeze(0), (1, 1)).squeeze()
                    batch_features.append(pooled)
            
            # Stack features into a single tensor
            try:
                feature_tensor = torch.stack(batch_features)
                
                # Handle feature dimensionality if needed
                feat_dim = feature_tensor.size(1)
                if feat_dim != self.feature_dim:
                    if str(feat_dim) in self.projections:
                        feature_tensor = self.projections[str(feat_dim)](feature_tensor)
                    else:
                        # Create projection on the fly if not available
                        projection = nn.Sequential(
                            nn.Linear(feat_dim, self.feature_dim),
                            nn.BatchNorm1d(self.feature_dim),
                            nn.LeakyReLU(inplace=True)
                        ).to(self.device)
                        feature_tensor = projection(feature_tensor)
                
                # Compute unary potentials for entire batch
                unary_potentials = self.feature_encoder(feature_tensor)
                
                # Assign unary potentials to nodes
                for idx, node in enumerate(batch_leaves):
                    node.unary_potentials = unary_potentials[idx]
                
            except Exception as e:
                print(f"Warning in batch_unary_potentials: {str(e)}")
                # Fallback for problematic batches
                for node in batch_leaves:
                    # Provide a simple prior favoring background class
                    node.unary_potentials = torch.zeros(self.n_classes, device=self.device)
                    node.unary_potentials[0] = 0.1  # Slight bias to background
        
        return trees
    
    def optimized_belief_propagation(self, trees, max_iterations=None):
        """
        Run highly optimized belief propagation with early stopping
        and efficient message caching
        
        Args:
            trees: List of tree root nodes
            max_iterations: Maximum BP iterations (overrides self.bp_iterations)
            
        Returns:
            Predicted segmentation maps
        """
        batch_size = len(trees)
        height = width = trees[0].size  # Assuming square images
        n_iterations = max_iterations if max_iterations is not None else self.bp_iterations
        
        # Initialize segmentation maps
        segmentations = torch.zeros(batch_size, height, width, device=self.device, dtype=torch.long)
        
        # Process each tree in the batch
        for b, tree in enumerate(trees):
            # Get all leaf nodes
            leaves = self._get_leaf_nodes(tree)
            
            # Skip if no leaves
            if not leaves:
                continue
                
            # Initialize beliefs with unary potentials
            for leaf in leaves:
                if hasattr(leaf, 'unary_potentials'):
                    # Initialize log-beliefs with unary potentials
                    leaf.log_beliefs = leaf.unary_potentials.clone()
                else:
                    # Uniform distribution in log space
                    leaf.log_beliefs = torch.zeros(self.n_classes, device=self.device)
                
                # Initialize messages dictionary
                leaf.log_messages = {}
            
            # Prepare for message passing
            # Pre-allocate message storage for ALL node pairs to avoid repeated dict lookups
            message_storage = {}
            for leaf in leaves:
                for neighbor in leaf.neighbors:
                    message_storage[(leaf.node_idx, neighbor.node_idx)] = torch.zeros(
                        self.n_classes, device=self.device
                    )
            
            # Run optimized belief propagation
            converged = False
            for iter_idx in range(n_iterations):
                max_diff = 0.0
                
                # Update all messages in parallel when possible
                for leaf in leaves:
                    for neighbor in leaf.neighbors:
                        # Start with unary potential
                        if hasattr(leaf, 'unary_potentials'):
                            msg = leaf.unary_potentials.clone()
                        else:
                            msg = torch.zeros(self.n_classes, device=self.device)
                        
                        # Add messages from other neighbors (in log space)
                        for other in leaf.neighbors:
                            if other.node_idx != neighbor.node_idx:
                                key = (other.node_idx, leaf.node_idx)
                                if key in message_storage:
                                    msg = msg + message_storage[key]
                        
                        # Compute outgoing message (optimized for GPU)
                        msg_out = torch.zeros_like(msg)
                        log_pw = torch.log(self.pairwise_weights + 1e-10)
                        
                        for c in range(self.n_classes):
                            # Vectorized computation
                            msg_out[c] = torch.logsumexp(msg + log_pw[c], dim=0)
                        
                        # Normalize to prevent numerical issues
                        msg_out = msg_out - msg_out.max()
                        
                        # Check for convergence
                        old_msg = message_storage.get((leaf.node_idx, neighbor.node_idx), 
                                                    torch.zeros_like(msg_out))
                        diff = torch.abs(msg_out - old_msg).max().item()
                        max_diff = max(max_diff, diff)
                        
                        # Store updated message
                        message_storage[(leaf.node_idx, neighbor.node_idx)] = msg_out
                
                # Check for convergence
                if max_diff < self.bp_convergence_threshold:
                    converged = True
                    break
                
                # Update beliefs after each round
                for leaf in leaves:
                    # Start with unary potentials
                    if hasattr(leaf, 'unary_potentials'):
                        log_belief = leaf.unary_potentials.clone()
                    else:
                        log_belief = torch.zeros(self.n_classes, device=self.device)
                    
                    # Add messages from all neighbors
                    for neighbor in leaf.neighbors:
                        key = (neighbor.node_idx, leaf.node_idx)
                        if key in message_storage:
                            log_belief = log_belief + message_storage[key]
                    
                    # Store updated beliefs
                    leaf.log_beliefs = log_belief
            
            # Create segmentation map based on final beliefs
            for leaf in leaves:
                # Get predicted class
                pred_class = torch.argmax(leaf.log_beliefs)
                
                # Fill region with predicted class
                segmentations[b, leaf.y:leaf.y+leaf.size, leaf.x:leaf.x+leaf.size] = pred_class
        
        return segmentations
    
    def forward(self, features, cvae_latent=None, initial_segmentation=None):
        """
        Forward pass through the Optimized Quadtree MRF
        
        Args:
            features: Feature maps from base network [B, C, H, W]
            cvae_latent: Latent features from CVAE [B, C, H, W]
            initial_segmentation: Optional initial segmentation [B, H, W]
            
        Returns:
            Refined segmentation maps [B, H, W]
        """
        # Get batch dimensions
        batch_size, n_features, height, width = features.shape
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Create initial segmentation if not provided
            if initial_segmentation is None:
                # Use a simple conv to create initial segmentation
                initial_logits = self.fallback_cnn(features)
                initial_segmentation = initial_logits.argmax(dim=1)
            
            # Ensure initial_segmentation is on the same device
            initial_segmentation = initial_segmentation.to(self.device)
            
            # Set up CVAE features
            if cvae_latent is not None:
                # Reshape if needed
                if len(cvae_latent.shape) == 2:  # [B, C]
                    cvae_latent = cvae_latent.unsqueeze(-1).unsqueeze(-1)
                    cvae_latent = F.interpolate(
                        cvae_latent.expand(-1, -1, 2, 2), 
                        size=(height, width), 
                        mode='bilinear',
                        align_corners=False
                    )
            else:
                # If no cvae_latent provided, use features
                cvae_latent = features
            
            # Build the optimized quadtree structure
            trees, all_nodes = self.vectorized_quadtree_construction(features, initial_segmentation)
            
            # Compute optimized adjacency relationships
            trees = self.batch_compute_adjacency(trees, all_nodes)
            
            # Compute unary potentials in batches
            trees = self.batch_unary_potentials(trees, cvae_latent)
            
            # Run optimized belief propagation
            refined_segmentation = self.optimized_belief_propagation(trees)
            
            # Report timing for performance analysis
            elapsed = time.time() - start_time
            print(f"OptimizedQuadtreeMRF completed in {elapsed:.3f}s")
            
            return refined_segmentation
            
        except Exception as e:
            # Graceful fallback with error reporting
            print(f"Error in OptimizedQuadtreeMRF.forward: {str(e)}")
            print("Falling back to CNN segmentation")
            
            # Use the fallback CNN
            return self.fallback_cnn(features).argmax(dim=1)


# For backward compatibility
QuadtreeMRF = OptimizedQuadtreeMRF