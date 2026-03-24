"""
Differentiable Hierarchical Belief Propagation (DHBP) on a Quadtree MRF.

Implements exact sum-product belief propagation on a tree-structured
graphical model, where each level of the image pyramid is a level of
the quadtree. Messages operate in CLASS-PROBABILITY space (not feature
space), and pairwise potentials are spatially-varying K×K compatibility
matrices predicted from contrastive-pretrained encoder features.

BP equations (log-space):

    Unary potential:
        log φ_i(x_i) = UnaryHead(features_i)

    Pairwise potential (NOVEL — spatially-varying, feature-dependent):
        log ψ(x_p, x_c) = PairwiseHead(features)    [K × K matrix]

    Child → Parent message:
        log m_{c→p}(x_p) = logsumexp_{x_c} [log ψ(x_p, x_c) + log b_c(x_c)]

    Parent → Child message (with cavity):
        cavity_c = log b_p - log m_{c→p}
        log m_{p→c}(x_c) = logsumexp_{x_p} [log ψ^T(x_c, x_p) + cavity_c(x_p)]

    Belief:
        log b_i = log φ_i + Σ_j log m_{j→i}

Single bottom-up + top-down pass is exact on a tree (no loops).

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Encoder features → Unary potentials (1×1 conv, K)   │
    │  Encoder features → Pairwise potentials (1×1, K×K)   │
    │                                                       │
    │  Bottom-up: 2×2 block reshape + logsumexp (quadtree) │
    │  Top-down:  cavity + ψ^T + reassemble                │
    │                                                       │
    │  All operations in log-space, all GPU-parallel        │
    └──────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnaryPotentialHead(nn.Module):
    """Maps encoder features to log-unary potentials over K classes.

    Output is log φ_i(x_i) — the local evidence for each class at
    each spatial location, before any message passing.
    """

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        mid = max(in_channels // 2, n_classes)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_classes, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C_in, H, W] encoder features

        Returns:
            [B, K, H, W] log-unary potentials (log_softmax normalized)
        """
        return F.log_softmax(self.net(feat), dim=1)


class PairwisePotentialHead(nn.Module):
    """Maps encoder features to spatially-varying K×K log-compatibility matrices.

    Output is log ψ(x_parent, x_child) — a K×K matrix at each spatial
    location representing how compatible each pair of parent-child states is.

    This is the KEY NOVELTY: pairwise potentials are not fixed (Potts model)
    or globally learned, but are spatially varying and predicted from
    contrastive-pretrained encoder features. Contrastive learning ensures
    that semantically similar regions produce high-compatibility potentials,
    directly improving structured prediction.
    """

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        mid = max(in_channels // 2, n_classes * n_classes)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_classes * n_classes, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C_in, H, W] encoder features at parent resolution

        Returns:
            [B, K, K, H, W] log-pairwise potentials.
            Dimension 1 = parent state, dimension 2 = child state.
            Normalized over child states (dim=2) via log_softmax.
        """
        K = self.n_classes
        B, _, H, W = feat.shape
        raw = self.net(feat)                         # [B, K*K, H, W]
        psi = raw.view(B, K, K, H, W)               # [B, K_parent, K_child, H, W]
        return F.log_softmax(psi, dim=2)             # normalize over child states


def _child_to_parent(
    child_belief: torch.Tensor,
    log_psi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bottom-up message: aggregate 4 children in a 2×2 quadtree block.

    BP equation (log-space):
        log m_{c→p}(x_p) = logsumexp_{x_c} [log ψ(x_p, x_c) + log b_c(x_c)]
        Total message = sum over 4 children (product in prob-space).

    Args:
        child_belief: [B, K, H, W] log-beliefs at child (fine) level
        log_psi: [B, K, K, H_p, W_p] log-pairwise at parent resolution
                 where H_p = H//2, W_p = W//2

    Returns:
        msg_total: [B, K, H_p, W_p] total upward message
        per_child: [B, K, H_p, W_p, 4] individual child messages (for cavity)
    """
    B, K, H, W = child_belief.shape
    H_p, W_p = H // 2, W // 2

    # Reshape child beliefs into 2×2 quadtree blocks → 4 children per parent
    # [B, K, H, W] → [B, K, H_p, 2, W_p, 2] → [B, K, H_p, W_p, 4]
    cb = child_belief.reshape(B, K, H_p, 2, W_p, 2)
    cb = cb.permute(0, 1, 2, 4, 3, 5)               # [B, K, H_p, W_p, 2, 2]
    cb = cb.reshape(B, K, H_p, W_p, 4)              # [B, K, H_p, W_p, 4]

    # For each child, compute message via logsumexp over child states
    # log_psi: [B, K_p, K_c, H_p, W_p]
    # child_c: [B, K_c, H_p, W_p] → [B, 1, K_c, H_p, W_p] for broadcast
    per_child_list = []
    for c in range(4):
        child_c = cb[:, :, :, :, c]                  # [B, K, H_p, W_p]
        # log_psi[b, k_p, k_c, h, w] + child_c[b, k_c, h, w]
        # → logsumexp over k_c (dim=2)
        combined = log_psi + child_c.unsqueeze(1)    # [B, K_p, K_c, H_p, W_p]
        msg_c = torch.logsumexp(combined, dim=2)     # [B, K, H_p, W_p]
        per_child_list.append(msg_c)

    per_child = torch.stack(per_child_list, dim=-1)  # [B, K, H_p, W_p, 4]
    msg_total = per_child.sum(dim=-1)                # [B, K, H_p, W_p]

    return msg_total, per_child


def _parent_to_child(
    parent_belief: torch.Tensor,
    per_child_msgs: torch.Tensor,
    log_psi: torch.Tensor,
) -> torch.Tensor:
    """Top-down message: parent sends to each of 4 children via cavity.

    BP equation (log-space):
        cavity_c = log b_p - log m_{c→p}    (exclude child's own contribution)
        log m_{p→c}(x_c) = logsumexp_{x_p} [log ψ^T(x_c, x_p) + cavity_c(x_p)]

    Args:
        parent_belief: [B, K, H_p, W_p] log-beliefs at parent level
        per_child_msgs: [B, K, H_p, W_p, 4] per-child upward messages
        log_psi: [B, K, K, H_p, W_p] log-pairwise (same as bottom-up)

    Returns:
        msg_down: [B, K, H, W] downward messages reassembled at child resolution
                  where H = H_p * 2, W = W_p * 2
    """
    B, K, H_p, W_p = parent_belief.shape

    # Transpose pairwise: swap parent/child dims
    log_psi_T = log_psi.transpose(1, 2)              # [B, K_c, K_p, H_p, W_p]

    child_msgs = []
    for c in range(4):
        # Cavity: parent belief minus this child's upward message
        cavity = parent_belief - per_child_msgs[:, :, :, :, c]
        cavity = torch.clamp(cavity, min=-1e6)        # prevent -inf → NaN

        # log_psi_T[b, k_c, k_p, h, w] + cavity[b, k_p, h, w]
        # → logsumexp over k_p (dim=2)
        combined = log_psi_T + cavity.unsqueeze(1)    # [B, K_c, K_p, H_p, W_p]
        msg_c = torch.logsumexp(combined, dim=2)      # [B, K, H_p, W_p]
        child_msgs.append(msg_c)

    # Reassemble 4 children back into 2×2 spatial blocks
    # Stack: [B, K, H_p, W_p, 4] → reshape to [B, K, H_p, W_p, 2, 2]
    stacked = torch.stack(child_msgs, dim=-1)         # [B, K, H_p, W_p, 4]
    stacked = stacked.reshape(B, K, H_p, W_p, 2, 2)
    # Permute to interleave: [B, K, H_p, 2, W_p, 2]
    stacked = stacked.permute(0, 1, 2, 4, 3, 5)
    # Reshape to full child resolution: [B, K, H, W]
    msg_down = stacked.reshape(B, K, H_p * 2, W_p * 2)

    return msg_down


class DHBPModule(nn.Module):
    """Differentiable Hierarchical Belief Propagation on a Quadtree MRF.

    Implements exact sum-product BP with:
        - Unary potentials from encoder features (3 levels)
        - Spatially-varying pairwise potentials from encoder features (2 edges)
        - Single bottom-up + top-down pass (exact on tree)
        - All operations in log-space, GPU-parallel

    Input:  p1 [B, 64, 128, 128], p2 [B, 128, 64, 64], p3 [B, 256, 32, 32]
    Output: logits [B, n_classes, 128, 128]
    """

    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.n_classes = n_classes

        # Unary potential heads (features → log class probabilities)
        self.unary_1 = UnaryPotentialHead(64, n_classes)    # fine: 128×128
        self.unary_2 = UnaryPotentialHead(128, n_classes)   # mid: 64×64
        self.unary_3 = UnaryPotentialHead(256, n_classes)   # coarse: 32×32

        # Pairwise potential heads (features → K×K log-compatibility)
        # Computed at parent resolution from parent-level features
        self.pairwise_12 = PairwisePotentialHead(128, n_classes)  # edge 1↔2, uses p2
        self.pairwise_23 = PairwisePotentialHead(256, n_classes)  # edge 2↔3, uses p3

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> torch.Tensor:
        """Run exact belief propagation on the quadtree.

        Args:
            p1: [B, 64,  128, 128] fine-scale encoder features
            p2: [B, 128,  64,  64] mid-scale encoder features
            p3: [B, 256,  32,  32] coarse-scale encoder features

        Returns:
            logits: [B, n_classes, 128, 128] — final beliefs at finest level,
                    valid as logits for cross_entropy (unnormalized log-probs)
        """
        # === Step 0: Compute potentials from encoder features ===
        phi_1 = self.unary_1(p1)          # [B, K, 128, 128]
        phi_2 = self.unary_2(p2)          # [B, K,  64,  64]
        phi_3 = self.unary_3(p3)          # [B, K,  32,  32]

        psi_12 = self.pairwise_12(p2)     # [B, K, K, 64, 64]
        psi_23 = self.pairwise_23(p3)     # [B, K, K, 32, 32]

        # === Step 1: Initialize leaf beliefs ===
        b1 = phi_1                        # [B, K, 128, 128]

        # === Step 2: Bottom-up pass (leaves → root) ===
        # Level 1 → Level 2
        msg_up_12, per_child_12 = _child_to_parent(b1, psi_12)
        b2 = phi_2 + msg_up_12            # [B, K, 64, 64]

        # Level 2 → Level 3
        msg_up_23, per_child_23 = _child_to_parent(b2, psi_23)
        b3 = phi_3 + msg_up_23            # [B, K, 32, 32]  (root belief)

        # === Step 3: Top-down pass (root → leaves) ===
        # Level 3 → Level 2
        msg_dn_23 = _parent_to_child(b3, per_child_23, psi_23)
        b2_final = phi_2 + msg_up_12 + msg_dn_23   # [B, K, 64, 64]

        # Level 2 → Level 1
        msg_dn_12 = _parent_to_child(b2_final, per_child_12, psi_12)
        b1_final = phi_1 + msg_dn_12      # [B, K, 128, 128]

        return b1_final

    @torch.no_grad()
    def forward_diagnostic(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> dict:
        """Same as forward(), but returns ALL intermediate tensors for analysis.

        Does NOT affect training — this is a read-only diagnostic method.
        """
        # Step 0: Potentials
        phi_1 = self.unary_1(p1)
        phi_2 = self.unary_2(p2)
        phi_3 = self.unary_3(p3)
        psi_12 = self.pairwise_12(p2)
        psi_23 = self.pairwise_23(p3)

        # Step 1: Leaf beliefs
        b1 = phi_1

        # Step 2: Bottom-up
        msg_up_12, per_child_12 = _child_to_parent(b1, psi_12)
        b2 = phi_2 + msg_up_12
        msg_up_23, per_child_23 = _child_to_parent(b2, psi_23)
        b3 = phi_3 + msg_up_23

        # Step 3: Top-down
        msg_dn_23 = _parent_to_child(b3, per_child_23, psi_23)
        b2_final = phi_2 + msg_up_12 + msg_dn_23
        msg_dn_12 = _parent_to_child(b2_final, per_child_12, psi_12)
        b1_final = phi_1 + msg_dn_12

        return {
            'phi_1': phi_1,           'phi_2': phi_2,           'phi_3': phi_3,
            'psi_12': psi_12,         'psi_23': psi_23,
            'b1': b1,                 'b2': b2,                 'b3': b3,
            'b2_final': b2_final,     'b1_final': b1_final,
            'msg_up_12': msg_up_12,   'msg_up_23': msg_up_23,
            'msg_dn_23': msg_dn_23,   'msg_dn_12': msg_dn_12,
        }
