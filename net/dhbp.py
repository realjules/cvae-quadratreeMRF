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


class SimpleUnaryHead(nn.Module):
    """Minimal linear projection from features to class log-probabilities.

    Single Conv1x1(C→K) + log_softmax. No hidden layer, no BN, no ReLU.
    This is the minimum bridge between feature space and probability space.
    """

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, n_classes, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(feat), dim=1)


class DiagonalPairwiseHead(nn.Module):
    """Hard diagonal pairwise: ψ = diag(d), no class mixing.

    Each class gets an independent positive scaling factor per location.
    d > 1 amplifies belief, d < 1 suppresses, d = 1 passes through.
    Off-diagonal is zero — no class transitions possible.

    This isolates whether BP's value comes from gradient amplification
    alone (multi-path computation) or from the pairwise class mixing.
    """

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        mid = max(in_channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_classes, 1),
        )
        # Initialize bias to 0 → softplus(0) = ln(2) ≈ 0.69 → near-identity scaling
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C_in, H, W] encoder features at parent resolution

        Returns:
            [B, K, K, H, W] log-pairwise potentials (diagonal only).
        """
        K = self.n_classes
        B, _, H, W = feat.shape

        # K positive scalars per location
        d = F.softplus(self.net(feat))  # [B, K, H, W], all positive

        # Build diagonal K×K matrix: off-diagonal = -inf (log(0))
        log_psi = torch.full((B, K, K, H, W), -1e8, device=feat.device)
        for k in range(K):
            log_psi[:, k, k, :, :] = torch.log(d[:, k, :, :] + 1e-8)

        return log_psi


class PairwisePotentialHead(nn.Module):
    """Constrained diagonal-dominant pairwise potential.

    Decomposes the K×K compatibility matrix as:

        ψ = α · I + (1-α) · R

    where:
        α ∈ [0,1] = per-location consistency strength (predicted from features)
        I = identity matrix (same-class → same-class)
        R = learned residual transition matrix (softmax-normalized rows)

    This constrains the pairwise to START as pure spatial consistency (α≈0.8)
    and only learn small transition corrections where needed (boundaries).
    Prevents the failure mode where unconstrained K×K matrices learn class
    remapping instead of spatial consistency.

    Reference: generalizes the Potts model (Boykov & Jolly 2001) with a
    learned, spatially-varying consistency strength.
    """

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        mid = max(in_channels // 4, 16)

        # α: consistency strength per location → [0,1] via sigmoid
        self.alpha_net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )
        # Initialize bias so sigmoid(1.4) ≈ 0.8 — consistency-biased start
        nn.init.constant_(self.alpha_net[-1].bias, 1.4)

        # R: residual transition matrix (small corrections)
        self.residual_net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_classes * n_classes, 1),
        )

        # Fixed identity matrix
        self.register_buffer('_identity', torch.eye(n_classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C_in, H, W] encoder features at parent resolution

        Returns:
            [B, K, K, H, W] log-pairwise potentials (diagonal-dominant).
        """
        K = self.n_classes
        B, _, H, W = feat.shape

        # Consistency strength: high α → strong same-class enforcement
        alpha = torch.sigmoid(self.alpha_net(feat))          # [B, 1, H, W]
        alpha = alpha.unsqueeze(1)                           # [B, 1, 1, H, W]

        # Residual transition matrix: softmax rows → valid distribution
        residual = self.residual_net(feat)                   # [B, K*K, H, W]
        residual = residual.view(B, K, K, H, W)             # [B, K, K, H, W]
        residual = F.softmax(residual, dim=2)                # normalize over child states

        # ψ = α·I + (1-α)·R  — diagonal-dominant by construction
        identity = self._identity.view(1, K, K, 1, 1)       # [1, K, K, 1, 1]
        psi = alpha * identity + (1.0 - alpha) * residual   # [B, K, K, H, W]

        return torch.log(psi + 1e-8)                        # log-space for BP


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

    def __init__(self, n_classes: int = 6, simple_unary: bool = False,
                 diagonal_pairwise: bool = False):
        super().__init__()
        self.n_classes = n_classes

        # Unary potential heads (features → log class probabilities)
        UHead = SimpleUnaryHead if simple_unary else UnaryPotentialHead
        if simple_unary:
            print("DHBP: using SimpleUnaryHead (Conv1x1 linear projection)")
        self.unary_1 = UHead(64, n_classes)    # fine: 128×128
        self.unary_2 = UHead(128, n_classes)   # mid: 64×64
        self.unary_3 = UHead(256, n_classes)   # coarse: 32×32

        # Pairwise potential heads (features → K×K log-compatibility)
        PHead = DiagonalPairwiseHead if diagonal_pairwise else PairwisePotentialHead
        if diagonal_pairwise:
            print("DHBP: using DiagonalPairwiseHead (no class mixing)")
        self.pairwise_12 = PHead(128, n_classes)  # edge 1↔2, uses p2
        self.pairwise_23 = PHead(256, n_classes)  # edge 2↔3, uses p3

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
