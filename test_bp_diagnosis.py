"""
Diagnose WHY BP isn't helping significantly.

Test 1: Does BP propagate horizontally?
  Set one pixel's unary to "building", neighbors to "uncertain."
  After BP, check if building belief spread to neighbors.

Test 2: Where does BP change predictions?
  Compare unary vs BP predictions spatially.
  Count changes at scale boundaries vs spatial boundaries.

Test 3: What does the pairwise alpha learn?
  Visualize alpha (consistency strength) spatially.
  Is it uniform (~0.8 everywhere) or does it vary at boundaries?

Usage:
    python test_bp_diagnosis.py \
        --contrastive_ckpt /path/to/contrastive_best.pth \
        --seg_ckpt /path/to/best_segmentation.pth \
        --output_dir ./output/bp_diagnosis
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule


def load_models(contrastive_ckpt, seg_ckpt, device, diagonal_pairwise=False):
    """Load encoder and DHBP from checkpoints."""
    encoder = ContrastiveEncoder(pretrained=True)

    # Load contrastive encoder
    ckpt_path = contrastive_ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = glob.glob(os.path.join(ckpt_path, "*.pth"))[0]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])

    dhbp = DHBPModule(n_classes=6, diagonal_pairwise=diagonal_pairwise)

    if seg_ckpt:
        seg_path = seg_ckpt
        if os.path.isdir(seg_path):
            seg_path = glob.glob(os.path.join(seg_path, "*.pth"))[0]
        seg = torch.load(seg_path, map_location=device, weights_only=False)
        if 'dhbp_state_dict' in seg:
            dhbp.load_state_dict(seg['dhbp_state_dict'])
        if 'encoder_state_dict' in seg:
            encoder.load_state_dict(seg['encoder_state_dict'])
        print("Loaded trained encoder + DHBP")
    else:
        print("WARNING: No seg_ckpt — using random DHBP weights")

    return encoder.to(device).eval(), dhbp.to(device).eval()


# =========================================================================
# TEST 1: Does BP propagate horizontally?
# =========================================================================

@torch.no_grad()
def test_horizontal_propagation(dhbp, device, output_dir):
    """Inject a strong signal at one pixel. Does it spread to neighbors?"""
    print("\n" + "=" * 60)
    print("TEST 1: Does BP propagate information horizontally?")
    print("=" * 60)

    K = 6
    # Create synthetic unary potentials: all uncertain except one pixel
    # Level 1: 128x128, Level 2: 64x64, Level 3: 32x32
    phi_1 = torch.zeros(1, K, 128, 128, device=device)  # uniform = uncertain
    phi_2 = torch.zeros(1, K, 64, 64, device=device)
    phi_3 = torch.zeros(1, K, 32, 32, device=device)

    # Inject strong "building" signal (class 1) at center pixel of level 1
    center_h, center_w = 64, 64
    phi_1[0, 1, center_h, center_w] = 10.0  # strong building evidence

    # Apply log_softmax (as DHBP forward does)
    phi_1_norm = F.log_softmax(phi_1, dim=1)
    phi_2_norm = F.log_softmax(phi_2, dim=1)
    phi_3_norm = F.log_softmax(phi_3, dim=1)

    # Run BP manually using the DHBP's pairwise heads
    # We need to create fake encoder features to get pairwise potentials
    # Use the actual pairwise from the trained model with dummy features
    # Instead, just use forward_diagnostic with modified unary

    # Hack: temporarily replace unary heads to output our synthetic potentials
    class FakeUnary(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = val
        def forward(self, x):
            return self.val

    orig_u1 = dhbp.unary_1
    orig_u2 = dhbp.unary_2
    orig_u3 = dhbp.unary_3

    # We need real encoder features for the pairwise heads
    # Create random features (pairwise depends on features, not unary)
    p1_fake = torch.randn(1, 64, 128, 128, device=device)
    p2_fake = torch.randn(1, 128, 64, 64, device=device)
    p3_fake = torch.randn(1, 256, 32, 32, device=device)

    dhbp.unary_1 = FakeUnary(phi_1_norm)
    dhbp.unary_2 = FakeUnary(phi_2_norm)
    dhbp.unary_3 = FakeUnary(phi_3_norm)

    diag = dhbp.forward_diagnostic(p1_fake, p2_fake, p3_fake)
    b1_final = diag['b1_final']  # [1, K, 128, 128]

    # Restore original unary heads
    dhbp.unary_1 = orig_u1
    dhbp.unary_2 = orig_u2
    dhbp.unary_3 = orig_u3

    # Analyze: how far did the building signal spread?
    building_belief = b1_final[0, 1].cpu().numpy()  # [128, 128] building belief
    building_unary = phi_1_norm[0, 1].cpu().numpy()

    # Check horizontal neighbors at same level
    print(f"\n  Center pixel ({center_h},{center_w}):")
    print(f"    Unary building: {phi_1_norm[0, 1, center_h, center_w]:.4f}")
    print(f"    After BP:       {b1_final[0, 1, center_h, center_w]:.4f}")

    print(f"\n  Horizontal neighbors (same row, same quadtree level):")
    for offset in [1, 2, 4, 8, 16]:
        w = center_w + offset
        if w < 128:
            unary_val = phi_1_norm[0, 1, center_h, w].item()
            bp_val = b1_final[0, 1, center_h, w].item()
            diff = bp_val - unary_val
            print(f"    Pixel ({center_h},{w}) [offset +{offset}]: "
                  f"unary={unary_val:.4f}, after_bp={bp_val:.4f}, change={diff:+.4f}")

    print(f"\n  Vertical neighbors (same column):")
    for offset in [1, 2, 4, 8, 16]:
        h = center_h + offset
        if h < 128:
            unary_val = phi_1_norm[0, 1, h, center_w].item()
            bp_val = b1_final[0, 1, h, center_w].item()
            diff = bp_val - unary_val
            print(f"    Pixel ({h},{center_w}) [offset +{offset}]: "
                  f"unary={unary_val:.4f}, after_bp={bp_val:.4f}, change={diff:+.4f}")

    # Key question: does the signal spread to pixels in the SAME 2x2 block?
    # Pixel (64,64) shares a 2x2 block with (64,65), (65,64), (65,65)
    print(f"\n  Same 2x2 quadtree block (should be affected by BP):")
    for dh, dw in [(0,1), (1,0), (1,1)]:
        h, w = center_h + dh, center_w + dw
        unary_val = phi_1_norm[0, 1, h, w].item()
        bp_val = b1_final[0, 1, h, w].item()
        diff = bp_val - unary_val
        print(f"    Pixel ({h},{w}): unary={unary_val:.4f}, after_bp={bp_val:.4f}, change={diff:+.4f}")

    print(f"\n  DIFFERENT 2x2 block (only connected through parent):")
    for dh, dw in [(0,2), (0,3), (2,0), (2,2)]:
        h, w = center_h + dh, center_w + dw
        unary_val = phi_1_norm[0, 1, h, w].item()
        bp_val = b1_final[0, 1, h, w].item()
        diff = bp_val - unary_val
        print(f"    Pixel ({h},{w}): unary={unary_val:.4f}, after_bp={bp_val:.4f}, change={diff:+.4f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Building unary
    ax = axes[0]
    im = ax.imshow(building_unary[56:72, 56:72], cmap='hot', interpolation='nearest')
    ax.set_title("Unary (building class)\n16x16 crop around center")
    ax.set_xlabel("Only center pixel is hot")
    plt.colorbar(im, ax=ax)

    # Building belief after BP
    ax = axes[1]
    im = ax.imshow(building_belief[56:72, 56:72], cmap='hot', interpolation='nearest')
    ax.set_title("After BP (building class)\n16x16 crop around center")
    ax.set_xlabel("Did signal spread?")
    plt.colorbar(im, ax=ax)

    # Difference
    ax = axes[2]
    diff_map = building_belief - building_unary
    im = ax.imshow(diff_map[56:72, 56:72], cmap='RdBu_r', interpolation='nearest',
                   vmin=-0.5, vmax=0.5)
    ax.set_title("Change from BP\nRed=increased, Blue=decreased")
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    path = os.path.join(output_dir, "test1_horizontal_propagation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Verdict
    same_block_change = abs(b1_final[0, 1, center_h, center_w+1].item() -
                            phi_1_norm[0, 1, center_h, center_w+1].item())
    diff_block_change = abs(b1_final[0, 1, center_h, center_w+2].item() -
                            phi_1_norm[0, 1, center_h, center_w+2].item())
    far_change = abs(b1_final[0, 1, center_h, center_w+16].item() -
                     phi_1_norm[0, 1, center_h, center_w+16].item())

    print(f"\n  VERDICT:")
    print(f"    Same 2x2 block change:    {same_block_change:.6f}")
    print(f"    Adjacent block change:    {diff_block_change:.6f}")
    print(f"    16 pixels away change:    {far_change:.6f}")
    if same_block_change > 0.01 and diff_block_change < 0.001:
        print(f"    → BP propagates ONLY within 2x2 blocks (vertical only)")
        print(f"    → NO horizontal propagation to neighboring blocks")
        print(f"    → This confirms the quadtree structural limitation")
    elif diff_block_change > 0.001:
        print(f"    → BP propagates beyond 2x2 blocks (through parent)")
        print(f"    → Some horizontal effect exists via parent-child chain")
    else:
        print(f"    → BP barely changes anything — pairwise too weak")


# =========================================================================
# TEST 2: Where does BP change predictions on real data?
# =========================================================================

@torch.no_grad()
def test_prediction_changes(encoder, dhbp, device, output_dir, data_dir="./input"):
    """Compare unary vs BP predictions on real images."""
    print("\n" + "=" * 60)
    print("TEST 2: Where does BP change predictions?")
    print("=" * 60)

    from dataset.dataset import ISPRS_dataset
    from torch.utils.data import DataLoader

    # Load a few test patches
    area_ids = ['32']  # single test area
    top_pat = os.path.join(data_dir, "top", "top_mosaic_09cm_area{}.tif")
    gt_pat = os.path.join(data_dir, "gt", "top_mosaic_09cm_area{}.tif")

    try:
        ds = ISPRS_dataset(
            ids=area_ids, ids_type='TEST', gt_type='full', gt_modification=None,
            data_files=top_pat, label_files=gt_pat,
            window_size=256, cache=False, augmentation=False,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"  Could not load data: {e}")
        print(f"  Skipping Test 2")
        return

    # Collect statistics
    total_pixels = 0
    bp_changed = 0
    changed_at_boundary = 0
    changed_at_interior = 0
    total_boundary = 0
    total_interior = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= 10:
            break

        images = images.to(device)
        labels = labels.long()
        labels_ds = F.interpolate(
            labels.unsqueeze(1).float(), size=(128, 128), mode='nearest'
        ).squeeze(1).long()

        p1, p2, p3 = encoder.encode(images)
        diag = dhbp.forward_diagnostic(p1, p2, p3)

        phi_1 = diag['phi_1']      # [B, 6, 128, 128]
        b1_final = diag['b1_final']  # [B, 6, 128, 128]

        unary_preds = phi_1.argmax(dim=1).cpu()     # [B, 128, 128]
        bp_preds = b1_final.argmax(dim=1).cpu()      # [B, 128, 128]

        # Where did BP change the prediction?
        changed = (unary_preds != bp_preds)  # [B, 128, 128]

        # Detect spatial boundaries in ground truth
        pad_lab = F.pad(labels_ds.float().unsqueeze(1), (1,1,1,1), mode='replicate').squeeze(1)
        boundary = (
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 1:-1, 2:]) |
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 1:-1, :-2]) |
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 2:, 1:-1]) |
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, :-2, 1:-1])
        )  # [B, 128, 128]

        interior = ~boundary

        total_pixels += changed.numel()
        bp_changed += changed.sum().item()
        changed_at_boundary += (changed & boundary).sum().item()
        changed_at_interior += (changed & interior).sum().item()
        total_boundary += boundary.sum().item()
        total_interior += interior.sum().item()

    pct_changed = bp_changed / total_pixels * 100
    pct_boundary_changed = changed_at_boundary / max(total_boundary, 1) * 100
    pct_interior_changed = changed_at_interior / max(total_interior, 1) * 100

    print(f"\n  Total pixels analyzed: {total_pixels:,}")
    print(f"  Pixels where BP changed prediction: {bp_changed:,} ({pct_changed:.1f}%)")
    print(f"  Changes at BOUNDARIES: {changed_at_boundary:,} / {total_boundary:,} ({pct_boundary_changed:.1f}%)")
    print(f"  Changes at INTERIOR:   {changed_at_interior:,} / {total_interior:,} ({pct_interior_changed:.1f}%)")

    print(f"\n  VERDICT:")
    if pct_changed < 5:
        print(f"    → BP barely changes any predictions ({pct_changed:.1f}%) — pairwise too weak")
    elif pct_boundary_changed > pct_interior_changed * 2:
        print(f"    → BP primarily changes BOUNDARY pixels — working as intended (spatial refinement)")
    elif pct_interior_changed > pct_boundary_changed:
        print(f"    → BP primarily changes INTERIOR pixels — it's reclassifying, not refining boundaries")
    else:
        print(f"    → BP changes boundaries and interior roughly equally — no spatial preference")


# =========================================================================
# TEST 3: What does the pairwise alpha look like spatially?
# =========================================================================

@torch.no_grad()
def test_alpha_spatial(encoder, dhbp, device, output_dir, data_dir="./input"):
    """Visualize the pairwise alpha (consistency strength) spatially."""
    print("\n" + "=" * 60)
    print("TEST 3: Pairwise alpha (consistency strength) spatial pattern")
    print("=" * 60)

    from dataset.dataset import ISPRS_dataset
    from torch.utils.data import DataLoader

    area_ids = ['32']
    top_pat = os.path.join(data_dir, "top", "top_mosaic_09cm_area{}.tif")
    gt_pat = os.path.join(data_dir, "gt", "top_mosaic_09cm_area{}.tif")

    try:
        ds = ISPRS_dataset(
            ids=area_ids, ids_type='TEST', gt_type='full', gt_modification=None,
            data_files=top_pat, label_files=gt_pat,
            window_size=256, cache=False, augmentation=False,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"  Could not load data: {e}")
        return

    # Get one batch
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.long()

    p1, p2, p3 = encoder.encode(images)

    # Get pairwise info — depends on head type
    has_alpha = hasattr(dhbp.pairwise_12, 'alpha_net')

    if has_alpha:
        # α·I + (1-α)·R pairwise: extract alpha
        alpha_12 = torch.sigmoid(dhbp.pairwise_12.alpha_net(p2))  # [1, 1, 64, 64]
        alpha_23 = torch.sigmoid(dhbp.pairwise_23.alpha_net(p3))  # [1, 1, 32, 32]
        alpha_12_np = alpha_12[0, 0].cpu().numpy()
        alpha_23_np = alpha_23[0, 0].cpu().numpy()
    else:
        # DiagonalPairwiseHead: extract diagonal scaling values
        d_12 = F.softplus(dhbp.pairwise_12.net(p2))  # [1, K, 64, 64]
        d_23 = F.softplus(dhbp.pairwise_23.net(p3))  # [1, K, 32, 32]
        # Use mean across classes as a proxy for "consistency strength"
        alpha_12_np = d_12[0].mean(dim=0).cpu().numpy()  # [64, 64]
        alpha_23_np = d_23[0].mean(dim=0).cpu().numpy()  # [32, 32]
        print(f"\n  Diagonal scaling values (d) per class:")
        for k, name in enumerate(["Imp", "Bldg", "Low", "Tree", "Car", "Clut"]):
            d_mean = d_12[0, k].mean().item()
            d_std = d_12[0, k].std().item()
            print(f"    {name}: mean={d_mean:.4f}, std={d_std:.4f}")

    # Detect boundaries at level 2 resolution
    labels_64 = F.interpolate(
        labels.unsqueeze(1).float(), size=(64, 64), mode='nearest'
    ).squeeze(1).long()
    pad_lab = F.pad(labels_64.float().unsqueeze(1), (1,1,1,1), mode='replicate').squeeze(1)
    boundary_64 = (
        (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 1:-1, 2:]) |
        (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 1:-1, :-2]) |
        (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 2:, 1:-1]) |
        (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, :-2, 1:-1])
    )[0].cpu().numpy()

    print(f"\n  Alpha statistics (pairwise_12, level 1↔2):")
    print(f"    Mean:    {alpha_12_np.mean():.4f}")
    print(f"    Std:     {alpha_12_np.std():.4f}")
    print(f"    Min:     {alpha_12_np.min():.4f}")
    print(f"    Max:     {alpha_12_np.max():.4f}")

    alpha_at_boundary = alpha_12_np[boundary_64]
    alpha_at_interior = alpha_12_np[~boundary_64]

    if len(alpha_at_boundary) > 0 and len(alpha_at_interior) > 0:
        print(f"    At boundaries: mean={alpha_at_boundary.mean():.4f}, std={alpha_at_boundary.std():.4f}")
        print(f"    At interior:   mean={alpha_at_interior.mean():.4f}, std={alpha_at_interior.std():.4f}")
        diff = alpha_at_interior.mean() - alpha_at_boundary.mean()
        print(f"    Difference:    {diff:.4f} (positive = interior more consistent, good)")

    print(f"\n  VERDICT:")
    if alpha_12_np.std() < 0.01:
        print(f"    → Alpha is nearly UNIFORM ({alpha_12_np.std():.4f} std)")
        print(f"    → Pairwise learned NOTHING about spatial structure")
        print(f"    → Every location gets the same consistency strength")
    elif abs(alpha_at_interior.mean() - alpha_at_boundary.mean()) > 0.02:
        print(f"    → Alpha VARIES spatially and differs at boundaries vs interior")
        print(f"    → Pairwise IS learning spatial structure")
    else:
        print(f"    → Alpha varies slightly but doesn't distinguish boundaries from interior")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: alpha maps
    ax = axes[0, 0]
    im = ax.imshow(alpha_12_np, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title(f"Alpha ψ₁₂ (mean={alpha_12_np.mean():.3f})")
    ax.set_xlabel("Green=high consistency, Red=low")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(alpha_23_np, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title(f"Alpha ψ₂₃ (mean={alpha_23_np.mean():.3f})")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    ax.hist(alpha_12_np.flatten(), bins=50, color='steelblue', alpha=0.7)
    ax.set_title("Alpha distribution (ψ₁₂)")
    ax.set_xlabel("Alpha value")
    ax.axvline(x=0.8, color='red', linestyle='--', label='init=0.8')
    ax.legend()

    # Row 2: alpha vs boundaries
    labels_64_np = labels_64[0].cpu().numpy()

    ax = axes[1, 0]
    ax.imshow(labels_64_np, cmap='tab10', vmin=0, vmax=5)
    ax.set_title("Ground truth labels (64×64)")

    ax = axes[1, 1]
    ax.imshow(boundary_64, cmap='gray')
    ax.set_title("Class boundaries")

    ax = axes[1, 2]
    # Overlay: alpha with boundary contours
    ax.imshow(alpha_12_np, cmap='RdYlGn', vmin=0, vmax=1)
    ax.contour(boundary_64, levels=[0.5], colors='red', linewidths=0.5)
    ax.set_title("Alpha + boundary overlay\nDoes alpha drop at boundaries?")

    fig.tight_layout()
    path = os.path.join(output_dir, "test3_alpha_spatial.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='BP Diagnosis')
    parser.add_argument('--contrastive_ckpt', required=True)
    parser.add_argument('--seg_ckpt', default=None)
    parser.add_argument('--data_dir', default='./input')
    parser.add_argument('--output_dir', default='./output/bp_diagnosis')
    parser.add_argument('--diagonal_pairwise', action='store_true',
                        help='Use DiagonalPairwiseHead (must match saved checkpoint)')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    encoder, dhbp = load_models(args.contrastive_ckpt, args.seg_ckpt, device,
                                diagonal_pairwise=args.diagonal_pairwise)

    # Test 1: Horizontal propagation (synthetic)
    test_horizontal_propagation(dhbp, device, args.output_dir)

    # Test 2: Where does BP change predictions (real data)
    test_prediction_changes(encoder, dhbp, device, args.output_dir, args.data_dir)

    # Test 3: Alpha spatial pattern (real data)
    test_alpha_spatial(encoder, dhbp, device, args.output_dir, args.data_dir)

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
