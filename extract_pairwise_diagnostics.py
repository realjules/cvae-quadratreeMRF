"""
Extract pairwise potential diagnostics from trained DHBP checkpoints.

For each checkpoint, computes:
  - Alpha (consistency strength) distribution: mean, std, range
  - Effective ψ = α·I + (1-α)·R averaged across spatial locations
  - Diagonal ratio = trace(ψ) / sum(ψ)
  - Max off-diagonal entry and which class pair
  - Per-class diagonal strength
  - K×K heatmap saved as PNG

Usage:
    # Single checkpoint
    python extract_pairwise_diagnostics.py \
        --seg_ckpt output/best_segmentation.pth

    # Multiple checkpoints (3-seed batch)
    python extract_pairwise_diagnostics.py \
        --seg_ckpt output_pct10_seed0/best_segmentation.pth \
                   output_pct10_seed1/best_segmentation.pth \
                   output_pct10_seed2/best_segmentation.pth \
        --output_dir paper_figures/

    # With real data (computes alpha on test images, not random)
    python extract_pairwise_diagnostics.py \
        --seg_ckpt output/best_segmentation.pth \
        --data_dir ./input
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule


CLASS_NAMES = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]


def detect_pairwise_head(dhbp_state):
    """Detect which pairwise head a checkpoint was trained with.

    Returns one of: 'constrained' (alpha*I + (1-alpha)*R),
    'unconstrained' (raw K*K conv, Experiment H), 'diagonal' (Potts-style).
    """
    if any(k.startswith('pairwise_12.alpha_net') for k in dhbp_state):
        return 'constrained'
    w = dhbp_state.get('pairwise_12.net.3.weight')
    if w is not None:
        # Last conv outputs K*K=36 channels for unconstrained, K=6 for diagonal
        return 'unconstrained' if w.shape[0] == 36 else 'diagonal'
    raise ValueError("Cannot detect pairwise head type from checkpoint keys: "
                     + ", ".join(sorted(k for k in dhbp_state if k.startswith('pairwise_12'))[:6]))


def load_model(seg_ckpt, device):
    """Load encoder + DHBP from a segmentation checkpoint (any pairwise head type)."""
    ckpt = torch.load(seg_ckpt, map_location=device, weights_only=False)

    head_type = 'constrained'
    if 'dhbp_state_dict' in ckpt:
        head_type = detect_pairwise_head(ckpt['dhbp_state_dict'])
    print(f"  Pairwise head type: {head_type}")

    encoder = ContrastiveEncoder(pretrained=True).to(device)
    dhbp = DHBPModule(
        n_classes=6,
        unconstrained_pairwise=(head_type == 'unconstrained'),
        diagonal_pairwise=(head_type == 'diagonal'),
    ).to(device)

    if 'encoder_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    if 'dhbp_state_dict' in ckpt:
        dhbp.load_state_dict(ckpt['dhbp_state_dict'])

    encoder.eval()
    dhbp.eval()
    return encoder, dhbp


def get_input_batch(data_dir, device, batch_size=4):
    """Get a batch of images. Uses real data if available, random otherwise."""
    if data_dir and os.path.exists(os.path.join(data_dir, "top")):
        try:
            from complete_training import create_real_dataloaders
            _, _, test_loader = create_real_dataloaders(
                data_dir=data_dir, batch_size=batch_size, labeled_percent=10,
            )
            images, labels = next(iter(test_loader))
            return images.to(device), labels.to(device)
        except Exception as e:
            print(f"  Could not load real data: {e}. Using random input.")

    torch.manual_seed(42)
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    labels = torch.randint(0, 6, (batch_size, 256, 256), device=device)
    return images, labels


@torch.no_grad()
def extract_diagnostics(encoder, dhbp, images):
    """Extract pairwise potential diagnostics from a batch of images.

    Computes the effective ψ via the head's own forward() so the SAME protocol
    applies to every head type (constrained, unconstrained, diagonal) — the
    March 2026 numbers were computed by two different scripts, which the
    June 2026 audit flagged as a protocol inconsistency.

    Chance level for diag_ratio under row-softmax normalization is 1/K ≈ 0.167
    (NOT 0.5): a random-init unconstrained head sits at ~0.167; values below
    it indicate actively learned anti-diagonal (remapping) structure.

    Returns dict with the average K×K pairwise matrix, diagonal metrics, and
    alpha stats when the head has them (constrained head only).
    """
    p1, p2, p3 = encoder.encode(images)

    pairwise = dhbp.pairwise_12
    K = pairwise.n_classes

    # Universal protocol: effective ψ from the head's forward (log-space → prob)
    log_psi = pairwise(p2)                            # [B, K, K, H, W]
    psi = torch.exp(log_psi)

    # Average across batch and spatial dimensions → K×K
    avg_psi = psi.mean(dim=(0, 3, 4)).cpu().numpy()  # [K, K]

    # Diagonal ratio (chance level = 1/K)
    diag_sum = np.trace(avg_psi)
    total_sum = avg_psi.sum()
    diag_ratio = diag_sum / total_sum if total_sum > 0 else 0.0

    # Max off-diagonal
    off_diag = avg_psi.copy()
    np.fill_diagonal(off_diag, 0)
    max_off_idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
    max_off_val = off_diag[max_off_idx]
    max_off_pair = (CLASS_NAMES[max_off_idx[0]], CLASS_NAMES[max_off_idx[1]])

    # Per-class diagonal strength
    per_class_diag = {CLASS_NAMES[i]: avg_psi[i, i] for i in range(K)}

    result = {
        'avg_psi': avg_psi,
        'diag_ratio': diag_ratio,
        'chance_diag_ratio': 1.0 / K,
        'max_off_diagonal': max_off_val,
        'max_off_pair': max_off_pair,
        'per_class_diag': per_class_diag,
    }

    # Alpha statistics — constrained head only
    if hasattr(pairwise, 'alpha_net'):
        alpha = torch.sigmoid(pairwise.alpha_net(p2))
        alpha_np = alpha.cpu().numpy().flatten()
        result.update({
            'alpha_mean': float(alpha_np.mean()),
            'alpha_std': float(alpha_np.std()),
            'alpha_min': float(alpha_np.min()),
            'alpha_max': float(alpha_np.max()),
        })
    else:
        result.update({'alpha_mean': None, 'alpha_std': None,
                       'alpha_min': None, 'alpha_max': None})

    return result


def plot_heatmap(avg_psi, title, output_path):
    """Plot K×K pairwise matrix as an annotated heatmap."""
    K = avg_psi.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(avg_psi, cmap='Blues', vmin=0, vmax=avg_psi.max())

    # Annotations
    for i in range(K):
        for j in range(K):
            val = avg_psi[i, j]
            color = 'white' if val > avg_psi.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    short_names = ["Imp", "Bld", "LVg", "Tre", "Car", "Clu"]
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Child class (j)", fontsize=11)
    ax.set_ylabel("Parent class (i)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def print_diagnostics(diag, ckpt_name):
    """Print diagnostics for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"  Checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    if diag.get('alpha_mean') is not None:
        print(f"\n  Alpha (consistency strength):")
        print(f"    Mean:  {diag['alpha_mean']:.4f}")
        print(f"    Std:   {diag['alpha_std']:.4f}")
        print(f"    Range: [{diag['alpha_min']:.4f}, {diag['alpha_max']:.4f}]")
    else:
        print(f"\n  Alpha: n/a (head has no alpha_net — unconstrained/diagonal arm)")

    print(f"\n  Pairwise matrix (avg across spatial locations):")
    print(f"    Diagonal ratio: {diag['diag_ratio']:.4f}  (chance level = 1/K = {diag['chance_diag_ratio']:.3f})")
    print(f"    Max off-diagonal: {diag['max_off_diagonal']:.4f} "
          f"({diag['max_off_pair'][0]} → {diag['max_off_pair'][1]})")

    print(f"\n  Per-class diagonal strength:")
    for name, val in diag['per_class_diag'].items():
        bar = '█' * int(val * 40)
        print(f"    {name:<12} {val:.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract pairwise potential diagnostics from trained checkpoints')
    parser.add_argument('--seg_ckpt', nargs='+', required=True,
                        help='Path(s) to segmentation checkpoint(s)')
    parser.add_argument('--data_dir', default=None,
                        help='ISPRS data directory (for real test images). Uses random if not provided.')
    parser.add_argument('--output_dir', default='./paper_figures',
                        help='Directory to save heatmap PNGs and JSON results')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Get input batch (shared across all checkpoints for consistency)
    images, labels = get_input_batch(args.data_dir, device)
    print(f"Input: {images.shape} ({'real data' if args.data_dir else 'random'})")

    all_diagnostics = []

    for ckpt_path in args.seg_ckpt:
        # Handle directory paths (Kaggle model format)
        if os.path.isdir(ckpt_path):
            import glob
            matches = glob.glob(os.path.join(ckpt_path, "*.pth"))
            if matches:
                ckpt_path = matches[0]
            else:
                print(f"  SKIP: no .pth file found in {ckpt_path}")
                continue

        ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
        if not ckpt_name or ckpt_name == '.':
            ckpt_name = os.path.basename(ckpt_path).replace('.pth', '')

        encoder, dhbp = load_model(ckpt_path, device)
        diag = extract_diagnostics(encoder, dhbp, images)
        print_diagnostics(diag, ckpt_name)

        # Save heatmap
        head_label = ("ψ = α·I + (1-α)·R" if diag['alpha_mean'] is not None
                      else "ψ (unconstrained / diagonal head)")
        plot_heatmap(
            diag['avg_psi'],
            f"Pairwise {head_label}\n{ckpt_name} (diag ratio: {diag['diag_ratio']:.3f}, chance 0.167)",
            os.path.join(args.output_dir, f"pairwise_heatmap_{ckpt_name}.png"),
        )

        # Store for multi-seed summary (convert numpy types for JSON)
        serializable = {
            'checkpoint': ckpt_path,
            'name': ckpt_name,
            'diag_ratio': float(diag['diag_ratio']),
            'chance_diag_ratio': float(diag['chance_diag_ratio']),
            'max_off_diagonal': float(diag['max_off_diagonal']),
            'max_off_pair': list(diag['max_off_pair']),
            'per_class_diag': {k: float(v) for k, v in diag['per_class_diag'].items()},
            'alpha_mean': None if diag['alpha_mean'] is None else float(diag['alpha_mean']),
            'alpha_std': None if diag['alpha_std'] is None else float(diag['alpha_std']),
            'alpha_min': None if diag['alpha_min'] is None else float(diag['alpha_min']),
            'alpha_max': None if diag['alpha_max'] is None else float(diag['alpha_max']),
            'avg_psi': diag['avg_psi'].tolist(),
        }
        all_diagnostics.append(serializable)

        del encoder, dhbp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Multi-seed summary (if more than 1 checkpoint)
    if len(all_diagnostics) > 1:
        print(f"\n{'='*60}")
        print(f"  MULTI-SEED SUMMARY ({len(all_diagnostics)} checkpoints)")
        print(f"{'='*60}")

        diag_ratios = [d['diag_ratio'] for d in all_diagnostics]
        max_offs = [d['max_off_diagonal'] for d in all_diagnostics]
        alpha_means = [d['alpha_mean'] for d in all_diagnostics if d['alpha_mean'] is not None]

        print(f"  Diagonal ratio:    {np.mean(diag_ratios):.4f} ± {np.std(diag_ratios):.4f}  (chance = 0.167)")
        print(f"  Max off-diagonal:  {np.mean(max_offs):.4f} ± {np.std(max_offs):.4f}")
        if alpha_means:
            print(f"  Alpha mean:        {np.mean(alpha_means):.4f} ± {np.std(alpha_means):.4f}")

        # Average pairwise matrix across seeds
        avg_matrices = np.array([d['avg_psi'] for d in all_diagnostics])
        mean_matrix = avg_matrices.mean(axis=0)
        std_matrix = avg_matrices.std(axis=0)

        plot_heatmap(
            mean_matrix,
            f"Pairwise ψ (mean of {len(all_diagnostics)} seeds)\n"
            f"diag ratio: {np.mean(diag_ratios):.3f} ± {np.std(diag_ratios):.3f}",
            os.path.join(args.output_dir, "pairwise_heatmap_mean.png"),
        )

    # Save JSON results
    json_path = os.path.join(args.output_dir, "pairwise_diagnostics.json")
    with open(json_path, 'w') as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
