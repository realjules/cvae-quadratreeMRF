"""
Full pipeline diagnostic: encoder → DHBP input → DHBP output.

Evaluates where the bottleneck is:
  Stage A: Are the encoder features discriminative? (linear probe + t-SNE)
  Stage B: Are the DHBP potentials meaningful? (unary accuracy + pairwise analysis)
  Stage C: Does BP improve predictions? (belief accuracy vs unary, entropy, per-class)

Usage:
    python evaluate_encoder.py \
        --contrastive_ckpt /path/to/contrastive_best.pth \
        --output_dir ./output/eval

    # With trained segmentation model:
    python evaluate_encoder.py \
        --contrastive_ckpt /path/to/contrastive_best.pth \
        --seg_ckpt /path/to/best_segmentation.pth \
        --output_dir ./output/eval
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
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset.dataset import ISPRS_dataset
from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule


CLASS_NAMES = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]
CLASS_COLORS = ['#ffffff', '#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff0000']


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_area_ids(data_dir="./input"):
    top_files = glob.glob(os.path.join(data_dir, "top", "top_mosaic_09cm_area*.tif"))
    gt_files = glob.glob(os.path.join(data_dir, "gt", "top_mosaic_09cm_area*.tif"))
    top_ids = [f.split('area')[1].split('.')[0] for f in top_files]
    gt_ids = [f.split('area')[1].split('.')[0] for f in gt_files]
    return sorted(set(top_ids) & set(gt_ids), key=lambda x: int(x))


def make_loader(ids, data_dir, batch_size=4, augmentation=False):
    top_pat = os.path.join(data_dir, "top", "top_mosaic_09cm_area{}.tif")
    gt_pat = os.path.join(data_dir, "gt", "top_mosaic_09cm_area{}.tif")
    ds = ISPRS_dataset(
        ids=ids, ids_type='TEST', gt_type='full', gt_modification=None,
        data_files=top_pat, label_files=gt_pat,
        window_size=256, cache=False, augmentation=augmentation,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=False)


def load_encoder(ckpt_path, device):
    encoder = ContrastiveEncoder(pretrained=True)
    if os.path.isdir(ckpt_path):
        pth_files = glob.glob(os.path.join(ckpt_path, "*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth in {ckpt_path}")
        ckpt_path = pth_files[0]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    epoch = ckpt.get('epoch', '?')
    print(f"Encoder loaded from {ckpt_path} (epoch {epoch})")
    return encoder.to(device)


# ---------------------------------------------------------------------------
# Stage A: Encoder evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_pixel_features(encoder, loader, device, max_patches=500):
    """Extract per-pixel features from p1 and corresponding labels."""
    encoder.eval()
    all_feats = []
    all_labels = []
    count = 0

    for images, labels in loader:
        if count >= max_patches:
            break
        images = images.to(device)
        labels = labels.long()

        p1, _, _ = encoder.encode(images)  # [B, 64, 128, 128]
        # Downsample labels to match p1 spatial size
        labels_ds = F.interpolate(
            labels.unsqueeze(1).float(), size=p1.shape[2:], mode='nearest'
        ).squeeze(1).long()  # [B, 128, 128]

        all_feats.append(p1.cpu())
        all_labels.append(labels_ds.cpu())
        count += images.size(0)

    feats = torch.cat(all_feats, dim=0)    # [N, 64, 128, 128]
    labels = torch.cat(all_labels, dim=0)  # [N, 128, 128]
    return feats, labels


def pixel_linear_probe(train_feats, train_labels, test_feats, test_labels,
                       n_classes=6, epochs=50, lr=0.01, device='cpu'):
    """Train a Conv1x1(64→6) on frozen features, evaluate per-pixel."""
    probe = nn.Conv2d(64, n_classes, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Train on random subset of patches
    n_train = min(200, train_feats.size(0))
    for epoch in range(epochs):
        idx = torch.randperm(train_feats.size(0))[:n_train]
        for i in range(0, n_train, 8):
            batch_idx = idx[i:i+8]
            f = train_feats[batch_idx].to(device)
            l = train_labels[batch_idx].to(device)
            logits = probe(f)
            loss = F.cross_entropy(logits, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(n_classes)
    class_total = torch.zeros(n_classes)

    with torch.no_grad():
        for i in range(0, test_feats.size(0), 8):
            f = test_feats[i:i+8].to(device)
            l = test_labels[i:i+8]
            preds = probe(f).argmax(dim=1).cpu()
            correct += (preds == l).sum().item()
            total += l.numel()
            for c in range(n_classes):
                mask = l == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()

    overall = correct / total * 100
    per_class = []
    for c in range(n_classes):
        if class_total[c] > 0:
            per_class.append((class_correct[c] / class_total[c] * 100).item())
        else:
            per_class.append(0.0)

    return overall, per_class


def compute_feature_stats(feats, labels, n_classes=6):
    """Mean and std of features per class."""
    # feats: [N, 64, H, W], labels: [N, H, W]
    stats = {}
    feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, 64)  # [N*H*W, 64]
    labels_flat = labels.reshape(-1)

    for c in range(n_classes):
        mask = labels_flat == c
        if mask.sum() > 100:
            cf = feats_flat[mask]
            stats[CLASS_NAMES[c]] = {
                'mean_norm': cf.mean(dim=0).norm().item(),
                'std': cf.std().item(),
                'count': mask.sum().item(),
            }
        else:
            stats[CLASS_NAMES[c]] = {'mean_norm': 0, 'std': 0, 'count': mask.sum().item()}
    return stats


def plot_tsne(feats, labels, output_path, n_classes=6, max_points=3000):
    """t-SNE of pixel features colored by class."""
    # feats: [N, 64, H, W], labels: [N, H, W]
    feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, 64).numpy()
    labels_flat = labels.reshape(-1).numpy()

    # Subsample
    if len(feats_flat) > max_points:
        idx = np.random.choice(len(feats_flat), max_points, replace=False)
        feats_flat = feats_flat[idx]
        labels_flat = labels_flat[idx]

    print("  Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000).fit_transform(feats_flat)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for c in range(n_classes):
        mask = labels_flat == c
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=CLASS_COLORS[c], label=f"{CLASS_NAMES[c]} ({mask.sum()})",
                       s=5, alpha=0.5, edgecolors='none')
    ax.legend(loc='best', fontsize=9)
    ax.set_title("t-SNE of Encoder Pixel Features (p1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_stage_a(encoder, train_loader, test_loader, device, output_dir):
    """Stage A: Encoder feature quality."""
    print("\n" + "=" * 60)
    print("STAGE A: ENCODER FEATURE QUALITY")
    print("=" * 60)

    # Extract features
    print("Extracting train features...")
    train_feats, train_labels = extract_pixel_features(encoder, train_loader, device, max_patches=300)
    print("Extracting test features...")
    test_feats, test_labels = extract_pixel_features(encoder, test_loader, device, max_patches=200)
    print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")

    # Linear probe
    print("Training pixel-level linear probe...")
    acc, per_class = pixel_linear_probe(train_feats, train_labels, test_feats, test_labels, device=device)
    print(f"  Linear probe accuracy: {acc:.2f}%")
    for name, a in zip(CLASS_NAMES, per_class):
        print(f"    {name}: {a:.2f}%")

    # Random encoder baseline
    print("Baseline: random encoder...")
    random_enc = ContrastiveEncoder(pretrained=False).to(device)
    rand_train_feats, _ = extract_pixel_features(random_enc, train_loader, device, max_patches=300)
    rand_test_feats, _ = extract_pixel_features(random_enc, test_loader, device, max_patches=200)
    rand_acc, rand_per_class = pixel_linear_probe(
        rand_train_feats, train_labels, rand_test_feats, test_labels, device=device
    )
    print(f"  Random encoder accuracy: {rand_acc:.2f}%")
    del random_enc, rand_train_feats, rand_test_feats
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Feature stats
    stats = compute_feature_stats(test_feats, test_labels)
    print("Feature stats per class:")
    for name, s in stats.items():
        print(f"    {name}: mean_norm={s['mean_norm']:.3f}, std={s['std']:.3f}, pixels={s['count']}")

    # t-SNE
    plot_tsne(test_feats, test_labels, os.path.join(output_dir, "tsne_encoder.png"))

    # Verdict
    improvement = acc - rand_acc
    if acc >= 75:
        verdict = "GOOD"
    elif acc >= 60:
        verdict = "MODERATE"
    else:
        verdict = "POOR"

    results = {
        'linear_probe_acc': acc,
        'random_baseline_acc': rand_acc,
        'improvement': improvement,
        'per_class': per_class,
        'verdict': verdict,
    }
    print(f"\n  Verdict: {verdict} (contrastive={acc:.1f}%, random={rand_acc:.1f}%, delta=+{improvement:.1f}%)")
    return results, test_feats, test_labels


# ---------------------------------------------------------------------------
# Stage B: DHBP input — potentials
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_stage_b(encoder, dhbp, test_loader, device, output_dir, max_batches=50):
    """Stage B: DHBP potential quality."""
    print("\n" + "=" * 60)
    print("STAGE B: DHBP POTENTIAL QUALITY")
    print("=" * 60)

    encoder.eval()
    dhbp.eval()

    unary_correct = 0
    unary_total = 0
    unary_class_correct = torch.zeros(6)
    unary_class_total = torch.zeros(6)

    all_psi_12 = []
    all_psi_12_boundary = []
    all_psi_12_interior = []
    all_diag_ratios = []

    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.long()
        labels_ds = F.interpolate(
            labels.unsqueeze(1).float(), size=(128, 128), mode='nearest'
        ).squeeze(1).long()

        p1, p2, p3 = encoder.encode(images)
        diag = dhbp.forward_diagnostic(p1, p2, p3)

        # Unary-only accuracy (φ₁ without BP)
        phi_1 = diag['phi_1']  # [B, 6, 128, 128]
        unary_preds = phi_1.argmax(dim=1).cpu()
        unary_correct += (unary_preds == labels_ds).sum().item()
        unary_total += labels_ds.numel()
        for c in range(6):
            mask = labels_ds == c
            if mask.sum() > 0:
                unary_class_correct[c] += (unary_preds[mask] == c).sum().item()
                unary_class_total[c] += mask.sum().item()

        # Pairwise analysis (ψ₁₂)
        psi_12 = diag['psi_12']  # [B, 6, 6, 64, 64]
        # Convert to probability for analysis
        psi_prob = torch.exp(psi_12)  # [B, 6, 6, 64, 64]
        avg_psi = psi_prob.mean(dim=(0, 3, 4))  # [6, 6] average matrix
        all_psi_12.append(avg_psi.cpu())

        # Diagonal ratio per pixel: sum(diag) / sum(all) in the 6×6 matrix
        diag_vals = torch.diagonal(psi_prob, dim1=1, dim2=2)  # [B, 64, 64, 6]
        diag_sum = diag_vals.sum(dim=-1)  # [B, 64, 64]
        total_sum = psi_prob.sum(dim=(1, 2))  # [B, 64, 64]
        diag_ratio = (diag_sum / (total_sum + 1e-8))  # [B, 64, 64]
        all_diag_ratios.append(diag_ratio.mean().cpu().item())

        # Boundary vs interior analysis
        labels_64 = F.interpolate(
            labels.unsqueeze(1).float(), size=(64, 64), mode='nearest'
        ).squeeze(1).long()
        # Detect boundaries: where label changes between neighbors
        pad_lab = F.pad(labels_64.float().unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
        boundary = (
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 1:-1, 2:]) |
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 1:-1, :-2]) |
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, 2:, 1:-1]) |
            (pad_lab[:, 1:-1, 1:-1] != pad_lab[:, :-2, 1:-1])
        )  # [B, 64, 64] bool

        for b in range(images.size(0)):
            bnd = boundary[b]  # [64, 64]
            psi_b = psi_prob[b]  # [6, 6, 64, 64]
            if bnd.sum() > 0:
                bnd_psi = psi_b[:, :, bnd].mean(dim=-1)  # [6, 6]
                all_psi_12_boundary.append(bnd_psi.cpu())
            interior = ~bnd
            if interior.sum() > 0:
                int_psi = psi_b[:, :, interior].mean(dim=-1)  # [6, 6]
                all_psi_12_interior.append(int_psi.cpu())

    # Aggregate results
    unary_acc = unary_correct / unary_total * 100
    unary_per_class = []
    for c in range(6):
        if unary_class_total[c] > 0:
            unary_per_class.append((unary_class_correct[c] / unary_class_total[c] * 100).item())
        else:
            unary_per_class.append(0.0)

    avg_psi_global = torch.stack(all_psi_12).mean(dim=0).numpy()  # [6, 6]
    avg_diag_ratio = np.mean(all_diag_ratios)

    print(f"  Unary-only accuracy (φ₁, no BP): {unary_acc:.2f}%")
    for name, a in zip(CLASS_NAMES, unary_per_class):
        print(f"    {name}: {a:.2f}%")

    print(f"\n  Pairwise diagonal ratio: {avg_diag_ratio:.4f} (>0.5 = diagonal-dominant)")
    print(f"  Average ψ₁₂ matrix (probability space):")
    print("         " + "  ".join(f"{n[:4]:>6}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{avg_psi_global[i, j]:6.3f}" for j in range(6))
        print(f"    {name[:4]:>4}: {row}")

    # Boundary vs interior
    if all_psi_12_boundary and all_psi_12_interior:
        bnd_avg = torch.stack(all_psi_12_boundary).mean(dim=0).numpy()
        int_avg = torch.stack(all_psi_12_interior).mean(dim=0).numpy()
        bnd_diag = np.diag(bnd_avg).mean()
        int_diag = np.diag(int_avg).mean()
        diff = int_diag - bnd_diag
        print(f"\n  Boundary diagonal avg: {bnd_diag:.4f}")
        print(f"  Interior diagonal avg: {int_diag:.4f}")
        print(f"  Difference: {diff:.4f} (>0 means interior is more consistent, good)")
    else:
        bnd_avg = int_avg = None
        diff = 0

    # Plot pairwise heatmaps
    fig, axes = plt.subplots(1, 3 if bnd_avg is not None else 1, figsize=(15 if bnd_avg is not None else 6, 5))
    if bnd_avg is None:
        axes = [axes]

    ax = axes[0] if bnd_avg is not None else axes[0]
    im = ax.imshow(avg_psi_global, cmap='Blues', vmin=0)
    ax.set_xticks(range(6)); ax.set_xticklabels([n[:3] for n in CLASS_NAMES], rotation=45)
    ax.set_yticks(range(6)); ax.set_yticklabels([n[:3] for n in CLASS_NAMES])
    ax.set_title("Average ψ₁₂ (global)")
    ax.set_xlabel("Child class"); ax.set_ylabel("Parent class")
    plt.colorbar(im, ax=ax)

    if bnd_avg is not None:
        ax = axes[1]
        im = ax.imshow(bnd_avg, cmap='Reds', vmin=0)
        ax.set_xticks(range(6)); ax.set_xticklabels([n[:3] for n in CLASS_NAMES], rotation=45)
        ax.set_yticks(range(6)); ax.set_yticklabels([n[:3] for n in CLASS_NAMES])
        ax.set_title("ψ₁₂ at BOUNDARIES")
        plt.colorbar(im, ax=ax)

        ax = axes[2]
        im = ax.imshow(int_avg, cmap='Greens', vmin=0)
        ax.set_xticks(range(6)); ax.set_xticklabels([n[:3] for n in CLASS_NAMES], rotation=45)
        ax.set_yticks(range(6)); ax.set_yticklabels([n[:3] for n in CLASS_NAMES])
        ax.set_title("ψ₁₂ at INTERIOR")
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    path = os.path.join(output_dir, "pairwise_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return {
        'unary_acc': unary_acc,
        'unary_per_class': unary_per_class,
        'diag_ratio': avg_diag_ratio,
        'boundary_interior_diff': diff,
    }


# ---------------------------------------------------------------------------
# Stage C: DHBP output — does BP help?
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_stage_c(encoder, dhbp, test_loader, device, output_dir,
                unary_per_class, max_batches=50):
    """Stage C: Does belief propagation improve predictions?"""
    print("\n" + "=" * 60)
    print("STAGE C: DOES BP IMPROVE PREDICTIONS?")
    print("=" * 60)

    encoder.eval()
    dhbp.eval()

    belief_correct = 0
    belief_total = 0
    belief_class_correct = torch.zeros(6)
    belief_class_total = torch.zeros(6)
    unary_entropies = []
    belief_entropies = []

    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.long()
        labels_ds = F.interpolate(
            labels.unsqueeze(1).float(), size=(128, 128), mode='nearest'
        ).squeeze(1).long()

        p1, p2, p3 = encoder.encode(images)
        diag = dhbp.forward_diagnostic(p1, p2, p3)

        # Final belief accuracy
        b1_final = diag['b1_final']  # [B, 6, 128, 128]
        belief_preds = b1_final.argmax(dim=1).cpu()
        belief_correct += (belief_preds == labels_ds).sum().item()
        belief_total += labels_ds.numel()
        for c in range(6):
            mask = labels_ds == c
            if mask.sum() > 0:
                belief_class_correct[c] += (belief_preds[mask] == c).sum().item()
                belief_class_total[c] += mask.sum().item()

        # Entropy comparison
        phi_1 = diag['phi_1']
        unary_probs = F.softmax(phi_1, dim=1)
        belief_probs = F.softmax(b1_final, dim=1)
        unary_ent = -(unary_probs * torch.log(unary_probs + 1e-8)).sum(dim=1).mean().cpu().item()
        belief_ent = -(belief_probs * torch.log(belief_probs + 1e-8)).sum(dim=1).mean().cpu().item()
        unary_entropies.append(unary_ent)
        belief_entropies.append(belief_ent)

    # Results
    belief_acc = belief_correct / belief_total * 100
    belief_per_class = []
    for c in range(6):
        if belief_class_total[c] > 0:
            belief_per_class.append((belief_class_correct[c] / belief_class_total[c] * 100).item())
        else:
            belief_per_class.append(0.0)

    avg_unary_ent = np.mean(unary_entropies)
    avg_belief_ent = np.mean(belief_entropies)

    print(f"  Final belief accuracy (after BP): {belief_acc:.2f}%")
    print(f"  Entropy: unary={avg_unary_ent:.4f}, belief={avg_belief_ent:.4f}, "
          f"reduction={avg_unary_ent - avg_belief_ent:.4f}")

    print(f"\n  Per-class comparison (unary → belief):")
    deltas = []
    for name, u, b in zip(CLASS_NAMES, unary_per_class, belief_per_class):
        delta = b - u
        deltas.append(delta)
        arrow = "+" if delta >= 0 else ""
        flag = " ← BP HURTING" if delta < -2 else ""
        print(f"    {name}: {u:.1f}% → {b:.1f}% ({arrow}{delta:.1f}%){flag}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(6)
    w = 0.35
    ax.bar(x - w/2, unary_per_class, w, label='Unary only (no BP)', color='#ff9999')
    ax.bar(x + w/2, belief_per_class, w, label='After BP', color='#66b3ff')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class: Unary vs After BP")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "unary_vs_belief.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return {
        'belief_acc': belief_acc,
        'belief_per_class': belief_per_class,
        'unary_entropy': avg_unary_ent,
        'belief_entropy': avg_belief_ent,
        'per_class_deltas': deltas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Diagnostic')
    parser.add_argument('--contrastive_ckpt', required=True)
    parser.add_argument('--seg_ckpt', default=None,
                        help='Trained segmentation model (encoder+DHBP). '
                             'If not provided, uses random DHBP weights.')
    parser.add_argument('--data_dir', default='./input')
    parser.add_argument('--output_dir', default='./output/eval')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("FULL PIPELINE DIAGNOSTIC")
    print(f"Device: {device}")

    # Load encoder
    encoder = load_encoder(args.contrastive_ckpt, device)

    # Load DHBP
    dhbp = DHBPModule(n_classes=6).to(device)
    if args.seg_ckpt:
        print(f"Loading trained DHBP from {args.seg_ckpt}")
        seg_ckpt_path = args.seg_ckpt
        if os.path.isdir(seg_ckpt_path):
            pth_files = glob.glob(os.path.join(seg_ckpt_path, "*.pth"))
            seg_ckpt_path = pth_files[0]
        ckpt = torch.load(seg_ckpt_path, map_location=device, weights_only=False)
        if 'dhbp_state_dict' in ckpt:
            dhbp.load_state_dict(ckpt['dhbp_state_dict'])
            print("  DHBP weights loaded from segmentation checkpoint.")
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            print("  Encoder weights updated from segmentation checkpoint (fine-tuned).")
    else:
        print("No seg_ckpt provided — using RANDOM DHBP weights (Stage B/C will reflect untrained DHBP).")

    # Data
    valid_ids = get_area_ids(args.data_dir)
    if not valid_ids:
        raise ValueError(f"No data in {args.data_dir}")
    split = max(1, int(0.2 * len(valid_ids)))
    train_ids = valid_ids[:-split]
    test_ids = valid_ids[-split:]
    print(f"Train: {train_ids}, Test: {test_ids}")

    train_loader = make_loader(train_ids, args.data_dir, batch_size=4)
    test_loader = make_loader(test_ids, args.data_dir, batch_size=4)

    # Run all stages
    stage_a, _, _ = run_stage_a(encoder, train_loader, test_loader, device, args.output_dir)
    stage_b = run_stage_b(encoder, dhbp, test_loader, device, args.output_dir)
    stage_c = run_stage_c(encoder, dhbp, test_loader, device, args.output_dir,
                          stage_b['unary_per_class'])

    # Final summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)

    print(f"\nENCODER (Stage A):")
    print(f"  Linear probe accuracy:     {stage_a['linear_probe_acc']:.2f}%")
    print(f"  Random encoder baseline:   {stage_a['random_baseline_acc']:.2f}%")
    print(f"  Improvement over random:   +{stage_a['improvement']:.2f}%")
    print(f"  Verdict: {stage_a['verdict']}")

    print(f"\nDHBP INPUT (Stage B):")
    print(f"  Unary-only accuracy:       {stage_b['unary_acc']:.2f}%")
    print(f"  Pairwise diagonal ratio:   {stage_b['diag_ratio']:.4f}")
    print(f"  Boundary-interior diff:    {stage_b['boundary_interior_diff']:.4f}")

    print(f"\nDHBP OUTPUT (Stage C):")
    print(f"  Final belief accuracy:     {stage_c['belief_acc']:.2f}%")
    bp_delta = stage_c['belief_acc'] - stage_b['unary_acc']
    print(f"  BP improvement over unary: {'+' if bp_delta >= 0 else ''}{bp_delta:.2f}%")
    print(f"  Entropy reduction:         {stage_c['unary_entropy'] - stage_c['belief_entropy']:.4f} bits")
    print(f"  Per-class delta:")
    for name, d in zip(CLASS_NAMES, stage_c['per_class_deltas']):
        flag = " ← BP HURTING" if d < -2 else ""
        print(f"    {name}: {'+' if d >= 0 else ''}{d:.1f}%{flag}")

    # Bottleneck identification
    print(f"\nBOTTLENECK ANALYSIS:")
    if stage_a['linear_probe_acc'] < 60:
        print("  → Encoder features are POOR. Fix contrastive pre-training first.")
    elif stage_b['unary_acc'] < stage_a['linear_probe_acc'] - 10:
        print("  → Unary heads are losing information. Fix unary potential architecture.")
    elif bp_delta < 0:
        print("  → BP is HURTING accuracy. Pairwise potentials are wrong. Fix pairwise training.")
    elif bp_delta < 3:
        print("  → BP is barely helping. Pairwise potentials are uninformative.")
    else:
        print("  → Pipeline is working. Focus on training longer or more labeled data.")


if __name__ == "__main__":
    main()
