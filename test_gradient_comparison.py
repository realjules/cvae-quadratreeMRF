"""
Measure gradient amplification through BP at multiple training stages.

Tests whether the 10-32x gradient amplification we observed is:
  - A real property of the BP computation graph (persists across stages)
  - An initialization artifact (disappears with training)

Stages:
  1. Random init (encoder + DHBP both random)
  2. After contrastive (trained encoder, random DHBP)
  3. After segmentation (trained encoder + trained DHBP)

For each stage, compares:
  Path A: loss → BP chain → unary head (how training works)
  Path B: loss → unary head directly (no BP)

Usage:
    python test_gradient_comparison.py
    python test_gradient_comparison.py \
        --contrastive_ckpt /path/to/contrastive_best.pth \
        --seg_ckpt /path/to/best_segmentation.pth
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule
from net.loss import FocalLoss


def _grad_norm(param):
    """Safe gradient norm — returns 0.0 if no gradient."""
    if param.grad is not None:
        return param.grad.norm().item()
    return 0.0


def measure_gradients(encoder, dhbp, focal, x, labels, stage_name):
    """Measure gradient norms through BP chain vs direct path."""
    # PATH A: Through BP chain
    encoder.zero_grad()
    dhbp.zero_grad()

    p1, p2, p3 = encoder.encode(x)
    b1_final = dhbp(p1, p2, p3)
    b1_up = F.interpolate(b1_final, size=(256, 256), mode='bilinear', align_corners=False)
    loss_bp = focal(b1_up, labels)
    loss_bp.backward()

    grad_a = {
        'unary_1.net[0]': _grad_norm(dhbp.unary_1.net[0].weight),
        'unary_1.net[-1]': _grad_norm(dhbp.unary_1.net[-1].weight),
        'encoder.layer1': _grad_norm(encoder.encoder.layer1[0].conv1.weight),
    }

    # PATH B: Direct to unary (no BP)
    # Only uses p1 → unary_1, so only layer1 and unary_1 get gradients
    encoder.zero_grad()
    dhbp.zero_grad()

    p1, p2, p3 = encoder.encode(x)
    phi_1 = dhbp.unary_1(p1)
    phi_1 = F.log_softmax(phi_1, dim=1)
    phi_1_up = F.interpolate(phi_1, size=(256, 256), mode='bilinear', align_corners=False)
    loss_direct = F.cross_entropy(phi_1_up, labels)
    loss_direct.backward()

    grad_b = {
        'unary_1.net[0]': _grad_norm(dhbp.unary_1.net[0].weight),
        'unary_1.net[-1]': _grad_norm(dhbp.unary_1.net[-1].weight),
        'encoder.layer1': _grad_norm(encoder.encoder.layer1[0].conv1.weight),
    }

    # Print results
    print(f"\n  {stage_name}")
    print(f"  {'Component':<25} {'BP chain':>10} {'Direct':>10} {'Ratio':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

    ratios = []
    for name in grad_a:
        a = grad_a[name]
        b = grad_b[name]
        ratio = a / b if b > 1e-10 else float('inf')
        ratios.append(ratio)
        print(f"  {name:<25} {a:>10.6f} {b:>10.6f} {ratio:>7.1f}x")

    finite_ratios = [r for r in ratios if r != float('inf')]
    avg_ratio = sum(finite_ratios) / len(finite_ratios) if finite_ratios else 0.0
    return avg_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrastive_ckpt', default=None)
    parser.add_argument('--seg_ckpt', default=None)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    focal = FocalLoss(gamma=2.0, class_weights=torch.tensor([1.0, 1.5, 1.0, 1.0, 3.0, 1.2], device=device))

    # Fixed input for fair comparison across stages
    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256, device=device)
    labels = torch.randint(0, 6, (2, 256, 256), device=device)

    print("=" * 60)
    print("GRADIENT AMPLIFICATION: BP chain vs Direct")
    print("Is the 10-32x amplification real or an init artifact?")
    print("=" * 60)

    results = {}

    # =========================================================
    # Stage 1: Random init
    # =========================================================
    encoder = ContrastiveEncoder(pretrained=True).to(device)  # ImageNet pretrained
    dhbp = DHBPModule(n_classes=6).to(device)
    ratio = measure_gradients(encoder, dhbp, focal, x, labels, "STAGE 1: Random DHBP + pretrained encoder")
    results['random_init'] = ratio
    del encoder, dhbp
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================
    # Stage 2: After contrastive (if checkpoint provided)
    # =========================================================
    if args.contrastive_ckpt:
        encoder = ContrastiveEncoder(pretrained=True).to(device)
        ckpt_path = args.contrastive_ckpt
        if os.path.isdir(ckpt_path):
            ckpt_path = glob.glob(os.path.join(ckpt_path, "*.pth"))[0]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        dhbp = DHBPModule(n_classes=6).to(device)  # DHBP still random
        ratio = measure_gradients(encoder, dhbp, focal, x, labels,
                                  f"STAGE 2: Random DHBP + contrastive encoder (epoch {ckpt.get('epoch', '?')})")
        results['after_contrastive'] = ratio
        del encoder, dhbp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================
    # Stage 3: After segmentation (if checkpoint provided)
    # =========================================================
    if args.seg_ckpt:
        encoder = ContrastiveEncoder(pretrained=True).to(device)
        dhbp = DHBPModule(n_classes=6).to(device)
        seg_path = args.seg_ckpt
        if os.path.isdir(seg_path):
            seg_path = glob.glob(os.path.join(seg_path, "*.pth"))[0]
        seg = torch.load(seg_path, map_location=device, weights_only=False)
        if 'encoder_state_dict' in seg:
            encoder.load_state_dict(seg['encoder_state_dict'])
        if 'dhbp_state_dict' in seg:
            dhbp.load_state_dict(seg['dhbp_state_dict'])
        ratio = measure_gradients(encoder, dhbp, focal, x, labels,
                                  "STAGE 3: Trained DHBP + fine-tuned encoder (50 epochs)")
        results['after_segmentation'] = ratio
        del encoder, dhbp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================
    # Stage 4: Multiple random seeds (same stage 1, different inits)
    # =========================================================
    print(f"\n  STAGE 4: Random seed variation (5 seeds)")
    print(f"  {'Seed':<10} {'Avg ratio':>10}")
    print(f"  {'-'*10} {'-'*10}")
    seed_ratios = []
    for seed in [0, 1, 2, 3, 4]:
        torch.manual_seed(seed)
        encoder = ContrastiveEncoder(pretrained=True).to(device)
        dhbp = DHBPModule(n_classes=6).to(device)
        # Also randomize input
        x_seed = torch.randn(2, 3, 256, 256, device=device)
        labels_seed = torch.randint(0, 6, (2, 256, 256), device=device)

        # Quick measure (no print)
        encoder.zero_grad(); dhbp.zero_grad()
        p1, p2, p3 = encoder.encode(x_seed)
        b1 = dhbp(p1, p2, p3)
        b1_up = F.interpolate(b1, size=(256, 256), mode='bilinear', align_corners=False)
        focal(b1_up, labels_seed).backward()
        ga = dhbp.unary_1.net[0].weight.grad.norm().item()

        encoder.zero_grad(); dhbp.zero_grad()
        p1, p2, p3 = encoder.encode(x_seed)
        phi = F.log_softmax(dhbp.unary_1(p1), dim=1)
        phi_up = F.interpolate(phi, size=(256, 256), mode='bilinear', align_corners=False)
        F.cross_entropy(phi_up, labels_seed).backward()
        gb = dhbp.unary_1.net[0].weight.grad.norm().item()

        r = ga / gb if gb > 1e-10 else float('inf')
        seed_ratios.append(r)
        print(f"  {seed:<10} {r:>10.1f}x")

        del encoder, dhbp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results['seed_mean'] = sum(seed_ratios) / len(seed_ratios)
    results['seed_std'] = (sum((r - results['seed_mean'])**2 for r in seed_ratios) / len(seed_ratios)) ** 0.5

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY: Gradient amplification through BP")
    print(f"{'='*60}")
    print(f"  Random init:            {results.get('random_init', 'N/A'):.1f}x" if isinstance(results.get('random_init'), float) else "")
    if 'after_contrastive' in results:
        print(f"  After contrastive:      {results['after_contrastive']:.1f}x")
    if 'after_segmentation' in results:
        print(f"  After segmentation:     {results['after_segmentation']:.1f}x")
    print(f"  Across 5 random seeds:  {results['seed_mean']:.1f}x ± {results['seed_std']:.1f}x")

    print(f"\n  VERDICT:")
    all_ratios = [results.get('random_init', 0)]
    if 'after_contrastive' in results:
        all_ratios.append(results['after_contrastive'])
    if 'after_segmentation' in results:
        all_ratios.append(results['after_segmentation'])

    min_r = min(r for r in all_ratios if r > 0)
    max_r = max(all_ratios)

    if min_r > 3.0:
        print(f"    → Amplification is CONSISTENT across all stages ({min_r:.1f}x - {max_r:.1f}x)")
        print(f"    → This is a REAL property of the BP computation graph")
        print(f"    → Not an initialization artifact")
    elif min_r > 1.5:
        print(f"    → Amplification exists but VARIES across stages ({min_r:.1f}x - {max_r:.1f}x)")
        print(f"    → Partially a graph property, partially affected by training")
    else:
        print(f"    → Amplification DISAPPEARS with training ({min_r:.1f}x - {max_r:.1f}x)")
        print(f"    → It was an initialization artifact")


if __name__ == "__main__":
    main()
