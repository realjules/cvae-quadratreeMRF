"""
Measure gradient amplification as a function of quadtree depth.

Tests 2, 3, and 4 levels to see if amplification scales with depth.
Prediction: more levels = more gradient paths = more amplification.

Usage:
    python test_gradient_depth.py
"""

import torch
import torch.nn.functional as F
from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule
from net.loss import FocalLoss


def _grad_norm(param):
    if param.grad is not None:
        return param.grad.norm().item()
    return 0.0


def measure_at_depth(n_levels, encoder, focal, x, labels, device):
    """Measure gradient amplification for a given depth."""
    dhbp = DHBPModule(n_classes=6, n_levels=n_levels).to(device)

    # PATH A: Through BP
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x)
    b1_final = dhbp(p1, p2, p3)
    b1_up = F.interpolate(b1_final, size=(256, 256), mode='bilinear', align_corners=False)
    focal(b1_up, labels).backward()

    bp_unary = _grad_norm(dhbp.unary_1.net[0].weight)
    bp_encoder = _grad_norm(encoder.encoder.layer1[0].conv1.weight)

    # PATH B: Direct
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x)
    phi_1 = F.log_softmax(dhbp.unary_1(p1), dim=1)
    phi_1_up = F.interpolate(phi_1, size=(256, 256), mode='bilinear', align_corners=False)
    F.cross_entropy(phi_1_up, labels).backward()

    direct_unary = _grad_norm(dhbp.unary_1.net[0].weight)
    direct_encoder = _grad_norm(encoder.encoder.layer1[0].conv1.weight)

    ratio_unary = bp_unary / direct_unary if direct_unary > 1e-10 else float('inf')
    ratio_encoder = bp_encoder / direct_encoder if direct_encoder > 1e-10 else float('inf')

    del dhbp
    return ratio_unary, ratio_encoder, bp_unary, direct_unary


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    focal = FocalLoss(gamma=2.0, class_weights=torch.tensor([1.0, 1.5, 1.0, 1.0, 3.0, 1.2], device=device))

    print("=" * 60)
    print("GRADIENT AMPLIFICATION vs QUADTREE DEPTH")
    print("=" * 60)

    # Run each depth with 3 seeds for stability
    for n_levels in [2, 3, 4]:
        ratios_unary = []
        ratios_encoder = []

        for seed in range(3):
            torch.manual_seed(seed)
            encoder = ContrastiveEncoder(pretrained=True).to(device)
            x = torch.randn(2, 3, 256, 256, device=device)
            labels = torch.randint(0, 6, (2, 256, 256), device=device)

            ru, re, bp_u, dir_u = measure_at_depth(n_levels, encoder, focal, x, labels, device)
            ratios_unary.append(ru)
            ratios_encoder.append(re)

            del encoder
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        mean_u = sum(ratios_unary) / len(ratios_unary)
        mean_e = sum(ratios_encoder) / len(ratios_encoder)

        print(f"\n  {n_levels} LEVELS:")
        print(f"    Unary grad ratio:   {mean_u:.1f}x (seeds: {', '.join(f'{r:.1f}' for r in ratios_unary)})")
        print(f"    Encoder grad ratio: {mean_e:.1f}x (seeds: {', '.join(f'{r:.1f}' for r in ratios_encoder)})")

    print(f"\n{'='*60}")
    print("Does amplification scale with depth?")
    print("If 4 levels >> 3 levels >> 2 levels → yes, it scales")
    print("If roughly equal → amplification is per-edge, not cumulative")
    print("=" * 60)


if __name__ == "__main__":
    main()
