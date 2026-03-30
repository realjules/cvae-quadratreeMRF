"""
Measure gradient DIRECTION alignment between BP and direct supervision paths.

Prior work measured gradient MAGNITUDE amplification (7-10x) but this was
confounded by loss magnitude difference (BP loss=4.3 vs direct loss=1.8).
This script answers: does BP change gradient DIRECTION?

Key design decisions:
  - SAME loss function (FocalLoss) on both paths to isolate BP as the only variable
  - Random baseline: cosine between two batches through the SAME path = null distribution
  - Pairwise params have NO direct gradient (BP-only) — this is a highlighted finding
  - Sweep data samples (not model inits) to measure stability across inputs

Interpretation:
  cos(∇BP, ∇Direct) >> random baseline → BP preserves gradient direction (scaling only)
  cos(∇BP, ∇Direct) ≈ random baseline → BP redirects gradients (different optimization)
  Pairwise params = BP-ONLY → BP introduces entirely new optimization dimensions

Note on FocalLoss input: Both dhbp() and unary_1() output log-softmax, but FocalLoss
expects raw logits (it uses F.cross_entropy which applies softmax internally). This
means softmax is applied to log-softmax on BOTH paths — same treatment, so the cosine
comparison is fair. This matches the training loop behavior (train.py:97).

Usage:
    python test_gradient_cosine.py
    python test_gradient_cosine.py --sanity-check
    python test_gradient_cosine.py \
        --contrastive_ckpt output/contrastive_best.pth \
        --seg_ckpt output/best_segmentation.pth
"""

import argparse
import glob
import os

import torch
import torch.nn.functional as F
from net.cvae import ContrastiveEncoder
from net.dhbp import DHBPModule
from net.loss import FocalLoss


# Parameter groups to measure: (display_name, accessor_function)
PARAMS_TO_MEASURE = [
    ('unary_1.net[0]',              lambda e, d: d.unary_1.net[0].weight),
    ('unary_1.net[-1]',             lambda e, d: d.unary_1.net[-1].weight),
    ('encoder.layer1',              lambda e, d: e.encoder.layer1[0].conv1.weight),
    ('encoder.layer3',              lambda e, d: e.encoder.layer3[0].conv1.weight),
    ('pairwise_12.alpha_net[0]',    lambda e, d: d.pairwise_12.alpha_net[0].weight),
    ('pairwise_12.residual_net[0]', lambda e, d: d.pairwise_12.residual_net[0].weight),
]

SHARED_PARAMS = [name for name, _ in PARAMS_TO_MEASURE if 'pairwise' not in name]
PAIRWISE_PARAMS = [name for name, _ in PARAMS_TO_MEASURE if 'pairwise' in name]


def _collect_grads(encoder, dhbp):
    """Clone full gradient tensors for all measured parameters.

    Returns dict mapping name -> flattened gradient tensor or None.
    Gradients are detached and cloned so they survive zero_grad().
    """
    grads = {}
    for name, accessor in PARAMS_TO_MEASURE:
        param = accessor(encoder, dhbp)
        if param.grad is not None:
            grads[name] = param.grad.detach().clone().flatten()
        else:
            grads[name] = None
    return grads


def _cosine_sim(a, b):
    """Cosine similarity between two flat tensors. Returns None if either is zero."""
    if a is None or b is None:
        return None
    if a.norm() < 1e-10 or b.norm() < 1e-10:
        return None
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def measure_cosine(encoder, dhbp, focal, x, labels):
    """Measure gradient direction alignment between BP and direct paths.

    Both paths use the SAME FocalLoss to isolate BP as the only variable.

    Returns:
        dict with keys: loss_bp, loss_direct, cosines, norms_bp, norms_direct
        cosines maps name -> float, 'BP_ONLY', or None
    """
    # PATH A: Through BP chain
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x)
    b1_final = dhbp(p1, p2, p3)
    b1_up = F.interpolate(b1_final, size=(256, 256), mode='bilinear', align_corners=False)
    loss_bp = focal(b1_up, labels)
    loss_bp.backward()
    grads_bp = _collect_grads(encoder, dhbp)
    loss_bp_val = loss_bp.item()

    # PATH B: Direct unary (SAME FocalLoss, no BP)
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x)
    phi_1 = dhbp.unary_1(p1)
    phi_1_up = F.interpolate(phi_1, size=(256, 256), mode='bilinear', align_corners=False)
    loss_direct = focal(phi_1_up, labels)
    loss_direct.backward()
    grads_direct = _collect_grads(encoder, dhbp)
    loss_direct_val = loss_direct.item()

    # Compute cosine similarity and norms per parameter
    cosines = {}
    norms_bp = {}
    norms_direct = {}

    for name, _ in PARAMS_TO_MEASURE:
        g_bp = grads_bp[name]
        g_dir = grads_direct[name]

        norms_bp[name] = g_bp.norm().item() if g_bp is not None else 0.0
        norms_direct[name] = g_dir.norm().item() if g_dir is not None else 0.0

        if g_bp is not None and g_dir is not None:
            cos = _cosine_sim(g_bp, g_dir)
            cosines[name] = cos
        elif g_bp is not None and g_dir is None:
            # Pairwise params: gradient only through BP path
            cosines[name] = 'BP_ONLY'
        else:
            cosines[name] = None

    return {
        'loss_bp': loss_bp_val,
        'loss_direct': loss_direct_val,
        'cosines': cosines,
        'norms_bp': norms_bp,
        'norms_direct': norms_direct,
    }


def measure_random_baseline(encoder, dhbp, focal, device):
    """Cosine between gradients from two different random batches through the SAME BP path.

    This establishes the null distribution: how similar are gradients when only
    the input data changes? In high-dimensional spaces (~295K params), random
    vectors are nearly orthogonal (cos ≈ 0).
    """
    torch.manual_seed(100)
    x1 = torch.randn(2, 3, 256, 256, device=device)
    labels1 = torch.randint(0, 6, (2, 256, 256), device=device)
    torch.manual_seed(200)
    x2 = torch.randn(2, 3, 256, 256, device=device)
    labels2 = torch.randint(0, 6, (2, 256, 256), device=device)

    # Grads from batch 1 through BP
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x1)
    b1 = dhbp(p1, p2, p3)
    b1_up = F.interpolate(b1, size=(256, 256), mode='bilinear', align_corners=False)
    focal(b1_up, labels1).backward()
    grads_1 = _collect_grads(encoder, dhbp)

    # Grads from batch 2 through BP
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x2)
    b1 = dhbp(p1, p2, p3)
    b1_up = F.interpolate(b1, size=(256, 256), mode='bilinear', align_corners=False)
    focal(b1_up, labels2).backward()
    grads_2 = _collect_grads(encoder, dhbp)

    baseline = {}
    for name, _ in PARAMS_TO_MEASURE:
        baseline[name] = _cosine_sim(grads_1.get(name), grads_2.get(name))
    return baseline


def measure_sanity_check(encoder, dhbp, focal, x, labels):
    """Both paths compute the SAME thing (direct unary). Cosine must be 1.0."""
    # Path A: direct unary
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x)
    phi_1 = dhbp.unary_1(p1)
    phi_1_up = F.interpolate(phi_1, size=(256, 256), mode='bilinear', align_corners=False)
    focal(phi_1_up, labels).backward()
    grads_a = _collect_grads(encoder, dhbp)

    # Path B: same direct unary
    encoder.zero_grad()
    dhbp.zero_grad()
    p1, p2, p3 = encoder.encode(x)
    phi_1 = dhbp.unary_1(p1)
    phi_1_up = F.interpolate(phi_1, size=(256, 256), mode='bilinear', align_corners=False)
    focal(phi_1_up, labels).backward()
    grads_b = _collect_grads(encoder, dhbp)

    print("\n  SANITY CHECK: Both paths = direct unary (cosine must be 1.0)")
    print(f"  {'Component':<30} {'Cosine':>10}")
    print(f"  {'-'*30} {'-'*10}")
    all_pass = True
    for name in SHARED_PARAMS:
        cos = _cosine_sim(grads_a.get(name), grads_b.get(name))
        status = "PASS" if cos is not None and abs(cos - 1.0) < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        cos_str = f"{cos:.6f}" if cos is not None else "N/A"
        print(f"  {name:<30} {cos_str:>10}  {status}")

    for name in PAIRWISE_PARAMS:
        has_grad = grads_a.get(name) is not None
        status = "PASS (no grad)" if not has_grad else "UNEXPECTED"
        if has_grad:
            all_pass = False
        print(f"  {name:<30} {'N/A':>10}  {status}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


def print_stage_results(stage_name, results, baseline=None):
    """Print three tables for a single stage."""
    print(f"\n  {stage_name}")
    print(f"  Loss: BP={results['loss_bp']:.4f}, Direct={results['loss_direct']:.4f}, "
          f"ratio={results['loss_bp']/results['loss_direct']:.2f}x")

    # Table 1: Cosine Similarity
    print(f"\n  Cosine Similarity (gradient direction alignment):")
    if baseline:
        print(f"  {'Component':<30} {'cos(BP,Direct)':>14} {'Baseline':>10} {'Interpretation'}")
        print(f"  {'-'*30} {'-'*14} {'-'*10} {'-'*30}")
    else:
        print(f"  {'Component':<30} {'cos(BP,Direct)':>14} {'Interpretation'}")
        print(f"  {'-'*30} {'-'*14} {'-'*30}")

    for name, _ in PARAMS_TO_MEASURE:
        cos = results['cosines'][name]
        if cos == 'BP_ONLY':
            cos_str = "BP-ONLY"
            interp = "New optimization dimension (no direct gradient)"
        elif cos is None:
            cos_str = "N/A"
            interp = "Zero gradient"
        else:
            cos_str = f"{cos:+.4f}"
            base = baseline.get(name) if baseline else None
            interp = _interpret_cosine(cos, base)

        if baseline:
            base_val = baseline.get(name)
            base_str = f"{base_val:.4f}" if base_val is not None else "—"
            print(f"  {name:<30} {cos_str:>14} {base_str:>10} {interp}")
        else:
            print(f"  {name:<30} {cos_str:>14} {interp}")

    # Table 2: Raw Gradient Norms
    print(f"\n  Gradient Norms (raw):")
    print(f"  {'Component':<30} {'BP norm':>12} {'Direct norm':>12} {'Ratio':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    for name, _ in PARAMS_TO_MEASURE:
        nbp = results['norms_bp'][name]
        ndir = results['norms_direct'][name]
        ratio = nbp / ndir if ndir > 1e-10 else float('inf')
        ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "inf"
        print(f"  {name:<30} {nbp:>12.6f} {ndir:>12.6f} {ratio_str:>8}")

    # Table 3: Loss-Normalized Norms
    loss_bp = results['loss_bp']
    loss_dir = results['loss_direct']
    print(f"\n  Loss-Normalized Gradient Norms (grad_norm / loss_value):")
    print(f"  {'Component':<30} {'BP norm/L':>12} {'Direct/L':>12} {'Norm ratio':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    for name, _ in PARAMS_TO_MEASURE:
        nbp = results['norms_bp'][name]
        ndir = results['norms_direct'][name]
        nbp_norm = nbp / loss_bp if loss_bp > 1e-10 else 0.0
        ndir_norm = ndir / loss_dir if loss_dir > 1e-10 else 0.0
        ratio = nbp_norm / ndir_norm if ndir_norm > 1e-10 else float('inf')
        ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "inf"
        print(f"  {name:<30} {nbp_norm:>12.6f} {ndir_norm:>12.6f} {ratio_str:>10}")


def _interpret_cosine(cos, baseline=None):
    """Interpret cosine value relative to random baseline."""
    if baseline is not None and baseline > 1e-6:
        ratio = cos / baseline
        if cos > 0.9:
            return f"SIMILAR direction ({ratio:.0f}x baseline)"
        elif cos > 0.5:
            return f"MODERATE redirection ({ratio:.0f}x baseline)"
        else:
            return f"STRONG redirection ({ratio:.0f}x baseline)"
    else:
        if cos > 0.9:
            return "SIMILAR direction"
        elif cos > 0.5:
            return "MODERATE redirection"
        else:
            return "STRONG redirection"


def print_verdict(all_results, baseline):
    """Print final interpretation across all stages."""
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    # Collect cosine values for shared params across all stages
    all_cosines = {}
    for name in SHARED_PARAMS:
        vals = []
        for r in all_results:
            cos = r['cosines'].get(name)
            if isinstance(cos, float):
                vals.append(cos)
        if vals:
            all_cosines[name] = vals

    if all_cosines:
        print("\n  Shared parameters (receive gradient from both paths):")
        for name, vals in all_cosines.items():
            mean_cos = sum(vals) / len(vals)
            base = baseline.get(name)
            base_str = f" (baseline: {base:.4f})" if base is not None else ""
            print(f"    {name:<30} mean cos={mean_cos:+.4f}{base_str}")

        all_vals = [v for vals in all_cosines.values() for v in vals]
        overall_mean = sum(all_vals) / len(all_vals)
        overall_min = min(all_vals)
        overall_max = max(all_vals)

        print(f"\n  Overall: mean={overall_mean:+.4f}, range=[{overall_min:+.4f}, {overall_max:+.4f}]")

        base_vals = [v for v in baseline.values() if v is not None]
        mean_base = sum(base_vals) / len(base_vals) if base_vals else 0.0

        if overall_min > 0.9:
            print("  → BP and direct produce SIMILAR gradient directions.")
            print("    BP acts primarily as a gradient scaler, not a preconditioner.")
        elif overall_max < 0.3 or (mean_base > 0 and overall_mean < mean_base * 5):
            print("  → BP STRONGLY redirects gradients for shared parameters.")
            print("    Optimization landscape is qualitatively different with BP.")
        else:
            print("  → BP PARTIALLY redirects gradients. Effect varies by parameter:")
            for name, vals in all_cosines.items():
                m = sum(vals) / len(vals)
                if m > 0.7:
                    print(f"      {name}: minimal redirection (cos={m:+.4f})")
                elif m > 0.3:
                    print(f"      {name}: moderate redirection (cos={m:+.4f})")
                else:
                    print(f"      {name}: strong redirection (cos={m:+.4f})")

    # Pairwise params
    bp_only_count = 0
    for r in all_results:
        for name in PAIRWISE_PARAMS:
            if r['cosines'].get(name) == 'BP_ONLY':
                bp_only_count += 1
                break
        break  # Only need to check once

    bp_only_params = [n for n in PAIRWISE_PARAMS
                      if any(r['cosines'].get(n) == 'BP_ONLY' for r in all_results)]

    if bp_only_params:
        print(f"\n  BP-only parameters (no direct gradient signal):")
        for name in bp_only_params:
            norms = [r['norms_bp'].get(name, 0.0) for r in all_results]
            mean_norm = sum(norms) / len(norms)
            print(f"    {name:<30} mean BP norm={mean_norm:.6f}")
        print(f"  → BP introduces {len(bp_only_params)} optimization dimensions")
        print(f"    that receive ZERO signal from direct supervision.")
        print(f"    These control spatial consistency and are entirely")
        print(f"    shaped by BP's message-passing structure.")


def main():
    parser = argparse.ArgumentParser(
        description='Measure gradient direction alignment between BP and direct paths')
    parser.add_argument('--contrastive_ckpt', default=None,
                        help='Path to contrastive-pretrained encoder checkpoint')
    parser.add_argument('--seg_ckpt', default=None,
                        help='Path to trained segmentation checkpoint')
    parser.add_argument('--device', default='auto',
                        help='Device: auto, cuda, or cpu')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check (both paths identical, cos must be 1.0)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    focal = FocalLoss(
        gamma=2.0,
        class_weights=torch.tensor([1.0, 1.5, 1.0, 1.0, 3.0, 1.2], device=device),
    )

    # Fixed input for reproducibility
    torch.manual_seed(42)
    x = torch.randn(2, 3, 256, 256, device=device)
    labels = torch.randint(0, 6, (2, 256, 256), device=device)

    print("=" * 70)
    print("GRADIENT DIRECTION ANALYSIS: BP chain vs Direct supervision")
    print("Does BP change gradient DIRECTION or just MAGNITUDE?")
    print("Both paths use the SAME FocalLoss to isolate BP as the only variable.")
    print("=" * 70)

    # --- Sanity check ---
    if args.sanity_check:
        encoder = ContrastiveEncoder(pretrained=True).to(device)
        dhbp = DHBPModule(n_classes=6).to(device)
        passed = measure_sanity_check(encoder, dhbp, focal, x, labels)
        del encoder, dhbp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not passed:
            print("\nSANITY CHECK FAILED — something is wrong with gradient collection")
            return
        print("\nSanity check passed. Proceeding to measurements.\n")

    # --- Stage 0: Random baseline ---
    print(f"\n{'='*70}")
    print("RANDOM BASELINE: cosine between two batches through SAME BP path")
    print("(Establishes null distribution for high-dimensional gradient vectors)")
    print(f"{'='*70}")

    encoder = ContrastiveEncoder(pretrained=True).to(device)
    dhbp = DHBPModule(n_classes=6).to(device)
    baseline = measure_random_baseline(encoder, dhbp, focal, device)

    print(f"\n  {'Component':<30} {'Baseline cos':>12}")
    print(f"  {'-'*30} {'-'*12}")
    for name, _ in PARAMS_TO_MEASURE:
        val = baseline.get(name)
        val_str = f"{val:.6f}" if val is not None else "N/A"
        print(f"  {name:<30} {val_str:>12}")

    del encoder, dhbp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_results = []

    # --- Stage 1: Random DHBP + pretrained encoder ---
    print(f"\n{'='*70}")
    encoder = ContrastiveEncoder(pretrained=True).to(device)
    dhbp = DHBPModule(n_classes=6).to(device)
    results = measure_cosine(encoder, dhbp, focal, x, labels)
    print_stage_results("STAGE 1: Random DHBP + pretrained encoder", results, baseline)
    all_results.append(results)
    del encoder, dhbp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Stage 2: Contrastive encoder + random DHBP ---
    if args.contrastive_ckpt:
        print(f"\n{'='*70}")
        encoder = ContrastiveEncoder(pretrained=True).to(device)
        ckpt_path = args.contrastive_ckpt
        if os.path.isdir(ckpt_path):
            ckpt_path = glob.glob(os.path.join(ckpt_path, "*.pth"))[0]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        dhbp = DHBPModule(n_classes=6).to(device)
        results = measure_cosine(encoder, dhbp, focal, x, labels)
        epoch = ckpt.get('epoch', '?')
        print_stage_results(
            f"STAGE 2: Contrastive encoder (epoch {epoch}) + random DHBP",
            results, baseline)
        all_results.append(results)
        del encoder, dhbp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Stage 3: Trained encoder + trained DHBP ---
    if args.seg_ckpt:
        print(f"\n{'='*70}")
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
        results = measure_cosine(encoder, dhbp, focal, x, labels)
        print_stage_results(
            "STAGE 3: Trained encoder + trained DHBP", results, baseline)
        all_results.append(results)
        del encoder, dhbp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Stage 4: Data sample sweep (5 batches, same model) ---
    print(f"\n{'='*70}")
    print("DATA SAMPLE SWEEP: 5 random batches, same model")
    print("(Measures stability of direction change across inputs)")
    print(f"{'='*70}")

    encoder = ContrastiveEncoder(pretrained=True).to(device)
    dhbp = DHBPModule(n_classes=6).to(device)

    sweep_cosines = {name: [] for name in SHARED_PARAMS}

    for batch_idx in range(5):
        torch.manual_seed(batch_idx * 1000)
        x_batch = torch.randn(2, 3, 256, 256, device=device)
        labels_batch = torch.randint(0, 6, (2, 256, 256), device=device)
        r = measure_cosine(encoder, dhbp, focal, x_batch, labels_batch)
        for name in SHARED_PARAMS:
            cos = r['cosines'].get(name)
            if isinstance(cos, float):
                sweep_cosines[name].append(cos)

    print(f"\n  {'Component':<30} {'Mean cos':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name in SHARED_PARAMS:
        vals = sweep_cosines[name]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"  {name:<30} {mean:>+10.4f} {std:>10.4f} {min(vals):>+10.4f} {max(vals):>+10.4f}")
        else:
            print(f"  {name:<30} {'N/A':>10}")

    del encoder, dhbp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- VERDICT ---
    print_verdict(all_results, baseline)


if __name__ == "__main__":
    main()
