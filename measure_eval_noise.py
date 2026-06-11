"""
Week-1 gates: eval-noise floor + removal-gap (inference-offloading) measurement.

Two of the cheapest, highest-leverage measurements from the June 2026 review,
in one script. Both are EVALUATION-ONLY — no training, runs on a 4GB local GPU
or any Kaggle session.

GATE A — eval-noise floor (--k 10):
    Re-evaluates the same checkpoint k times. The test loader's dataset samples
    random patches per __getitem__, so repeated evals give the eval-noise
    distribution. This pins the actual noise floor (the 12-agent review
    ESTIMATED 5-8pp but never measured it; Exp 19 paired seeds suggest ~1.2pp).
    Every later claim is read against this number.

GATE B — removal gap (--unary_only on BP checkpoints):
    Evaluates a BP-trained checkpoint with BP stripped at inference
    (unary_1(p1) only). Compare against (a) the same checkpoint WITH BP and
    (b) the no-BP-trained checkpoints. This replicates the 36.7%-vs-62.0%
    dependency finding with seeds — note the original came from different
    (50-epoch) checkpoints, so this is a real test, not a formality.

Usage:
    # Noise floor: one checkpoint, 10 repeated evals
    python measure_eval_noise.py --seg_ckpt eval/all_results/output_pct10_seed0_best.pth --k 10

    # Removal gap across all 6 checkpoints (BP ckpts with and without BP at inference)
    python measure_eval_noise.py \
        --seg_ckpt eval/all_results/output_pct10_seed0_best.pth \
                   eval/all_results/output_pct10_seed1_best.pth \
                   eval/all_results/output_pct10_seed2_best.pth \
        --k 3 --both

    python measure_eval_noise.py \
        --seg_ckpt eval/all_results/output_pct10_nobp_seed0_best.pth \
                   eval/all_results/output_pct10_nobp_seed1_best.pth \
                   eval/all_results/output_pct10_nobp_seed2_best.pth \
        --k 3 --unary_only
"""

import argparse
import json
import os

import numpy as np
import torch

from net.cvae import ContrastiveEncoder
from train import SegmentationTrainer
from complete_training import create_real_dataloaders

CLASS_NAMES = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]


def build_trainer(seg_ckpt, device, use_bp):
    """Build a trainer whose module architecture matches the checkpoint.

    Auto-detects: dumb-pooling checkpoints (keys start with 'fuse.'),
    unconstrained/diagonal pairwise heads (via extract_pairwise_diagnostics),
    constrained default.
    """
    ckpt = torch.load(seg_ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('dhbp_state_dict', {})
    dumb = any(k.startswith('fuse.') for k in state)
    unconstrained, diagonal = False, False
    if not dumb and state:
        from extract_pairwise_diagnostics import detect_pairwise_head
        ht = detect_pairwise_head(state)
        unconstrained, diagonal = ht == 'unconstrained', ht == 'diagonal'
        print(f"    head type: {ht}")
    if dumb:
        print("    head type: dumb-pooling (no unary head — unary_only mode not applicable)")

    encoder = ContrastiveEncoder(pretrained=True)
    trainer = SegmentationTrainer(
        encoder=encoder, n_classes=6, device=str(device),
        use_bp=use_bp and not dumb, dumb_pooling=dumb,
        unconstrained_pairwise=unconstrained, diagonal_pairwise=diagonal,
    )
    trainer.load(seg_ckpt)
    return trainer


def run_evals(trainer, test_loader, k, label):
    accs, mean_accs, per_class_runs = [], [], []
    for i in range(k):
        m = trainer.evaluate(test_loader)
        accs.append(m['accuracy'])
        mean_accs.append(m['mean_accuracy'])
        per_class_runs.append(m['per_class_accuracy'])
        print(f"    eval {i+1}/{k}: overall {m['accuracy']:.2f}%  mean-class {m['mean_accuracy']:.2f}%")
    accs, mean_accs = np.array(accs), np.array(mean_accs)
    pc = np.array(per_class_runs)  # [k, 6]
    print(f"  {label}: overall {accs.mean():.2f} ± {accs.std():.2f}%  "
          f"(range [{accs.min():.2f}, {accs.max():.2f}], k={k})")
    print(f"  {label}: mean-class {mean_accs.mean():.2f} ± {mean_accs.std():.2f}%")
    for c, name in enumerate(CLASS_NAMES):
        print(f"      {name:<12} {pc[:, c].mean():6.2f} ± {pc[:, c].std():.2f}%")
    return {
        'overall_mean': float(accs.mean()), 'overall_std': float(accs.std()),
        'overall_runs': accs.tolist(),
        'mean_class_mean': float(mean_accs.mean()), 'mean_class_std': float(mean_accs.std()),
        'per_class_mean': pc.mean(axis=0).tolist(),
        'per_class_std': pc.std(axis=0).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description='Eval-noise floor + removal-gap measurement')
    p.add_argument('--seg_ckpt', nargs='+', required=True)
    p.add_argument('--data_dir', default='./input')
    p.add_argument('--k', type=int, default=10, help='Repeated evaluations per checkpoint')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--unary_only', action='store_true',
                   help='Strip BP at inference: evaluate unary_1(p1) only (removal gap)')
    p.add_argument('--both', action='store_true',
                   help='Evaluate each checkpoint BOTH with BP and unary-only')
    p.add_argument('--output_json', default='./output/eval_noise_removal_gap.json')
    p.add_argument('--device', default='auto')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else torch.device(args.device)
    print(f"Device: {device}")

    _, _, test_loader = create_real_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, labeled_percent=10,
    )

    results = {}
    for ckpt in args.seg_ckpt:
        name = os.path.basename(ckpt).replace('.pth', '')
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        results[name] = {}

        modes = [('with_bp', True), ('unary_only', False)] if args.both \
            else [('unary_only', False)] if args.unary_only \
            else [('with_bp', True)]
        for label, use_bp in modes:
            print(f"  mode: {label}")
            trainer = build_trainer(ckpt, device, use_bp=use_bp)
            results[name][label] = run_evals(trainer, test_loader, args.k, label)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.both and 'with_bp' in results[name]:
            gap = results[name]['with_bp']['overall_mean'] - results[name]['unary_only']['overall_mean']
            results[name]['removal_gap_pp'] = float(gap)
            print(f"\n  REMOVAL GAP (with_bp − unary_only): {gap:+.2f}pp")

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_json}")

    if len(args.seg_ckpt) > 1 and args.both:
        gaps = [v['removal_gap_pp'] for v in results.values() if 'removal_gap_pp' in v]
        if gaps:
            print(f"\nREMOVAL GAP across {len(gaps)} checkpoints: "
                  f"{np.mean(gaps):+.2f} ± {np.std(gaps):.2f}pp")


if __name__ == "__main__":
    main()
