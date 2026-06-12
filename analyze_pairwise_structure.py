"""
Metric-hardening pass for the pairwise-degeneracy finding (week-1 gates).

Session A (Experiment 20) showed the scalar diag ratio UNDERSELLS the
phenomenon: the unconstrained seed-0 matrix has diag ratio 0.281 (above the
0.167 chance level) yet its Trees row puts 0.659 on Impervious. The remapping
is row-wise/selective, so the paper needs structure-aware metrics:

  1. NEAREST-ASSIGNMENT ANALYSIS — the row-to-column assignment P* maximizing
     captured mass (Hungarian; K=6 so brute force over 720 perms is exact).
     `assignment_ratio` (mass on P*) vs `diag_ratio` (mass on identity):
     a matrix can be far from identity yet close to a PERMUTATION — that gap
     IS the remapping signature.
  2. PER-ROW VERDICTS — each parent row: diagonal mass, dominant target,
     REMAPPED flag (off-diagonal dominant).
  3. EMPIRICAL CO-OCCURRENCE NULL — P(child | parent) measured from Vaihingen
     ground truth (parent = majority label of each 2x2 block). This is what
     LEGITIMATE spatial statistics look like; some off-diagonal mass (e.g.
     LowVeg<->Trees) is real class adjacency, not degeneracy. Per-row KL of
     learned psi against this null separates "learned real transitions" from
     "learned remapping".

Inputs are pairwise_diagnostics.json files (produced by
extract_pairwise_diagnostics.py) — no GPU or checkpoints needed. The March
2026 unconstrained matrix (experiment_log.md, Stage B diagnostic) is built in
as a historical reference.

Usage:
    python analyze_pairwise_structure.py \
        --json eval/all_results/paper_figures_10pct/pairwise_diagnostics.json \
               output/week1_sessionA_results/diag_week1/pairwise_diagnostics.json \
        --gt_dir ./input/gt \
        --output_json output/pairwise_structure_analysis.json
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

CLASS_NAMES = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]
SHORT = ["Imp", "Bld", "LVg", "Tre", "Car", "Clu"]

# March 2026 unconstrained Stage-B diagnostic (docs/experiment_log.md, single
# run, collapsed-unary regime). Historical reference for the degeneracy claim.
MARCH_2026_UNCONSTRAINED = np.array([
    [0.325, 0.004, 0.347, 0.311, 0.013, 0.000],
    [0.668, 0.206, 0.047, 0.052, 0.027, 0.000],
    [0.508, 0.138, 0.024, 0.001, 0.330, 0.000],
    [0.190, 0.789, 0.016, 0.000, 0.005, 0.000],
    [0.302, 0.107, 0.318, 0.262, 0.011, 0.000],
    [0.080, 0.384, 0.475, 0.055, 0.004, 0.000],
])


def nearest_assignment(psi):
    """Exact best row->column assignment by brute force (K=6: 720 perms)."""
    K = psi.shape[0]
    best_perm, best_mass = None, -1.0
    for perm in itertools.permutations(range(K)):
        mass = sum(psi[i, perm[i]] for i in range(K))
        if mass > best_mass:
            best_mass, best_perm = mass, perm
    return list(best_perm), float(best_mass / psi.sum())


def row_verdicts(psi):
    """Per-row: diagonal mass, dominant target, remapped flag."""
    rows = []
    for i in range(psi.shape[0]):
        j = int(np.argmax(psi[i]))
        rows.append({
            'row': CLASS_NAMES[i],
            'diag': float(psi[i, i]),
            'dominant_target': CLASS_NAMES[j],
            'dominant_mass': float(psi[i, j]),
            'remapped': bool(j != i),
        })
    return rows


def kl_rows(psi, null, eps=1e-8):
    """Per-row KL( psi_row || null_row ), rows assumed ~stochastic."""
    out = []
    for i in range(psi.shape[0]):
        p = psi[i] / max(psi[i].sum(), eps)
        q = null[i] / max(null[i].sum(), eps)
        out.append(float(np.sum(p * np.log((p + eps) / (q + eps)))))
    return out


def cooccurrence_null(gt_dir):
    """Empirical P(child | parent) from GT: parent = majority of 2x2 block."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from skimage import io
    from utils.utils_dataset import convert_from_color

    K = len(CLASS_NAMES)
    C = np.zeros((K, K), dtype=np.float64)
    files = sorted(glob.glob(os.path.join(gt_dir, "*.tif")))
    if not files:
        raise FileNotFoundError(f"No GT tifs in {gt_dir}")
    for f in files:
        labels = convert_from_color(np.asarray(io.imread(f)))
        H, W = labels.shape[0] // 2 * 2, labels.shape[1] // 2 * 2
        lab = labels[:H, :W]
        blocks = lab.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
        valid = (blocks < K).all(axis=1)
        blocks = blocks[valid]
        # parent = majority label of the 4 children (ties -> lowest index)
        onehot_counts = np.stack([(blocks == k).sum(axis=1) for k in range(K)], axis=1)
        parent = onehot_counts.argmax(axis=1)
        np.add.at(C, (np.repeat(parent, 4), blocks.ravel()), 1.0)
        print(f"  co-occurrence: {os.path.basename(f)} ({valid.sum()} blocks)")
    C = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-8)
    return C


def analyze_matrix(name, psi, null=None):
    psi = np.asarray(psi, dtype=np.float64)
    K = psi.shape[0]
    diag_ratio = float(np.trace(psi) / psi.sum())
    perm, assignment_ratio = nearest_assignment(psi)
    rows = row_verdicts(psi)
    n_remapped = sum(r['remapped'] for r in rows)
    result = {
        'name': name,
        'diag_ratio': diag_ratio,
        'chance_diag_ratio': 1.0 / K,
        'assignment_ratio': assignment_ratio,
        'permutation_gap': float(assignment_ratio - diag_ratio),
        'best_assignment': {CLASS_NAMES[i]: CLASS_NAMES[perm[i]] for i in range(K)},
        'assignment_is_identity': perm == list(range(K)),
        'n_remapped_rows': n_remapped,
        'rows': rows,
    }
    if null is not None:
        kls = kl_rows(psi, null)
        result['kl_to_cooccurrence_per_row'] = {CLASS_NAMES[i]: kls[i] for i in range(K)}
        result['kl_to_cooccurrence_mean'] = float(np.mean(kls))
    return result


def print_report(r):
    print(f"\n{'='*68}\n  {r['name']}\n{'='*68}")
    print(f"  diag ratio:       {r['diag_ratio']:.3f}   (chance {r['chance_diag_ratio']:.3f})")
    print(f"  assignment ratio: {r['assignment_ratio']:.3f}   "
          f"(permutation gap: +{r['permutation_gap']:.3f} — large gap = permutation-like, not identity-like)")
    ident = "IDENTITY" if r['assignment_is_identity'] else "NON-IDENTITY (remapping!)"
    print(f"  best assignment:  {ident}")
    if not r['assignment_is_identity']:
        for src, dst in r['best_assignment'].items():
            if src != dst:
                print(f"        {src} -> {dst}")
    print(f"  rows off-diagonal-dominant: {r['n_remapped_rows']}/6")
    for row in r['rows']:
        flag = "  <-- REMAPPED" if row['remapped'] else ""
        print(f"    {row['row']:<12} diag {row['diag']:.3f}  dominant -> "
              f"{row['dominant_target']:<12} {row['dominant_mass']:.3f}{flag}")
    if 'kl_to_cooccurrence_mean' in r:
        print(f"  KL(psi || GT co-occurrence): mean {r['kl_to_cooccurrence_mean']:.3f}  per-row: "
              + ", ".join(f"{SHORT[i]}:{v:.2f}" for i, v in
                          enumerate(r['kl_to_cooccurrence_per_row'].values())))


def main():
    ap = argparse.ArgumentParser(description='Structure-aware pairwise-matrix analysis')
    ap.add_argument('--json', nargs='+', required=True,
                    help='pairwise_diagnostics.json file(s) from extract_pairwise_diagnostics.py')
    ap.add_argument('--gt_dir', default='./input/gt',
                    help='Vaihingen GT directory for the co-occurrence null (skipped if missing)')
    ap.add_argument('--output_json', default='./output/pairwise_structure_analysis.json')
    args = ap.parse_args()

    null = None
    if args.gt_dir and os.path.isdir(args.gt_dir):
        print("Computing GT co-occurrence null (parent = majority of 2x2 block)...")
        null = cooccurrence_null(args.gt_dir)

    results = []
    if null is not None:
        r = analyze_matrix("GT CO-OCCURRENCE NULL (what legitimate spatial statistics look like)",
                           null, None)
        print_report(r)
        r['matrix'] = null.tolist()
        results.append(r)

    results.append(analyze_matrix("march2026_unconstrained (single run, collapsed-unary regime)",
                                  MARCH_2026_UNCONSTRAINED, null))
    print_report(results[-1])

    for path in args.json:
        entries = json.load(open(path))
        for e in entries:
            r = analyze_matrix(e['name'], np.array(e['avg_psi']), null)
            print_report(r)
            results.append(r)

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
