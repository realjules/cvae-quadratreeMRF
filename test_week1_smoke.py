"""
Week-1 gate experiments — pre-flight smoke test.

Verifies every code path the gate experiments depend on, WITHOUT real data
or training, so no Kaggle hours are wasted on a crash:

  1. DumbPoolingModule forward/backward (the --dumb_pooling path has never run)
  2. UnconstrainedPairwiseHead forward/backward + init diag ratio ~ 1/K = 0.167
  3. UnconstrainedPairwiseHead(diag_init=True) init diag ratio ~ 0.80
  4. DHBPModule with each pairwise head, full BP forward/backward
  5. Mutual-exclusion guard (--unconstrained_pairwise + --diagonal_pairwise)
  6. extract_pairwise_diagnostics head-type detection + loading of the
     existing constrained 3-seed checkpoints (eval/all_results)
  7. Save → reload round-trip of an unconstrained checkpoint (what Kaggle
     produces must be loadable by the diagnostics script)

Run:  python test_week1_smoke.py            (CPU is fine, ~1-2 min)
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from net.dhbp import (
    DHBPModule, DumbPoolingModule,
    UnconstrainedPairwiseHead, PairwisePotentialHead,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


def diag_ratio_of_head(head, in_ch=128, K=6):
    """Effective psi diag ratio at init, via the head's own forward."""
    head.eval()
    with torch.no_grad():
        feat = torch.randn(2, in_ch, 16, 16)
        psi = torch.exp(head(feat))               # [B, K, K, H, W]
        avg = psi.mean(dim=(0, 3, 4)).numpy()
    return float(np.trace(avg) / avg.sum())


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}\n")

    # ---- 1. DumbPoolingModule (never executed before) ----
    print("1. DumbPoolingModule forward/backward")
    dumb = DumbPoolingModule(n_classes=6).to(DEVICE)
    p1 = torch.randn(2, 64, 128, 128, device=DEVICE, requires_grad=True)
    p2 = torch.randn(2, 128, 64, 64, device=DEVICE)
    p3 = torch.randn(2, 256, 32, 32, device=DEVICE)
    out = dumb(p1, p2, p3)
    check("output shape [2,6,128,128]", tuple(out.shape) == (2, 6, 128, 128))
    out.sum().backward()
    check("gradients flow to input", p1.grad is not None and p1.grad.abs().sum() > 0)

    # ---- 2/3. Unconstrained head init diag ratios ----
    print("\n2. UnconstrainedPairwiseHead init (chance level)")
    torch.manual_seed(0)
    uh = UnconstrainedPairwiseHead(128, 6, diag_init=False)
    dr = diag_ratio_of_head(uh)
    check(f"init diag ratio ~ 1/K=0.167", 0.10 < dr < 0.25, f"got {dr:.3f}")

    print("\n3. UnconstrainedPairwiseHead diag_init=True (control arm)")
    torch.manual_seed(0)
    uh_d = UnconstrainedPairwiseHead(128, 6, diag_init=True)
    dr_d = diag_ratio_of_head(uh_d)
    check(f"init diag ratio ~ 0.80 (matches constrained init)", 0.65 < dr_d < 0.90,
          f"got {dr_d:.3f}")

    # Constrained head init, for reference
    torch.manual_seed(0)
    ch = PairwisePotentialHead(128, 6)
    dr_c = diag_ratio_of_head(ch)
    print(f"  [info] constrained head init diag ratio: {dr_c:.3f} (target ~0.8)")

    # ---- 4. Full DHBP forward/backward with each head ----
    print("\n4. DHBPModule end-to-end with each pairwise head")
    for kwargs, label in [
        (dict(), "constrained (default)"),
        (dict(unconstrained_pairwise=True), "unconstrained"),
        (dict(unconstrained_pairwise=True, unconstrained_diag_init=True),
         "unconstrained + diag init"),
        (dict(diagonal_pairwise=True), "diagonal (Potts)"),
    ]:
        torch.manual_seed(0)
        m = DHBPModule(n_classes=6, **kwargs).to(DEVICE)
        a = torch.randn(2, 64, 128, 128, device=DEVICE)
        b = torch.randn(2, 128, 64, 64, device=DEVICE)
        c = torch.randn(2, 256, 32, 32, device=DEVICE)
        logits = m(a, b, c)
        loss = F.cross_entropy(
            logits, torch.randint(0, 6, (2, 128, 128), device=DEVICE))
        loss.backward()
        n_grads = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        n_params = sum(1 for _ in m.parameters())
        check(f"{label}: forward+backward, {n_grads}/{n_params} param tensors got gradient",
              tuple(logits.shape) == (2, 6, 128, 128) and torch.isfinite(loss)
              and n_grads > n_params * 0.8)

    # ---- 5. Mutual exclusion guard ----
    print("\n5. Flag guard")
    try:
        DHBPModule(n_classes=6, unconstrained_pairwise=True, diagonal_pairwise=True)
        check("unconstrained+diagonal raises ValueError", False)
    except ValueError:
        check("unconstrained+diagonal raises ValueError", True)

    # ---- 6. Diagnostics loader on existing checkpoints ----
    print("\n6. extract_pairwise_diagnostics on existing checkpoints")
    from extract_pairwise_diagnostics import detect_pairwise_head, load_model
    ckpt_bp = "eval/all_results/output_pct10_seed0_best.pth"
    ckpt_nobp = "eval/all_results/output_pct10_nobp_seed0_best.pth"
    if os.path.exists(ckpt_bp):
        raw = torch.load(ckpt_bp, map_location='cpu', weights_only=False)
        print(f"  [info] BP ckpt keys: {sorted(raw.keys())}")
        ht = detect_pairwise_head(raw['dhbp_state_dict'])
        check("BP seed0 detected as constrained", ht == 'constrained', ht)
        enc, dhbp = load_model(ckpt_bp, DEVICE)
        from extract_pairwise_diagnostics import extract_diagnostics, get_input_batch
        # IMPORTANT: diagnostics are input-dependent — random-noise input gives
        # ~0.32 on this checkpoint vs 0.784 recorded on real data. All gate
        # measurements must therefore use --data_dir (real test images).
        imgs, _ = get_input_batch('./input' if os.path.isdir('./input/top') else None, DEVICE)
        real = os.path.isdir('./input/top')
        d = extract_diagnostics(enc, dhbp, imgs)
        if real:
            check("universal-protocol diag ratio matches recorded 0.784 on real data",
                  0.65 < d['diag_ratio'] < 0.90, f"got {d['diag_ratio']:.3f} (real input)")
        else:
            check("diag ratio finite above chance (random input — install data for the real check)",
                  np.isfinite(d['diag_ratio']) and d['diag_ratio'] > 0.167,
                  f"got {d['diag_ratio']:.3f}")
        del enc, dhbp
    else:
        check("BP seed0 checkpoint exists", False, ckpt_bp)
    if os.path.exists(ckpt_nobp):
        raw = torch.load(ckpt_nobp, map_location='cpu', weights_only=False)
        has_enc = 'encoder_state_dict' in raw
        has_dhbp = 'dhbp_state_dict' in raw
        print(f"  [info] no-BP ckpt keys: {sorted(raw.keys())} "
              f"(encoder={has_enc}, dhbp={has_dhbp})")
        check("no-BP ckpt has encoder weights (needed for removal-gap eval)", has_enc)
    else:
        check("no-BP seed0 checkpoint exists", False, ckpt_nobp)

    # ---- 7. Unconstrained save→reload round trip ----
    print("\n7. Unconstrained checkpoint round-trip (what Kaggle will produce)")
    torch.manual_seed(0)
    m = DHBPModule(n_classes=6, unconstrained_pairwise=True)
    tmp = "/tmp/_smoke_unconstrained.pth"
    torch.save({'dhbp_state_dict': m.state_dict()}, tmp)
    raw = torch.load(tmp, map_location='cpu', weights_only=False)
    ht = detect_pairwise_head(raw['dhbp_state_dict'])
    check("detected as unconstrained", ht == 'unconstrained', ht)
    m2 = DHBPModule(n_classes=6, unconstrained_pairwise=True)
    m2.load_state_dict(raw['dhbp_state_dict'])
    check("reload OK", True)
    os.remove(tmp)

    print(f"\n{'='*50}\n  {PASS} passed, {FAIL} failed\n{'='*50}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
