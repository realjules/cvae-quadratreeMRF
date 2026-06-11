# Week-1 Gate Experiments — Runbook (June 2026)

The five experiments that decide which paper this project becomes (see
`docs/neurips_research_directions_2026-06.md`, Part 3). Everything below is
**code-complete and smoke-tested** (`python test_week1_smoke.py`, 14/14 pass,
2026-06-12). Kaggle hours are spent only on training.

## What was built/fixed for this (already in the repo)

| Item | Where |
|---|---|
| `UnconstrainedPairwiseHead` restored from `503db8a^` + `diag_init` control | `net/dhbp.py` |
| `--unconstrained_pairwise`, `--unconstrained_diag_init` flags | `complete_training.py`, `train.py` |
| Head-type auto-detection + **universal ψ protocol** (one script, all arms, chance level 1/K printed) | `extract_pairwise_diagnostics.py` |
| Eval-noise floor + removal-gap script (eval-only) | `measure_eval_noise.py` |
| Pre-flight smoke test (run on Kaggle before training too) | `test_week1_smoke.py` |
| `utils/__init__.py`, `dataset/__init__.py` (a pip `utils` module shadowed the repo package locally) | — |

**Init diag-ratios verified:** unconstrained starts at 0.169 (≈ chance 1/K = 0.167);
diag-init control starts at 0.785; constrained starts at 0.791. The two control
arms start from the same effective ψ — the comparison is clean.

**⚠ Protocol rule (verified empirically):** diagnostics are input-dependent —
seed-0 constrained gives **0.32 on random-noise input vs 0.76–0.78 on real
data**. Every diagnostic run MUST pass `--data_dir ./input`. Never compare a
real-data number against a noise-input number.

---

## Step 0 (before anything): back up the contrastive checkpoint

The encoder checkpoint exists ONLY as a Kaggle model
(`/kaggle/input/models/udaheju/contrastive-best/pytorch/default/1`). If it is
lost, every comparison to the existing 6 checkpoints becomes unreproducible.
In any Kaggle session: `cp` it to `/kaggle/working/`, then download → store in
Google Drive AND the repo machine. 5 minutes; do it first.

---

## Local gates (zero Kaggle quota)

> Note: local eval works (4GB RTX 2050) but tile I/O from the OneDrive-synced
> folder makes the FIRST eval slow (tiles cache in RAM afterwards). If local
> runs feel too slow, both gates run on Kaggle in ~1–2h total — they're
> eval-only and can ride along in either session.

### Gate N — eval-noise floor (k = 10 re-evals, one checkpoint)

```bash
python measure_eval_noise.py \
    --seg_ckpt eval/all_results/output_pct10_seed0_best.pth \
    --k 10 --output_json output/eval_noise_floor.json
```

**Why:** the 12-agent review *estimated* 5–8pp single-run noise but never
measured it; Exp 19's paired seeds suggest ~1.2pp. This number is the yardstick
every later claim is read against, and it pre-empts the reviewer question of
why ±1.2pp error bars coexist with a claimed 5–8pp floor.

### Gate R — removal-gap (inference offloading) seeded replication

```bash
# BP-trained checkpoints, each evaluated WITH BP and with BP stripped:
python measure_eval_noise.py \
    --seg_ckpt eval/all_results/output_pct10_seed0_best.pth \
               eval/all_results/output_pct10_seed1_best.pth \
               eval/all_results/output_pct10_seed2_best.pth \
    --k 3 --both --output_json output/removal_gap_bp.json

# No-BP-trained baselines (their unary IS the model):
python measure_eval_noise.py \
    --seg_ckpt eval/all_results/output_pct10_nobp_seed0_best.pth \
               eval/all_results/output_pct10_nobp_seed1_best.pth \
               eval/all_results/output_pct10_nobp_seed2_best.pth \
    --k 3 --unary_only --output_json output/removal_gap_nobp.json
```

**Why:** the headline 36.7%-vs-62.0% came from ONE pair of *50-epoch*
checkpoints; these are the *30-epoch* seeded ones — a real test, not a
formality. **Read-out:** removal gap = (with_bp − unary_only) per BP seed,
vs the no-BP arm's unary accuracy.

---

## Kaggle Session A (~8–9h of a 12h session)

Setup as in `docs/kaggle_session1.md` (clone repo, attach dataset + contrastive
model). Then:

```bash
# 0. Pre-flight (~5 min — verifies all new paths on the Kaggle image)
python test_week1_smoke.py

# 1. GATE D — dumb-pooling receptive-field baseline, 3 seeds (~6h)
#    The control flagged P0 "blocks paper" since April. Zero new code.
for SEED in 0 1 2; do
  python complete_training.py \
    --data_dir ./input \
    --output_dir /kaggle/working/output_pct10_dumbpool_seed${SEED} \
    --contrastive_ckpt /kaggle/input/models/udaheju/contrastive-best/pytorch/default/1 \
    --labeled_percent 10 --epochs_seg 30 --seed ${SEED} --dumb_pooling
done

# 2. GATE U begins — unconstrained reproduction, seed 0 (~2h)
python complete_training.py \
  --data_dir ./input \
  --output_dir /kaggle/working/output_pct10_unconstrained_seed0 \
  --contrastive_ckpt /kaggle/input/models/udaheju/contrastive-best/pytorch/default/1 \
  --labeled_percent 10 --epochs_seg 30 --seed 0 --unconstrained_pairwise

# 3. Diagnostics on whatever finished (minutes, REAL DATA mandatory)
python extract_pairwise_diagnostics.py \
  --seg_ckpt /kaggle/working/output_pct10_unconstrained_seed0/best_segmentation.pth \
  --data_dir ./input --output_dir /kaggle/working/diag_unconstrained
```

Download `/kaggle/working` outputs before the session ends (no resume support).

## Kaggle Session B (~10h)

```bash
# 4. GATE U — unconstrained seeds 1, 2 (~4h)
for SEED in 1 2; do
  python complete_training.py \
    --data_dir ./input \
    --output_dir /kaggle/working/output_pct10_unconstrained_seed${SEED} \
    --contrastive_ckpt /kaggle/input/models/udaheju/contrastive-best/pytorch/default/1 \
    --labeled_percent 10 --epochs_seg 30 --seed ${SEED} --unconstrained_pairwise
done

# 5. GATE U-INIT — diagonal-init control, 3 seeds (~6h)
#    Disambiguates "remapping is a training attractor" from "the constraint
#    just installs a good init" (audit: ~0.62 of the 0.094-vs-0.784 gap is init).
for SEED in 0 1 2; do
  python complete_training.py \
    --data_dir ./input \
    --output_dir /kaggle/working/output_pct10_uncdiag_seed${SEED} \
    --contrastive_ckpt /kaggle/input/models/udaheju/contrastive-best/pytorch/default/1 \
    --labeled_percent 10 --epochs_seg 30 --seed ${SEED} \
    --unconstrained_pairwise --unconstrained_diag_init
done

# 6. Diagnostics across ALL new arms in one protocol (minutes)
python extract_pairwise_diagnostics.py \
  --seg_ckpt /kaggle/working/output_pct10_unconstrained_seed*/best_segmentation.pth \
             /kaggle/working/output_pct10_uncdiag_seed*/best_segmentation.pth \
  --data_dir ./input --output_dir /kaggle/working/diag_week1
```

If quota is tight: run diag-init with 1 seed first; expand to 3 only if Gate U
reproduces. Total training: 9–11 runs ≈ 18–22 T4-hours across two sessions —
inside one week of the 30h quota with margin.

---

## Decision rules (write the outcome into docs/experiment_log.md either way)

**Gate U (unconstrained 3-seed) — the coin flip the headline rests on.**
Read diag_ratio (real-data protocol, chance = 0.167):
- mean ≲ 0.25 across seeds (esp. below chance) → **degeneracy reproduces**;
  the Compatibility Collapse paper is alive; replace every March n=1 number.
- mean > 0.4 → the March 0.094 was a collapsed-unary/era artifact → pivot per
  the decision tree (removal-gap direction + synthetic testbed).
- in between → report honestly, lean on per-class accuracy destruction and the
  permutation structure of ψ, run 2 more seeds before concluding.

**Gate U-init (diag-init control).**
- An initially-healthy ψ (0.785 at init) gets destroyed → remapping is a
  **training attractor** — the strong version of the claim.
- The diagonal survives training → the constrained head's effect is mostly
  initialization → reframe as "init selects the orbit member" (weaker but
  still publishable; be honest in the paper).

**Gate D (dumb pooling vs BP).** Compare best accuracy vs BP 65.27 ± 1.4 and
no-BP 60.62 ± 1.2 (Exp 19):
- within ~1.5pp of BP → the +4.65pp is receptive field/multi-scale features,
  NOT structured inference. Degeneracy paper survives (gets cleaner); drop all
  "benefit of structured prediction" wording; entropy-gating formalization dies.
- BP wins by > 2pp → structured-inference attribution survives; the
  entropy-gating direction (#6) earns its next gate.
- pooling wins outright → negative-result framing; SPIGM-type venues welcome it.

**Gate R (removal gap).** Robust if the gap is consistently large (≥ 5pp)
across the 3 BP seeds; the n=1 prior was 25.4pp on 50-epoch checkpoints.
Frame as inference offloading (descriptive), never bare pathology.

**Gate N (noise floor).** Report σ_eval next to every accuracy claim from now
on; claims must clear max(σ_eval, seed σ).

## After the gates

Update `docs/paper_plan.md`'s bracketed [TBD] numbers from
`diag_week1/pairwise_diagnostics.json`, log everything in
`docs/experiment_log.md` as "Experiment 20 (Gates)", and follow the decision
tree in `docs/neurips_research_directions_2026-06.md` Part 3. Target:
NeurIPS 2026 workshop 4-pager, deadline ~Aug 29, 2026.
