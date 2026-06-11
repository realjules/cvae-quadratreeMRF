# Paper Plan: Pairwise Potential Degeneracy in End-to-End Trained Graphical Models

**Target**: SPIGM workshop @ NeurIPS 2025 (4 pages + appendix, deadline TBD ~Aug/Sep 2025)
**Backup**: UAI 2026 or AISTATS 2026 (full 8-page paper)
**Working title**: "Pairwise Potential Degeneracy in End-to-End Trained Graphical Models"

**Last updated**: 2026-04-07 (post adversarial review — 4 structural vulnerabilities addressed)

---

## 1. Paper Structure (4 pages + appendix)

### Abstract (~150 words)
End-to-end training of structured prediction layers enables learning pairwise potentials directly from data, but we identify and characterize a previously undocumented failure mode: unconstrained learned K×K pairwise matrices converge to class remapping instead of spatial consistency. On aerial image segmentation (ISPRS Vaihingen), the unconstrained pairwise matrix exhibits Tree→Building remapping of [0.789 — single run, 2026-03-24; replace with 3-seed mean ± std after Experiment H] with diagonal ratio [0.094 — single run; 3-seed TBD after Experiment H], destroying minority classes ([Cars accuracy halved — single run; re-verify across seeds in Experiment H]). [TBD after Experiment G (dumb-pooling baseline, NOT YET RUN): whether this effect is distinct from receptive-field expansion, by comparing against a spatial-context baseline matching the same effective field of view — do not claim until the result exists.] Analysis of the gradient landscape reveals that end-to-end training naturally produces gradient divergence between the message-passing and direct supervision paths — an expected consequence of inference offloading, not a pathology. A constrained decomposition ψ = α·I + (1-α)·R prevents degeneracy (diagonal ratio 0.784 ± 0.003, 3 seeds, eval/all_results/paper_figures_10pct/pairwise_diagnostics.json) while preserving the +4.65 ± 1.1pp accuracy benefit of the message-passing module over the no-BP baseline (3 seeds, Experiment 19; attribution to *structured inference* specifically is pending Experiment G). Our findings explain why Potts potentials remain dominant and provide practical guidance for learning pairwise potentials safely.

> **⚠ NUMBERS POLICY (added 2026-06-11):** The previous version of this abstract reported unconstrained results "on 3 seeds" with values (Tree→Building 0.729, diagonal ratio 0.196) that exist nowhere in this repository — Experiment H has never been run and the unconstrained head was deleted in commit 503db8a. The only real unconstrained measurement is the single 2026-03-24 run (0.789 / 0.094, docs/experiment_log.md "Stage B"). Rule going forward: **no number enters paper text without a pointer to a results file in this repo.** Bracketed [TBD] markers must survive until the named experiment's output exists.

### Section 1: Introduction (~0.75 page)
- End-to-end CRF/MRF training: promise and fragility
- "Everyone uses Potts, nobody says why learned potentials fail"
- Our contribution: characterize the failure mode, distinguish it from receptive field effects, provide a constrained fix

### Section 2: Background (~0.5 page)
- Pairwise MRF/CRF for segmentation
- Message passing on quadtrees (note: our entropy-gated variant is approximate, not exact BP — see Section 5 discussion)
- Related work: Larsson et al. 2018 (arbitrary pairwise, no degeneracy analysis), E-CRF 2023 (boundary confusion, different mechanism)

### Section 3: Pairwise Degeneracy Characterization (~1 page)
- Unconstrained K×K matrix analysis: diagonal ratio, off-diagonal structure
- Class remapping visualization (Figure 1 heatmap — constrained vs unconstrained)
- Effect on per-class accuracy (Cars destruction)
- Comparison: constrained α·I+(1-α)·R prevents it
- 3-seed reproduction with error bars

### Section 4: Is It Just Receptive Field? (~0.75 page)
- Dumb pooling baseline matching ~103px receptive field
- If BP > dumb pooling: structured inference provides value beyond spatial context
- If BP ≈ dumb pooling: BP's value is receptive field, but degeneracy finding still holds
- Gradient landscape characterization: divergence between BP and direct paths is expected (inference offloading), not pathological. Cite MTL gradient conflict literature as related framing (Du et al. 2018, PCGrad).

### Section 5: Discussion (~0.5 page)
- Entropy-gated message passing: acknowledge departure from exact inference. Our modification weights children by confidence, breaking the sum-product derivation. Practical benefit (+4.65pp) but not exact marginals.
- Pairwise degeneracy likely generalizes to any end-to-end trained structured prediction with unconstrained potentials (DenseCRF, Neural MRF)
- Causal direction is ambiguous: degeneracy may cause gradient warping OR gradient freedom may enable degeneracy. We present descriptive characterization, not causal claim.

### Section 6: Conclusion (~0.25 page)
- First characterization of pairwise potential degeneracy in end-to-end trained graphical models
- Constrained decomposition is the practical fix
- Implications for Neural MRF (Guan et al. CVPR 2024) and future learned potentials

---

## 2. Final Experiments to Run

### Experiment A: 3-Seed Reproduction (CRITICAL, blocks paper)

**Purpose**: Error bars on all claims. Currently everything is single-run.

**Runs needed** (6 total, ~2 hours each on T4):

| Run | Labels | Seeds | Epochs | What to measure |
|-----|--------|-------|--------|-----------------|
| A1-A3 | 10% | 0, 1, 2 | 30 seg | Accuracy, per-class, pairwise matrix, cosine similarity |
| A4-A6 | 50% | 0, 1, 2 | 30 seg | Same |

**For each run, save**:
- `best_segmentation.pth` checkpoint
- Per-class accuracy at best epoch
- Pairwise matrix diagnostics (diagonal ratio, off-diagonal max)

**After all 6 runs, measure**:
- Run `test_gradient_cosine.py --seg_ckpt <each checkpoint>` (6 runs)
- Report mean ± std for cosine similarity per parameter per label fraction

**Expected Kaggle time**: 6 training runs × ~2h + 6 cosine measurements × ~5min ≈ 13 hours

```bash
# Template for each run (modify seed and output dir)
python complete_training.py \
    --data_dir ./input \
    --output_dir /kaggle/working/output_pct10_seed0 \
    --epochs_contrastive 50 \
    --epochs_seg 30 \
    --labeled_percent 10 \
    --seed 0 \
    --contrastive_ckpt /kaggle/input/models/udaheju/contrastive-best/pytorch/default/1

# Then measure cosine
python test_gradient_cosine.py \
    --contrastive_ckpt /kaggle/input/models/udaheju/contrastive-best/pytorch/default/1 \
    --seg_ckpt /kaggle/working/output_pct10_seed0/best_segmentation.pth
```

**NOTE**: `complete_training.py` may need a `--seed` argument added. Check if it exists, otherwise add one that sets `torch.manual_seed()`, `np.random.seed()`, and `torch.cuda.manual_seed_all()` before training.

---

### Experiment B: Pairwise Matrix Diagnostics (per seed)

**Purpose**: Show pairwise degeneracy is reproducible, not a single-run fluke.

**What to extract from each checkpoint** (write a script `extract_pairwise_diagnostics.py`):

```python
# For each trained checkpoint:
# 1. Load dhbp.pairwise_12
# 2. Compute on a batch of test images:
#    - alpha distribution (mean, std, boundary vs interior)
#    - Effective ψ = α·I + (1-α)·R for each spatial location
#    - Average ψ across all locations → K×K matrix
#    - Diagonal ratio = trace(ψ) / sum(ψ)
#    - Max off-diagonal entry and which class pair
#    - Per-class diagonal strength
```

**Output**: One K×K heatmap per run, plus summary statistics.

---

### Experiment C: Constrained vs Unconstrained Cosine Comparison (strengthens causal claim)

**Purpose**: If the constrained pairwise (α·I+(1-α)·R) produces MORE aligned gradients than unconstrained K×K, that's the causal link between gradient conflict and degeneracy.

**Runs needed**: 2 additional training runs at 10%, 1 seed:
- C1: Unconstrained K×K pairwise (remove α·I+(1-α)·R constraint, use raw K×K output)
- C2: Constrained (current default)

**Measure**: cosine similarity for both. Compare.

**Expected**: Unconstrained should have LOWER cosine (more conflict), because the unconstrained K×K matrix has more freedom to redirect gradients toward the remapping shortcut.

**NOTE**: Need to add an `--unconstrained_pairwise` flag to the training script that replaces `PairwisePotentialHead` with a raw `nn.Conv2d(in_channels, K*K, 1)` + reshape + log_softmax.

---

### Experiment D: Potsdam Dataset (for full paper, not workshop)

**Purpose**: Second dataset proves generalization.

**Runs needed**: Same as Experiment A but on ISPRS Potsdam.
- Requires: downloading Potsdam data, adapting data loader
- Same 6 classes, similar aerial imagery, different city

**Defer to full paper** unless time permits before workshop deadline.

---

### Experiment F: Convergence Speedup Characterization (P2, supporting evidence)

**Purpose**: Quantify BP's convergence speedup with error bars so it can be cited as a supporting observation. Prior single-run data showed ~3x speedup (10 epochs with BP ≈ 30 epochs without). If this holds across seeds, it's a useful sentence in the paper: "BP provides faster convergence via 9x gradient amplification, but creates dependency (the encoder trained with BP cannot function without it)."

**Runs needed** (piggyback on Experiment A — no extra training):

From the 3-seed runs at 10% labels, we already have BP-trained models. We additionally need 3 no-BP runs:

| Run | Config | Seeds | Purpose |
|-----|--------|-------|---------|
| F1-F3 | 10%, **no BP** (`--no_bp`) | 0, 1, 2 | No-BP accuracy curve baseline |

**For each of the 6 runs (3 BP + 3 no-BP), log**:
- Accuracy at every evaluation epoch (5, 10, 15, 20, 25, 30)
- This data should already come from the training logs

**After all runs, measure**:
- Epoch at which BP model reaches X% accuracy vs epoch at which no-BP model reaches same X%
- BP-trained encoder evaluated WITHOUT BP at inference (dependency test, as in Experiment 11)

**Analysis**:
```
Accuracy vs epoch (3-seed mean ± std):

  65% ┤        ●───BP (with BP)────────●
      │      ╱
  60% ┤    ●╱
      │   ╱           ○───No-BP────────○
  55% ┤  ╱          ╱
      │ ╱         ○╱
  50% ┤●        ╱
      │       ○
  45% ┤     ╱
      │   ○
  40% ┤
      └──┬────┬────┬────┬────┬────┬──
         5   10   15   20   25   30  epoch

  ▼ BP-trained encoder, BP removed at inference: ???%
    (If << No-BP final → dependency confirmed across seeds)
```

**What this gives the paper**:
- One sentence + one small figure (appendix) confirming convergence speedup with error bars
- Confirms the dependency finding (Experiment 11) across seeds
- Supports the gradient amplification narrative without overclaiming

**What this does NOT claim**:
- NOT a general fine-tuning technique (no evidence for generalization)
- NOT a novel contribution (auxiliary task speedup is well-known)
- Just a characterization of BP's training dynamics that supports the main degeneracy story

**Kaggle time**: 3 extra no-BP training runs × ~2h = ~6h. Can run in Session 1 alongside Experiment A if Kaggle quota allows, otherwise Session 2.

```bash
# No-BP runs (piggyback on Session 1)
for SEED in 0 1 2; do
    python complete_training.py \
        --contrastive_ckpt /kaggle/input/models/.../contrastive_best.pth \
        --output_dir /kaggle/working/output_pct10_nobp_seed${SEED} \
        --labeled_percent 10 --epochs_seg 30 --seed ${SEED} --no_bp
done
```

---

### Experiment E: Fix Data Sweep (minor, do during Experiment A)

**Purpose**: The current sweep in `test_gradient_cosine.py` uses random-init model weights. Fix it to use the trained model.

**Fix**: The sweep already uses the trained model from Stage 3 for the cosine measurement. But it creates a fresh encoder+dhbp for the sweep stage (line ~507). Change it to reuse the Stage 3 model. Quick code fix.

---

## 3. Figures and Tables

### Figure 1: Pairwise Matrix Heatmaps (main result visualization)

```
┌─────────────────────────────────────────────────┐
│  (a) Unconstrained K×K          (b) Constrained │
│  ┌─────────────────────┐  ┌─────────────────────┐
│  │  Imp Bld LVg Tre Car│  │  Imp Bld LVg Tre Car│
│  │ ■■■ ... ... ... ... │  │ ████ .   .   .   .  │
│  │ ... ■■■ ... ▓▓▓ ... │  │  .  ████ .   .   .  │
│  │ ... ... ■■■ ... ... │  │  .   .  ████ .   .  │
│  │ ... ▓▓▓ ... ■■■ ... │  │  .   .   .  ████ .  │
│  │ ... ... ... ... ■■■ │  │  .   .   .   .  ████│
│  └─────────────────────┘  └─────────────────────┘
│  diag ratio: 0.094           diag ratio: 0.270   │
│  Tree→Bld: 0.789             Tree→Bld: 0.113     │
└─────────────────────────────────────────────────┘
```

> **⚠ Stale sketch values:** the constrained-panel 0.270 is the March equal-weight-BP value, which is
> also mathematically inconsistent (the constrained head guarantees diag ratio ≥ α, and α was 0.63–0.80
> then — see audit note in docs/neurips_research_directions_2026-06.md). The real constrained 3-seed value
> is 0.784 ± 0.003 (pairwise_diagnostics.json). The unconstrained-panel 0.094/0.789 is the single March
> run — regenerate BOTH panels from Experiment H + B outputs with one script/protocol before drafting.

**Data source**: Extract from trained checkpoints (Experiment B)
**Format**: matplotlib heatmap with annotated values, 2-panel figure
**Show**: 3-seed average matrix with std annotations

---

### Figure 2: Per-Class Accuracy — Unconstrained vs Constrained

```
        Unconstrained    Constrained
Imp     ████████  52%    █████████  55%
Bld     ████████████ 75% ████████████ 78%
LVg     ████  22%        ████████  45%
Tre     ████████████ 90% ████████████ 88%
Cars    ███  18%         ████████  41%
Clu     ▏  0%            ▏  0%
```

**Data source**: Per-class accuracy from Experiment A
**Format**: Grouped bar chart, 2 conditions, 6 classes
**Show**: Mean ± std across 3 seeds

---

### Figure 3: Cosine Similarity — Dose Response (key gradient result)

```
cos(∇BP, ∇Direct)
  1.0 ┤                              ● baseline (same path, diff batch)
      │
  0.8 ┤                        ●─────── 50% labels
      │                  ╱
  0.6 ┤            ╱
      │      ╱
  0.4 ┤
      │
  0.2 ┤
      │
  0.0 ┤──●──────────────────────────── 10% labels
      │
 -0.1 ┤
      └──┬──────────┬──────────┬──
        10%        30%        50%     label fraction
```

**Data source**: Experiment A (3-seed cosine measurements at 10% and 50%)
**Format**: Line plot with error bars, one line per parameter group
**Show**: unary_1.net[0], unary_1.net[-1], encoder.layer1 as separate lines
**Add**: Horizontal dashed line for random baseline per parameter

---

### Figure 4: BP-Only Parameters (supplementary/appendix)

```
Gradient signal sources:

                   Direct path    BP path
                   ──────────     ───────
unary_1.net[0]     ████████       ████████████
unary_1.net[-1]    ████████       ████████████
encoder.layer1     ████           ████████████████
encoder.layer2     ░░░░░░░░       ████████████
encoder.layer3     ░░░░░░░░       ████████████
pairwise_12.α      ░░░░░░░░       ████████
pairwise_12.R      ░░░░░░░░       ████████

░ = zero gradient    █ = receives gradient
```

**Format**: Schematic diagram (not a data plot)
**Purpose**: Visual explanation of why BP dominates the optimization for deep layers

---

### Table 1: Main Results (3-seed)

| Config | Labels | Overall Acc | Mean Class | Cars | Diag Ratio | cos(net[0]) |
|--------|--------|-------------|------------|------|------------|-------------|
| Unconstrained | 10% | X ± Y | X ± Y | X ± Y | X ± Y | X ± Y |
| Constrained | 10% | X ± Y | X ± Y | X ± Y | X ± Y | X ± Y |
| Constrained | 50% | X ± Y | X ± Y | X ± Y | X ± Y | X ± Y |

**Data source**: Experiments A + C

---

### Table 2: Cosine Similarity Full Breakdown

| Parameter | 10% (3 seeds) | 50% (3 seeds) | Baseline |
|-----------|---------------|---------------|----------|
| unary_1.net[0] | X ± Y | X ± Y | 0.913 |
| unary_1.net[-1] | X ± Y | X ± Y | 1.000 |
| encoder.layer1 | X ± Y | X ± Y | 0.290 |
| encoder.layer3 | BP-ONLY | BP-ONLY | 0.208 |
| pairwise_12.alpha | BP-ONLY | BP-ONLY | 0.425 |
| pairwise_12.residual | BP-ONLY | BP-ONLY | 0.678 |

---

### Table 3: Loss-Normalized Gradient Norms (appendix)

Same format as Experiment 18 Table 3, but with 3-seed mean ± std.

---

## 4. Literature — Complete Citation List

### Primary citations (MUST appear in main text)

**Pairwise potentials in CRF/MRF:**
1. Krahenbuhl & Koltun 2011. "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials." NeurIPS. https://arxiv.org/abs/1210.5644
   — DenseCRF. Gaussian pairwise kernels. Standard post-processing. Motivates Potts-like structure for efficiency.

2. Zheng et al. 2015. "Conditional Random Fields as Recurrent Neural Networks." ICCV. https://arxiv.org/abs/1502.03240
   — CRF-as-RNN. End-to-end CRF training. Learns compatibility transform initialized from Potts. Does NOT analyze degeneracy.

3. Larsson, Arnab et al. 2018. "A Projected Gradient Descent Method for CRF Inference allowing End-To-End Training of Arbitrary Pairwise Potentials." https://arxiv.org/abs/1701.06805
   — Learns UNCONSTRAINED pairwise potentials. Closest prior work. Visualizes learned filters but does NOT analyze for class remapping degeneracy.

4. Knobelreiter & Pock 2020. "Belief Propagation Reloaded: Learning BP-Layers for Labeling Problems." CVPR. https://arxiv.org/abs/2003.06281
   — Differentiable BP as neural network layer. No degeneracy analysis.

**Gradient conflict in multi-task learning:**
5. Du, Czarnecki et al. 2018. "Adapting Auxiliary Losses Using Gradient Similarity." NeurIPS workshop. https://arxiv.org/abs/1812.02224
   — Measures cos(grad_aux, grad_primary). Zeros out auxiliary when negative. Closest methodological match to our gradient analysis.

6. Yu et al. 2020. "Gradient Surgery for Multi-Task Learning." NeurIPS. https://arxiv.org/abs/2001.06782
   — PCGrad. Defines "conflicting gradients" as cos < 0. Projects conflicting gradients. Establishes cosine similarity as standard diagnostic.

7. Liu et al. 2021. "Conflict-Averse Gradient Descent for Multi-task Learning." NeurIPS. https://arxiv.org/abs/2110.14048
   — CAGrad. Minimizes worst-case task loss under gradient conflict.

### Secondary citations (important context)

8. Chandra et al. 2017. "Dense and Low-Rank Gaussian CRFs Using Deep Embeddings." ICCV. https://arxiv.org/abs/1611.09051
   — Low-rank pairwise constraint for computational efficiency. NOT motivated by degeneracy.

9. Shamsian et al. 2023. "AuxiNash: Playing a Nash Equilibrium on Auxiliary Losses." ICML. https://arxiv.org/abs/2301.13501
   — Auxiliary learning as asymmetric bargaining game. Relevant framing for BP as auxiliary path.

10. Jiang et al. 2023. "ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning." NeurIPS. https://arxiv.org/abs/2301.12618
    — Negative transfer can occur even with aligned gradients. Important nuance.

11. Guan et al. 2024. "Neural Markov Random Field for Stereo Matching." CVPR. https://openaccess.thecvf.com/content/CVPR2024/html/Guan_Neural_Markov_Random_Field_for_Stereo_Matching_CVPR_2024_paper.html
    — CVPR 2024. Learned MRF potentials. Our degeneracy finding is directly relevant to this line of work.

12. Zhu et al. 2023. "E-CRF: Boundary-caused Class Weights Confusion in CRF." ICLR. https://arxiv.org/abs/2112.07106
    — Boundary class confusion in CNN classifier weights. Related but distinct from pairwise matrix degeneracy.

### Tertiary citations (background, appendix)

13. Domke 2012. "Generic Methods for Optimization-Based Modeling." AISTATS. http://proceedings.mlr.press/v22/domke12.html
    — Backprop through truncated optimization. Theoretical foundation for differentiable CRF training.

14. Wang et al. 2020. "Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models." ICLR. https://arxiv.org/abs/2010.05874
    — Negative cosine → negative transfer in multilingual models.

15. Sener & Koltun 2018. "Multi-Task Learning as Multi-Objective Optimization." NeurIPS. https://arxiv.org/abs/1810.04650
    — Pareto-optimal gradient combination. Foundational MTL optimization paper.

16. Vemulapalli et al. 2016. "Gaussian Conditional Random Field Network for Semantic Segmentation." CVPR.
    — Gaussian CRF with CNN pairwise. No degeneracy analysis.

17. Lin et al. 2016. "Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation." CVPR. https://arxiv.org/abs/1504.01013
    — Piecewise CRF training. No pairwise failure mode analysis.

18. Belanger & McCallum 2016. "Structured Prediction Energy Networks." ICML. https://proceedings.mlr.press/v48/belanger16.html
    — Energy-based models with deep networks. Uses Domke's truncated differentiation.

---

## 5. Experiment Priority and Timeline (updated 2026-04-07)

```
PRIORITY   EXPERIMENT                              TIME     STATUS
────────   ──────────────────────────────────────   ──────   ──────
DONE       A: 3-seed reproduction (10%)             ~12h    COMPLETE (Session 1)
DONE       B: Pairwise diagnostics per seed (10%)   ~30min  COMPLETE (Session 1)
DONE       F: No-BP baseline (10%, 3 seeds)         ~6h     COMPLETE (Session 1)
P0 NEW     G: Dumb pooling baseline (receptive field) ~2h   NEEDED — blocks paper
P0         A: 3-seed reproduction (50%)             ~6h     PENDING (Session 2)
P1         H: Unconstrained pairwise training       ~2h     NEEDED for Figure 1a real data
DROPPED    C: Constrained vs unconstrained cosine   —       DROPPED (reverse causality, Vuln #4)
P1         E: Fix data sweep in script              ~10min  PENDING
P2         D: Potsdam dataset                       ~2 days Full paper only
```

### Kaggle execution plan (2 sessions)

**Session 1** (~21h, overnight + day):
```
# Seed 0, 1, 2 at 10% labels (with BP)
for SEED in 0 1 2; do
    python complete_training.py \
        --contrastive_ckpt /kaggle/input/models/.../contrastive_best.pth \
        --output_dir /kaggle/working/output_pct10_seed${SEED} \
        --labeled_percent 10 --epochs_seg 30 --seed ${SEED}
done

# Seed 0, 1, 2 at 50% labels (with BP)
for SEED in 0 1 2; do
    python complete_training.py \
        --contrastive_ckpt /kaggle/input/models/.../contrastive_best.pth \
        --output_dir /kaggle/working/output_pct50_seed${SEED} \
        --labeled_percent 50 --epochs_seg 30 --seed ${SEED}
done

# Experiment F: Seed 0, 1, 2 at 10% labels WITHOUT BP (convergence baseline)
for SEED in 0 1 2; do
    python complete_training.py \
        --contrastive_ckpt /kaggle/input/models/.../contrastive_best.pth \
        --output_dir /kaggle/working/output_pct10_nobp_seed${SEED} \
        --labeled_percent 10 --epochs_seg 30 --seed ${SEED} --no_bp
done
```

**Session 2** (~6h):
```
# Cosine measurements for all 6 checkpoints
for DIR in output_pct10_seed0 output_pct10_seed1 output_pct10_seed2 \
           output_pct50_seed0 output_pct50_seed1 output_pct50_seed2; do
    python test_gradient_cosine.py \
        --contrastive_ckpt /kaggle/input/models/.../contrastive_best.pth \
        --seg_ckpt /kaggle/working/${DIR}/best_segmentation.pth
done

# Pairwise diagnostics for all 6 checkpoints
for DIR in output_pct10_seed0 ...; do
    python extract_pairwise_diagnostics.py \
        --seg_ckpt /kaggle/working/${DIR}/best_segmentation.pth
done

# Experiment C: unconstrained pairwise (1 run)
python complete_training.py \
    --contrastive_ckpt /kaggle/input/models/.../contrastive_best.pth \
    --output_dir /kaggle/working/output_pct10_unconstrained \
    --labeled_percent 10 --epochs_seg 30 --seed 0 --unconstrained_pairwise

python test_gradient_cosine.py \
    --seg_ckpt /kaggle/working/output_pct10_unconstrained/best_segmentation.pth
```

---

## 6. Scripts to Write

| Script | Purpose | Lines (est.) |
|--------|---------|-------------|
| `extract_pairwise_diagnostics.py` | Load checkpoint, compute K×K matrix stats, save heatmap PNG | ~120 |
| `plot_paper_figures.py` | Generate all paper figures from saved results | ~200 |
| Add `--seed` to `complete_training.py` | Reproducibility across runs | ~5 lines |
| Add `--unconstrained_pairwise` to `complete_training.py` | Experiment C | ~15 lines |

---

## 7. Paper Writing Checklist (updated 2026-04-07)

**Experiments:**
- [x] Experiment A: 3-seed, 10% with BP — COMPLETE
- [x] Experiment B: Pairwise diagnostics per seed (10%) — COMPLETE
- [x] Experiment F: 3-seed, 10% no-BP baseline — COMPLETE
- [ ] Experiment G: Dumb pooling baseline (receptive field match) — P0, BLOCKS PAPER
- [ ] Experiment A: 3-seed, 50% with BP — PENDING
- [ ] Experiment H: Unconstrained pairwise training (real data for Figure 1a)

**Figures:**
- [x] Figure 1: Pairwise heatmaps (constrained from real data, unconstrained needs Exp H)
- [ ] Figure 2: Per-class accuracy bar chart (BP vs no-BP vs dumb pooling)
- [ ] Figure 3: Cosine similarity (descriptive, not causal)
- [x] Figure 4: BP-only parameter diagram (appendix)

**Writing:**
- [ ] Abstract rewritten (removed causal gradient claim, added receptive field baseline)
- [ ] Acknowledge entropy-weighted BP is NOT exact inference
- [ ] Gradient analysis reframed as descriptive (inference offloading), not causal
- [ ] Receptive field baseline results integrated
- [ ] Causal direction acknowledged as ambiguous
- [ ] All 18+ citations formatted
- [ ] Submitted to SPIGM workshop

---

## 8. Key Arguments to Make (and Pitfalls to Avoid) — updated 2026-04-07

### DO say:
- "We characterize a previously undocumented failure mode of learned pairwise potentials"
- "Gradient divergence between BP and direct supervision is EXPECTED in end-to-end structured prediction (inference offloading), but unconstrained potentials exploit this freedom"
- "Our entropy-gated message passing is approximate, not exact BP — we acknowledge this"
- "We compare against a receptive-field-matched baseline to isolate the structured prediction contribution"
- "This explains why Potts potentials remain dominant despite decades of work on learned potentials"

### DO NOT say:
- "Gradient conflict CAUSES pairwise degeneracy" (causal direction is ambiguous)
- "Exact sum-product belief propagation" (entropy weighting breaks exactness)
- "We discovered that gradient directions can conflict" (PCGrad 2020 did)
- "7-10x gradient amplification" (confounded, debunked)
- "BP acts as a preconditioner" (too vague, wrong framing)
- "Novel gradient analysis technique" (standard MTL methodology)

### Anticipated reviewer objections and responses:
1. **"Just use Potts / DenseCRF"** → "That's our point. Everyone does, but nobody said WHY learned potentials fail. We provide the first characterization."
2. **"CRF is dead, who cares"** → "Neural MRF (Guan CVPR 2024) and learned message passing are active. This failure mode applies to any learned pairwise potential."
3. **"Single dataset"** → Valid for workshop paper. Potsdam for full paper.
4. **"Gradient conflict is known in MTL"** → "Yes. We cite PCGrad. Our contribution is not the measurement technique but the finding that structured prediction layers create gradient divergence as a natural consequence of inference offloading, and that unconstrained pairwise potentials exploit this."
5. **"No theory, just empirics"** → Fair. Theoretical analysis of when degeneracy occurs (conditions on K, graph structure, label fraction) is future work.
6. **"This isn't exact BP"** → "Correct. Our entropy-gated message passing is approximate. We acknowledge this and note the degeneracy occurs regardless of the weighting scheme."
7. **"BP improvement is just receptive field"** → "We compare against a dumb pooling baseline matching the same ~103px receptive field. [Result: BP beats/matches dumb pooling by X pp]."
8. **"Causal direction is unclear"** → "Agreed. We present descriptive characterization. Whether degeneracy causes gradient warping or gradient freedom enables degeneracy is an open question."
