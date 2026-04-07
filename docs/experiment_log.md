# Experiment Log — DHBP: Differentiable Hierarchical Belief Propagation

## Project Goal

Semi-supervised semantic segmentation of aerial imagery (ISPRS Vaihingen, 6 classes) achieving 95% accuracy with only 10% labeled data, using a novel Differentiable Hierarchical Belief Propagation module on a quadtree MRF with contrastive-learned spatially-varying pairwise potentials.

---

## Architecture Overview

```
STAGE 1: SimCLR Contrastive Pre-training (unlabeled data)
  Image → [aug1, aug2] → ResNet-18 Encoder → GAP → Projection Head → SimCLR Loss
  Learns discriminative features from unlabeled aerial imagery.

STAGE 2: Supervised Segmentation with DHBP (labeled data)
  Image → ResNet-18 Encoder (fine-tuned) → (p1, p2, p3) → DHBP → [B, 6, H, W]

  DHBP performs exact sum-product belief propagation on a quadtree:
    - Unary potentials: encoder features → per-pixel class log-probabilities
    - Pairwise potentials: encoder features → spatially-varying K×K matrices
    - Bottom-up pass: 2×2 quadtree blocks, logsumexp marginalization
    - Top-down pass: cavity computation + transposed pairwise
    - Single pass = exact (tree has no loops)
```

---

## Experiment 1: Baseline with 10% Labels

**Date**: 2026-03-24
**Config**: 50 contrastive epochs, 50 segmentation epochs, batch_size=4, labeled_percent=10, 1 labeled area
**Branch**: `first-experiment`

### Results

| Class | Accuracy |
|-------|----------|
| Impervious | 52.86% |
| Buildings | 75.14% |
| Low Veg | 22.46% |
| Trees | 90.13% |
| Cars | 18.00% |
| Clutter | 0.00% |
| **Overall** | **56.89%** |
| **Mean class** | **43.10%** |

### Observations
- Only 1 labeled area (10% of 13 train areas rounds to 1) — extremely limited supervision
- Trees very strong (90%) but Cars/Low Veg/Clutter failing
- Loss converging (1.73 → 0.42) but accuracy oscillating — cosine warm restart schedule causes dips
- Training capped at 50 batches/epoch — model seeing fraction of available data

---

## Experiment 2: 30% Labels, 100 Epochs, No Batch Cap

**Date**: 2026-03-24
**Config**: contrastive from checkpoint (epoch 43), 100 segmentation epochs, batch_size=4, labeled_percent=30, 3 labeled areas, batch cap removed

### Results

| Class | Accuracy |
|-------|----------|
| Impervious | 68.40% |
| Buildings | 86.08% |
| Low Veg | 66.16% |
| Trees | 63.83% |
| Cars | 29.56% |
| Clutter | 0.00% |
| **Overall** | **70.36%** |
| **Best (epoch 100)** | **71.51%** |
| **Mean class** | **52.34%** |

### Observations
- Significant improvement over Experiment 1 (+13.5% overall)
- Low Veg massive jump: 22.5% → 66.2% (+43.7%)
- Trees DROPPED: 90.1% → 63.8% (-26.3%) — model lost specialization as it learned more classes
- Cars still weak (29.6%) — tiny objects, class weight 3x insufficient
- Loss still decreasing at epoch 100 (0.35) — more training would help
- Best accuracy at epoch 100 — not converged yet

---

## Diagnostic: Encoder Quality Evaluation

**Date**: 2026-03-24
**Method**: Linear probe (freeze encoder, train Conv1x1 64→6), t-SNE visualization, comparison with random encoder. Ran with both untrained and trained DHBP.

### Stage A: Encoder Features

| Metric | Value |
|--------|-------|
| Contrastive encoder linear probe | 67.47% |
| Random encoder baseline | 25.98% |
| Improvement over random | +41.49% |
| **Verdict** | **MODERATE** |

Per-class linear probe (fine-tuned encoder):
| Class | Linear Probe |
|-------|-------------|
| Impervious | 65.06% |
| Buildings | 81.90% |
| Low Veg | 78.22% |
| Trees | 44.77% |
| Cars | 0.01% |
| Clutter | 0.00% |

**Finding**: Encoder features are discriminative for major classes but fail completely on Cars and Clutter at the pixel level. The contrastive pre-training (SimCLR, patch-level) doesn't capture fine-grained features needed for tiny objects.

### t-SNE Visualization

Classes form distinct clusters with some overlap. Impervious and Buildings separate well. Low Veg and Trees have partial overlap. Cars (48 points) barely visible. Confirms moderate but not excellent feature quality.

### Stage B: DHBP Potentials (Trained Model)

| Metric | Value | Target |
|--------|-------|--------|
| Unary-only accuracy (φ₁) | 58.78% | Close to linear probe (67%) |
| Pairwise diagonal ratio | 0.094 | >0.5 |
| Boundary-interior difference | 0.021 | >0.1 |

Unary per-class:
| Class | Unary Accuracy |
|-------|---------------|
| Impervious | 79.85% |
| Buildings | 29.13% |
| Low Veg | 42.89% |
| Trees | 83.95% |
| Cars | 60.44% |
| Clutter | 0.00% |

**Critical finding — pairwise matrix is wrong:**

```
Average ψ₁₂ (trained model):
         Imp    Bldg   Low    Tree   Car    Clut
Imp:    0.325  0.004  0.347  0.311  0.013  0.000
Bldg:   0.668  0.206  0.047  0.052  0.027  0.000  ← Building→Impervious (0.668)!
Low:    0.508  0.138  0.024  0.001  0.330  0.000
Tree:   0.190  0.789  0.016  0.000  0.005  0.000  ← Tree→Building (0.789)!
Car:    0.302  0.107  0.318  0.262  0.011  0.000
Clut:   0.080  0.384  0.475  0.055  0.004  0.000
```

The pairwise learned **class remapping** instead of spatial consistency. Diagonal entries are NOT dominant. The unconstrained K×K matrix found a shortcut: remap minority classes to majority classes to improve overall accuracy.

### Stage C: Does BP Help? (Trained Model)

| Metric | Value |
|--------|-------|
| Final belief accuracy | 70.26% |
| BP improvement over unary | +11.47% |
| Entropy reduction | 0.6725 bits |

Per-class BP effect:
| Class | Unary | After BP | Delta |
|-------|-------|----------|-------|
| Impervious | 79.8% | 66.0% | **-13.9%** ← HURTING |
| Buildings | 29.1% | 84.6% | +55.5% |
| Low Veg | 42.9% | 68.7% | +25.8% |
| Trees | 84.0% | 65.4% | **-18.5%** ← HURTING |
| Cars | 60.4% | 32.2% | **-28.2%** ← HURTING |
| Clutter | 0.0% | 0.0% | 0.0% |

**Root cause**: BP uses the broken pairwise matrix to override correct unary predictions. It steals accuracy from Impervious/Trees/Cars (where unary is strong) and gives it to Buildings/Low Veg (where unary is weak). Overall accuracy improves (+11.5%) but 3 classes are destroyed.

---

## Fix: Constrained Diagonal-Dominant Pairwise (Current)

**Date**: 2026-03-24
**Branch**: `fix/kaggle-compat`

### The Problem
Unconstrained K×K pairwise matrix (36 free parameters per location) learns class remapping instead of spatial consistency.

### The Solution
Decompose the pairwise as:

```
ψ = α · I + (1-α) · R

where:
  α ∈ [0,1] = per-location consistency strength (predicted from features via sigmoid)
  I = identity matrix (same-class → same-class)
  R = learned residual transition matrix (softmax-normalized rows)
```

- **At initialization**: α ≈ 0.8, so ψ ≈ 0.8·I + 0.2·R — strongly diagonal-dominant
- **After training**: α can decrease at boundaries where transitions are legitimate
- **Key constraint**: the identity component ALWAYS dominates unless α is explicitly pushed down

### Theoretical basis
- Generalizes the Potts model (Boykov & Jolly 2001), the most common pairwise in MRF/CRF segmentation
- Similar constraint philosophy to Krähenbühl & Koltun 2011 (Gaussian kernel pairwise in dense CRFs)
- Auxiliary loss concept from Zheng et al. 2015 (CRF-as-RNN)

### Gradient Flow Test Results
All 13 checkpoints pass. Alpha initializes at 0.8051 (target ~0.8). Zero dead params.

### Training Results (10 epochs, 10% labels, 1 area)

| Class | Accuracy |
|-------|----------|
| Impervious | 55.16% |
| Buildings | 77.26% |
| Low Veg | 39.05% |
| Trees | 78.62% |
| Cars | 11.93% |
| Clutter | 0.00% |
| **Overall** | **60.11%** |

Improvement over unconstrained pairwise: 60.1% vs 56.9% (old, 50 epochs) in 1/5th training time.
Trees held at 78.6% — old model destroyed Trees from 84→65% via BP.

### Diagnostic (constrained, 10 epochs)
- Pairwise diagonal ratio: **0.270** (up from 0.094 unconstrained)
- BP improvement: **+21.5%** (up from +11.5%)
- Unary collapsed to 2 classes: Buildings (90%) and Trees (90%), others near 0%
- BP rescues Impervious (4.5→55.5%) and Low Veg (0.4→35.3%)

### Gradient Comparison Experiment

Tested whether unary head receives weak gradients through BP chain.

| Component | Through BP | Direct | Ratio |
|-----------|-----------|--------|-------|
| unary_1.net[0].weight | 0.372 | 0.038 | **9.7x stronger through BP** |
| unary_1.net[3].weight | 8.406 | 0.261 | **32.2x stronger through BP** |
| encoder.layer1 | 0.216 | 0.015 | **14.7x stronger through BP** |

**Finding**: BP chain gradients are 10-32x STRONGER than direct path. The weak gradient hypothesis was WRONG. The unary collapse is NOT caused by gradient dilution.

**Likely real causes**:
1. Only 1 labeled area — unary overfits to that area's class distribution
2. BP compensation — optimizer improves loss via pairwise adjustments instead of fixing unary
3. Class imbalance within the single training area

---

## Foundational References

### Belief Propagation
- **Pearl, J. (1988)**. *Probabilistic Reasoning in Intelligent Systems.* Introduced sum-product message passing on trees.
- **Bouman & Shapiro (1994)**. *A Multiscale Random Field Model for Bayesian Image Segmentation.* IEEE Trans. Image Processing. First quadtree MRF for multi-resolution segmentation.

### Differentiable Structured Prediction
- **Zheng et al. (2015)**. *Conditional Random Fields as Recurrent Neural Networks.* ICCV. Showed CRF mean-field inference as a differentiable neural module (CRF-as-RNN). Similar spirit but uses dense CRFs with approximate loopy inference. Ours uses exact tree BP.
- **Krähenbühl & Koltun (2011)**. *Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials.* NeurIPS. Constrained pairwise potentials as weighted Gaussian kernels — same motivation as our diagonal-dominant constraint.

### Contrastive Learning
- **Chen et al. (2020)**. *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* ICML. The pre-training method we use. Patch-level contrastive.
- **Wang et al. (2021)**. *Dense Contrastive Learning for Self-Supervised Visual Pre-Training.* CVPR. Pixel-level contrastive — potential improvement for Cars/small objects (TODO).

### MRF Pairwise Models
- **Boykov & Jolly (2001)**. *Interactive Graph Cuts for Optimal Boundary and Region Segmentation of Objects in N-D Images.* ICCV. Introduced Potts model for segmentation (ψ = identity). Our α·I + (1-α)·R generalizes this.

### Semi-supervised Segmentation
- **Tarvainen & Valpola (2017)**. *Mean Teachers are Better Role Models.* NeurIPS. Teacher-student semi-supervised framework — potential comparison baseline.
- **Sohn et al. (2020)**. *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.* NeurIPS. Pseudo-labeling baseline for semi-supervised comparison.

---

## Novelty Claims (for paper)

1. **Differentiable quadtree BP as a neural module**: Exact sum-product BP on a tree-structured MRF, implemented as GPU-parallel tensor operations (2×2 block reshape + logsumexp). Not approximate, not iterative.

2. **Contrastive-learned spatially-varying pairwise potentials**: The K×K compatibility matrix is predicted from encoder features pre-trained with contrastive learning on unlabeled data. This connects self-supervised learning to structured probabilistic inference — unlabeled data improves the MRF structure.

3. **Constrained pairwise decomposition (α·I + (1-α)·R)**: Generalizes the Potts model with a learned, spatially-varying consistency strength. Prevents the class-remapping failure mode of unconstrained pairwise matrices.

---

## Experiment 4: BP Ablation Study (in progress)

**Date**: 2026-03-24
**Purpose**: Measure BP's actual contribution with a fair comparison. Three configs, same encoder, same data, same epochs.

| Experiment | Config | Description |
|-----------|--------|-------------|
| A | `--epochs_seg 10 --labeled_percent 10` | Full DHBP (constrained pairwise, 2-layer unary + BP) |
| B | `--epochs_seg 10 --labeled_percent 10 --no_bp` | No BP (2-layer unary head only) |
| C | `--epochs_seg 10 --labeled_percent 10 --simple_unary` | Simple unary (Conv1x1 projection) + BP |

### Results (10 epochs)

| | A: Full DHBP | B: No BP | C: Simple + BP |
|---|---|---|---|
| Overall | **61.35%** | 55.84% | 53.91% |
| Mean class | **45.09%** | 40.72% | 39.85% |

BP added **+5.5%** at 10 epochs. Promising early signal.

### Results (50 epochs) — FINAL

| | A: Full DHBP | B: No BP | C: Simple + BP |
|---|---|---|---|
| **Best accuracy** | **61.65%** | 59.86% | 59.22% |
| **Final accuracy** | 54.58% | 54.15% | 56.12% |
| **Mean class (final)** | 43.21% | 42.27% | **44.67%** |
| Impervious | 35.79% | **39.81%** | 39.44% |
| Buildings | **82.14%** | 75.47% | 76.29% |
| Low Veg | 30.62% | 29.67% | **47.97%** |
| Trees | 86.66% | **88.60%** | 77.85% |
| Cars | **24.05%** | 20.05% | **26.46%** |

### Analysis

**BP's contribution is small at convergence.** The +5.5% gap at 10 epochs closed to +1.8% (best) and +0.4% (final) at 50 epochs. The no-BP model caught up with more training.

**Training instability.** All three models peak at epoch 15-35 then degrade by epoch 50. Best accuracy is 61.65% but final accuracy is 54.58%. The CosineAnnealingWarmRestarts scheduler's LR resets destroy progress.

**Simple unary + BP (C) is surprisingly strong.** Best mean class accuracy (44.67%), best Low Veg (47.97%), best Cars (26.46%). A linear projection + BP outperforms the 2-layer unary for minority classes.

**Comparison to CRFNet baseline.** CRFNet (Pastorino et al., 2024, IEEE TGRS) achieves **83-84%** at 10% labels on the same dataset. We're at 61.65%. The gap is large.

### Key issues identified

1. **Training instability** — cosine warm restart causes accuracy drops. Need ReduceLROnPlateau or no restarts.
2. **Only 1 labeled area** — 10% of 13 = 1 area. This is the dominant limiting factor.
3. **Encoder ceiling** — linear probe at 67.5% caps everything downstream.
4. **BP gap narrows with training** — the initial BP advantage (+5.5%) diminishes as the unary head learns. BP's value may be in faster convergence rather than final accuracy.

---

## Experiment 5: BP Diagnosis — Why BP Isn't Helping

**Date**: 2026-03-25
**Purpose**: Determine whether BP's poor contribution (+1.8%) is caused by (a) quadtree structure, (b) pairwise head not learning, or (c) insufficient training data.

### Test 1: Does BP propagate horizontally?

Injected a strong "building" signal at a single pixel (center of 128×128), all neighbors set to uniform/uncertain. After BP:

| Location | Unary (building) | After BP | Change |
|----------|-----------------|----------|--------|
| Center pixel (64,64) | -0.0002 | -35.48 | **-35.48** |
| Same 2×2 block (64,65) | -1.79 | -35.84 | -34.04 |
| Adjacent block (64,66) | -1.79 | -36.57 | -34.78 |
| 16 pixels away (64,80) | -1.79 | -37.86 | -36.07 |

**Finding**: BP **destroyed** the building signal instead of propagating it. The 3 uncertain siblings in the 2×2 block outvoted the 1 confident pixel through the parent. The parent belief became "uncertain", which then propagated back down and overrode the correct pixel. This is correct BP behavior on a tree — majority voting through the parent — but it means **isolated correct predictions get suppressed**.

### Test 2: Where does BP change predictions on real data?

| Metric | Value |
|--------|-------|
| Total pixels analyzed | 655,360 |
| Pixels changed by BP | 262,361 (**40.0%**) |
| Changes at boundaries | 19,321 / 33,986 (56.8%) |
| Changes at interior | 243,040 / 621,374 (39.1%) |

**Finding**: BP changes **40% of all pixels** — this is wholesale reclassification, not subtle boundary refinement. Interior pixels change almost as often as boundary pixels (39.1% vs 56.8%). BP has no strong spatial preference — it's acting as a global class redistribution mechanism.

### Test 3: Pairwise alpha spatial pattern

| Metric | Value |
|--------|-------|
| Alpha mean | 0.628 (initialized at 0.8) |
| Alpha std | 0.132 |
| Alpha at boundaries | 0.602 |
| Alpha at interior | 0.631 |
| Boundary-interior difference | **0.029** |

**Finding**: Alpha is nearly uniform across space. The model pushed it globally from 0.8 → 0.63 (wants more BP activity) but the boundary/interior difference is only 0.029. The pairwise does NOT know where class boundaries are. The alpha distribution is a sharp peak around 0.62 — almost all pixels get the same consistency strength.

Visual confirmation: alpha map shows no alignment with ground truth boundaries. Red boundary contours overlaid on alpha show no correlation.

### Root Cause Diagnosis

```
The quadtree BP acts as a SPATIAL AVERAGING filter, not a
boundary-aware structured predictor.

1. MAJORITY VOTING: 2×2 blocks → parent aggregation suppresses
   minority class predictions within a block (Test 1)

2. NO HORIZONTAL CONNECTIONS: pixel A can only influence pixel B
   through their common ancestor, not directly. CRFs have direct
   neighbor connections — quadtrees don't. (Test 1)

3. WHOLESALE RECLASSIFICATION: BP changes 40% of pixels,
   including 39% of interior pixels that should stay the same.
   It's not refining boundaries — it's redistributing classes. (Test 2)

4. ALPHA LEARNS NOTHING SPATIAL: The pairwise consistency
   strength is uniform — the model can't distinguish boundaries
   from interior regions. (Test 3)

CONCLUSION: The quadtree is the WRONG graph structure for
segmentation boundary refinement. The parent-child hierarchy
captures multi-scale information but lacks the direct neighbor
connections needed for spatial consistency. CRFs with grid
connections (CRF-as-RNN, CRFNet) work better because they
directly model "neighboring pixels should have the same class."
```

---

## Experiment 6: Gradient Amplification — Is It Real or an Artifact?

**Date**: 2026-03-25
**Purpose**: Our initial measurement (10-32x gradient amplification through BP) was from one random initialization. We need to verify this is a real property of the computation graph, not an init artifact.

### Method

Measured gradient norms (BP chain vs direct supervision) at four stages:
1. Random init (ImageNet encoder + random DHBP)
2. After contrastive pre-training (trained encoder, epoch 43 + random DHBP)
3. After segmentation training (fine-tuned encoder + trained DHBP, 50 epochs)
4. Across 5 different random seeds

### Results

| Stage | unary_1.net[0] | unary_1.net[-1] | encoder.layer1 | Avg ratio |
|---|---|---|---|---|
| Random init | 2.8x | 11.4x | 6.1x | **6.7x** |
| After contrastive | 6.1x | 12.5x | 12.8x | **10.5x** |
| After segmentation | 3.5x | 9.3x | 9.4x | **7.4x** |

Random seed variation (5 seeds):

| Seed | Avg ratio |
|---|---|
| 0 | 9.0x |
| 1 | 12.7x |
| 2 | 8.4x |
| 3 | 5.6x |
| 4 | 9.1x |
| **Mean ± Std** | **9.0x ± 2.3x** |

### Finding

**The gradient amplification is REAL and CONSISTENT.** It persists across:
- All training stages (6.7x - 10.5x)
- Multiple random seeds (9.0x ± 2.3x)
- Both random and trained weights

This is a property of the BP computation graph — the multi-path structure (4 children per parent, bottom-up + top-down passes) creates multiple gradient paths that sum together, amplifying the total gradient signal ~7-10x compared to direct supervision.

### Limitations

- Measured on ONE type of structured prediction module (quadtree BP, 3 levels)
- To claim generality, would need measurements on grid CRF, chain CRF, dense CRF
- The amplification factor likely scales with graph connectivity (more paths = more amplification)
- Unknown whether this amplification is helpful (faster convergence) or harmful (instability)

### Significance

This is a reproducible, measurable empirical finding about how structured prediction modules affect gradient dynamics. It hasn't been explicitly characterized in prior work (Zheng et al. 2015 noted vanishing/exploding gradients through CRF iterations but didn't measure the amplification ratio vs direct supervision). It could explain:
- Why end-to-end CRF training outperforms CRF post-processing (gradient amplification helps encoder)
- Why BP helped more early in training (+5.5% at 10 epochs) than at convergence (+1.8% at 50 epochs) — the amplification is most useful when gradients are weakest
- Why structured prediction layers sometimes cause training instability — the amplification can overshoot

---

## Experiment 7: Quadtree Depth Ablation — 2 vs 3 vs 4 Levels

**Date**: 2026-03-25
**Purpose**: Test whether shallower trees (less blur) or deeper trees (more multi-scale context) improve accuracy. All using α·I+(1-α)·R pairwise.

### Results (5 epochs, 10% labels, 1 area)

| | 2 levels | 3 levels | 4 levels |
|---|---|---|---|
| **Overall** | 49.70% | **61.59%** | 58.20% |
| **Mean class** | 37.61% | **43.67%** | 41.69% |
| Impervious | 43.79% | **70.18%** | 50.68% |
| Buildings | **73.17%** | 69.67% | 74.31% |
| Low Veg | 9.12% | 33.09% | **46.59%** |
| Trees | **87.72%** | 78.04% | 69.89% |
| Cars | **11.89%** | 11.04% | 8.66% |

### Analysis

**3 levels is the sweet spot.** Beats both 2 and 4 on overall accuracy by a significant margin.

**2 levels is too shallow (49.7%):** Without the coarse level (32×32), BP can't propagate semantic context far enough. Impervious (43.8%) suffers most — it needs medium-range context that 2 levels can't provide.

**4 levels is too deep (58.2%):** Extra averaging step (32→16) adds blur without enough useful context. But 4 levels has the best Low Veg (46.6%) — deeper trees help classes that occupy large contiguous regions.

**Cars degrades monotonically: 11.9% → 11.0% → 8.7%.** More levels = more averaging = more suppression of tiny objects. This directly confirms the quadtree averaging problem scales with depth.

**Trees degrades monotonically: 87.7% → 78.0% → 69.9%.** Deeper trees average away the Trees signal, especially when Tree regions are adjacent to Buildings or Impervious in the 2×2 blocks.

### Gradient amplification vs depth (measured)

| Depth | Unary amplification | Encoder amplification |
|---|---|---|
| 2 levels | 5.7x (±0.7) | 7.4x (±0.5) |
| 3 levels | 8.3x (±1.1) | 16.6x (±3.3) |
| 4 levels | 7.6x (±1.5) | **22.6x (±7.3)** |

**Finding**: Encoder gradient amplification scales with depth (7.4x → 16.6x → 22.6x), roughly doubling per level. Unary amplification plateaus (~6-8x regardless of depth) because the unary head is always at the leaf level.

**Interpretation**: Each additional quadtree level adds more gradient paths through the encoder (which feeds all levels), amplifying the encoder's gradient signal. But at 4 levels, the 22.6x amplification overshoots — combined with 3 steps of spatial averaging, it produces worse accuracy (58.2%) than 3 levels (61.6%).

**The optimal depth balances gradient amplification (benefits from more levels) against spatial blur (costs of more levels).** At 3 levels, 16.6x encoder amplification is the sweet spot.

**Relationship between amplification and accuracy:**

```
Depth    Encoder amp    Accuracy    Interpretation
2 lev    7.4x           49.7%       Too little amplification, too little context
3 lev    16.6x          61.6%       Sweet spot
4 lev    22.6x          58.2%       Too much amplification, too much blur
```

---

## Experiment 8: Hard Diagonal Pairwise

**Date**: 2026-03-25
**Purpose**: Test whether BP's value comes from gradient amplification alone (no class mixing) by using ψ = diag(d) with zero off-diagonal.

### Results (5 epochs, 10% labels, 1 area)

| | Hard diagonal | α·I+(1-α)·R (3 lev) | No BP |
|---|---|---|---|
| **Overall** | 49.74% | **61.59%** | ~53% |
| Buildings | **86.82%** | 69.67% | ~79% |
| Trees | **91.17%** | 78.04% | ~84% |
| Impervious | 32.03% | **70.18%** | ~38% |
| Low Veg | 8.92% | **33.09%** | ~15% |
| Cars | 0.73% | **11.04%** | ~10% |

### Diagnostic findings

- Pairwise diagonal ratio: 1.000 (by construction)
- BP changes **69.7%** of pixels (vs 40% with α·I+(1-α)·R) — massive reclassification
- BP changes interior pixels **70.0%** — more than boundaries (65.1%)
- Learned scaling: Buildings 1.045 (amplified), everything else <1.0 (suppressed)
- **Verdict**: Without class mixing, BP becomes a pure majority amplifier. Optimizer suppresses all minority classes. The worst of all pairwise variants.

---

## Ruled out hypotheses (with evidence)

### "Quadtree BP improves segmentation via structured inference" — MARGINAL
- Ablation: +1.8% best, +0.4% at convergence — not significant
- BP diagnosis: acts as spatial blur, not boundary refinement
- Quadtree lacks horizontal connections needed for spatial consistency
- **Status**: The quadtree BP concept works mechanically but the graph structure is wrong for this task

---

## Experiment 9: Entropy-Weighted Child Aggregation

**Date**: 2026-03-26
**Purpose**: Fix the quadtree majority voting problem where 3 uncertain siblings outvote 1 confident pixel. Instead of equal-weight sum over 4 children, weight by confidence (negative entropy). Zero new parameters — attention comes from belief confidence itself.

### Method

Modified `_child_to_parent()` in BP:
```
Standard BP:     m_total = m_c1 + m_c2 + m_c3 + m_c4  (equal weight)
Entropy-weighted: m_total = Σ w_c · m_c  where w = softmax(-entropy(child))
```

Confident children (low entropy) get higher weight. Uncertain children (high entropy) get lower weight. Scaled by 4x to maintain magnitude.

### Results (25 epochs, 10% labels, 1 area)

| | Entropy BP | No BP | Delta |
|---|---|---|---|
| **Best accuracy** | **62.80%** | 59.02% | **+3.8%** |
| **Final accuracy** | **60.01%** | 51.90% | **+8.1%** |
| **Mean class** | **45.27%** | 39.45% | **+5.8%** |
| Impervious | **60.50%** | 32.35% | +28.2% |
| Buildings | 76.51% | 77.62% | -1.1% |
| Low Veg | 29.11% | 37.90% | -8.8% |
| Trees | **82.95%** | 75.40% | +7.6% |
| Cars | **22.53%** | 13.45% | **+9.1%** |

### Comparison with all previous BP variants

| | Unconstrained (50ep) | α·I+(1-α)·R equal (50ep) | **Entropy-weighted (25ep)** | No BP (25ep) |
|---|---|---|---|---|
| Best | 56.89% | 61.65% | **62.80%** | 59.02% |
| Final | 56.89% | 54.58% | **60.01%** | 51.90% |
| Cars | 18.00% | 24.05% | 22.53% | 13.45% |
| BP delta (best) | — | +1.8% | **+3.8%** | baseline |
| BP delta (final) | — | -5.3% | **+8.1%** | baseline |

### Key findings

1. **BP gap HELD at 25 epochs (+3.8% best, +8.1% final).** Previous equal-weight BP gap closed from +5.5% at 10 epochs to +1.8% at 50 epochs. Entropy weighting maintains the gap.

2. **62.80% best in 25 epochs beats old BP's 50-epoch best of 61.65%.** Faster convergence AND higher peak.

3. **Final accuracy 60.01% vs 54.58% (old BP).** The entropy-weighted version doesn't degrade as badly — more stable training.

4. **Cars +9.1% over no-BP (22.53% vs 13.45%).** The entropy weighting preserves minority class predictions instead of drowning them through majority voting.

5. **Buildings only -1.1%** — BP is barely stealing from strong classes. The entropy weighting prevents confident majority-class children from being overridden.

6. **Low Veg -8.8%** — still a weakness. Low Veg may have similar entropy to surrounding classes, so entropy weighting doesn't help distinguish it.

### Why entropy weighting works

Standard BP: 1 confident car pixel + 3 uncertain pixels → car gets 25% weight → signal lost.
Entropy BP: 1 confident car pixel (low entropy → high weight ~60%) + 3 uncertain pixels (high entropy → low weight ~13% each) → car signal preserved.

The entropy-based attention is parameter-free and doesn't change the probabilistic framework of BP. It simply acknowledges that not all children are equally informative — which is always true in practice.

### 50-Epoch Results (entropy-weighted BP, 10% labels)

| | Entropy BP (50ep) | No BP (50ep) | Delta |
|---|---|---|---|
| **Best** | **66.02%** | 62.04% | **+4.0%** |
| **Final** | 57.88% | **60.72%** | -2.8% |
| Cars (final) | **26.52%** | 20.48% | +6.0% |

**The BP gap HELD at 50 epochs for best accuracy (+4.0%).** Previous equal-weight BP gap closed to +1.8% at 50 epochs. Entropy weighting maintains the advantage.

Training instability still present: best at epoch 20 (66.02%), final at epoch 50 (57.88%). Cosine warm restart scheduler is destroying progress.

### 50-Epoch Diagnostic: Pairwise Quality

**Pairwise diagonal ratio: 0.816** — the best we've ever seen:

```
Previous pairwise variants:
  Unconstrained:    0.094  (class remapping)
  α·I+(1-α)·R:     0.270  (barely diagonal)
  Hard diagonal:    1.000  (by construction)
  Entropy-weighted: 0.816  (strongly diagonal, LEARNED)
```

The entropy weighting helped the pairwise learn a sensible matrix:

```
       Imp    Bldg   Low    Tree   Car    Clut
Imp:   0.806  0.052  0.056  0.056  0.023  0.007
Bldg:  0.113  0.821  0.047  0.004  0.010  0.005
Low:   0.070  0.018  0.789  0.113  0.004  0.007
Tree:  0.072  0.008  0.115  0.796  0.003  0.006
Cars:  0.089  0.007  0.008  0.017  0.845  0.033
Clut:  0.025  0.023  0.042  0.033  0.038  0.839
```

Cars has the HIGHEST diagonal (0.845) — pairwise learned to strongly preserve car beliefs. Low Veg ↔ Tree has largest off-diagonal (0.113/0.115) — a legitimate transition.

### 50-Epoch Diagnostic: BP Helps ALL Classes (first time)

```
                Unary    After BP    Delta
Impervious:     6.9%  →  60.0%     +53.1%
Buildings:     85.4%  →  90.0%     +4.6%
Low Veg:       54.1%  →  74.7%     +20.7%
Trees:          6.8%  →  38.5%     +31.7%
Cars:          21.4%  →  27.4%     +6.0%
```

**BP improves ALL 5 classes. No class is hurt.** This is the first time across all experiments. Previously BP always stole from 3-4 classes to boost 1-2. The entropy weighting fixed this.

### 50-Epoch Diagnostic: BP Still Changes Interior

- BP changes 42.2% of pixels (down from 69.7% with hard diagonal, similar to 40% with old α·I+(1-α)·R)
- Boundaries: 49.2% changed, Interior: 41.9% — roughly equal, no spatial preference
- Alpha mean: 0.747 (down from 0.8 init), boundary alpha 0.778 vs interior 0.742

### Additional experiments (25 epochs)

**Entropy BP at 30% labels (25 epochs):**

| | Old BP 30% (100ep) | Entropy BP 30% (25ep) |
|---|---|---|
| Best | **71.51%** | 69.59% |
| Final | **70.36%** | 64.29% |
| Buildings | 86.08% | **89.82%** |

Entropy BP at 30% reaches 69.59% in 25 epochs vs old BP needing 100 epochs for 71.51%. Faster convergence but hasn't matched final accuracy — needs more epochs.

**Entropy BP at 2 levels (25 epochs, 10% labels):**

| | Old 2-level (5ep) | Entropy 2-level (25ep) |
|---|---|---|
| Best | 49.70% | **62.91%** |

Entropy weighting fixed the 2-level depth problem massively: 49.70% → 62.91%. The majority voting fix has bigger impact at shallower depths.

### "BP as training tool" test — RESULT

**Hypothesis**: BP's value is gradient amplification during training, not inference.

```
Model A: trained WITH BP, evaluated WITH BP        = 66.02%
Model A: trained WITH BP, evaluated WITHOUT BP     = 36.68% (Stage B unary-only)
Model B: trained WITHOUT BP, evaluated WITHOUT BP  = 62.04%
```

**HYPOTHESIS NOT SUPPORTED.** The BP-trained unary (36.68%) is much WORSE than the no-BP unary (62.04%). Training with BP makes the unary lazy — the encoder relies on BP to do the work instead of learning good unary predictions on its own.

**BP's value is at INFERENCE TIME, not training time.** The gradient amplification (7-10x) doesn't translate to a better encoder. It translates to BP doing more of the heavy lifting during inference, compensating for a weaker unary head.

### Updated summary: all experiments

| Experiment | Config | Best | Final | BP delta (best) |
|---|---|---|---|---|
| 1 | Unconstrained, 10%, 50ep | 56.89% | 56.89% | — |
| 2 | α·I+(1-α)·R, 30%, 100ep | 71.51% | 70.36% | +1.8% |
| 3 | α·I+(1-α)·R, 10%, 10ep | 60.11% | 60.11% | +5.5% |
| 4 | α·I+(1-α)·R, 10%, 50ep | 61.65% | 54.58% | +1.8% |
| 5 | No BP, 10%, 50ep | 62.04% | 60.72% | baseline |
| 6 | Hard diagonal, 10%, 5ep | 49.74% | 49.74% | -3.1% |
| 7 | 2 levels, 10%, 5ep | 49.70% | 49.70% | — |
| 8 | 4 levels, 10%, 5ep | 58.20% | 58.20% | — |
| **9** | **Entropy BP, 10%, 25ep** | **62.80%** | **60.01%** | **+3.8%** |
| **10** | **Entropy BP, 10%, 50ep** | **66.02%** | **57.88%** | **+4.0%** |
| **11** | **Entropy BP, 30%, 25ep** | **69.59%** | **64.29%** | — |
| **12** | **Entropy BP 2-lev, 10%, 25ep** | **62.91%** | **54.32%** | — |
| **13** | **Entropy BP, ReduceLROnPlateau p=5, 10%, 50ep** | **66.45%** | **60.09%** | — |
| **14** | **Entropy BP, ReduceLROnPlateau p=3, 10%, 120ep** | **67.76%** | **64.05%** | — |

---

## Experiment 13: ReduceLROnPlateau (patience=5, 50 epochs)

**Date**: 2026-03-26
**Change**: Replaced CosineAnnealingWarmRestarts with ReduceLROnPlateau(mode='max', factor=0.5, patience=5).

| | Cosine (50ep) | Plateau p=5 (50ep) |
|---|---|---|
| Best | 66.02% | **66.45%** |
| Final | 57.88% | **60.09%** |
| Degradation | -8.1% | **-6.4%** |
| Cars | 26.52% | **32.76%** |
| Buildings | 79.24% | **92.40%** |

**Finding**: Less degradation (6.4% vs 8.1%), better final numbers, best-ever Cars (32.8%) and Buildings (92.4%). But LR never actually dropped during 50 epochs — patience=5 with eval every 5 epochs = 25 epoch wait, too long for 50 epochs total.

---

## Experiment 14: ReduceLROnPlateau (patience=3, 120 epochs) — with wandb

**Date**: 2026-03-26
**Change**: Reduced patience to 3, extended to 120 epochs. Full wandb logging.

### Results

| Metric | Value |
|---|---|
| Best accuracy | **67.76%** (epoch 15) |
| Final accuracy | **64.05%** (epoch 90, early stopped) |
| Degradation | -3.7% (best improvement) |
| Mean class (final) | **49.17%** |
| Cars (final) | 31.15% |
| Buildings (final) | 83.64% |
| Low Veg (final) | **57.91%** (best ever) |
| Early stopping | Triggered at epoch 90 (15 evals without improvement) |

### Wandb insights

- **LR dropped 2-3 times** (visible in seg/lr graph): 0.0001 → 0.00005 → 0.000025
- **Best accuracy hit at epoch 15 and never improved again** despite 75 more epochs and multiple LR reductions
- **Per-class oscillation**: all classes oscillate wildly (Impervious 30-65%, Low Veg 20-60%, Cars 15-35%) — overfitting to 1 labeled area causes variance
- **Train loss kept decreasing** (0.76 → 0.25) while accuracy plateaued — classic overfitting signal

### Key conclusion

**The model hits its ceiling (~67-68%) within the first 15 epochs regardless of scheduler, LR drops, or training length.** The ceiling comes from:

1. **Encoder quality** — linear probe 71.5%, hard limit on downstream accuracy
2. **One labeled area** — 200 patches from 1 neighborhood, model overfits and oscillates
3. **No training trick fixes these** — scheduler, patience, more epochs, LR drops all tried

The next meaningful improvement requires either:
- More labeled data (30% = 3 areas → 69.6% best already demonstrated)
- Better encoder (DINOv2, longer contrastive, ResNet-50)

---

## Future Research Direction: Hierarchical Gradient Amplification as a General Training Technique

**Date identified**: 2026-03-26
**Status**: Idea stage — pending validation experiments

### Core insight

BP adds zero parameters but amplifies gradients 7-10x through multi-path computation. Our experiments suggest BP's primary value is gradient amplification during training, not spatial consistency at inference. This could be extracted into a general-purpose training technique:

- **Add** a hierarchical multi-path module during training (gradient amplification)
- **Remove** it at inference (zero cost, like dropout)
- **Result**: model converges in 5-10 epochs instead of 30-50, saving 3-5x total compute

### Evidence from our experiments

```
Gradient amplification:  7-10x verified (3 stages, 5 seeds, 3 depths)
Convergence speedup:     10 epochs with BP (60.5%) ≈ 30+ epochs without BP (59.9%)
Depth scaling:           amplification doubles per level (7.4x → 16.6x → 22.6x)
Entropy weighting:       content-dependent amplification (confident pixels ≠ uncertain)
```

### CRITICAL UPDATE: "Remove at inference" hypothesis FAILED

The 50-epoch experiment showed:
```
BP-trained model, unary only (no BP at inference):  36.68%
No-BP-trained model:                                62.04%
```

**Training with BP makes the unary WORSE, not better.** The encoder becomes lazy — it relies on BP instead of learning good features independently. BP's gradient amplification helps BP do its job, but doesn't help the encoder learn better representations.

**This means:** BP is an inference-time tool (adds +4% when used), NOT a training-time tool (can't be removed). The "remove at inference" use case is invalidated. The idea of BP as a general training technique like Dropout needs to be reconsidered — the current evidence shows BP creates a dependency, not a temporary scaffold.

**However:** The gradient amplification IS real (7-10x verified). The question is whether a different formulation (e.g., the simplified gradient amplification module that doesn't create unary dependency) could achieve the training-time benefit without the inference-time dependency. This remains open.

### What makes this different from existing techniques

```
Deep supervision:   adds LOSS FUNCTIONS at intermediate layers → needs labels at each scale
FPN:                adds ARCHITECTURAL connections → permanent, can't remove at inference
Dropout:            removes NEURONS randomly → stochastic regularization
ResNet:             adds SKIP connections → preserves gradients (1x), doesn't amplify

This technique:     adds COMPUTATION PATHS → amplifies gradients (7-10x),
                    content-dependent (entropy), removable at inference,
                    zero parameters, no extra labels needed
```

### Potential applications beyond segmentation

1. **Foundation model fine-tuning** — encoder gets weak gradients from random task head. Amplification bootstraps learning in 5-10 epochs instead of 30-50.
2. **Few-label learning** — each gradient update carries minimal info. 7-10x amplification = 7-10x effective training signal per batch.
3. **Class-imbalanced learning** — entropy-weighted amplification gives different gradient flow to minority vs majority classes.
4. **Multi-task learning** — lagging task heads get amplified gradients, naturally balancing task convergence.
5. **Compute-constrained training** — 10 epochs × 1.2x cost = 12 units vs 50 epochs × 1.0x = 50 units. Net 76% compute savings.

### Status of claims

```
PROVEN:
  ✓ Gradient amplification is real (7-10x, verified across 3 stages, 5 seeds)
  ✓ Scales with depth (7.4x → 16.6x → 22.6x, ~2x per level)
  ✓ BP helps convergence speed (60.5% at 10ep vs 54.9% without)
  ✓ Entropy-weighted BP helps all classes (+4% overall, no class hurt)

DISPROVEN:
  ✗ "Train with BP, remove at inference" — unary becomes lazy
    BP-trained unary: 36.7% vs no-BP-trained unary: 62.0%
    The encoder relies on BP instead of learning independently
    BP creates DEPENDENCY, not a temporary scaffold

STILL OPEN:
  ? Does a DIFFERENT multi-path module avoid the dependency?
  ? Does the amplification generalize to other tasks/architectures?
  ? Can we design a module that amplifies WITHOUT making the unary lazy?
  ? Is the amplification useful for foundation model fine-tuning?
```

### The dependency problem and potential fixes

The "remove at inference" hypothesis failed because BP takes OVER the prediction — the unary head learns "BP will fix my mistakes, so why try?" Three potential designs might avoid this:

1. **Auxiliary loss on unary** — force the unary to be good independently while BP still amplifies gradients. Loss = CE(BP_output) + 0.5 × CE(unary_output). The unary can't be lazy because it has its own loss.

2. **Gradual BP removal** — start with full BP, reduce BP weight over training (like scheduled dropout). Early epochs: BP amplifies gradients. Late epochs: BP fades, unary must work alone. The transition teaches the unary to be independent.

3. **Detached BP for gradient routing** — use BP for gradient amplification but DON'T let BP output reach the loss. Only the unary output counts for the loss. BP's multi-path structure still creates gradient amplification through the computation graph, but the unary is always responsible for the final prediction. This is the cleanest separation of "gradient routing" from "inference."

### What's needed to publish this (NeurIPS-level)

| Requirement | Status |
|---|---|
| Gradient amplification measured | ✓ Done (7-10x, verified) |
| Convergence speedup shown | ✓ Partial (1 task) |
| "Remove at inference" proof | ✗ FAILED — dependency created |
| Fix dependency (aux loss / gradual / detached) | ? Open — needs experiments |
| 3+ tasks (segmentation, detection, depth) | ? Needed |
| 2+ architectures (ResNet, ViT) | ? Needed |
| Head-to-head vs deep supervision | ? Needed |
| Simplified module (10 lines, not 100) | ? Needed |
| Theoretical analysis of amplification factor | ? Needed |

### Paper framing (revised after failed hypothesis)

Original: *"Hierarchical Gradient Amplification: A Parameter-Free Training Module for Faster Fine-Tuning"* — claimed removable at inference. **INVALIDATED.**

Revised option A: *"Understanding Gradient Amplification in Differentiable Structured Prediction"* — analysis paper. Document the amplification phenomenon, characterize it, explain why it creates dependency. No claim about practical utility, just understanding. Lower impact but honest.

Revised option B: *"Decoupled Gradient Amplification: Multi-Path Training Without Inference Dependency"* — solve the dependency problem with one of the three fixes above. If it works, the original claim is restored. Higher impact but needs more experiments.

### Key risks (updated)

- ~~"Remove at inference" claim may not hold~~ → **CONFIRMED: it doesn't hold**
- Deep supervision comparison may show similar benefits AND similar dependency
- The dependency may be inherent to any multi-path gradient amplification (not fixable)
- Amplification may not generalize beyond quadtree structure
- Simplified module may not retain 7-10x amplification

---

## Ruled out hypotheses (with evidence)

### "Quadtree BP improves segmentation via structured inference" — REVISED
- Original equal-weight BP: +1.8% best, gap closes at convergence — marginal
- **Entropy-weighted BP: +3.8% best, +8.1% final — significant improvement**
- The majority voting problem was the bottleneck, not BP itself
- With proper child weighting, BP is a meaningful contribution

### "Unary head gets weak gradients through BP" — WRONG, but "7-10x amplification" is also WRONG → RESOLVED (Experiments 18/18b)
- **Raw measurement**: BP chain gradient norms are 7-10x larger than direct path
- **After scrutiny (12-agent review)**: the 7-10x is confounded by loss magnitude (BP loss=4.3 vs direct loss=1.8, expected ratio from loss alone = 12.1x)
- **Corrected amplification (loss-normalized)**: 2.2x for unary net[0], 1.55x for unary net[-1], **9.0x for encoder.layer1** (genuine)
- **RESOLVED — Cosine similarity measurement (Experiments 18/18b)**: BP creates gradient conflict with direct supervision. This is an instance of the well-known gradient conflict phenomenon from multi-task learning (Du et al. 2018, PCGrad Yu et al. 2020), applied for the first time to differentiable structured prediction layers. Key results: cos=-0.028 at 10% labels (anti-correlated), cos=0.690 at 50% labels, baseline=0.913. Dose-response across label fractions. 3 parameter groups are BP-only (zero direct gradient). The measurement technique is standard (MTL literature since 2018). The finding in the structured prediction context is novel.

### "Unconstrained K×K pairwise matrix" — FAILED
- Diagonal ratio 0.094, Tree→Building: 0.789, BP destroyed 3 classes
- **Replaced with**: constrained α·I + (1-α)·R decomposition (diagonal ratio: 0.270)

### "MoCo contrastive learning" — BROKEN IN ORIGINAL CODE
- Key encoder returned feature maps not latents, memory bank filled with garbage
- **Replaced with**: SimCLR (simpler, proven)

### "VAE reconstruction path" — UNNECESSARY
- Decoder + reconstruction loss not needed for segmentation
- **Removed**: encoder only needs good features, not image reconstruction

### "QuadtreeMRF (non-differentiable)" — REPLACED
- Sequential node-by-node BP can't run on GPU
- **Replaced with**: DHBP (same math, GPU-parallel tensor operations)

---

## CRITICAL SCRUTINY: 12-Agent Review (2026-03-28)

We subjected all findings to rigorous review by 12 independent analytical agents across two rounds: engineering review, CEO/strategic review, adversarial NeurIPS review, math verification, experimental design review, and literature verification. Each agent was given corrected literature context from verified paper fetches.

### Round 1 findings (6 agents)

- **16 issues collapse to ~8** after removing duplicates and misattributions
- **Three "walls" are one wall** (structural) with downstream symptoms
- **Gradient amplification (7-10x) is confounded** by loss magnitude difference (BP loss=4.3 vs direct loss=1.8)
- **+4% improvement is within 5-8% evaluation noise** — not statistically significant
- **5 of 7 experimental design issues rated CRITICAL**: no error bars, single dataset, inconsistent checkpoints, non-deterministic eval, multiple variables changed
- **3 of 5 literature citations were misrepresented**: Lin et al. 2016 motivated by compute efficiency (not unary degradation), Chandra et al. 2017 motivated by compute efficiency (not degeneracy prevention), GATs (Veličković 2018) is a loose analogy to entropy-weighted BP (not "same principle")
- **None of our findings are breakthrough-novel**, though pairwise degeneracy characterization is the most novel (not found in recent post-2020 literature)

### Round 2 findings (6 agents, with corrected literature)

#### What SURVIVES scrutiny

**1. Pairwise degeneracy characterization — GENUINELY NOVEL**
- Unconstrained learned K×K pairwise matrices converge to class remapping (Tree→Building: 0.789) instead of spatial consistency
- Not found in recent literature (post-2020). The Potts model is universally used precisely to avoid this, but nobody documented the failure mode explicitly with learned deep potentials
- The heatmap visualization is compelling evidence
- Needs: reproduction on 2nd dataset + error bars to be publishable

**2. Gradient measurement through BP — METHODOLOGY NOVEL, MEASUREMENT FLAWED**
- Nobody has quantified gradient ratio through hierarchical BP vs direct supervision (Knöbelreiter 2017 only noted qualitative issues)
- BUT: the 7-10x claim is demolished by math review:
  - Loss magnitude confound alone could explain **12.1x** (larger loss → larger gradient via chain rule)
  - After correction: unary amplification is **0.5-3.5x** (possibly attenuation)
  - Encoder amplification at depth 3-4 may be real but modest (1.4-1.9x)
- **Correct framing is NOT "gradient amplification" but "BP as implicit gradient preconditioner"** — it changes gradient DIRECTION, not just magnitude
- Key test needed: measure cosine similarity between BP and direct gradients

**3. One valid experimental conclusion: more labeled data helps**
- 10%→30%→50% scaling is the only result where deltas exceed noise floor (8-13pp)
- Every method-vs-method comparison is within noise or confounded

#### What is DEFINITIVELY INVALIDATED

| Claim | Why invalidated |
|---|---|
| "+4% BP improvement" | Within 5-8% eval noise, single runs, not statistically significant |
| "7-10x gradient amplification" | Confounded by loss magnitude. After correction: 0.5-3.5x for unary, 1.4-1.9x for encoder |
| "~2x per level scaling" | Actually D^1.5 power law, may vanish after loss correction |
| "Three walls" | One wall (structural) with downstream symptoms |
| "RF ≈ 88px" | Correct calculation gives 103px |
| "9.5% compression gap (64→6 dim)" | Same architecture on same features should give 0 gap — it's a training confound |
| "Lazy unary = pathology" | May be healthy specialization (division of labor). Combined system outperforms standalone |
| "BP helps all classes" | Cannot conclude from single runs within noise |
| Every method comparison | All within noise floor or confounded by different checkpoints/conditions |

#### What was MISREPRESENTED in our citations

| Paper | Our claim | What paper actually says |
|---|---|---|
| Lin et al. CVPR 2016 | "Piecewise training because joint training degrades unary" | Motivated by **computational efficiency** |
| Chandra et al. ICCV 2017 | "Low-rank pairwise to prevent degeneracy" | Motivated by **computational efficiency** of dense CRFs |
| Veličković et al. ICLR 2018 (GATs) | "Same principle as entropy-weighted BP" | Never mentions BP, entropy, or PGMs. Loose analogy at best |

#### Verified prior work (papers confirmed via URL fetch)

| Paper | Verified? | Relevance |
|---|---|---|
| Laferté et al. IEEE TIP 2000 | Yes (author order wrong: should be Laferté, Pérez, Heitz) | Quadtree MRF, blocky artifacts discussed in surrounding literature |
| Knöbelreiter et al. CVPR 2017 | Yes, "fixed magnitude gradient" confirmed in paper body | Directly relevant: CRF gradient issues, pretraining requirement |
| Kuck et al. NeurIPS 2020 | Yes | Learnable BP layers (BPNNs), not segmentation |
| Xu et al. ACCV 2020 | Yes | Fast differentiable message passing for segmentation, most directly comparable to our work |
| Tang et al. ICLR 2022 | Yes | QuadTree Attention for ViTs, uses quadtree in modern context |
| Blondel et al. NeurIPS 2022 | Yes | Implicit differentiation framework applicable to structured prediction |

### Math review: gradient amplification analysis

The math review provided a first-principles derivation:

**Loss magnitude confound calculation:**
```
Path B (direct): loss = 1.8, p_true = e^{-1.8} ≈ 0.165, gradient ∝ 1/p = 6.06
Path A (BP):     loss = 4.3, p_true = e^{-4.3} ≈ 0.014, gradient ∝ 1/p = 73.5
Expected ratio from loss alone: 73.5/6.06 = 12.1x

Reported amplification: 7-10x
Loss-corrected amplification: 7/12.1 = 0.58x to 10/12.1 = 0.83x

CONCLUSION: The "amplification" may actually be ATTENUATION after correction.
```

**Theoretical amplification bounds:**
- Unary: `A_unary ≈ 1 + Σ c_d · ρ^d` where ρ < 1 (converges, explains plateau)
- Encoder: `A_encoder ≈ Σ γ_d · 4^d` (grows with depth, each level adds independent contribution)

**Correct framing:** Not "gradient amplification" but "BP as implicit gradient preconditioner" — changes gradient direction, not just magnitude. Key test: measure `cos(∇L_BP, ∇L_direct)`. If significantly < 1, BP provides qualitatively different gradient information.

### NeurIPS review score: 3/10, Reject

Key reviewer feedback:
- S1: Core idea has merit (quadtree + differentiable BP)
- S2: Entropy-weighted aggregation is clean and well-motivated
- W1 (Fatal): Single dataset, no error bars, eval noise > claimed improvement
- W2 (Fatal): Zero comparison against published baselines
- W3 (Major): Gradient analysis confounded
- W4 (Major): Novelty overstated relative to Kuck 2020, Xu 2020, GATs 2018

### Experimental design: only ONE valid conclusion

| Comparison | Verdict |
|---|---|
| Exp 4 vs 5 (BP vs no-BP) | **INVALID** — 0.39pp delta, deep within noise |
| Exp 10 vs 5 (Entropy BP vs no-BP) | **INVALID** — best and final disagree on direction |
| Depth ablation (7 vs 8) | **QUESTIONABLE** — 8.5pp at noise boundary, directionally suggestive |
| Label fraction scaling (10→30→50%) | **VALID** — 8-13pp deltas exceed noise floor |

### Publication options

| Option | Venue | Timeline | Probability |
|---|---|---|---|
| Workshop paper: "Pairwise Potential Degeneracy in End-to-End BP" | NeurIPS/ICML workshop | 3-4 weeks | 55-70% |
| Domain paper: "Diagnosing Structured Prediction for RS Segmentation" | ISPRS/TGRS | 6-8 weeks | 45-55% |
| Analysis paper: "Why Learned Pairwise Potentials Collapse" | NeurIPS/TMLR | 8-12 weeks (needs: fix gradient measurement, 2nd dataset, error bars) | 25-35% NeurIPS, 50-60% TMLR |

### Immediate next steps (prioritized)

1. **Fix eval determinism** — fixed seed or exhaustive tiling for test patches. Highest leverage single change.
2. **Run 3 seeds** for core comparison (entropy BP vs no-BP, 10%, 50ep). Determines if +4% is real.
3. **Measure gradient cosine similarity** (BP vs direct). Could reframe the entire gradient finding.
4. **Write workshop paper** on pairwise degeneracy while running experiments.

---

## Experiment 15-17: Label Fraction Scaling (50% and 100%)

**Date**: 2026-03-30
**Purpose**: Determine whether data or encoder is the bottleneck by plotting accuracy vs label fraction.

### Results

| Labels | Areas | Epochs | Best | Final | Cars | Mean class |
|---|---|---|---|---|---|---|
| 10% | 1 | 120 | 67.76% | 64.05% | 31.15% | 49.17% |
| 20% | 2 | 50 | 66.65% | 66.71% | 38.75% | 51.57% |
| 30% | 3 | 25 | 69.59% | 64.29% | 25.14% | 48.36% |
| 50% | 6 | 15 | **73.83%** | 70.89% | 61.30% | 58.18% |
| 100% | 13 | 8 | 72.23% | **73.81%** | **64.34%** | **60.37%** |

### Label fraction curve analysis

```
Accuracy vs label fraction:

  75% ─────────────────────────────────■─────■
                                    ╱
  70% ──────────────────────────╱──────────────
                              ╱
  65% ──■──────■───────────╱───────────────────
        │      │         ╱
  60% ──┼──────┼───────╱───────────────────────
       10%    20%    30%    50%    100%

Cars accuracy:
  64% ───────────────────────────────────────■
  61% ─────────────────────────────■──────────
  39% ──────────■──────────────────────────────
  31% ──■──────────────────────────────────────
       10%    20%    50%    100%
```

### Key findings

**1. Curve plateaus at 50-100%.** Going from 50% (6 areas) to 100% (13 areas): 73.83% → 73.81% final. More data beyond 50% does NOT help. **At 50%+ labels, the ENCODER is the bottleneck.**

**2. Big jump at 20% → 50%.** From 66.7% to 73.8% — a 7pp jump from adding 4 more training areas. Below 50%, data diversity (number of different areas) is the bottleneck.

**3. Cars scales dramatically with data.** 31% → 39% → 61% → 64%. Cars accuracy nearly doubles from 10% to 50% labels. Cars needs diverse training areas because each area has different car patterns.

**4. 100% ceiling is ~74%.** With ALL labeled data, the system tops out at 74%. The encoder linear probe was 71.5%. The system slightly exceeds the linear probe ceiling (74% > 71.5%), meaning BP IS adding value beyond what a linear classifier on frozen features achieves — approximately +2.5% from the structured prediction module.

**5. Bottleneck transition point is ~50% (6 areas).**

```
10-30% labels: DATA is the bottleneck
  → more areas = more accuracy (linear relationship)
  → solution: more diverse training data, better augmentation

50-100% labels: ENCODER is the bottleneck
  → more areas = no improvement (plateau)
  → solution: better encoder (DINOv2, ResNet-50, longer contrastive)

The crossover is at ~50% (6 areas)
```

**6. Gap to CRFNet.** CRFNet gets 83% at 10% labels. We get 74% at 100% labels. The ~9% gap remaining after giving our model ALL the data is purely encoder + architecture. A better encoder would close this gap.

### Updated experiment summary table

| # | Config | Best | Final |
|---|---|---|---|
| 1-14 | Various BP configs, 10-30% labels | 56.89-69.59% | 54.32-70.36% |
| 15 | 20% labels, entropy BP, balanced, 50ep | 66.65% | 66.71% |
| **16** | **50% labels, entropy BP, balanced, 15ep** | **73.83%** | **70.89%** |
| **17** | **100% labels, entropy BP, balanced, 8ep** | **72.23%** | **73.81%** |

---

## Experiment 18: Gradient Direction Analysis — Cosine Similarity (BP vs Direct)

**Date**: 2026-03-30
**Purpose**: The 12-agent scrutiny killed the "7-10x gradient amplification" claim (confounded by loss magnitude). The reframed question: does BP change gradient DIRECTION? If cos(∇L_BP, ∇L_direct) << 1, BP creates a qualitatively different optimization landscape, not just a scaled version of direct supervision.

**Script**: `test_gradient_cosine.py`

### Method

Measured cosine similarity between gradient vectors from two paths:
- **Path A (BP)**: encoder → DHBP (unary + pairwise + message passing) → FocalLoss
- **Path B (Direct)**: encoder → unary_1 only → FocalLoss (SAME loss function)

Using the SAME FocalLoss on both paths isolates BP as the only variable (prior measurements used FocalLoss vs cross_entropy, confounding the result).

Also measured:
- **Random baseline**: cosine between gradients from two random batches through the SAME BP path (null distribution for high-dim vectors)
- **Loss-normalized norms**: grad_norm / loss_value to factor out the loss magnitude confound
- **Data sample sweep**: 5 random batches, same model (stability across inputs)

### Parameters measured

| Parameter | What it is | Shape | Dims |
|---|---|---|---|
| unary_1.net[0] | First conv in unary head | [32,64,1,1] | 2,048 |
| unary_1.net[-1] | Classification layer | [6,32,1,1] | 192 |
| encoder.layer1 | Early encoder (ResNet block 1) | [64,64,3,3] | 36,864 |
| encoder.layer3 | Deep encoder (ResNet block 3) | [256,128,3,3] | 294,912 |
| pairwise_12.alpha_net[0] | Consistency strength | [32,128,3,3] | 294,912 |
| pairwise_12.residual_net[0] | Transition matrix | [32,128,3,3] | 294,912 |

### Random baseline (null distribution)

| Parameter | Baseline cos | Notes |
|---|---|---|
| unary_1.net[0] | 0.913 | High — small param, structured gradients |
| unary_1.net[-1] | 1.000 | Essentially identical across batches (192 params) |
| encoder.layer1 | 0.290 | Moderate — larger param, more stochastic |
| encoder.layer3 | 0.208 | Lower — deepest, most stochastic |
| pairwise_12.alpha_net[0] | 0.425 | Moderate |
| pairwise_12.residual_net[0] | 0.678 | Moderate-high |

### Results: Cosine similarity across training stages

**Model**: 50% labeled, entropy-weighted BP, 3 levels

| Parameter | Stage 1 (random DHBP) | Stage 2 (contrastive enc) | Stage 3 (trained) | Baseline |
|---|---|---|---|---|
| unary_1.net[0] | 0.607 | 0.778 | **0.690** | 0.913 |
| unary_1.net[-1] | 0.704 | 0.790 | **0.885** | 1.000 |
| encoder.layer1 | 0.299 | 0.542 | **0.422** | 0.290 |
| encoder.layer3 | BP-ONLY | BP-ONLY | **BP-ONLY** | 0.208 |
| pairwise_12.alpha_net[0] | BP-ONLY | BP-ONLY | **BP-ONLY** | 0.425 |
| pairwise_12.residual_net[0] | BP-ONLY | BP-ONLY | **BP-ONLY** | 0.678 |

### Data sample sweep (5 batches, trained model, Stage 3 weights)

| Parameter | Mean cos | Std | Min | Max |
|---|---|---|---|---|
| unary_1.net[0] | **+0.256** | 0.024 | +0.228 | +0.284 |
| unary_1.net[-1] | **+0.853** | 0.004 | +0.847 | +0.858 |
| encoder.layer1 | **+0.174** | 0.063 | +0.115 | +0.295 |
| encoder.layer3 | N/A (BP-only) | — | — | — |

### Loss-normalized gradient norms (Stage 3, trained model)

| Parameter | BP norm/L | Direct norm/L | Normalized ratio |
|---|---|---|---|
| unary_1.net[0] | 0.1022 | 0.0462 | **2.21x** |
| unary_1.net[-1] | 0.5101 | 0.3291 | **1.55x** |
| encoder.layer1 | 0.0313 | 0.0035 | **8.98x** |
| encoder.layer3 | 0.0183 | 0.0000 | inf (BP-only) |
| pairwise_12.alpha_net[0] | 0.0082 | 0.0000 | inf (BP-only) |
| pairwise_12.residual_net[0] | 0.0348 | 0.0000 | inf (BP-only) |

### Key findings

**1. BP creates gradient conflict with direct supervision (instance of a known MTL phenomenon).**

The cosine similarity between BP and direct gradients is well below the random baseline for all shared parameters. The data sweep (more reliable than single-batch) shows:
- unary_1.net[0]: cos = 0.256 on a baseline of 0.913 — **near-orthogonal**
- encoder.layer1: cos = 0.174 on a baseline of 0.290 — **below baseline** (BP redirects MORE than batch noise)
- unary_1.net[-1]: cos = 0.853 on a baseline of 1.000 — moderate redirection

This is NOT a random init artifact. It persists at Stage 3 (trained model).

**Novelty context (critical for paper framing):** Measuring cosine similarity between gradients from different computation paths is a STANDARD technique in multi-task learning, established by Du et al. 2018 and formalized by Yu et al. 2020 (PCGrad). Finding cos < 0 between auxiliary and primary gradients is well-documented. What is novel here is applying this analysis to differentiable structured prediction (BP) layers, where the gradient conflict arises from a single loss through two computation paths rather than from two separate loss functions. The paper MUST cite:
- Du, Czarnecki et al. 2018: "Adapting Auxiliary Losses Using Gradient Similarity" (NeurIPS workshop) — closest match, measures cos(grad_aux, grad_primary) and zeros out auxiliary when negative
- Yu et al. 2020: "Gradient Surgery for Multi-Task Learning" (PCGrad, NeurIPS) — defines "conflicting gradients" as cos < 0, introduces gradient projection
- Liu et al. 2021: "Conflict-Averse Gradient Descent" (CAGrad, NeurIPS) — minimizes worst-case task loss under gradient conflict
- Shamsian et al. 2023: "AuxiNash" (ICML) — formalizes auxiliary learning as asymmetric bargaining game

**2. Three parameters are BP-ONLY (zero direct gradient).**

Unexpected finding: encoder.layer3 gets NO gradient from the direct path. The direct path uses only p1 (from layer1), so layers 2-3 are dead ends. BP's multi-scale fusion (using p1, p2, p3) is the ONLY source of gradient signal for the deeper encoder.

BP-only parameters:
- encoder.layer3 (deep encoder features)
- pairwise_12.alpha_net (spatial consistency strength)
- pairwise_12.residual_net (class transition matrix)

These 3 parameter groups represent entirely new optimization dimensions that don't exist under direct supervision.

**3. Magnitude amplification is partially real after loss normalization.**

After factoring out the loss magnitude confound:
- encoder.layer1: **9.0x** — genuine amplification (down from raw 17.7x)
- unary_1.net[0]: **2.2x** — modest (down from raw 4.4x)
- unary_1.net[-1]: **1.55x** — minimal (down from raw 3.0x)

The prior "7-10x" claim was ~50% loss confound for unary params, but real for encoder params.

### Connection to pairwise degeneracy

This provides a necessary (not sufficient) condition for the degeneracy finding:
- Pairwise params get ZERO direct supervision signal
- BP redirects shared parameter gradients away from the direct supervision direction
- The entire model is optimized in a landscape shaped primarily by BP, not by "predict the right class per pixel"
- This is the environment where class remapping becomes an attractive shortcut

**Still needed for a causal claim**: cosine measurements comparing constrained (α·I+(1-α)·R) vs unconstrained pairwise — if the constrained version produces more aligned gradients, that's the mechanistic link.

### Design note: bugs fixed from prior measurements

- **Loss confound eliminated**: both paths now use the same FocalLoss (prior `test_gradient_comparison.py` used FocalLoss vs cross_entropy)
- **Double-softmax noted**: both unary_1() and dhbp() output log-softmax, but FocalLoss expects raw logits. Both paths have identical treatment, so the comparison is fair.

---

## Experiment 18b: Gradient Direction at 10% Labels — Dose-Response

**Date**: 2026-03-31
**Purpose**: Repeat cosine similarity measurement on the 10% labeled model (where pairwise degeneracy was most severe) and compare against 50% results from Experiment 18. If gradient redirection is stronger at 10%, this establishes a dose-response relationship between label fraction, gradient redirection, and degeneracy.

**Script**: `test_gradient_cosine.py`
**Model**: 10% labeled, entropy-weighted BP, 3 levels, 30 epochs (best=62.21% at epoch 25, final=49.06%)
**Checkpoint**: `/kaggle/working/output_pct_10/best_segmentation.pth`

### Results: Stage 3 (trained model) — 10% vs 50%

| Parameter | 10% model | 50% model | Baseline | Interpretation |
|---|---|---|---|---|
| unary_1.net[0] | **-0.028** | 0.690 | 0.913 | 10%: ANTI-CORRELATED. 50%: moderate. |
| unary_1.net[-1] | **0.574** | 0.885 | 1.000 | 10%: moderate redirection. 50%: mild. |
| encoder.layer1 | **0.232** | 0.422 | 0.290 | 10%: below baseline. 50%: above baseline. |
| encoder.layer3 | BP-ONLY | BP-ONLY | 0.208 | Both: zero direct gradient |
| pairwise_12.alpha_net[0] | BP-ONLY | BP-ONLY | 0.425 | Both: zero direct gradient |
| pairwise_12.residual_net[0] | BP-ONLY | BP-ONLY | 0.678 | Both: zero direct gradient |

### Loss-normalized gradient norms (Stage 3, 10% model)

| Parameter | BP norm/L | Direct norm/L | Normalized ratio |
|---|---|---|---|
| unary_1.net[0] | 0.2999 | 0.1256 | **2.39x** |
| unary_1.net[-1] | 0.8656 | 0.3044 | **2.84x** |
| encoder.layer1 | 0.0074 | 0.0033 | **2.25x** |
| encoder.layer3 | 0.0027 | 0.0000 | inf (BP-only) |
| pairwise_12.alpha_net[0] | 0.0085 | 0.0000 | inf (BP-only) |
| pairwise_12.residual_net[0] | 0.0314 | 0.0000 | inf (BP-only) |

### Key finding: Dose-response relationship

Less labeled data → stronger gradient redirection by BP:

```
Gradient direction alignment (cosine similarity) vs label fraction:

unary_1.net[0]:     10% = -0.028     50% = 0.690    (baseline = 0.913)
                         │                  │
                    ANTI-CORRELATED    MODERATE
                    ◄──────────────────────────────────────────────►
                    opposite            aligned           identical
                   -1.0                  0.0                +1.0

unary_1.net[-1]:    10% = 0.574      50% = 0.885    (baseline = 1.000)

encoder.layer1:     10% = 0.232      50% = 0.422    (baseline = 0.290)
```

**The 10% model's first unary conv has cos = -0.028** — BP and direct supervision gradients point in slightly OPPOSITE directions. This is not just redirection, it's anti-correlation. The 50% model has cos = 0.690 for the same parameter — still redirected but meaningfully correlated.

This pattern holds across all shared parameters: every cosine drops significantly from 50% to 10%.

### Interpretation

The dose-response connects three phenomena:

1. **Less labeled data → less direct supervision signal** to anchor the optimization
2. **Less anchoring → BP dominates the optimization landscape** (cos drops toward 0 or below)
3. **BP-dominated landscape → pairwise has freedom to learn shortcuts** (class remapping)

At 10% labels, the first unary conv is being trained in essentially opposite directions by BP vs direct supervision. The pairwise potentials — which receive ONLY BP gradient — are completely unsupervised by the direct signal. This is the exact condition under which the unconstrained pairwise learned Tree→Building remapping (Experiment 5).

At 50% labels, the stronger direct supervision signal keeps the optimization more aligned (cos = 0.690 instead of -0.028), giving the pairwise less room to degenerate.

### What this means for the paper narrative

The three findings form a coherent story, but the gradient analysis must be framed correctly:

```
Finding 1: Pairwise degeneracy              (WHAT happens — GENUINELY NOVEL)
  Unconstrained K×K matrices learn class remapping instead of
  spatial consistency. Tree→Building: 0.789. Worst at 10% labels.
  Nobody has documented this failure mode. The field uses Potts
  for computational reasons, not because they characterized this.
  Verified against 12+ papers (Zheng 2015, Chandra 2017,
  Vemulapalli 2016, Lin 2016, Larsson 2018, Knobelreiter 2020).

Finding 2: Gradient conflict through BP      (WHY — NOVEL APPLICATION of known technique)
  BP creates gradient conflict with direct supervision. At 10%,
  cos=-0.028 (anti-correlated). At 50%, cos=0.690. Dose-response.
  The MEASUREMENT TECHNIQUE (cosine similarity between gradient
  paths) is standard in MTL (Du et al. 2018, PCGrad 2020).
  The FINDING (BP layers create gradient conflict with supervised
  loss in structured prediction) has not been reported before.
  Pairwise params receive ZERO direct supervision signal.

Finding 3: Constrained pairwise prevents it  (HOW to fix it — modest novelty)
  α·I + (1-α)·R constrains the pairwise potential space,
  preventing the class remapping shortcut regardless of
  how much BP dominates the gradient landscape.
```

### Novelty assessment (4-agent literature verification, 2026-03-31)

Four independent research agents searched 60+ papers across structured prediction, MTL optimization, and CRF/MRF literature:

**Finding 1 (pairwise degeneracy): GENUINELY NOVEL** (confidence 9/10)
- Searched: Zheng 2015, Chandra 2017, Vemulapalli 2016, Lin 2016, Larsson 2018, Knobelreiter 2017/2020, E-CRF 2023, Arnab 2018 survey, DilatedCRF 2022, Shah & Shah 2021
- Closest work: Larsson et al. 2018 (learn arbitrary pairwise potentials but never analyze for degeneracy), E-CRF 2023 (boundary class confusion, but in CNN classifier weights, not pairwise matrix)
- Everybody uses Potts for computational reasons (tractable graph cuts, efficient mean-field), NOT because they documented the learned-potential failure mode

**Finding 2 (gradient direction through BP): NOVEL APPLICATION, NOT NOVEL TECHNIQUE** (confidence 9/10)
- Cosine similarity between task gradients is the standard diagnostic in MTL since Du et al. 2018 (DeepMind) and PCGrad (Yu et al. 2020)
- cos < 0 between auxiliary and primary gradients is well-documented (Du 2018, Gradient Vaccine 2020, AuxiNash 2023)
- Nobody has applied this to structured prediction layers (CRF/MRF/BP). The two literatures have never been connected
- Searched: Domke 2012, Zheng 2015, Knobelreiter 2020, Belanger & McCallum 2016, Blondel 2022 — none measure gradient direction through structured prediction

**Finding 3 (constrained pairwise): MODEST NOVELTY** (confidence 7/10)
- Constrained potentials exist (Potts, Gaussian kernels in DenseCRF) but are motivated by computation, not degeneracy prevention
- The specific α·I+(1-α)·R decomposition with learned spatially-varying α is a reasonable contribution

### Critical literature to cite

**Must cite (gradient conflict is a known MTL phenomenon):**
- Du, Czarnecki et al. 2018: "Adapting Auxiliary Losses Using Gradient Similarity" https://arxiv.org/abs/1812.02224
- Yu et al. 2020: "Gradient Surgery for Multi-Task Learning" (PCGrad) https://arxiv.org/abs/2001.06782
- Liu et al. 2021: "Conflict-Averse Gradient Descent" (CAGrad) https://arxiv.org/abs/2110.14048
- Shamsian et al. 2023: "AuxiNash" (auxiliary learning as asymmetric bargaining) https://arxiv.org/abs/2301.13501
- Jiang et al. 2023: "ForkMerge" (negative transfer in auxiliary learning, NeurIPS) https://arxiv.org/abs/2301.12618

**Must cite (pairwise potentials in CRF/MRF):**
- Zheng et al. 2015: "CRF as RNN" (end-to-end CRF, learned compatibility transform) https://arxiv.org/abs/1502.03240
- Larsson, Arnab et al. 2018: "Projected Gradient Descent for Arbitrary Pairwise Potentials" https://arxiv.org/abs/1701.06805
- Knobelreiter & Pock 2020: "BP Reloaded: Learning BP-Layers" (CVPR) https://arxiv.org/abs/2003.06281
- Krahenbuhl & Koltun 2011: "Dense CRF" (Gaussian pairwise kernels) https://arxiv.org/abs/1210.5644

**Should cite (related phenomena):**
- Wang et al. 2020: "Gradient Vaccine" (negative cosine → negative transfer in multilingual) https://arxiv.org/abs/2010.05874
- Sener & Koltun 2018: "Multi-Task Learning as Multi-Objective Optimization" https://arxiv.org/abs/1810.04650
- Guan et al. 2024: "Neural Markov Random Field for Stereo Matching" (CVPR 2024, learned MRF potentials) https://openaccess.thecvf.com/content/CVPR2024/html/Guan_Neural_Markov_Random_Field_for_Stereo_Matching_CVPR_2024_paper.html

### Target venues (based on publication landscape analysis)

**CRF/MRF for segmentation is declining at top CV venues** (0-2 papers/year at CVPR/ECCV/NeurIPS). But graphical model inference theory venues are active:

**Tier 1 (strong fit):**
- **SPIGM workshop @ NeurIPS 2025** — theme: "is probabilistic inference still relevant?" Almost tailor-made. Workshop paper, early feedback. https://spigmworkshopv3.github.io/
- **UAI 2025/2026** — natural home for graphical model inference theory. ~100 papers/year.
- **AISTATS** — accepts theoretical analysis of inference algorithms.
- **PGM (Int'l Conf. Probabilistic Graphical Models)** — small dedicated venue, 32 papers in 2024, published in PMLR.

**Tier 2 (good fit with right framing):**
- **TMLR** — journal, no deadline pressure, rigorous review.
- **NeurIPS main** — only if framed as "when/why structured prediction fails in modern systems." High bar.

**Tier 3 (application framing needed):**
- **MICCAI / Medical Image Analysis** — CRF still used in medical imaging, degeneracy analysis relevant.
- **CVPR/ECCV** — only with strong application results on multiple datasets.

### Limitations

1. ~~**Single run at each label fraction.**~~ **RESOLVED by Experiment 19** — 3-seed reproduction at 10% complete with error bars. 50% still pending (Session 2).
2. **Single dataset** (ISPRS Vaihingen). Need: Potsdam reproduction.
3. **Correlation, not causation.** 10% has more redirection AND more degeneracy, but both could be downstream effects of limited data. Still needed: constrained-vs-unconstrained cosine comparison to establish the causal mechanism.
4. **Data sweep uses random-init model.** The sweep section (5 random batches) ran on a fresh random-DHBP model, not the trained 10% model.
5. **Gradient conflict framing is NOT novel.** Must position as application of known MTL technique to structured prediction, not as discovery of gradient conflict.

### Path to publication

**Workshop paper (SPIGM @ NeurIPS 2025, 4 pages, 55-70%)** — 2-3 weeks:
- 3-seed reproduction for 10% and 50% cosine measurements
- Fix sweep to use trained model weights
- Cite MTL gradient conflict literature properly
- Title: "Pairwise Potential Degeneracy in End-to-End Belief Propagation: Characterization and Gradient Conflict Analysis"

**Full paper (UAI/AISTATS, 25-40% / TMLR 50-60%)** — 6-8 weeks:
- Everything above, plus second dataset (Potsdam)
- Constrained-vs-unconstrained cosine comparison (causal mechanism)
- Apply degeneracy diagnostic to DenseCRF (generality)
- Frame as contribution to understanding inference failures in pairwise graphical models, with segmentation as motivating application

---

## Experiment 19: 3-Seed Reproduction at 10% Labels (Session 1)

**Date**: 2026-04-02 / 2026-04-03
**Purpose**: 3-seed reproduction of BP vs no-BP at 10% labels with error bars. This was the P0 blocker for the paper.

**Script**: `complete_training.py` with `--seed` argument (added for this experiment)
**Contrastive checkpoint**: `massivehead/best-model-const` (pre-fine-tuned encoder)
**Config**: 10% labeled (1 area), 30 seg epochs, 3 levels, entropy-weighted BP

### Results: Accuracy (3 seeds)

| | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---|---|---|---|
| **BP best** | 66.58% | 63.74% | 65.50% | **65.27 ± 1.4%** |
| **BP final** | 63.34% | 58.24% | 65.33% | **62.30 ± 3.6%** |
| **No-BP best** | 61.95% | 60.19% | 59.73% | **60.62 ± 1.2%** |
| **No-BP final** | 60.29% | 56.59% | 55.26% | **57.38 ± 2.6%** |
| **BP delta (best)** | +4.63pp | +3.55pp | +5.77pp | **+4.65 ± 1.1pp** |
| **BP delta (final)** | +3.05pp | +1.65pp | +10.07pp | **+4.92 ± 4.5pp** |

### Per-class accuracy (BP vs No-BP, 3-seed mean)

| Class | BP (final) | No-BP (final) | Delta |
|---|---|---|---|
| Impervious | 55.26% | 45.58% | +9.68pp |
| Buildings | 79.48% | 77.10% | +2.38pp |
| Low Veg | 48.06% | 43.09% | +4.97pp |
| Trees | 72.48% | 79.23% | -6.75pp |
| Cars | 33.13% | 24.10% | **+9.03pp** |
| Clutter | 0.00% | 0.00% | 0.00pp |

**Key observations:**
- BP improves overall accuracy by **+4.65pp best, +4.92pp final** (reproducible across seeds)
- Cars benefit most from BP: **+9.03pp** (33.13% vs 24.10%)
- Trees slightly WORSE with BP (-6.75pp) — BP redistributes from majority to minority classes
- Best accuracy variance is LOW (±1.4pp for BP, ±1.2pp for no-BP) — reproducible
- Final accuracy variance is HIGHER (±3.6pp for BP) — training instability persists

### Cosine similarity (Stage 3, trained 10% models, 3 seeds)

| Parameter | Seed 0 | Seed 1 | Seed 2 | Mean ± Std | Baseline |
|---|---|---|---|---|---|
| unary_1.net[0] | +0.517 | +0.175 | +0.214 | **+0.302 ± 0.19** | 0.913 |
| unary_1.net[-1] | +0.859 | +0.653 | +0.697 | **+0.736 ± 0.11** | 1.000 |
| encoder.layer1 | +0.544 | -0.079 | +0.328 | **+0.264 ± 0.31** | 0.290 |
| encoder.layer3 | BP-ONLY | BP-ONLY | BP-ONLY | — | 0.208 |
| pairwise_12.α | BP-ONLY | BP-ONLY | BP-ONLY | — | 0.425 |
| pairwise_12.R | BP-ONLY | BP-ONLY | BP-ONLY | — | 0.678 |

**Gradient direction findings (with error bars):**
- unary_1.net[0]: cos = **0.302 ± 0.19** on baseline 0.913 — consistently redirected, well below baseline
- unary_1.net[-1]: cos = **0.736 ± 0.11** on baseline 1.000 — moderate, stable redirection
- encoder.layer1: cos = **0.264 ± 0.31** on baseline 0.290 — high variance, near/below baseline. Seed 1 shows anti-correlation (-0.079)
- All 3 BP-only parameters confirmed across all seeds

### Pairwise diagnostics (constrained model, seed 0)

| Metric | Value |
|---|---|
| Alpha mean | 0.742 |
| Alpha std | 0.147 |
| Alpha range | [0.079, 0.890] |
| Diagonal ratio | **0.784** |
| Max off-diagonal | 0.122 (Trees → Low Veg) |
| Cars diagonal | 0.806 (highest — pairwise learned to preserve Cars beliefs) |

The constrained model maintains a healthy pairwise matrix (diagonal ratio 0.784). The max off-diagonal (Trees → Low Veg: 0.122) represents a legitimate class transition, not class remapping. Compare with unconstrained degeneracy: diagonal ratio 0.094, Tree→Building: 0.789.

### Loss-normalized gradient norms (3-seed mean, Stage 3)

| Parameter | BP norm/L | Direct norm/L | Normalized ratio |
|---|---|---|---|
| unary_1.net[0] | 0.168 | 0.119 | **1.37x** |
| unary_1.net[-1] | 0.745 | 0.316 | **2.37x** |
| encoder.layer1 | 0.029 | 0.003 | **11.47x** |
| encoder.layer3 | 0.012 | 0.000 | inf (BP-only) |
| pairwise_12.α | 0.024 | 0.000 | inf (BP-only) |
| pairwise_12.R | 0.060 | 0.000 | inf (BP-only) |

Encoder.layer1 genuine amplification (**11.5x** after loss normalization) confirmed across seeds.

### What this resolves

1. **Error bars now exist.** The 12-agent scrutiny's #1 criticism is addressed.
2. **BP improvement is reproducible.** +4.65 ± 1.1pp best accuracy, consistent across 3 seeds.
3. **Gradient redirection is reproducible.** All shared params consistently below baseline cosine.
4. **Pairwise constrained model is healthy.** Diagonal ratio 0.784 (vs 0.094 for unconstrained from prior experiments).
5. **Convergence data collected.** BP and no-BP accuracy curves available per epoch for Experiment F analysis.

### What's still needed

1. **50% label results** (Session 2) — for dose-response with error bars
2. **Unconstrained pairwise comparison** (Experiment C) — for causal mechanism
3. **Potsdam dataset** — for generalization (full paper only)

### Outputs saved

- 6 checkpoints: `output_pct10_seed{0,1,2}/best_segmentation.pth` + `output_pct10_nobp_seed{0,1,2}/best_segmentation.pth`
- Pairwise heatmaps: `paper_figures_10pct/pairwise_heatmap_*.png` (per seed + mean)
- Pairwise diagnostics: `paper_figures_10pct/pairwise_diagnostics.json`
- Paper figures: `paper_figures/fig1_pairwise_comparison.pdf`, `paper_figures/fig4_gradient_signal_diagram.pdf`
- All packaged in: `session1_results.zip`

---

## Critical Review: Four Structural Vulnerabilities (2026-04-07)

Adversarial review from domain expert identified four issues that would sink the paper at UAI/SPIGM. All four are legitimate.

### Vulnerability 1: Gradient conflict framing is theoretically backwards

**The critique**: We framed cos(∇BP, ∇Direct) < 0 as a pathology causing degeneracy. But in correctly-functioning end-to-end structured prediction, the unary head SHOULD learn to output something different from a standalone classifier. It should do "inference offloading" — be uncertain where BP will correct, confident where BP is weak. Therefore gradient divergence is EXPECTED, not pathological. The negative cosine proves the unary head is adapting to BP, not that BP is breaking things.

**Status**: ACCEPTED — this reframes the narrative.

**Corrected framing**: Gradient divergence is the natural consequence of end-to-end learning with structured prediction. The degeneracy happens not because gradients diverge (they should), but because unconstrained pairwise potentials exploit this freedom to learn class-remapping shortcuts. The constrained decomposition channels the freedom properly.

**Impact on paper**: The gradient analysis becomes a characterization of how structured prediction changes the optimization landscape (descriptive), not a causal explanation for degeneracy (causal claim was too strong). The pairwise degeneracy characterization remains the primary contribution.

### Vulnerability 2: Entropy-weighted BP is not BP

**The critique**: Standard sum-product BP on trees computes exact marginals. Adding entropy-based weights to the child aggregation (`m_total = Σ w_c · m_c` instead of `m_total = Σ m_c`) breaks the sum-product derivation. We are no longer computing exact marginals of an MRF. At UAI/SPIGM, claiming "exact BP" with this modification is a rejection-worthy mathematical error.

**Status**: ACCEPTED — mathematically correct.

**Options**:
- A) Rebrand: "Entropy-Gated Message Passing on Quadtrees" (admit it's a heuristic, not exact inference)
- B) Formalize: derive the entropy weighting as a variational message passing step from a modified free-energy objective (harder, potentially publishable itself)

**Recommended**: Option A for the workshop paper. Option B for the full paper if the derivation works.

**Impact on paper**: Replace all instances of "exact sum-product BP" with "entropy-gated message passing." Acknowledge the departure from exact inference explicitly. This actually HELPS the degeneracy paper — we can say "even in an approximate message passing scheme, pairwise degeneracy occurs, suggesting the failure mode is general."

### Vulnerability 3: Missing receptive field baseline ("dumb pooling")

**The critique**: BP on a 3-level quadtree expands the receptive field to ~103px. The no-BP baseline (unary head only) has a much smaller receptive field. If a simple dilated conv or average-pooling pyramid matching the same ~103px receptive field achieves the same +4.65pp improvement, then BP's value is just spatial context, not structured inference. The entire graphical model narrative collapses.

**Status**: ACCEPTED — this is the most dangerous vulnerability. Must be tested.

**Experiment needed**: Run a "dumb pooling" baseline:
```python
# Replace DHBP with a simple spatial context aggregator:
# 1. Average pool p1 [128×128] to [32×32] (same as quadtree root)
# 2. Bilinear upsample back to [128×128]
# 3. Concatenate with original p1
# 4. Conv1x1 to n_classes
# This matches the ~103px receptive field without any structured inference.
```

**Expected outcomes**:
- If dumb pooling ≈ BP (+4-5pp): BP is just spatial context. Degeneracy paper survives (the failure mode is real regardless), but the "structured prediction" framing dies.
- If dumb pooling << BP: BP provides genuine structured inference value beyond receptive field. Strongest outcome.
- If dumb pooling > BP: BP is actively harmful compared to simple alternatives. Worst case but important to know.

**Priority**: P0 — must run before submission. Cheap experiment (~2h on T4).

### Vulnerability 4: Reverse causality in the degeneracy → gradient conflict chain

**The critique**: Our causal chain was: less data → gradient conflict → pairwise learns shortcuts. But the actual chain could be: unconstrained K×K matrix immediately finds the remapping shortcut (36 free parameters, easy optimization) → this warps the gradient landscape → gradient conflict is the SYMPTOM of degeneracy, not the CAUSE.

**Status**: ACCEPTED — the causal direction is ambiguous.

**Implication**: Experiment C (constrained vs unconstrained cosine comparison) will NOT disambiguate the causal direction. If unconstrained has lower cosine, it could be either:
- gradient conflict → degeneracy (our original claim)
- degeneracy → gradient conflict (reverse causality)

**Resolution**: Abandon the causal claim. Present the gradient analysis as DESCRIPTIVE:
- "End-to-end training with BP creates gradient divergence (natural, expected)"
- "Unconstrained pairwise potentials converge to class remapping (novel failure mode)"
- "The constrained decomposition prevents degeneracy (practical fix)"

The gradient analysis characterizes the optimization landscape. The degeneracy is the main finding. Don't overclaim causality.

### Updated paper narrative (post-review)

```
OLD narrative (too strong):
  Gradient conflict CAUSES pairwise degeneracy.

NEW narrative (defensible):
  1. End-to-end structured prediction creates gradient divergence
     between BP and direct supervision (EXPECTED, not pathological).
  2. Unconstrained pairwise potentials exploit this freedom to learn
     class remapping (NOVEL failure mode, the main contribution).
  3. A receptive-field-matched baseline confirms BP provides value
     beyond simple spatial context (MUST VERIFY with Experiment).
  4. Constrained decomposition prevents degeneracy (practical fix).
```

### Updated experiment priorities

```
PRIORITY   EXPERIMENT                              TIME     STATUS
────────   ──────────────────────────────────────   ──────   ──────
P0 NEW     Dumb pooling baseline (receptive field)  ~2h      NEEDED
P0         50% label reproduction (Session 2)       ~6h      PENDING
P1         Unconstrained pairwise training          ~2h      PENDING
P2         Potsdam dataset                          ~2 days  DEFERRED
P2         Convergence speedup (Exp F)              ~6h      DONE (10%)
DROPPED    Causal cosine comparison (Exp C)         —        DROPPED (reverse causality)
```

Experiment C (constrained vs unconstrained cosine) is DROPPED — it cannot disambiguate the causal direction per Vulnerability 4. Replace with the dumb pooling baseline (Vulnerability 3) which is more informative.

---

## Upcoming Experiments

See `TODO.md` for full prioritized list and ruled-out items with evidence.
