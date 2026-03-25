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

## Ruled out hypotheses (with evidence)

### "Quadtree BP improves segmentation via structured inference" — MARGINAL
- Ablation: +1.8% best, +0.4% at convergence — not significant
- BP diagnosis: acts as spatial blur, not boundary refinement
- Quadtree lacks horizontal connections needed for spatial consistency
- **Status**: The quadtree BP concept works mechanically but the graph structure is wrong for this task

### "Unary head gets weak gradients through BP" — WRONG (opposite is true)
- **Gradient comparison test**: BP chain gradients are 7-10x STRONGER than direct path
- **Verified across**: 3 training stages + 5 random seeds (9.0x ± 2.3x average)
- **Implication**: Auxiliary unary loss was proposed to fix a problem that doesn't exist. The BP computation graph actually AMPLIFIES gradients, it doesn't dilute them.
- Unary collapse is caused by limited data (1 area) + BP compensation, not gradient dilution

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

## Upcoming Experiments

See `TODO.md` for full prioritized list and ruled-out items with evidence.
