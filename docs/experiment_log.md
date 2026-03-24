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

## Upcoming Experiments

See `TODO.md` for full list. Priority order:

1. **Constrained pairwise training** (current) — 10 epochs at a time, 10% labels
2. **Auxiliary unary loss** — direct supervision of unary heads
3. **Dense contrastive learning** — pixel-level SimCLR for better Cars/small object features
4. **More contrastive epochs** — current encoder is MODERATE, room to improve
5. **Ablation study** — BP on vs off, constrained vs unconstrained pairwise, contrastive vs random encoder
