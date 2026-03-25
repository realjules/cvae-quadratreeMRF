# Research Ideas — NeurIPS-Level Directions

Generated 2026-03-25 from adversarial literature search across 60+ papers and 19 query angles.

**Context**: Our DHBP (quadtree BP) experiments showed BP adds only +1.8% over no-BP baseline at convergence. The architecture works mechanically but the contribution is too small for a top venue. We also found that the "contrastive features encode MRF structure" intuition is already common knowledge (Google AI can articulate it). These ideas pivot to genuinely open problems identified in recent literature.

---

## Idea 1: Semi-Supervised Segmentation is Miscalibrated — A Reliability Theory

**The gap:** Semi-supervised methods optimize accuracy but produce badly calibrated confidence estimates. SOTA methods are *less calibrated* than supervised baselines — they trade reliability for accuracy. No theory explains when or why this happens.

**The idea:** Develop theory for when and why semi-supervised segmentation becomes miscalibrated. Prove that pseudo-labeling with high-confidence thresholds systematically creates overconfident predictions in a self-reinforcing loop. Derive calibration bounds as a function of labeled data fraction.

**Why NeurIPS:** Theory paper with safety implications. Calibration matters for deployment (medical, autonomous driving). Nobody has formalized this.

**Novelty:** High
**Feasibility:** Medium
**NeurIPS fit:** Strong

**Key papers:**
- Landgraf et al. 2025, "Rethinking Semi-supervised Segmentation Beyond Accuracy: Reliability and Robustness"
  https://arxiv.org/abs/2506.05917
- Liu & Liu 2026, "A Confidence-Variance Theory for Pseudo-Label Selection in Semi-Supervised Learning"
  https://arxiv.org/abs/2601.11670
- Arazo et al. 2020, "Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning"
  https://arxiv.org/abs/1908.02983

---

## Idea 2: Sample Complexity for Dense Prediction — How Many Pixel Labels Do You Actually Need?

**The gap:** PAC-learning theory for semi-supervised classification exists (NeurIPS 2022), but has NEVER been extended to structured output spaces like segmentation. We have no formal answer to "how many labels are necessary and sufficient for segmentation at quality X?" The TPAMI survey (Shen et al. 2022) explicitly identifies this as the central unsolved question.

**The idea:** Derive PAC-style sample complexity bounds for semi-supervised pixel-level prediction, accounting for spatial label correlations (neighboring pixels are likely the same class). Show that spatial structure reduces the effective number of independent labels needed — formalizing the intuition that you don't need to label every pixel.

**Why NeurIPS:** Foundational theory. Extends a NeurIPS 2022 paper to structured outputs.

**Novelty:** Very high
**Feasibility:** Hard (pure theory)
**NeurIPS fit:** Strong

**Key papers:**
- Attias et al. 2022, "A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability" (NeurIPS 2022)
  https://arxiv.org/abs/2202.05420
- Shen et al. 2022, "A Survey on Label-efficient Deep Image Segmentation" (IEEE TPAMI)
  https://arxiv.org/abs/2207.01223
- Chandrasekaran et al. 2025, "Learning Neural Networks with Distribution Shift" (ICLR 2025)
  https://arxiv.org/abs/2502.16021

---

## Idea 3: The Optimization Landscape of Structured Prediction Layers

**The gap:** All loss landscape theory (NeurIPS 2018, NeurIPS 2024) applies to standard losses (CE, MSE). How CRF/MRF/energy-based output layers change the optimization landscape is COMPLETELY unstudied. Nobody knows if adding a CRF layer makes the landscape smoother, creates more saddle points, or changes the connectivity of minima.

**The idea:** Characterize how structured prediction layers modify the loss landscape. Use the tools from Li et al. (NeurIPS 2018) to visualize landscapes with and without CRF layers. Prove conditions under which structured prediction layers improve the Polyak-Lojasiewicz condition that governs convergence.

**Why NeurIPS:** Directly extends two NeurIPS papers (2018, 2024) to a new setting. Our gradient amplification finding (10-32x) is an empirical observation that this theory would explain.

**Connection to our work:** We measured that BP chain gradients are 10-32x stronger than direct gradients. This is an empirical data point about how structured prediction changes the optimization landscape.

**Novelty:** Very high
**Feasibility:** Medium (extends existing tools)
**NeurIPS fit:** Strong

**Key papers:**
- Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018)
  https://arxiv.org/abs/1712.09913
- Islamov et al. 2024, "Loss Landscape Characterization without Over-Parametrization" (NeurIPS 2024)
  https://arxiv.org/abs/2410.12455
- Kim et al. 2024, "Exploring the Loss Landscape via Convex Duality"
  https://arxiv.org/abs/2411.07729

**Potential threats (prior work that partially addresses this):**
- Zheng et al. 2015 (CRF-as-RNN) noted vanishing/exploding gradients through CRF iterations
- Domke 2012 derived backprop through iterative inference
- Knobelreiter et al. 2020 "Belief Propagation Reloaded" (CVPR) computes backprop through BP layers
- Larsson et al. 2018 studied gradient flow through CRF modules
- None of these formally characterize the LANDSCAPE — they analyze gradients at specific points, not the global geometry

---

## Idea 4: Neural Collapse Meets Dense Prediction

**The gap:** Neural collapse (features converging to simplex ETF geometry) is well-studied for classification. Whether pixel-level features in segmentation encoders exhibit the same collapse is UNKNOWN. Minority collapse under class imbalance (common in segmentation — Cars = 1.5% of pixels) has been shown for classification but not dense prediction.

**The idea:** Study whether pixel embeddings from segmentation encoders undergo neural collapse, and how spatial correlations between pixels affect the collapse dynamics. Show that class imbalance in segmentation causes minority collapse where rare classes (Cars, Clutter) become undetectable — and propose a fix based on ETF regularization of pixel features.

**Why NeurIPS:** Extends a hot topic (neural collapse) to a new setting (dense prediction with spatial structure).

**Connection to our work:** Our encoder linear probe showed Cars at 0.01% and Clutter at 0.00%. This is exactly what minority collapse predicts — rare classes collapse into a single indistinguishable representation.

**Novelty:** High
**Feasibility:** Medium (experiments + analysis)
**NeurIPS fit:** Strong

**Key papers:**
- Li et al. 2024, "Preventing Collapse in Contrastive Learning with Orthonormal Prototypes" (ICLR 2025)
  https://arxiv.org/abs/2403.18699
- "Neural and Minority Collapse in Contrastive Learning with Imbalanced Datasets" (2025)
  https://sites.bu.edu/pi/files/2025/09/Neural-Minority-Collapse-preprint-Sep-2025.pdf
- Liu 2024, "Leveraging Intermediate Neural Collapse with Simplex ETFs"
  https://arxiv.org/abs/2412.00884

---

## Idea 5: Spatially-Aware Active Learning — Which Pixels Are Worth Labeling?

**The gap:** Active learning for segmentation selects pixels independently, ignoring spatial structure. No theory exists for spatially-aware acquisition functions that account for the fact that labeling a boundary pixel is more informative than labeling an interior pixel.

**The idea:** Derive an acquisition function that accounts for spatial label correlations (MRF prior over labels). Prove that boundary pixels have higher expected information gain than interior pixels. Show that with optimal spatial-aware selection, you need 5-10x fewer labels than random selection.

**Why NeurIPS:** Combines active learning theory with structured prediction. NeurIPS 2025 paper on diffusion-driven active learning identifies the lack of theory as the key gap.

**Novelty:** Medium-high
**Feasibility:** Medium
**NeurIPS fit:** Medium-strong

**Key papers:**
- Kim et al. 2025, "Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation" (NeurIPS 2025)
  https://arxiv.org/abs/2510.22229
- Didari et al. 2024, "Bayesian Active Learning for Semantic Segmentation"
  https://arxiv.org/abs/2408.01694
- ESA 2024, "Annotation-Efficient Active Learning for Semantic Segmentation"
  https://arxiv.org/abs/2408.13491

---

## Idea 6: Confirmation Bias Has Spatial Structure — Why Pseudo-Labels Fail at Boundaries

**The gap:** Confirmation bias in pseudo-labeling is well-documented for classification. But in segmentation, confirmation bias has SPATIAL STRUCTURE — errors concentrate at class boundaries and propagate outward. Nobody has formalized this spatial confirmation bias or shown how it compounds across training iterations.

**The idea:** Model pseudo-label error propagation as a spatial process (diffusion on a graph). Prove that boundary errors expand at a rate proportional to the local class uncertainty. Derive the number of training iterations before boundary errors consume the interior of small objects (explaining why Cars accuracy degrades).

**Why NeurIPS:** Novel formalization of a known problem (confirmation bias) in a new setting (spatial prediction). Connects to our empirical observation that small classes are systematically destroyed.

**Connection to our work:** We observed that Cars went from 60.4% (unary) to 32.2% (after BP) — the structured prediction propagated errors from boundaries inward, destroying small objects. This is spatial confirmation bias in action.

**Novelty:** High
**Feasibility:** Medium (theory + experiments)
**NeurIPS fit:** Strong

**Key papers:**
- Liu & Liu 2025, "When Confidence Fails: Revisiting Pseudo-Label Selection" (ICCV 2025)
  https://arxiv.org/abs/2509.16704
- Liu & Liu 2026, "A Confidence-Variance Theory for Pseudo-Label Selection"
  https://arxiv.org/abs/2601.11670
- Arazo et al. 2020, "Pseudo-Labeling and Confirmation Bias"
  https://arxiv.org/abs/1908.02983

---

## Idea 7: Do Foundation Models Need Structured Prediction?

**The gap:** SAM and DINOv2 achieve impressive segmentation, but SAM "is not always perfect" — it lacks tight boundaries in specialized domains. MarkovGen (Google 2023) showed MRF layers still help even with modern models. Whether foundation models subsume structured prediction is empirically contested and theoretically uncharacterized.

**The idea:** Systematic study across 5+ foundation models (SAM, DINOv2, CLIP, MAE, EVA) measuring the marginal benefit of adding CRF/MRF post-processing. Derive a "spatial coherence score" from features and show it predicts CRF benefit.

**Why NeurIPS:** Timely question the community cares about. Answers "should I use a CRF with my foundation model?" with a principled framework.

**Novelty:** Medium
**Feasibility:** Easy (empirical study)
**NeurIPS fit:** Medium

**Key papers:**
- Ji et al. 2023, "Segment Anything Is Not Always Perfect" (CVPRW 2023)
  https://arxiv.org/abs/2304.05750
- Jayasumana et al. 2023, "MarkovGen: Structured Prediction for Efficient Text-to-Image Generation"
  https://arxiv.org/abs/2308.10997
- Brehmer et al. 2024, "Does Equivariance Matter at Scale?" (TMLR)
  https://arxiv.org/abs/2410.23179

---

## Idea 8: Attention IS Message Passing — But on the Wrong Graph

**The gap:** Transformers implement message passing on a fully-connected graph. Structured attention (ICLR 2017) showed CRF inference can be an attention layer. But for images, full attention ignores spatial locality. The formal equivalence between spatial attention patterns and grid-structured MRF inference is unexplored.

**The idea:** Prove that spatially-restricted attention (Swin Transformer local windows) is equivalent to message passing on a specific graph structure. Characterize what is lost vs gained by full attention (dense graph) vs local attention (sparse graph) vs CRF (grid graph). Show the optimal attention pattern for segmentation corresponds to an adaptively-constructed graph.

**Why NeurIPS:** Theoretical unification of two dominant paradigms (transformers + graphical models).

**Novelty:** High
**Feasibility:** Medium (theory)
**NeurIPS fit:** Strong

**Key papers:**
- Kim et al. 2017, "Structured Attention Networks" (ICLR 2017)
  https://arxiv.org/abs/1702.00887
- Piotrowski et al. 2025, "Constrained Belief Updates Explain Geometric Structures in Transformer Representations"
  https://arxiv.org/abs/2502.01954
- Joshi 2025, "Transformers are Graph Neural Networks"
  https://arxiv.org/abs/2506.22084

---

## Idea 9: Test-Time Structured Prediction — MRF Priors as Self-Supervised Signals

**The gap:** Test-time adaptation (TTA) for segmentation uses entropy minimization or self-training, but these ignore spatial structure. Using structured prediction constraints (spatial consistency, boundary sharpness) as the TTA objective is unexplored.

**The idea:** At test time, enforce MRF-like spatial consistency as a self-supervised signal — no labels needed, just the prior that "neighboring pixels should agree." Show this outperforms entropy minimization because spatial consistency is a stronger inductive bias.

**Why NeurIPS:** Practical contribution (better TTA) with a principled framework (MRF energy as TTA loss).

**Novelty:** Medium
**Feasibility:** Easy (builds on existing)
**NeurIPS fit:** Medium

**Key papers:**
- Zhang et al. 2025, "Progressive Test Time Energy Adaptation" (ICCV 2025)
  https://arxiv.org/abs/2503.16616
- Hubotter et al. 2025, "Specialization after Generalization: TTT in Foundation Models" (NeurIPS 2025 Oral)
  https://arxiv.org/abs/2509.24510
- TTA Benchmark for Medical Segmentation 2025
  https://arxiv.org/abs/2512.02497

---

## Idea 10: Energy-Based Segmentation — Connecting Diffusion Score Functions to MRF Potentials

**The gap:** Diffusion models for segmentation treat label maps as objects to denoise, but the learned score function's relationship to classical MRF energy functions is unknown. Energy Matching (NeurIPS 2025) unifies flow matching and EBMs but hasn't been applied to structured prediction.

**The idea:** Show formally that the score function learned by a label diffusion model decomposes into unary potentials (per-pixel class evidence) and pairwise potentials (spatial coherence) — recovering an implicit MRF. Derive conditions under which the diffusion process converges to the MAP estimate of this MRF. This bridges classical structured prediction with modern generative models.

**Why NeurIPS:** Theoretical unification of two hot areas (diffusion models + graphical models). If the score function IS an MRF, it explains why diffusion-based segmentation produces spatially coherent results.

**Novelty:** Very high
**Feasibility:** Hard (theory)
**NeurIPS fit:** Strong

**Key papers:**
- Balcerak et al. 2025, "Energy Matching: Unifying Flow Matching and Energy-Based Models" (NeurIPS 2025)
  https://arxiv.org/abs/2504.10612
- LDSeg 2024, "Denoising Diffusions in Latent Space for Medical Image Segmentation"
  https://arxiv.org/abs/2407.12952
- Schroeder et al. 2024, "Energy-Based Modelling on Structured Spaces" (NeurIPS 2024)
  https://arxiv.org/abs/2412.01019

---

## Summary Ranking

| # | Idea | Novelty | Feasibility | NeurIPS fit |
|---|---|---|---|---|
| 2 | Sample complexity for dense prediction | Very high | Hard | Strong |
| 3 | Optimization landscape of structured prediction | Very high | Medium | Strong |
| 6 | Spatial confirmation bias in pseudo-labels | High | Medium | Strong |
| 10 | Diffusion score ↔ MRF potential equivalence | Very high | Hard | Strong |
| 4 | Neural collapse in dense prediction | High | Medium | Strong |
| 8 | Attention = message passing on wrong graph | High | Medium | Strong |
| 1 | Semi-supervised calibration theory | High | Medium | Strong |
| 5 | Spatially-aware active learning | Medium-high | Medium | Medium-strong |
| 9 | Test-time MRF adaptation | Medium | Easy | Medium |
| 7 | Foundation models + structured prediction | Medium | Easy | Medium |

---

## What we built that's reusable

Regardless of which idea we pursue, the following tools from our DHBP work are reusable:
- **Diagnostic pipeline** (evaluate_encoder.py): linear probe, t-SNE, unary/pairwise analysis, gradient comparison
- **BP diagnosis tests** (test_bp_diagnosis.py): horizontal propagation, spatial prediction changes, alpha visualization
- **Contrastive encoder** (SimCLR + ResNet-18 on ISPRS)
- **ISPRS training pipeline** with data loading, evaluation, ablation flags
- **Experiment log** documenting what works and what doesn't

---

## Ruled-out directions (from our experiments)

These are documented in `docs/experiment_log.md` with full evidence:
- "Contrastive features encode MRF structure" — common knowledge, not novel
- "Quadtree BP improves segmentation" — +1.8%, not significant
- "Gradient amplification through BP" — partially preempted by Zheng 2015, Domke 2012, Knobelreiter 2020
- "Unconstrained pairwise potentials" — learn class remapping, not spatial consistency
