# TODO — Future Experiments

## Active experiments (running now)

### BP ablation study (3 experiments, 10 epochs each, 10% labels)
- **A**: Full DHBP (constrained pairwise, 2-layer unary + BP)
- **B**: No BP (2-layer unary only, no message passing)
- **C**: Simple unary + BP (Conv1x1 linear projection + BP)
- **Purpose**: Measure BP's actual contribution for the paper

---

## Upcoming experiments (prioritized)

### 1. Longer training with best configuration
- Run winner of ablation A/B/C for 100 epochs
- Loss was still decreasing at 10 epochs (0.84 → 0.55)

### 2. Confidence-gated BP (fix BP hurting strong classes)
- **Problem**: BP adds +5% overall but hurts Buildings (-5%) and Trees (-2.1%) — classes where unary is already confident and correct. BP messages from weaker neighbors drag down correct predictions.
- **Fix**: Feed unary confidence (entropy) into the pairwise alpha_net, so the model learns when to trust unary vs BP.
  - If unary is confident (low entropy) → α → 1 → BP preserves unary
  - If unary is uncertain (high entropy) → α → lower → BP corrects
- **Implementation**: Concatenate per-pixel unary entropy [B, 1, H, W] with encoder features before alpha_net. Changes alpha_net input channels from C to C+1.
- **Expected result**: BP stops hurting Buildings/Trees, keeps helping Impervious/Low Veg/Cars. Net improvement should increase from +5% to +7-8%.
- **Evidence that this matters**: Ablation showed Buildings 80%→75% and Trees 84%→82% after BP. These are avoidable losses.

### 3. Dense contrastive learning (PixelCL)
- Current SimCLR pulls patch-level features — Cars (1.5% of pixels) get lost
- Dense contrastive pulls pixel-level features at same location across augmented views
- Should improve Cars (currently 0.01% linear probe) and fine-grained features
- **Reference**: Wang et al. 2021, "Dense Contrastive Learning for Self-Supervised Visual Pre-Training"

### 3. Encoder upgrade
- Linear probe is 67.5% — MODERATE. This is the accuracy ceiling.
- Try: longer contrastive training (200 epochs), lower temperature (0.05), larger batch (8)
- Try: ResNet-34 or ResNet-50 (more capacity)

### 4. Class-balanced patch sampling
- During contrastive pre-training, oversample patches containing Cars/Clutter
- Doesn't need labels — can use image statistics to identify small bright objects

### 5. Multi-scale cropping for contrastive
- Add 128×128 and 64×64 crops alongside 256×256
- Forces encoder to learn fine-grained features (important for Cars)

### 6. ReduceLROnPlateau scheduler
- Current CosineAnnealingWarmRestarts causes accuracy oscillations
- ReduceLROnPlateau(mode='max', patience=5) would be more stable

### 7. Multiple BP iterations
- Single pass is exact on tree, but stacking 2-3 passes with different pairwise heads
  could act as iterative refinement (similar to CRF-as-RNN unrolling)
- Only explore if single pass hits a ceiling

---

## Ruled out (with evidence)

### Auxiliary unary loss
- **Hypothesis**: Unary head gets weak gradients through BP, needs direct supervision
- **Tested**: Gradient comparison experiment showed BP chain gradients are 10-32x STRONGER than direct path, not weaker
- **Finding**: Weak gradients was WRONG. Unary collapse is caused by limited data (1 training area) and BP compensation, not gradient dilution
- **Status**: RULED OUT as a fix. May still be useful for other reasons but not for the stated hypothesis

### Unconstrained K×K pairwise matrix
- **Tested**: Trained model showed diagonal ratio 0.094 (should be >0.5)
- **Finding**: The unconstrained matrix learned class remapping (Tree→Building: 0.789) instead of spatial consistency. BP destroyed 3 classes: Impervious (-14%), Trees (-19%), Cars (-28%)
- **Status**: REPLACED with constrained α·I + (1-α)·R decomposition

### Unary head architecture changes (Conv3x3, deeper)
- **Hypothesis**: 1×1 conv unary head lacks spatial context
- **Finding**: The gradient comparison showed the unary head receives STRONG gradients (10x stronger through BP). The collapse is a data/optimization issue, not architecture.
- **Status**: DEFERRED. Testing simple Conv1x1 projection first (Experiment C). If simple projection + BP works well, complex unary head is unnecessary.

### MoCo contrastive learning
- **Tested**: Original codebase had broken MoCo implementation (key encoder returned feature maps not latents, memory bank filled with garbage)
- **Status**: REPLACED with SimCLR (simpler, proven). MoCo adds complexity without clear benefit for this use case.

### VAE reconstruction path
- **From original codebase**: CVAE had decoder + reconstruction loss
- **Status**: REMOVED. Decoder/reconstruction is not needed for segmentation. The encoder only needs to produce good features, not reconstruct images.

### QuadtreeMRF (original, non-differentiable)
- **From original proposal**: Sequential node-by-node belief propagation
- **Status**: REPLACED with DHBP (GPU-parallel, differentiable). Same math, GPU-friendly implementation.
