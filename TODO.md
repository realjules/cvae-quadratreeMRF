# TODO — Future experiments

## After pairwise constraint fix (current PR)

### Auxiliary unary loss
- Add `0.4 * CE(phi_1, labels)` to loss function alongside focal + boundary
- Gives unary head direct gradient signal instead of routing through BP
- **Why**: unary accuracy is 58.8% but linear probe is 67.7% — the unary head underperforms
- **Reference**: Zheng et al. 2015 (CRF-as-RNN) uses auxiliary losses at intermediate layers

### Unary head architecture
- Replace Conv1x1 with Conv3x3 for spatial context
- Current head: `Conv1x1(C→C/2) + BN + ReLU + Conv1x1(C/2→K) + log_softmax`
- Proposed: `Conv3x3(C→C/2) + BN + ReLU + Conv3x3(C/2→C/4) + BN + ReLU + Conv1x1(C/4→K)`

### Dense contrastive learning (PixelCL)
- Current SimCLR pulls patch-level features together — Cars (1.5% of pixels) get lost
- Dense contrastive pulls pixel-level features at same location across augmented views
- **Reference**: Wang et al. 2021, "Dense Contrastive Learning for Self-Supervised Visual Pre-Training"

### Class-balanced patch sampling
- During contrastive pre-training, oversample patches containing Cars/Clutter
- Doesn't need labels — can use image statistics to identify small bright objects

### Multi-scale cropping for contrastive
- Add 128×128 and 64×64 crops alongside 256×256
- Forces encoder to learn fine-grained features (important for Cars)

### Encoder upgrade
- Linear probe is 67.7% — MODERATE. Room to improve.
- Try ResNet-34 or ResNet-50 (more capacity)
- Try longer contrastive training (100+ epochs)
- Try lower temperature (0.05 instead of 0.1)

### ReduceLROnPlateau scheduler
- Current CosineAnnealingWarmRestarts causes accuracy oscillations
- ReduceLROnPlateau(mode='max', patience=5) would be more stable

### Multiple BP iterations
- Single pass is exact on tree, but stacking 2-3 passes with different pairwise heads
  could act as iterative refinement (similar to CRF-as-RNN unrolling)
- Only explore if single pass hits a ceiling
