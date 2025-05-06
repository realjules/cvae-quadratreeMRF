# Model Enhancement Documentation

This document outlines the implemented enhancements to our aerial image segmentation framework. We've developed three separate model improvements that can be trained and evaluated independently for ablation studies.

## Table of Contents

1. [Attention Fusion Enhanced CVAE](#1-attention-fusion-enhanced-cvae)
2. [Supervised Contrastive CVAE](#2-supervised-contrastive-cvae)
3. [Cross-Scale Attention MRF](#3-cross-scale-attention-mrf)
4. [Training Instructions](#4-training-instructions)
5. [Implementation Details](#5-implementation-details)

## 1. Attention Fusion Enhanced CVAE

### Overview
The Attention Fusion Enhanced CVAE improves feature integration in the decoder by replacing simple skip connections with sophisticated cross-attention mechanisms.

### Key Components
- **CrossAttentionFusion Module**: Implements a transformer-style attention mechanism that allows decoder features to selectively attend to encoder features.
- **Query-Key-Value Attention**: Features from the decoder (query) attend to features from the encoder (key/value) using scaled dot-product attention.
- **Content-Adaptive Integration**: Unlike standard concatenation, this approach allows for dynamic, content-aware feature integration.

### Benefits
- **Selective Feature Utilization**: The model learns which encoder features are most relevant for each spatial position in the decoder.
- **Improved Detail Preservation**: Better preservation of important details from encoder features during reconstruction.
- **Non-Local Relationships**: Attention mechanism allows the model to capture relationships beyond just spatial alignment.

### Files
- `net/attention_fusion_cvae.py`: Main model implementation
- `run_attention_fusion_cvae.py`: Training script

## 2. Supervised Contrastive CVAE

### Overview
This enhancement significantly improves the contrastive learning component of the CVAE by increasing its weight and adding supervised contrastive learning when labels are available.

### Key Components
- **Higher Contrastive Weight**: Increases the default weight from 0.5 to 1.0, emphasizing the importance of contrastive learning.
- **Supervised Projection Head**: Separate pathway for projecting features when class labels are available.
- **Enhanced Memory Bank**: Larger memory bank (8192 vs 4096) and separate memory bank for labeled samples.
- **Harder Negative Mining**: More aggressive weighting of hard negative examples.
- **Focal Weighting**: Applies focal-loss-inspired weighting to emphasize challenging examples.

### Benefits
- **Stronger Discriminative Features**: Higher weight on contrastive learning creates more discriminative latent representations.
- **Better Semantic Alignment**: Supervised contrastive loss directly aligns the latent space with semantic classes.
- **Improved Generalization**: The enhanced contrastive learning leads to better generalization to unseen data.
- **Faster Convergence**: Stronger contrastive signal helps the model learn more quickly.

### Files
- `net/supervised_contrast_cvae.py`: Main model implementation
- `run_supervised_contrast_cvae.py`: Training script

## 3. Cross-Scale Attention MRF

### Overview
This enhancement adds explicit cross-scale interactions to the SimplifiedMRF through multiple attention mechanisms that relate features at different scales.

### Key Components
- **CrossScaleAttention**: Core module that allows features at different scales to attend to each other.
- **FeaturePyramidAttention**: Integrates multi-scale context within a single feature map using pyramid pooling.
- **ScaleAwareAttention**: Dynamically weights features from different scales based on content.
- **Deep Supervision**: Scale-specific classifiers for improved intermediate representations.

### Benefits
- **Better Multi-Scale Understanding**: Improved handling of objects at different scales in aerial imagery.
- **Content-Adaptive Scale Integration**: Dynamic weighting of features based on their importance at each scale.
- **Enhanced Boundary Precision**: Better boundary delineation by integrating fine details with contextual information.
- **Scale Invariance**: Improved robustness to variations in object scale within aerial imagery.

### Files
- `net/cross_scale_mrf.py`: Main model implementation
- `run_cross_scale_mrf.py`: Training script

## 4. Training Instructions

### Attention Fusion Enhanced CVAE
```bash
python run_attention_fusion_cvae.py \
  --data_path input/dataset \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --mixed_precision
```

### Supervised Contrastive CVAE
```bash
python run_supervised_contrast_cvae.py \
  --data_path input/dataset \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --contrastive_weight 1.0 \
  --supervised_weight 1.0 \
  --mixed_precision
```

### Cross-Scale Attention MRF
```bash
python run_cross_scale_mrf.py \
  --data_path input/dataset \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --cvae_weights path/to/pretrained_cvae.pth \
  --class_weights \
  --mixed_precision
```

## 5. Implementation Details

All implementations follow the same architectural principles as the existing codebase but add specific enhancements. Each implementation includes:

### Training Infrastructure
- **Metrics Tracking**: Comprehensive logging of loss components and performance metrics.
- **Visualization**: Periodic visualization of model outputs during training.
- **Checkpointing**: Proper saving and resuming of training state.
- **Mixed Precision**: Support for faster training with mixed precision arithmetic.

### Model Architecture
- Each enhancement maintains compatibility with the existing pipeline.
- Models are designed to be drop-in replacements for their counterparts in the original implementation.
- Careful attention has been paid to computational efficiency, with optional parameters to control memory usage.

### Loss Functions 
- Enhanced loss functions with adaptive weighting based on training progress.
- Support for both supervised and unsupervised training modes.
- Comprehensive loss component tracking for analysis.

## Ablation Studies

These separate implementations allow for thorough ablation studies to evaluate the impact of each enhancement:

1. **Baseline**: Original implementation (EnhancedCVAE + SimplifiedMRF)
2. **Attention Fusion**: AttentionFusionCVAE + SimplifiedMRF
3. **Supervised Contrastive**: SupervisedContrastCVAE + SimplifiedMRF
4. **Cross-Scale MRF**: EnhancedCVAE + CrossScaleMRF
5. **Full Enhancement**: Combination of all three enhancements

By comparing the performance of these different combinations, we can quantify the contribution of each enhancement to the overall performance improvement.