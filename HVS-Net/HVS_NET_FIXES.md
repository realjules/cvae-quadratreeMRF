# HVS-Net Refactoring and Debugging Guide

This document outlines the critical fixes and improvements required to get the HVS-Net model running correctly and to ensure the results are reliable. The original codebase contains several issues that would prevent successful training.

## 1. Critical Bug Fixes in `core/architecture.py`

The architecture file contains dimensional mismatches that will cause runtime errors.

### 1.1. Fix Encoder Dimensionality

*   **Issue:** The `SharedEncoder` hardcodes the flattened layer size for an `8x8` feature map, but the bottleneck actually produces a `16x16` map.
*   **Solution:** The linear layers (`fc_mu`, `fc_log_var`) must be resized to accept a `16x16` input. Better yet, calculate the size dynamically based on the input image size to make the model more flexible.

    ```python
    # In SharedEncoder.__init__
    final_size = input_size // (2**4) # 4 downsampling layers (2x2 stride)
    self.flattened_size = hidden_dims[2] * final_size * final_size
    self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
    self.fc_log_var = nn.Linear(self.flattened_size, latent_dim)
    ```

### 1.2. Fix Decoder Symmetry and Output Size

*   **Issue:** The `GenerativeDecoder` produces a `128x128` output, which will mismatch a `256x256` input image during loss calculation. The `SegmentationDecoder` uses `F.interpolate` as a workaround.
*   **Solution:** Create a `BaseDecoder` class that both decoders inherit from. This ensures they share the same symmetric upsampling path. Add the necessary `ConvTranspose2d` layers to both decoders to properly upsample to the final `256x256` resolution.

## 2. Robustness and Metrics in `core/trainer.py`

The trainer logic is not robust to variations in batch composition and lacks essential evaluation metrics.

### 2.1. Implement Robust Batch Handling

*   **Issue:** The `_train_epoch` function concatenates all images and splits them based on a fixed size. This will fail if a batch doesn't contain all expected image types (e.g., the last batch of an epoch).
*   **Solution:** Refactor the training loop to handle labeled and unlabeled data in separate forward passes. This is more robust and easier to debug.

    ```python
    # In _train_epoch
    # Instead of concatenating, process separately
    if labeled_image is not None:
        labeled_output = self.model(labeled_image)
        # ... calculate labeled loss

    if unlabeled_image1 is not None:
        unlabeled_output1 = self.model(unlabeled_image1)
        unlabeled_output2 = self.model(unlabeled_image2)
        # ... calculate consistency loss
    ```

### 2.2. Add mIoU Validation and Checkpointing

*   **Issue:** The validation loop only calculates loss, which is not a reliable indicator of segmentation quality. Checkpoints are saved based on this unreliable metric.
*   **Solution:**
    1.  Create a new utility file `utils/evaluation.py` with functions to compute per-class IoU and mean IoU (mIoU).
    2.  In `_validate_epoch`, call the `compute_miou` function.
    3.  Log the mIoU and save checkpoints based on the **best validation mIoU**, not the best loss.

## 3. Flexible Loss Calculation in `core/losses.py`

The `CombinedLoss` class needs to be updated to work with the more robust trainer.

*   **Issue:** The `forward` pass assumes that `labeled` and `unlabeled` data are always present in the input dictionaries.
*   **Solution:** Modify the `forward` signature and logic to accept potentially `None` values for labeled or unlabeled data and only compute the relevant losses.

    ```python
    # In CombinedLoss.forward
    def forward(self, labeled_output, labeled_data, unlabeled_output1, unlabeled_output2, unlabeled_data):
        total_loss = 0
        if labeled_output is not None:
            # ... compute supervised loss
            total_loss += ...
        if unlabeled_output1 is not None:
            # ... compute consistency loss
            total_loss += ...
        # ... etc.
    ```

By following this guide, the HVS-Net codebase will be significantly more robust, debuggable, and capable of producing meaningful, reliable results.
