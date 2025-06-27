# HVS-Net/utils/visualizer.py

"""
This file contains functions for visualizing the model's predictions.

It is useful for:
1.  Saving example segmentation outputs during training to monitor progress.
2.  Comparing the ground truth, the model's prediction, and the input image side-by-side.
3.  Visualizing the reconstructed images from the generative decoder.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Define a color map for the 6 classes of the ISPRS dataset
# 0: Impervious surfaces (white)
# 1: Building (blue)
# 2: Low vegetation (cyan)
# 3: Tree (green)
# 4: Car (yellow)
# 5: Clutter/background (red)
COLOR_MAP = np.array(
    [
        [255, 255, 255],  # Impervious surfaces
        [0, 0, 255],      # Building
        [0, 255, 255],    # Low vegetation
        [0, 255, 0],      # Tree
        [255, 255, 0],    # Car
        [255, 0, 0],      # Clutter/background
    ]
)

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image with mean and standard deviation."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_segmentation(
    image_tensor, 
    pred_mask_tensor, 
    gt_mask_tensor, 
    save_path,
    epoch
):
    """Saves a visualization of input, prediction, and ground truth."""
    # Move tensors to CPU and convert to numpy
    image = denormalize(image_tensor.clone().cpu()).numpy().transpose(1, 2, 0)
    pred_mask = pred_mask_tensor.cpu().numpy()
    gt_mask = gt_mask_tensor.cpu().numpy()

    # Convert masks to color images
    pred_colored_mask = COLOR_MAP[pred_mask].astype(np.uint8)
    gt_colored_mask = COLOR_MAP[gt_mask].astype(np.uint8)

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Epoch {epoch}', fontsize=16)

    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')

    ax2.imshow(gt_colored_mask)
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    ax3.imshow(pred_colored_mask)
    ax3.set_title('Prediction')
    ax3.axis('off')

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)