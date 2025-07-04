# HVS-Net/utils/evaluation.py
import torch
import numpy as np

def compute_iou(pred, target, n_classes):
    """Computes the Intersection over Union (IoU) for each class."""
    iou = []
    # Using argmax to get the predicted class index
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou.append(float('nan'))  # Class not present in target, NaN so it's ignored in mean
        else:
            iou.append(intersection / union)
    return iou

def compute_miou(pred, target, n_classes):
    """Computes the Mean Intersection over Union (mIoU) across all classes."""
    iou_per_class = compute_iou(pred, target, n_classes)
    # Using nanmean to ignore NaN values (classes not present in the batch)
    miou = np.nanmean(iou_per_class)
    return miou, iou_per_class
