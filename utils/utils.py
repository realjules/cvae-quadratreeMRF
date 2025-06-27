# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:37:46 2022

@author: marti
"""

import time
import numpy as np
import random
import itertools
from sklearn.metrics import confusion_matrix
import json
import matplotlib.pyplot as plt




############# Utils ###############


def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    # Handle both integer (square window) and tuple (rectangular window) inputs
    if isinstance(window_shape, int):
        w, h = window_shape, window_shape  # Convert int to square dimensions
    else:
        w, h = window_shape  # Use provided (width, height) tuple
    
    # FIXED: Auto-detect tensor format to handle both (H,W,C) and (C,H,W)
    if len(img.shape) == 3:
        if img.shape[0] <= 4:  # (C, H, W) format - channels first
            C, H, W = img.shape
        else:  # (H, W, C) format - channels last
            H, W, C = img.shape
    else:  # 2D image (H, W)
        H, W = img.shape
    
    # FIXED: Robust validation and safe random position generation
    if H < h or W < w:
        # Image too small - use safe fallback positioning
        x1 = max(0, min((W - w) // 2, W - w)) if W >= w else 0
        y1 = max(0, min((H - h) // 2, H - h)) if H >= h else 0
        # Ensure we don't exceed bounds
        w = min(w, W)
        h = min(h, H)
    else:
        # FIXED: Safe random position calculation with proper bounds checking
        max_x = max(0, W - w)  # Ensure non-negative
        max_y = max(0, H - h)  # Ensure non-negative
        
        if max_x > 0:
            x1 = random.randint(0, max_x - 1)
        else:
            x1 = 0
            
        if max_y > 0:
            y1 = random.randint(0, max_y - 1)
        else:
            y1 = 0
    
    x2 = x1 + w
    y2 = y1 + h
    
    # Ensure bounds are within image limits
    x2 = min(x2, W)
    y2 = min(y2, H)
    x1 = max(0, x2 - w)
    y1 = max(0, y2 - h)
    
    return x1, x2, y1, y2


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def validate_image_size(img, window_size, min_size_factor=1.1):
    """
    Validate if image is large enough for random patch extraction
    
    Args:
        img: Input image array
        window_size: Required window size (int or tuple)
        min_size_factor: Minimum size factor (1.1 means 10% larger than window)
    
    Returns:
        bool: True if image is large enough, False otherwise
    """
    if isinstance(window_size, int):
        w, h = window_size, window_size
    else:
        w, h = window_size
    
    # Get spatial dimensions correctly
    if len(img.shape) == 3:
        if img.shape[0] <= 4:  # Likely (C, H, W) format
            H, W = img.shape[-2:]
        else:  # Likely (H, W, C) format
            H, W = img.shape[:2]
    else:  # 2D image
        H, W = img.shape
    
    min_required_w = int(w * min_size_factor)
    min_required_h = int(h * min_size_factor)
    
    return W >= min_required_w and H >= min_required_h


def resize_image_if_needed(img, window_size, target_size_factor=1.2):
    """
    Resize image if it's too small for patch extraction
    
    Args:
        img: Input image array
        window_size: Required window size
        target_size_factor: Target size factor (1.2 means 20% larger than window)
    
    Returns:
        Resized image or original if already large enough
    """
    if validate_image_size(img, window_size):
        return img
    
    if isinstance(window_size, int):
        w, h = window_size, window_size
    else:
        w, h = window_size
    
    target_W = int(w * target_size_factor)
    target_H = int(h * target_size_factor)
    
    # This would require scipy or cv2, so for now just return original
    # In a real implementation, you'd use cv2.resize or similar
    print(f"Warning: Image too small for window size {window_size}. Consider resizing to at least {target_W}x{target_H}")
    return img


def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values):
    cm = confusion_matrix(
            gts,
            predictions)
    
    print("Confusion matrix :")
    print(cm)
    
    print("---")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    return accuracy
