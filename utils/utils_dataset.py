# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:37:47 2022

@author: marti
Enhanced with additional data augmentation for improved model training
"""

import numpy as np
import random
import torch
import torchvision.transforms as transforms
from scipy import ndimage
from scipy.ndimage import map_coordinates
from skimage.morphology import erosion, disk

# ISPRS standard color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def erode_gt(gt, kernel):
    gt1 = erosion(gt, kernel)
    gt2 = np.where(gt-gt1 != 0, 6, gt)
    return gt2

def conn_comp(gt, kernel): # taking the GT already converted in 1 channel
    """ removal of connected components in GTs """
    """ change the number of remaining components to lower the percentage of labels"""
    new_gt = 6 * np.ones(gt.shape) # completely black image of the size of the original GT
    threshold = 0.5 # stating the threshold for binarized 0,1 images
    for i in np.unique(gt):    # finding connected components for each class
        bin_gt = np.zeros(gt.shape)
        bin_gt[np.where(gt==i)] = 1 # binarizing the image discarding any class non-currently considered
        labeled, nr_objects = ndimage.label(bin_gt > threshold) 

        #print("Number of objects for class " + str(i) + " is {}".format(nr_objects))
        
        if i == 4 or i == 0: # in this case, cars and streets
            num = np.argsort(np.unique(labeled, return_counts=True)[1])[:-1][::2] # reorder the elements of the array of connected components 
                                                                                  # (from smaller to bigger, discarding the background)
                                                                                  # take one element every two
        else:
            num = np.argsort(np.unique(labeled, return_counts=True)[1])[:-1][::3]  
            labeled = erosion(labeled, kernel)                                    # further erosion                                                     
                                                                         
        for n in num:
            new_gt[np.where(labeled==n)] = i  # give back the right class number to the saved connected components

    return new_gt
    
# Enhanced augmentation functions for improving training
def get_augmentation_transforms(p=0.5):
    """
    Create a set of image augmentation transforms with probability p
    """
    color_jitter = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )
    
    # Only apply to image, not mask
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([color_jitter], p=p),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()
    ])
    
    return image_transforms

def elastic_transform(image, mask, alpha=50, sigma=5):
    """
    Apply elastic transform to image and mask simultaneously
    """
    # Get image dimensions
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image.transpose(1, 2, 0)
    
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    height, width = image_np.shape[:2]
    
    # Create displacement field
    dx = np.random.rand(height, width) * 2 - 1
    dy = np.random.rand(height, width) * 2 - 1
    
    # Smooth displacement field
    from scipy.ndimage import gaussian_filter
    dx = gaussian_filter(dx, sigma) * alpha
    dy = gaussian_filter(dy, sigma) * alpha
    
    # Create meshgrid for sampling
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Apply transform to image and mask
    distorted_image = np.zeros_like(image_np)
    for c in range(image_np.shape[2]):
        distorted_image[:, :, c] = map_coordinates(image_np[:, :, c], indices, order=1).reshape(height, width)
    
    distorted_mask = map_coordinates(mask_np, indices, order=0).reshape(height, width)
    
    # Convert back to original format
    if isinstance(image, torch.Tensor):
        distorted_image = torch.from_numpy(distorted_image.transpose(2, 0, 1))
    else:
        distorted_image = distorted_image.transpose(2, 0, 1)
    
    if isinstance(mask, torch.Tensor):
        distorted_mask = torch.from_numpy(distorted_mask)
    
    return distorted_image, distorted_mask

def cutmix_augmentation(image1, mask1, image2, mask2, alpha=0.5):
    """
    Apply CutMix augmentation between two images and their masks
    """
    # Ensure tensors
    if not isinstance(image1, torch.Tensor):
        image1 = torch.from_numpy(image1)
    if not isinstance(mask1, torch.Tensor):
        mask1 = torch.from_numpy(mask1)
    if not isinstance(image2, torch.Tensor):
        image2 = torch.from_numpy(image2)
    if not isinstance(mask2, torch.Tensor):
        mask2 = torch.from_numpy(mask2)
    
    # Get dimensions
    _, h, w = image1.shape
    
    # Generate random box
    r_x = random.randint(0, w)
    r_y = random.randint(0, h)
    r_w = random.randint(1, w - r_x)
    r_h = random.randint(1, h - r_y)
    
    # Create box
    x1 = r_x
    y1 = r_y
    x2 = r_x + r_w
    y2 = r_y + r_h
    
    # Apply mixing
    mixed_image = image1.clone()
    mixed_mask = mask1.clone()
    
    mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
    mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
    
    return mixed_image, mixed_mask