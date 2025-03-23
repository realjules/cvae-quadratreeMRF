# -*- coding: utf-8 -*-
"""
Testing function for Semi-Supervised Hierarchical PGM with Contrastive Learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
from utils.utils_dataset import convert_to_color
from utils.utils import metrics, sliding_window, count_sliding_window, grouper


def test(net, test_ids, test_images, test_labels, eroded_labels, classes, stride, batch_size, window_size, all=True):
    """
    Test the Semi-Supervised Hierarchical PGM model
    
    Parameters:
    -----------
    net: torch.nn.Module
        The hierarchical PGM model
    test_ids: list
        List of test image IDs
    test_images: generator or list
        Test image data
    test_labels: generator or list
        Test label data
    eroded_labels: generator or list
        Eroded test label data
    classes: list
        Class names
    stride: int
        Stride for sliding window
    batch_size: int
        Batch size for processing
    window_size: tuple
        Window size for patches
    all: bool
        Whether to return all predictions and ground truths
    
    Returns:
    --------
    accuracy: float
        Overall accuracy
    all_preds: list (optional)
        List of all predictions
    all_gts: list (optional)
        List of all ground truths
    """
    all_preds = []
    all_gts = []
    
    # Switch to evaluation mode
    net.eval()

    for img, gt, gt_e, tile_id in zip(test_images, test_labels, eroded_labels, test_ids):
        print(f"Processing test tile {tile_id}...")
        
        # Initialize prediction map
        pred = np.zeros(img.shape[:2] + (len(classes),))
        
        # Process image in patches using sliding window
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), 
                                       total=total, desc=f"Tile {tile_id}")):
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())
                else:
                    image_patches = Variable(torch.from_numpy(image_patches))
                
                # Forward pass in inference mode
                outputs = net(image_patches, mode='inference')
                
                # Get segmentation predictions
                if 'final_segmentation' in outputs:
                    outs = outputs['final_segmentation'].cpu().numpy()
                else:
                    outs = outputs['hierarchical_segmentations'][-1].cpu().numpy()
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out
        
        # Get final prediction
        pred = np.argmax(pred, axis=-1)
        
        # Visualize prediction
        fig = plt.figure(figsize=(15, 5))
        fig.add_subplot(1,3,1)
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        plt.title(f'Original (Tile {tile_id})')
        fig.add_subplot(1,3,2)
        plt.imshow(convert_to_color(pred))
        plt.title('Prediction')
        fig.add_subplot(1,3,3)
        plt.imshow(convert_to_color(gt_e))
        plt.title('Ground Truth')
        plt.show()
        
        all_preds.append(pred)
        all_gts.append(gt_e)
        
        # Compute metrics for this tile
        print(f"Metrics for tile {tile_id}:")
        metrics(pred.ravel(), gt_e.ravel(), classes)
    
    # Compute overall metrics
    print("\nOverall metrics:")
    accuracy = metrics(
        np.concatenate([p.ravel() for p in all_preds]), 
        np.concatenate([p.ravel() for p in all_gts]).ravel(), 
        classes
    )
    
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy