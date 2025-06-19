import torch
import numpy as np
from utils.utils import sliding_window, grouper, accuracy
from utils.utils_dataset import convert_to_color
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score

def test(net, test_ids, test_images, test_labels, eroded_labels, labels, stride, batch_size, window_size=(256, 256)):
    """
    Test the network on the provided test data
    
    Args:
        net: The trained network
        test_ids: List of test image IDs
        test_images: Generator of test images
        test_labels: Generator of test labels  
        eroded_labels: Generator of eroded labels
        labels: Class labels
        stride: Stride for sliding window
        batch_size: Batch size for inference
        window_size: Window size for patches
        
    Returns:
        acc_test: Test accuracy
        all_preds: All predictions
        all_gts: All ground truths
    """
    net.eval()
    
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for img_id, img, gt, eroded in zip(test_ids, test_images, test_labels, eroded_labels):
            print(f"Testing on image {img_id}")
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Initialize prediction array
            prediction = np.zeros((height, width), dtype=np.uint8)
            count_map = np.zeros((height, width), dtype=np.float32)
            
            # Sliding window inference
            for x, y, w, h in sliding_window(img, step=stride, window_size=window_size):
                # Extract patch
                patch = img[x:x+w, y:y+h, :]
                
                # Normalize and convert to tensor
                patch = patch / 255.0 if patch.max() > 1.0 else patch
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)
                
                if torch.cuda.is_available():
                    patch_tensor = patch_tensor.cuda()
                
                # Forward pass
                outputs = net(patch_tensor, mode='inference')
                
                # Get prediction
                if 'final_segmentation' in outputs:
                    pred = outputs['final_segmentation'].argmax(dim=1).cpu().numpy()[0]
                else:
                    # Fallback to first available output
                    pred = outputs['hierarchical_segmentations'][-1].argmax(dim=1).cpu().numpy()[0]
                
                # Accumulate predictions
                prediction[x:x+w, y:y+h] += pred
                count_map[x:x+w, y:y+h] += 1
            
            # Normalize by count
            prediction = prediction / np.maximum(count_map, 1)
            prediction = prediction.astype(np.uint8)
            
            all_preds.append(prediction)
            all_gts.append(gt if isinstance(gt, np.ndarray) else np.array(gt))
    
    # Calculate overall accuracy
    total_pixels = 0
    correct_pixels = 0
    
    for pred, gt in zip(all_preds, all_gts):
        # Convert gt if needed
        if gt.ndim == 3:
            from utils.utils_dataset import convert_from_color
            gt = convert_from_color(gt)
        
        # Mask out ignore pixels
        valid_mask = gt != 6
        total_pixels += valid_mask.sum()
        correct_pixels += ((pred == gt) & valid_mask).sum()
    
    acc_test = 100.0 * correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    print(f"Test Accuracy: {acc_test:.2f}%")
    
    return acc_test, all_preds, all_gts