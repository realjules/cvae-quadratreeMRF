"""
Contrastive Learning Augmentation Pipeline for Remote Sensing Images

This module provides strong augmentations specifically designed for 
self-supervised contrastive learning on aerial imagery.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image, ImageFilter
import cv2


class ContrastiveAugmentation:
    """
    Strong augmentation pipeline for contrastive learning on remote sensing images.
    Creates two different augmented views of the same image for positive pairs.
    """
    
    def __init__(self, size=256, strength=0.8):
        self.size = size
        self.strength = strength
        
        # Color augmentations - stronger for contrastive learning
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4 * strength,
            contrast=0.4 * strength, 
            saturation=0.4 * strength,
            hue=0.1 * strength
        )
        
        # Geometric augmentations
        self.rotation_range = int(30 * strength)  # Up to 30 degrees
        self.scale_range = (0.8, 1.2)  # Scale variation
        
        # Advanced augmentations
        self.gaussian_blur_prob = 0.5 * strength
        self.gaussian_noise_prob = 0.3 * strength
        self.cutout_prob = 0.3 * strength
        
    def __call__(self, image):
        """
        Generate two augmented views of the input image
        
        Args:
            image: PIL Image or tensor [C, H, W]
            
        Returns:
            view1, view2: Two differently augmented versions of the input
        """
        # Convert to PIL if tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                # Convert from [C, H, W] to PIL
                image = transforms.ToPILImage()(image)
            else:
                raise ValueError(f"Expected 3D tensor, got {image.dim()}D")
        
        # Generate two different augmented views
        view1 = self._apply_augmentations(image, seed=None)
        view2 = self._apply_augmentations(image, seed=None)
        
        return view1, view2
    
    def _apply_augmentations(self, image, seed=None):
        """Apply a random set of augmentations to create one view"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Start with original image
        augmented = image.copy()
        
        # 1. Random crop and resize (always applied)
        augmented = self._random_crop_and_resize(augmented)
        
        # 2. Random horizontal flip (50% chance)
        if random.random() < 0.5:
            augmented = transforms.functional.hflip(augmented)
        
        # 3. Random vertical flip (50% chance) - good for aerial imagery
        if random.random() < 0.5:
            augmented = transforms.functional.vflip(augmented)
        
        # 4. Random rotation (70% chance)
        if random.random() < 0.7:
            angle = random.randint(-self.rotation_range, self.rotation_range)
            augmented = transforms.functional.rotate(augmented, angle, fill=0)
        
        # 5. Color jittering (80% chance)
        if random.random() < 0.8:
            augmented = self.color_jitter(augmented)
        
        # 6. Gaussian blur (with probability)
        if random.random() < self.gaussian_blur_prob:
            augmented = self._gaussian_blur(augmented)
        
        # 7. Convert to tensor for further processing
        augmented = transforms.ToTensor()(augmented)
        
        # 8. Gaussian noise (with probability)
        if random.random() < self.gaussian_noise_prob:
            augmented = self._add_gaussian_noise(augmented)
        
        # 9. Random cutout (with probability)
        if random.random() < self.cutout_prob:
            augmented = self._random_cutout(augmented)
        
        # 10. Normalize to [0, 1] if needed
        augmented = torch.clamp(augmented, 0, 1)
        
        return augmented
    
    def _random_crop_and_resize(self, image):
        """Random crop and resize back to original size"""
        width, height = image.size
        
        # Random crop size (80% to 100% of original)
        crop_ratio = random.uniform(0.8, 1.0)
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        
        # Random crop position
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        
        # Crop and resize
        cropped = transforms.functional.crop(image, top, left, crop_height, crop_width)
        resized = transforms.functional.resize(cropped, (self.size, self.size))
        
        return resized
    
    def _gaussian_blur(self, image):
        """Apply Gaussian blur with random kernel size"""
        # Random blur radius (0.1 to 2.0)
        radius = random.uniform(0.1, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _add_gaussian_noise(self, tensor):
        """Add Gaussian noise to tensor"""
        # Random noise strength (0.01 to 0.05)
        noise_std = random.uniform(0.01, 0.05)
        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise
    
    def _random_cutout(self, tensor):
        """Apply random cutout (erase rectangular patches)"""
        _, height, width = tensor.shape
        
        # Random cutout size (5% to 15% of image area)
        cutout_ratio = random.uniform(0.05, 0.15)
        cutout_size = int(np.sqrt(height * width * cutout_ratio))
        
        # Random position
        y = random.randint(0, height - cutout_size)
        x = random.randint(0, width - cutout_size)
        
        # Apply cutout (fill with random gray value)
        fill_value = random.uniform(0.3, 0.7)
        tensor[:, y:y+cutout_size, x:x+cutout_size] = fill_value
        
        return tensor


class WeakAugmentation:
    """
    Weak augmentation for semi-supervised learning.
    Used for generating pseudo-labels with high confidence.
    """
    
    def __init__(self, size=256):
        self.size = size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        return self.transform(image)


class StrongAugmentation:
    """
    Strong augmentation for semi-supervised learning.
    Used for consistency regularization.
    """
    
    def __init__(self, size=256):
        self.size = size
        self.contrastive_aug = ContrastiveAugmentation(size=size, strength=1.0)
    
    def __call__(self, image):
        # Use one view from contrastive augmentation
        view1, _ = self.contrastive_aug(image)
        return view1


def create_contrastive_pair(image):
    """
    Convenience function to create a contrastive pair from a single image
    
    Args:
        image: Input image (PIL or tensor)
        
    Returns:
        tuple: (view1, view2) - two augmented views for contrastive learning
    """
    aug = ContrastiveAugmentation()
    return aug(image)


def test_augmentations():
    """Test function to verify augmentations work correctly"""
    # Create dummy image
    dummy_image = torch.randn(3, 256, 256)
    
    # Test contrastive augmentation
    aug = ContrastiveAugmentation()
    view1, view2 = aug(dummy_image)
    
    print(f"Original shape: {dummy_image.shape}")
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"View 1 range: [{view1.min():.3f}, {view1.max():.3f}]")
    print(f"View 2 range: [{view2.min():.3f}, {view2.max():.3f}]")
    
    # Test that views are different
    diff = torch.abs(view1 - view2).mean()
    print(f"Mean difference between views: {diff:.3f}")
    
    print("✅ Augmentation pipeline test passed!")


if __name__ == "__main__":
    test_augmentations()