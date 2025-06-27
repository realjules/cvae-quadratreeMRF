# HVS-Net/data/dataset.py

"""
This file contains the PyTorch Dataset class for loading the ISPRS data.

It handles:
1.  Finding and loading the image and ground truth tiles.
2.  A robust data augmentation pipeline using Albumentations for the segmentation task.
3.  A separate, strong augmentation pipeline for the unsupervised consistency loss.
4.  Returning both labeled and unlabeled data as needed by the trainer.
"""

import os
import random
import glob
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISPRSDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode

        # Get all image paths
        self.image_paths = sorted(glob.glob(os.path.join(config['data']['data_dir'], '*.tif')))
        self.label_paths = sorted(glob.glob(os.path.join(config['data']['label_dir'], '*.tif')))

        # Create a map of image IDs to label paths for easy lookup
        label_map = {os.path.basename(lp).replace('_noBoundary', ''): lp for lp in self.label_paths}
        
        # Split into labeled and unlabeled sets
        num_labeled = int(len(self.image_paths) * (config['data']['labeled_percentage'] / 100.0))
        
        if mode == 'train':
            self.labeled_image_paths = self.image_paths[:num_labeled]
            self.unlabeled_image_paths = self.image_paths
            print(f"Training with {len(self.labeled_image_paths)} labeled images and {len(self.unlabeled_image_paths)} unlabeled images.")
        elif mode == 'validate':
            # For validation, we only need labeled data
            self.labeled_image_paths = self.image_paths[num_labeled:]
            self.unlabeled_image_paths = []
            print(f"Validating with {len(self.labeled_image_paths)} images.")

        self.labeled_map = {p: label_map.get(os.path.basename(p)) for p in self.labeled_image_paths}

        # Define augmentation pipelines
        self.base_aug = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.strong_aug = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8),
            A.ColorJitter(p=0.8),
            A.GaussianBlur(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        # In each epoch, we want to see a fixed number of samples
        return 10000 if self.mode == 'train' else len(self.labeled_image_paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # Get a labeled sample
            labeled_img_path = random.choice(self.labeled_image_paths)
            labeled_mask_path = self.labeled_map[labeled_img_path]
            labeled_img = io.imread(labeled_img_path)
            labeled_mask = io.imread(labeled_mask_path)
            
            augmented_labeled = self.base_aug(image=labeled_img, mask=labeled_mask)
            labeled_img, labeled_mask = augmented_labeled['image'], augmented_labeled['mask']

            # Get an unlabeled sample for consistency loss
            unlabeled_img_path = random.choice(self.unlabeled_image_paths)
            unlabeled_img = io.imread(unlabeled_img_path)
            
            aug1 = self.strong_aug(image=unlabeled_img)['image']
            aug2 = self.strong_aug(image=unlabeled_img)['image']

            return {
                'labeled_image': labeled_img,
                'labeled_mask': labeled_mask,
                'unlabeled_image1': aug1,
                'unlabeled_image2': aug2
            }
        else: # Validation mode
            img_path = self.labeled_image_paths[idx]
            mask_path = self.labeled_map[img_path]
            image = io.imread(img_path)
            mask = io.imread(mask_path)

            augmented = self.base_aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

            return {'image': image, 'mask': mask}