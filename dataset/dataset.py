# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:11:27 2022

@author: marti

modified from the original code available at https://github.com/nshaud/DeepNetsForEO 

"""

import torch
import random
import numpy as np
import os
from skimage import io
from utils.utils_dataset import *
from utils.utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Dataset class

class ISPRS_dataset(torch.utils.data.Dataset):
    # Rare classes to oversample: Cars=4, Clutter=5
    RARE_CLASSES = {4, 5}

    def __init__(self, ids, ids_type, gt_type, gt_modification, data_files, label_files,
                            window_size, cache=False, augmentation=False,
                            class_balanced=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.ids_type = ids_type
        self.gt_type = gt_type
        self.gt_modification = gt_modification
        self.window_size = window_size
        self.class_balanced = class_balanced


        # List of files
        self.data_files = [data_files.format(id) for id in ids]
        self.label_files = [label_files.format(id) for id in ids]

        # Check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

        # Pre-compute positions containing rare classes for balanced sampling
        self.rare_positions = {}
        if self.class_balanced and self.ids_type == 'TRAIN':
            self._precompute_rare_positions()

        # Albumentations pipeline
        if self.augmentation:
            self.aug_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, p=1),
                ], p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
                ToTensorV2(),
            ])
        else:
            self.aug_pipeline = A.Compose([
                ToTensorV2(),
            ])
            
    
    def _precompute_rare_positions(self):
        """Scan label files to find patch positions containing rare classes (Cars, Clutter).
        Stores (x1, y1) positions where a 256×256 patch contains at least 1% rare class pixels."""
        w = self.window_size
        for idx, lf in enumerate(self.label_files):
            label = np.asarray(convert_from_color(io.imread(lf)), dtype='int64')
            h_img, w_img = label.shape[:2]
            positions = []
            # Scan with stride of window_size//2 for overlap
            stride = max(w // 2, 1)
            for y in range(0, h_img - w, stride):
                for x in range(0, w_img - w, stride):
                    patch = label[y:y+w, x:x+w]
                    rare_count = sum((patch == c).sum() for c in self.RARE_CLASSES)
                    # At least 1% of patch is rare class (~655 pixels out of 65536)
                    if rare_count > w * w * 0.01:
                        positions.append((y, x))
            self.rare_positions[idx] = positions
            print(f"  Class-balanced sampling: file {idx} has {len(positions)} rare-class patches "
                  f"(out of {((h_img - w) // stride) * ((w_img - w) // stride)} total)")

    def __len__(self):
        # Reasonable epoch size for faster training (was 10,000)
        # Use smaller epoch size to prevent extremely long training times
        return min(2000, len(self.data_files) * 200)

    def _load_data(self, random_idx):
        """Load and cache image data for a given file index."""
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            data = np.asarray(io.imread(self.data_files[random_idx]), dtype='float32')
            data = 1/255 * data
            if self.cache:
                self.data_cache_[random_idx] = data
        return data

    def _load_label(self, random_idx):
        """Load and cache label data for a given file index."""
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            if self.ids_type == 'TRAIN':
                if self.gt_type == 'conncomp':
                    label = np.asarray(conn_comp(convert_from_color(io.imread(self.label_files[random_idx])), self.gt_modification), dtype='int64')
                elif self.gt_type == 'full':
                    label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label
        return label

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        data = self._load_data(random_idx)
        label = self._load_label(random_idx)

        try:
            # Class-balanced sampling: 50% chance to sample from rare-class positions
            use_rare = (self.class_balanced
                        and random_idx in self.rare_positions
                        and len(self.rare_positions[random_idx]) > 0
                        and random.random() < 0.5)

            if use_rare:
                # Sample from pre-computed rare-class positions (stored as row, col)
                row, col = random.choice(self.rare_positions[random_idx])
            else:
                # get_random_pos returns (y1, y2, x1, x2) = (row1, row2, col1, col2)
                row, _, col, _ = get_random_pos(data, self.window_size)

            w = self.window_size
            data_p = data[row:row+w, col:col+w, :]
            label_p = label[row:row+w, col:col+w]
        except (ValueError, IndexError):
            return self.__getitem__(i)

        # Data augmentation & tensor conversion
        if self.augmentation:
            augmented = self.aug_pipeline(image=data_p, mask=label_p)
            data_p = augmented['image']
            label_p = augmented['mask']
        else:
            # When no augmentation, manually convert to tensor and permute channels
            data_p = torch.from_numpy(data_p).permute(2, 0, 1)
            label_p = torch.from_numpy(label_p)


        # Return the torch.Tensor values
        return (data_p, label_p)