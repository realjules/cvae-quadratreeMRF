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
    def __init__(self, ids, ids_type, gt_type, gt_modification, data_files, label_files,
                            window_size, cache=False, augmentation=False):
        super(ISPRS_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        self.ids_type = ids_type
        self.gt_type = gt_type
        self.gt_modification = gt_modification
        self.window_size = window_size

        
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
            
    
    def __len__(self):
        # Reasonable epoch size for faster training (was 10,000)
        # Use smaller epoch size to prevent extremely long training times
        return min(2000, len(self.data_files) * 200)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = np.asarray(io.imread(self.data_files[random_idx]), dtype='float32')
            data = 1/255 * data
            
            # vaihinger
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            if self.ids_type == 'TRAIN':
                if self.gt_type == 'conncomp':
                    label = np.asarray(conn_comp(convert_from_color(io.imread(self.label_files[random_idx])), self.gt_modification), dtype='int64')
                elif self.gt_type == 'full':
                    label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        try:
            # Get a random patch
            x1, x2, y1, y2 = get_random_pos(data, self.window_size)
            data_p = data[x1:x2, y1:y2, :]
            label_p = label[x1:x2, y1:y2]
        except ValueError:
            # Image was too small for the window, try another one
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