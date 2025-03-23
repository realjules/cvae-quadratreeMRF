# -*- coding: utf-8 -*-
"""
Dataset class for unlabeled data in semi-supervised learning
"""

import torch
import random
import numpy as np
import os
from skimage import io
from utils.utils import get_random_pos


class ISPRS_unsupervised_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files, window_size, cache=False, augmentation=False):
        super(ISPRS_unsupervised_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        self.window_size = window_size
        
        # List of files
        self.data_files = [data_files.format(id) for id in ids]

        # Check : raise an error if some files do not exist
        for f in self.data_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dict
        self.data_cache_ = {}
    
    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            data = 1/255 * data
            
            if self.cache:
                self.data_cache_[random_idx] = data
        
        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2,y1:y2] 
        
        # Data augmentation
        if self.augmentation:
            data_p = self.data_augmentation(data_p)[0]

        # Return the torch.Tensor values (with a dummy target)
        return torch.from_numpy(data_p), torch.zeros(1, dtype=torch.long)