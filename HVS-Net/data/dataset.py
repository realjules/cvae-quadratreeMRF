# HVS-Net/data/dataset.py

"""
This file will contain the PyTorch Dataset class for loading the ISPRS data.

It will handle:
1.  Finding and loading the image and ground truth tiles.
2.  A robust data augmentation pipeline using Albumentations for the segmentation task.
3.  A separate, strong augmentation pipeline for the unsupervised consistency loss.
4.  Returning both labeled and unlabeled data as needed by the trainer.
"""

from torch.utils.data import Dataset

class ISPRSDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        # TODO: Implement dataset loading and augmentation pipelines.
        pass

    def __len__(self):
        # TODO: Return the number of samples in the dataset.
        pass

    def __getitem__(self, idx):
        # TODO: Return a single data sample (image, mask, etc.).
        pass
