# HVS-Net/train.py

"""
This is the main entry point for training the HVS-Net.

This script will:
1.  Parse command-line arguments (e.g., path to config file).
2.  Load the configuration using the utility from utils/config_loader.py.
3.  Initialize the dataset and dataloaders.
4.  Initialize the HVSTrainer.
5.  Start the training process.
"""

import argparse
from utils.config_loader import load_config
from core.trainer import HVSTrainer
from data.dataset import ISPRSDataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train HVS-Net')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    config = load_config(args.config)

    # TODO: Initialize datasets and dataloaders
    # train_dataset = ISPRSDataset(config, mode='train')
    # train_loader = DataLoader(train_dataset, ...)

    # TODO: Initialize and run the trainer
    # trainer = HVSTrainer(config)
    # trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
