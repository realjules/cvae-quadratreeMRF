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
    parser = argparse.ArgumentParser(description='Train HVS-Net for semi-supervised semantic segmentation')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize datasets and dataloaders
    print("Initializing datasets...")
    train_dataset = ISPRSDataset(config, mode='train')
    val_dataset = ISPRSDataset(config, mode='validate')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize and run the trainer
    print("Initializing trainer...")
    trainer = HVSTrainer(config)
    trainer.train(train_loader, val_loader)

    print("Training finished.")

if __name__ == '__main__':
    main()