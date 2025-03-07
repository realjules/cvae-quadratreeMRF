# Notebook script to setup and run main.py
# This is designed to be run in a Kaggle or Colab notebook

# Cell 1: Clone the repository
"""
# Remove old version if exists
!rm -rf cvae-quadratreeMRF 
# Clone the repo
!git clone https://github.com/realjules/cvae-quadratreeMRF.git
# Add to path
import sys
sys.path.append('/kaggle/working/cvae-quadratreeMRF')
"""

# Cell 2: Install dependencies
"""
# Install required packages
!pip install -r /kaggle/working/cvae-quadratreeMRF/requirements.txt
"""

# Cell 3: Import dependencies and run main
"""
import sys
import os
import argparse
from main import main

# Create args object with default values
parser = argparse.ArgumentParser()
args = parser.parse_args([])

# Set required argument
args.gt_type = 'ero'  # Options: 'ero', 'full', 'conncomp'

# Set optional arguments
args.input = '/kaggle/working/cvae-quadratreeMRF/input/'
args.output = '/kaggle/working/cvae-quadratreeMRF/output/'
args.retrain = True
args.window = (256, 256)
args.batch_size = 10
args.ero_disk = 8
args.experiment_name = 'kaggle_experiment'
args.base_lr = 0.01
args.epochs = 30
args.save_epoch = 10

# Run the main function with the args
main(args)
"""