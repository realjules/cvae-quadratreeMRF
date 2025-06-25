# HVS-Net/utils/config_loader.py

"""
This file will contain a helper function to load the YAML configuration file.
This keeps our main training script clean and organized.
"""

import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
