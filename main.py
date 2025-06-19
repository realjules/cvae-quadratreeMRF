#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified main script that works with the cleaned codebase
"""

import torch
import argparse
import numpy as np
import os
import sys
from train import main as run_simplified_main

def main():
    """Main function that delegates to the working simplified implementation"""
    parser = argparse.ArgumentParser(description='Semi-Supervised Hierarchical PGM with Contrastive Learning')
    
    # Use the same arguments as run_simplified_mrf.py
    parser.add_argument('-i', '--input', help='Path of input directory', 
                      default="./input/")
    parser.add_argument('-o', '--output', help='Path of output directory',
                      default="./output/SimplifiedMRF/")
    parser.add_argument('-c', '--cvae', help='Path to pre-trained CVAE model',
                      default="./output/Enhanced-CVAE/model_best.pth")
    parser.add_argument('-w', '--window', nargs=2, type=int, default=[256, 256],
                      help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('-ld', '--latent_dim', default=256, type=int, help='CVAE latent dimension')
    parser.add_argument('-nc', '--n_classes', default=6, type=int, help='Number of classes')
    parser.add_argument('-m', '--mode', choices=['train', 'validate'], default='train',
                      help='Mode: train or validate')
    parser.add_argument('-cp', '--checkpoint', help='Path to model checkpoint for validation',
                      default=None)
    parser.add_argument('-lp', '--labeled_percentage', default=100, type=int, 
                      help='Percentage of labeled data to use (10, 30, 75, 100)')
    parser.add_argument('-s', '--seed', default=42, type=int,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Call the working implementation
    sys.argv = ['main.py'] + [
        '-i', args.input,
        '-o', args.output,
        '-c', args.cvae,
        '-w'] + [str(x) for x in args.window] + [
        '-b', str(args.batch_size),
        '-lr', str(args.learning_rate),
        '-e', str(args.epochs),
        '-ld', str(args.latent_dim),
        '-nc', str(args.n_classes),
        '-m', args.mode,
        '-lp', str(args.labeled_percentage),
        '-s', str(args.seed)
    ]
    
    if args.checkpoint:
        sys.argv.extend(['-cp', args.checkpoint])
    
    # Call the working main function
    run_simplified_main()

if __name__ == "__main__":
    main()