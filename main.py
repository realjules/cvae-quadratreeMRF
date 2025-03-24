# -*- coding: utf-8 -*-
"""
Main script for Semi-Supervised Hierarchical PGM with Contrastive Learning
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from datetime import datetime

from net.net import HierarchicalPGM
from net.loss import HierarchicalPGMLoss
from dataset.dataset import ISPRS_dataset
from dataset.unsupervised_dataset import ISPRS_unsupervised_dataset
from utils.utils_dataset import *
from utils.utils import *
from utils.export_result import *
from train import train
from test_network import test


def main(args):
    # Parameters
    WINDOW_SIZE = args.window
    IN_CHANNELS = 3
    LATENT_DIM = args.latent_dim
    MAX_DEPTH = args.max_depth
    
    FOLDER = args.input
    OUTPUT_FOLDER = args.output
    batch_size = args.batch_size
    epochs = args.epochs
    save_epoch = args.save_epoch
    
    # Define labels
    labels = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
    N_CLASSES = len(labels)
    WEIGHTS = torch.ones(N_CLASSES)
    CACHE = True
    
    # Data paths
    DATA_FOLDER = f"{FOLDER}/top/top_mosaic_09cm_area{{}}.tif"
    LABEL_FOLDER = f"{FOLDER}/gt/top_mosaic_09cm_area{{}}.tif"
    ERODED_FOLDER = f"{FOLDER}/gt_eroded/top_mosaic_09cm_area{{}}_noBoundary.tif"
    
    # Initialize the model
    net = HierarchicalPGM(n_channels=IN_CHANNELS, n_classes=N_CLASSES, 
                          latent_dim=LATENT_DIM, max_depth=MAX_DEPTH)
    
    # Initialize loss function
    criterion = HierarchicalPGMLoss(N_CLASSES, weights=WEIGHTS, 
                                    kld_weight=args.kld_weight,
                                    contrastive_weight=args.contrastive_weight,
                                    consistency_weight=args.consistency_weight)
    
    # Set up optimizer
    base_lr = args.base_lr
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0005)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
    
    # Use GPU if available
    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()
        WEIGHTS = WEIGHTS.cuda()
    
    # Define train and test data
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '15', '21', '30']
    
    # For semi-supervised learning, we use a portion of labeled data
    labeled_percentage = args.labeled_percentage
    if labeled_percentage < 100:
        # Use only a subset of training data as labeled
        num_labeled = int(len(train_ids) * labeled_percentage / 100)
        labeled_ids = train_ids[:num_labeled]
        unlabeled_ids = train_ids[num_labeled:]
        print(f"Using {len(labeled_ids)} tiles as labeled data: {labeled_ids}")
        print(f"Using {len(unlabeled_ids)} tiles as unlabeled data: {unlabeled_ids}")
    else:
        labeled_ids = train_ids
        unlabeled_ids = []
        print(f"Using all {len(labeled_ids)} tiles as labeled data")
    
    # Create datasets
    labeled_set = ISPRS_dataset(labeled_ids, ids_type='TRAIN', gt_type=args.gt_type,
                                gt_modification=disk(args.ero_disk),
                                data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                                window_size=WINDOW_SIZE, cache=CACHE)
    
    test_set = ISPRS_dataset(test_ids, ids_type='TEST', gt_type=args.gt_type,
                             gt_modification=disk(args.ero_disk),
                             data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                             window_size=WINDOW_SIZE, cache=CACHE)
    
    # Create unlabeled dataset if needed
    if unlabeled_ids:
        unlabeled_set = ISPRS_unsupervised_dataset(unlabeled_ids, data_files=DATA_FOLDER,
                                                 window_size=WINDOW_SIZE, cache=CACHE)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size)
    else:
        unlabeled_loader = None
    
    # Create data loaders
    labeled_loader = torch.utils.data.DataLoader(labeled_set, batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size)
    
    # Set up experiment name and output location
    experiment_name = args.experiment_name
    output_path = set_output_location(experiment_name, OUTPUT_FOLDER)
    
    # Train or load the model
    if args.retrain:
        train(net, criterion, optimizer, scheduler, labeled_loader, unlabeled_loader,
             epochs, save_epoch, WEIGHTS, batch_size, WINDOW_SIZE, output_path)
    else:
        model_weights = args.model_weights
        net.load_state_dict(torch.load(model_weights))
    
    # Test the model
    test_images = (1/255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id))) for id in test_ids)
    
    stride = args.stride
    acc_test, all_preds, all_gts = test(net, test_ids, test_images, test_labels, eroded_labels,
                                       labels, stride, batch_size, window_size=WINDOW_SIZE)
    
    # Export results
    title = f"Results for Semi-Supervised Hierarchical PGM - {experiment_name}"
    export_results(all_preds, all_gts, OUTPUT_FOLDER, experiment_name,
                  confusionMat=True, prodAccuracy=True, averageAccuracy=True,
                  kappaCoeff=True, title=title)
    
    # Save segmentation results
    for pred, ids in zip(all_preds, test_ids):
        img = convert_to_color(pred)
        plt.imshow(img)
        plt.show()
        io.imsave(f"{output_path}/segmentation_result_area{ids}.png", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semi-Supervised Hierarchical PGM with Contrastive Learning')
    parser.add_argument('-i', '--input', help='Path of input directory', 
                        metavar='INPUT_DIR_PATH', default="./input/")
    parser.add_argument('-o', '--output', help='Path of output directory',
                        metavar='OUTPUT_DIR_PATH', default="./output/")
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain the model')
    parser.add_argument('-w', '--window', default=(256, 256), type=tuple, nargs='?',
                        help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('-d', '--ero_disk', default=8, type=int, help='Size of erosion disk')
    parser.add_argument('-g', '--gt_type', required=True, choices=['ero', 'full', 'conncomp'],
                        help='Ground truth type')
    parser.add_argument('-exp', '--experiment_name', default='hierarchical_pgm_experiment', 
                        type=str, help='Experiment name')
    parser.add_argument('-lr', '--base_lr', default=0.0001, type=float, help='Base learning rate')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('-se', '--save_epoch', default=5, type=int, help='Save model every N epochs')
    parser.add_argument('-s', '--stride', default=32, type=int, help='Stride for testing')
    parser.add_argument('-ld', '--latent_dim', default=128, type=int, help='Latent dimension size')
    parser.add_argument('-md', '--max_depth', default=4, type=int, help='Maximum depth of quadtree')
    parser.add_argument('-kw', '--kld_weight', default=0.005, type=float, help='KL divergence weight')
    parser.add_argument('-cw', '--contrastive_weight', default=0.1, type=float, help='Contrastive loss weight')
    parser.add_argument('-hcw', '--consistency_weight', default=0.05, type=float, help='Hierarchical consistency weight')
    parser.add_argument('-lp', '--labeled_percentage', default=100, type=int, help='Percentage of labeled data to use')
    parser.add_argument('-mw', '--model_weights', default='./output/model_final.pth', type=str, help='Path to pretrained weights')
    
    args = parser.parse_args()
    main(args)