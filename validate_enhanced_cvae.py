#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for Enhanced CVAE model
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
import cv2

from net.enhanced_cvae import EnhancedCVAE
from run_enhanced_cvae import UnsupervisedDataset

def load_model(model_path, input_channels=3, latent_dim=256):
    """
    Load a trained Enhanced CVAE model
    """
    # Create model instance
    model = EnhancedCVAE(input_channels=input_channels, latent_dim=latent_dim)
    
    # Load weights with weights_only=True to address the FutureWarning
    model.load_state_dict(torch.load(
        model_path, 
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        weights_only=True  # Only load the weights, not the entire pickle
    ))
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Set to evaluation mode
    model.eval()
    
    return model

def calculate_metrics(original, reconstructed):
    """
    Calculate image quality metrics between original and reconstructed images
    """
    # Convert from torch tensor to numpy array if needed
    if torch.is_tensor(original):
        original = original.cpu().detach().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().detach().numpy()
    
    # Ensure arrays are in the correct shape (H,W,C)
    if original.shape[0] == 3:  # If in format (C,H,W)
        original = np.transpose(original, (1, 2, 0))
    if reconstructed.shape[0] == 3:  # If in format (C,H,W)
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
    
    # Clip values to valid range [0,1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Calculate MSE manually (sklearn's doesn't handle 3D arrays well)
    mse = np.mean((original - reconstructed) ** 2)
    
    # Calculate PSNR
    max_pixel = 1.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse) if mse > 0 else float('inf')
    
    # Calculate SSIM (for each channel and then average)
    ssim_value = 0
    for i in range(original.shape[2]):
        ssim_value += structural_similarity(
            original[:,:,i], 
            reconstructed[:,:,i],
            data_range=1.0
        )
    ssim_value /= original.shape[2]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_value
    }

def validate_model(model, data_loader, output_path, num_samples=16):
    """
    Validate the Enhanced CVAE model and generate reconstruction visualizations
    """
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Metrics storage
    metrics_list = []
    
    # Process a subset of validation images
    with torch.no_grad():
        batch_idx = 0
        for data, _ in tqdm(data_loader, desc="Validating"):
            if torch.cuda.is_available():
                data = data.cuda()
            
            # Get model output
            outputs = model(data)
            reconstructions = outputs['reconstruction']
            
            # Calculate metrics for each image in batch
            for i in range(data.size(0)):
                metrics = calculate_metrics(data[i], reconstructions[i])
                metrics_list.append(metrics)
            
            # Visualize first few batches
            if batch_idx < num_samples // data.size(0) + 1:
                visualize_batch(data, reconstructions, batch_idx, output_path)
            
            batch_idx += 1
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'psnr': np.mean([m['psnr'] for m in metrics_list]),
        'ssim': np.mean([m['ssim'] for m in metrics_list])
    }
    
    # Print and save metrics
    print(f"Average MSE: {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_path, 'validation_metrics.txt'), 'w') as f:
        f.write(f"Average MSE: {avg_metrics['mse']:.6f}\n")
        f.write(f"Average PSNR: {avg_metrics['psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {avg_metrics['ssim']:.4f}\n")
    
    # Create metrics histogram
    plot_metrics_histogram(metrics_list, output_path)
    
    return avg_metrics

def visualize_batch(originals, reconstructions, batch_idx, output_path):
    """
    Visualize a batch of original and reconstructed images
    """
    batch_size = originals.size(0)
    n_samples = min(8, batch_size)  # Display up to 8 samples per batch
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    for i in range(n_samples):
        # Original
        orig = originals[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].set_title("Original" if i == 0 else "")
        axes[0, i].axis('off')
        
        # Reconstruction
        recon = reconstructions[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(np.clip(recon, 0, 1))
        axes[1, i].set_title("Reconstruction" if i == 0 else "")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/reconstructions_batch{batch_idx}.png")
    plt.close()

def plot_metrics_histogram(metrics_list, output_path):
    """
    Plot histograms of the metrics
    """
    mse_values = [m['mse'] for m in metrics_list]
    psnr_values = [m['psnr'] for m in metrics_list]
    ssim_values = [m['ssim'] for m in metrics_list]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE histogram
    axes[0].hist(mse_values, bins=20, alpha=0.7, color='blue')
    axes[0].set_title('MSE Distribution')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # PSNR histogram
    axes[1].hist(psnr_values, bins=20, alpha=0.7, color='green')
    axes[1].set_title('PSNR Distribution (dB)')
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[2].hist(ssim_values, bins=20, alpha=0.7, color='red')
    axes[2].set_title('SSIM Distribution')
    axes[2].set_xlabel('SSIM')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/metrics_distribution.png")
    plt.close()

def generate_latent_space_visualization(model, data_loader, output_path):
    """
    Generate t-SNE visualization of the latent space
    """
    model.eval()
    
    # Sample latent vectors
    latent_vectors = []
    original_images = []
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Generating latent vectors"):
            if torch.cuda.is_available():
                data = data.cuda()
            
            # Encode images to get latent vectors
            outputs = model(data)
            z = outputs['z']
            
            # Store batch of latent vectors and original images
            latent_vectors.append(z.cpu().numpy())
            original_images.append(data.cpu().numpy())
            
            # Limit to a reasonable number for visualization
            if len(latent_vectors) * data.size(0) >= 500:
                break
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    original_images = np.concatenate(original_images, axis=0)
    
    # Reduce dimensionality for visualization using t-SNE
    try:
        from sklearn.manifold import TSNE
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], s=5, alpha=0.6)
        plt.title('t-SNE Visualization of Latent Space')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}/latent_space_tsne.png")
        plt.close()
        
        # Plot grid of reconstructed images organized by t-SNE coordinates
        plot_latent_space_samples(model, latent_vectors, latent_2d, original_images, output_path)
        
    except ImportError:
        print("scikit-learn not available for t-SNE visualization")

def plot_latent_space_samples(model, latent_vectors, latent_2d, original_images, output_path):
    """
    Plot sample images arranged by their position in the latent space
    """
    # Select a grid of points in the t-SNE space
    x_min, x_max = latent_2d[:, 0].min(), latent_2d[:, 0].max()
    y_min, y_max = latent_2d[:, 1].min(), latent_2d[:, 1].max()
    
    grid_size = 5  # 5x5 grid
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    
    # Find nearest points in the latent space for each grid point
    grid_indices = []
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            # Find closest point
            distances = np.sqrt((latent_2d[:, 0] - x)**2 + (latent_2d[:, 1] - y)**2)
            idx = np.argmin(distances)
            grid_indices.append(idx)
    
    # Create figure with grid of original images only
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    
    for i, idx in enumerate(grid_indices):
        row = i // grid_size
        col = i % grid_size
        
        # Get original image
        orig_img = original_images[idx]
        
        # Display original image
        if len(orig_img.shape) == 3 and orig_img.shape[0] == 3:
            # Convert from CHW to HWC format
            img_display = np.transpose(orig_img, (1, 2, 0))
        else:
            img_display = orig_img
            
        axes[row, col].imshow(np.clip(img_display, 0, 1))
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/latent_space_grid.png")
    plt.close()

def interpolate_latent_space(model, data_loader, output_path, num_steps=10):
    """
    Generate interpolations between two random points in the latent space
    """
    model.eval()
    
    # Skip interpolation since it requires full encoder-decoder pipeline
    # and we can't directly decode without encoder features
    print("Skipping latent space interpolation due to model architecture constraints.")
    
    # Just save some original images instead
    with torch.no_grad():
        for data, _ in data_loader:
            if data.size(0) >= 4:
                # Get a few sample images
                sample_images = data[:4].cpu().numpy()
                
                # Create a simple visualization
                fig, axes = plt.subplots(1, 4, figsize=(10, 3))
                
                for i in range(4):
                    img = np.transpose(sample_images[i], (1, 2, 0))
                    axes[i].imshow(np.clip(img, 0, 1))
                    axes[i].set_title(f"Sample {i+1}")
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{output_path}/sample_images.png")
                plt.close()
                break
    
    return

def main():
    """Main function for validation"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Validate Enhanced CVAE model')
    parser.add_argument('-m', '--model', help='Path to the model checkpoint', 
                      default="./output/Enhanced-CVAE/model_best.pth")
    parser.add_argument('-i', '--input', help='Path of input directory', 
                      default="./input/")
    parser.add_argument('-o', '--output', help='Path for validation output', 
                      default="./output/Enhanced-CVAE-Validation/")
    parser.add_argument('-w', '--window', nargs=2, type=int, default=[256, 256],
                      help='Dimension of image patches')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-ld', '--latent_dim', default=256, type=int, help='Latent dimension size')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Parameters
    WINDOW_SIZE = tuple(args.window)
    IN_CHANNELS = 3
    LATENT_DIM = args.latent_dim
    FOLDER = args.input
    OUTPUT_FOLDER = args.output
    batch_size = args.batch_size
    
    # Data paths
    DATA_FOLDER = f"{FOLDER}/top/top_mosaic_09cm_area{{}}.tif"
    
    # Load the model
    model = load_model(args.model, input_channels=IN_CHANNELS, latent_dim=LATENT_DIM)
    
    # Use all IDs for validation
    # Focus on IDs not used during training
    val_ids = ['5', '15', '21', '30']
    
    # Create dataset without data augmentation
    val_set = UnsupervisedDataset(val_ids, data_files=DATA_FOLDER, window_size=WINDOW_SIZE, augment=False)
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    print("Starting validation...")
    
    # Validate the model
    metrics = validate_model(model, val_loader, OUTPUT_FOLDER)
    
    # Generate latent space visualization
    print("Generating latent space visualization...")
    generate_latent_space_visualization(model, val_loader, OUTPUT_FOLDER)
    
    # Generate latent space interpolation
    print("Generating latent space interpolation...")
    interpolate_latent_space(model, val_loader, OUTPUT_FOLDER)
    
    print(f"Validation completed! Results saved to {OUTPUT_FOLDER}")
    print(f"MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")

if __name__ == "__main__":
    main()