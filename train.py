# -*- coding: utf-8 -*-
"""
Training function for Semi-Supervised Hierarchical PGM with Contrastive Learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
from utils.utils_dataset import convert_to_color
from utils.utils import accuracy


def train(net, criterion, optimizer, scheduler, labeled_loader, unlabeled_loader, 
          epochs, save_epoch, weights, batch_size, window_size, output_path):
    """
    Train the Semi-Supervised Hierarchical PGM model
    
    Parameters:
    -----------
    net: torch.nn.Module
        The hierarchical PGM model
    criterion: torch.nn.Module
        The loss function (HierarchicalPGMLoss)
    optimizer: torch.optim.Optimizer
        The optimizer
    scheduler: torch.optim.lr_scheduler
        Learning rate scheduler
    labeled_loader: torch.utils.data.DataLoader
        DataLoader for labeled data
    unlabeled_loader: torch.utils.data.DataLoader or None
        DataLoader for unlabeled data (None if all data is labeled)
    epochs: int
        Number of training epochs
    save_epoch: int
        Save model every N epochs
    weights: torch.Tensor
        Class weights for loss function
    batch_size: int
        Batch size
    window_size: tuple
        Window size for patches
    output_path: str
        Path to save model and results
    """
    # Loss tracking
    epoch_losses = []
    
    # Use tqdm only for epoch-level progress, not batch-level
    for e in tqdm(range(1, epochs + 1), desc="Epochs"):
        # Training mode
        net.train()
        
        # Track losses for this epoch
        epoch_supervised_losses = []
        epoch_unsupervised_losses = []
        epoch_total_loss = 0.0
        num_batches = 0
        
        # Train with labeled data - no tqdm progress bar here
        for batch_idx, (data, target) in enumerate(labeled_loader):
            if torch.cuda.is_available():
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)
            
            optimizer.zero_grad()
            
            # Forward pass with labeled data (supervised mode)
            outputs = net(data, mode='full')
            
            # Calculate loss
            loss, loss_components = criterion(outputs, target, mode='full')
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record loss for this batch
            epoch_supervised_losses.append(loss.item())
            epoch_total_loss += loss.item()
            num_batches += 1
        
        # Train with unlabeled data if available - no tqdm progress bar here
        if unlabeled_loader is not None:
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                if torch.cuda.is_available():
                    data = Variable(data.cuda())
                else:
                    data = Variable(data)
                
                optimizer.zero_grad()
                
                # Forward pass with unlabeled data (unsupervised mode)
                outputs = net(data, mode='unsupervised')
                
                # Calculate unsupervised loss
                loss, loss_components = criterion(outputs, targets=None, mode='unsupervised')
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Record loss for this batch
                epoch_unsupervised_losses.append(loss.item())
                epoch_total_loss += loss.item()
                num_batches += 1
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        
        # Evaluate model on a sample batch at the end of epoch
        if len(labeled_loader) > 0:
            with torch.no_grad():
                # Get a sample batch
                data, target = next(iter(labeled_loader))
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                    
                # Forward pass
                outputs = net(data, mode='full')
                
                # Get segmentation prediction for first image in batch
                if 'final_segmentation' in outputs:
                    pred = outputs['final_segmentation'][0].argmax(dim=0).cpu().numpy()
                else:
                    pred = outputs['hierarchical_segmentations'][-1][0].argmax(dim=0).cpu().numpy()
                
                # Convert to visualizable format
                rgb = np.asarray(255 * np.transpose(data.cpu().numpy()[0], (1,2,0)), dtype='uint8')
                gt = target.cpu().numpy()[0]
                
                # Calculate accuracy on this sample
                sample_acc = accuracy(pred, gt)
                
                # Report epoch-level stats
                print(f'Epoch {e}/{epochs} completed - Avg Loss: {avg_epoch_loss:.4f}, Sample Accuracy: {sample_acc:.2f}%')
                
                # Plot loss curve
                plt.figure(figsize=(10, 4))
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
                plt.title('Average Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(f"{output_path}/epoch_loss_curve.png")
                plt.close()
                
                # Only save sample visualization at end of epoch
                if e % save_epoch == 0 or e == epochs:
                    # Visualize predictions
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax1.imshow(rgb)
                    ax1.set_title('RGB Input')
                    ax2.imshow(convert_to_color(gt))
                    ax2.set_title('Ground Truth')
                    ax3.imshow(convert_to_color(pred))
                    ax3.set_title('Prediction')
                    plt.savefig(f"{output_path}/prediction_epoch{e}.png")
                    plt.close()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save model checkpoint
        if e % save_epoch == 0:
            torch.save(net.state_dict(), f'{output_path}/model_epoch{e}.pth')
            print(f"Model saved at epoch {e}")
    
    # Save final model
    torch.save(net.state_dict(), f'{output_path}/model_final.pth')
    print("Training completed. Final model saved.")