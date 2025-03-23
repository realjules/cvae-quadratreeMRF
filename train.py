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
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 0
    
    for e in tqdm(range(1, epochs + 1), desc="Epochs"):
        # Training mode
        net.train()
        
        # Train with labeled data
        for batch_idx, (data, target) in enumerate(tqdm(labeled_loader, desc=f"Labeled data (Epoch {e})", leave=False)):
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
            
            # Record loss
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_-100):iter_+1])
            
            # Visualize progress
            if iter_ % 100 == 0:
                # Display current results
                with torch.no_grad():
                    # Get segmentation prediction
                    if 'final_segmentation' in outputs:
                        pred = outputs['final_segmentation'][0].argmax(dim=0).cpu().numpy()
                    else:
                        pred = outputs['hierarchical_segmentations'][-1][0].argmax(dim=0).cpu().numpy()
                    
                    # Convert to visualizable format
                    rgb = np.asarray(255 * np.transpose(data.cpu().numpy()[0], (1,2,0)), dtype='uint8')
                    gt = target.cpu().numpy()[0]
                    
                    # Calculate accuracy
                    acc = accuracy(pred, gt)
                    
                    # Print current stats
                    print(f'Epoch {e}/{epochs} [{batch_idx}/{len(labeled_loader)} ({100*batch_idx/len(labeled_loader):.0f}%)]')
                    print(f'Total Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%')
                    
                    for component, value in loss_components.items():
                        print(f'{component}: {value.item():.4f}')
                    
                    # Plot loss curve
                    plt.figure(figsize=(10, 4))
                    plt.plot(mean_losses[:iter_+1])
                    plt.title('Mean Loss')
                    plt.grid(True)
                    plt.savefig(f"{output_path}/loss_curve.png")
                    plt.close()
                    
                    # Visualize predictions
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax1.imshow(rgb)
                    ax1.set_title('RGB Input')
                    ax2.imshow(convert_to_color(gt))
                    ax2.set_title('Ground Truth')
                    ax3.imshow(convert_to_color(pred))
                    ax3.set_title('Prediction')
                    plt.savefig(f"{output_path}/prediction_epoch{e}_iter{iter_}.png")
                    plt.close()
            
            iter_ += 1
        
        # Train with unlabeled data if available
        if unlabeled_loader is not None:
            for batch_idx, (data, _) in enumerate(tqdm(unlabeled_loader, desc=f"Unlabeled data (Epoch {e})", leave=False)):
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
                
                # Record loss (but don't visualize for unlabeled data)
                losses[iter_] = loss.item()
                mean_losses[iter_] = np.mean(losses[max(0, iter_-100):iter_+1])
                
                if batch_idx % 100 == 0:
                    print(f'Unsupervised Loss: {loss.item():.4f}')
                    for component, value in loss_components.items():
                        print(f'{component}: {value.item():.4f}')
                
                iter_ += 1
        
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