# -*- coding: utf-8 -*-
"""
Enhanced Training function for Semi-Supervised Hierarchical PGM with Contrastive Learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.autograd import Variable
from utils.utils_dataset import convert_to_color
from utils.utils import accuracy
from torch.cuda.amp import autocast, GradScaler
import time


def train(net, criterion, optimizer, scheduler, labeled_loader, unlabeled_loader, 
          val_loader, epochs, save_epoch, weights, batch_size, window_size, output_path):
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
    val_loader: torch.utils.data.DataLoader
        DataLoader for validation data
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
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Loss tracking
    epoch_losses = []
    epoch_val_losses = []
    epoch_val_acc = []
    
    # Track individual loss components
    component_losses = {
        'seg_loss': [],
        'hier_loss': [],
        'recon_loss': [],
        'kld_loss': [],
        'contrastive_loss': [],
        'hier_consistency_loss': []
    }
    
    # Gradient norm tracking for debugging
    grad_norms = []
    
    # Track best validation accuracy
    best_val_acc = 0.0
    
    # Early stopping
    patience = 10
    early_stop_counter = 0
    prev_val_loss = float('inf')
    
    # Use tqdm only for epoch-level progress, not batch-level
    for e in tqdm(range(1, epochs + 1), desc="Epochs"):
        # Training mode
        net.train()
        
        # Track losses for this epoch
        epoch_supervised_losses = []
        epoch_unsupervised_losses = []
        epoch_total_loss = 0.0
        batch_component_losses = {k: [] for k in component_losses.keys()}
        num_batches = 0
        
        # Train with labeled data - no tqdm progress bar here
        if len(labeled_loader) > 0:  # Only if we have labeled data
            for batch_idx, (data, target) in enumerate(labeled_loader):
                if torch.cuda.is_available():
                    data, target = Variable(data.cuda()), Variable(target.cuda())
                else:
                    data, target = Variable(data), Variable(target)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass with labeled data
                with autocast():
                    # Forward pass with labeled data (supervised mode)
                    outputs = net(data, mode='full')
                    
                    # Calculate loss
                    loss, loss_components = criterion(outputs, target, mode='full')
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Calculate gradient norm for debugging
                total_norm = 0
                for p in net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)
                
                # Gradient clipping with stricter limit to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                # Update with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                
                # Record loss for this batch
                epoch_supervised_losses.append(loss.item())
                epoch_total_loss += loss.item()
                num_batches += 1
                
                # Track individual loss components
                for component, value in loss_components.items():
                    if component in batch_component_losses:
                        batch_component_losses[component].append(value.item() if torch.is_tensor(value) else value)
        
        # Train with unlabeled data if available - no tqdm progress bar here
        if unlabeled_loader is not None:
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                if torch.cuda.is_available():
                    data = Variable(data.cuda())
                else:
                    data = Variable(data)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass with unlabeled data
                with autocast():
                    # Forward pass with unlabeled data (unsupervised mode)
                    outputs = net(data, mode='unsupervised')
                    
                    # Calculate unsupervised loss
                    loss, loss_components = criterion(outputs, targets=None, mode='unsupervised')
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping with stricter limit to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                # Update with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                
                # Record loss for this batch
                epoch_unsupervised_losses.append(loss.item())
                epoch_total_loss += loss.item()
                num_batches += 1
                
                # Track individual loss components
                for component, value in loss_components.items():
                    if component in batch_component_losses:
                        batch_component_losses[component].append(value.item() if torch.is_tensor(value) else value)
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        
        # Calculate average component losses
        for component, values in batch_component_losses.items():
            if values:
                component_losses[component].append(sum(values) / len(values))
            else:
                component_losses[component].append(0)
        
        # Validation step at the end of each epoch (if we have validation data)
        if len(val_loader) > 0:
            val_loss, val_acc = validate(net, criterion, val_loader)
            epoch_val_losses.append(val_loss)
            epoch_val_acc.append(val_acc)
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(net.state_dict(), f'{output_path}/model_best.pth')
                print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
                early_stop_counter = 0  # Reset early stopping counter
            else:
                early_stop_counter += 1
                
            # Early stopping check
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
                
            # Update loss weights based on current epoch
            if hasattr(criterion, 'update_epoch'):
                criterion.update_epoch(e, epochs)
        else:
            # If no validation data, just use training loss
            val_loss = avg_epoch_loss
            val_acc = 0.0
            epoch_val_losses.append(val_loss)
            epoch_val_acc.append(val_acc)
        
        # Report epoch-level stats
        print(f'Epoch {e}/{epochs} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Plot losses
        plot_training_curves(epoch_losses, epoch_val_losses, epoch_val_acc, 
                           component_losses, grad_norms, output_path)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save model checkpoint
        if e % save_epoch == 0:
            torch.save(net.state_dict(), f'{output_path}/model_epoch{e}.pth')
            print(f"Model saved at epoch {e}")
        
        # Visualize a sample from validation set at regular intervals
        if e % save_epoch == 0 or e == epochs:
            visualize_sample(net, val_loader, e, output_path)
    
    # Save final model
    torch.save(net.state_dict(), f'{output_path}/model_final.pth')
    print("Training completed. Final model saved.")


def validate(net, criterion, val_loader):
    """
    Validate the model on the validation set with mixed precision
    
    Parameters:
    -----------
    net: torch.nn.Module
        The hierarchical PGM model
    criterion: torch.nn.Module
        The loss function (HierarchicalPGMLoss)
    val_loader: torch.utils.data.DataLoader
        DataLoader for validation data
        
    Returns:
    --------
    avg_val_loss: float
        Average validation loss
    avg_val_acc: float
        Average validation accuracy (%)
    """
    net.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            # Use mixed precision for validation as well
            with autocast():
                # Forward pass in inference mode
                outputs = net(data, mode='inference')
                
                # Calculate loss
                loss, _ = criterion(outputs, target, mode='supervised')
            
            val_loss += loss.item()
            
            # Calculate accuracy and IoU
            pred = outputs['final_segmentation'].argmax(dim=1).cpu().numpy()
            target_np = target.cpu().numpy()
            
            # Calculate batch accuracy and IoU
            batch_acc = 0
            batch_iou = 0
            for i in range(len(pred)):
                batch_acc += accuracy(pred[i], target_np[i])
                batch_iou += calculate_miou(pred[i], target_np[i], n_classes=outputs['final_segmentation'].size(1))
            
            batch_acc /= len(pred)
            batch_iou /= len(pred)
            
            val_acc += batch_acc
            val_iou += batch_iou
            num_batches += 1
    
    # Calculate averages
    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
    avg_val_acc = val_acc / num_batches if num_batches > 0 else 0
    avg_val_iou = val_iou / num_batches if num_batches > 0 else 0
    
    print(f"Validation IoU: {avg_val_iou:.4f}")
    
    return avg_val_loss, avg_val_acc

def calculate_miou(pred, target, n_classes=6, ignore_index=6):
    """
    Calculate mean IoU for evaluation
    
    Parameters:
    -----------
    pred: numpy array
        Prediction mask
    target: numpy array
        Ground truth mask
    n_classes: int
        Number of classes
    ignore_index: int
        Index to ignore in calculation
        
    Returns:
    --------
    miou: float
        Mean IoU score
    """
    ious = []
    
    # Calculate IoU for each class
    for cls in range(n_classes):
        if cls == ignore_index:
            continue
            
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        # Skip if class is not present in ground truth
        if target_mask.sum() == 0:
            continue
            
        # Calculate intersection and union
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    # Return mean IoU
    return sum(ious) / len(ious) if ious else 0


def plot_training_curves(train_losses, val_losses, val_acc, component_losses, grad_norms, output_path):
    """
    Plot training curves
    
    Parameters:
    -----------
    train_losses: list
        List of training losses per epoch
    val_losses: list
        List of validation losses per epoch
    val_acc: list
        List of validation accuracies per epoch
    component_losses: dict
        Dictionary of individual loss components per epoch
    grad_norms: list
        List of gradient norms per batch
    output_path: str
        Path to save plots
    """
    # Skip if no data
    if len(train_losses) == 0:
        print("No training data to plot")
        return
    
    # Plot combined loss curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    if len(val_losses) > 0:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(val_acc) > 0:
        plt.plot(range(1, len(val_acc) + 1), val_acc)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No validation data", horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/training_curves.png")
    plt.close()
    
    # Plot individual loss components
    plt.figure(figsize=(15, 10))
    plotted = 0
    for i, (component, values) in enumerate(component_losses.items()):
        if any(values):  # Only plot non-zero components
            plt.subplot(3, 2, i + 1)
            plt.plot(range(1, len(values) + 1), values)
            plt.title(f'{component}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plotted += 1
    
    if plotted == 0:
        plt.text(0.5, 0.5, "No loss components to plot", horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/component_losses.png")
    plt.close()
    
    # Plot gradient norms
    if len(grad_norms) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(grad_norms)
        plt.title('Gradient Norms')
        plt.xlabel('Batch')
        plt.ylabel('Norm')
        plt.grid(True)
        plt.savefig(f"{output_path}/gradient_norms.png")
        plt.close()
    else:
        print("No gradient norms to plot")


def visualize_sample(net, data_loader, epoch, output_path):
    """
    Visualize a sample from the data loader
    
    Parameters:
    -----------
    net: torch.nn.Module
        The hierarchical PGM model
    data_loader: torch.utils.data.DataLoader
        DataLoader to get samples from
    epoch: int
        Current epoch number
    output_path: str
        Path to save visualizations
    """
    net.eval()
    
    # Skip visualization if no data
    if len(data_loader) == 0:
        print("No data available for visualization")
        return
    
    with torch.no_grad():
        try:
            # Get a sample batch
            data, target = next(iter(data_loader))
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            # Forward pass
            outputs = net(data, mode='full')
            
            # Take the first image in the batch
            img = data[0].cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # Convert from CxHxW to HxWxC
            img = (img * 255).astype(np.uint8)  # Assuming data is normalized to [0, 1]
            
            gt = target[0].cpu().numpy()
            
            # Get final segmentation
            pred = outputs['final_segmentation'][0].argmax(dim=0).cpu().numpy()
            
            # Visualize
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(convert_to_color(gt))
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(convert_to_color(pred))
            plt.title('Prediction')
            plt.axis('off')
            
            plt.savefig(f"{output_path}/sample_epoch_{epoch}.png")
            plt.close()
        except StopIteration:
            print("No data available for visualization")
        except Exception as e:
            print(f"Error during visualization: {e}")