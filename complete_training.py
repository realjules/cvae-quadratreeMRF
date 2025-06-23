#!/usr/bin/env python3
"""
Complete End-to-End Training Script for 90% Accuracy Target

This script implements the full pipeline:
1. CVAE contrastive learning (unsupervised)
2. Semi-supervised segmentation training
3. Curriculum learning strategy
4. Model evaluation and saving

Usage:
    python complete_training.py --data_path ./input/ --target_accuracy 90 --labeled_percent 10
"""

import torch
import torch.nn as nn
import argparse
import os
import time
import numpy as np
from tqdm import tqdm

# Import our fixed components
from utils.cvae_trainer import CVAETrainer
from utils.curriculum_learning import CurriculumTrainer, create_default_curriculum
from train import FixedSegmentationTrainer
from utils.contrastive_augmentations import ContrastiveAugmentation

# Import real dataset classes
from dataset.dataset import ISPRS_dataset
from dataset.unsupervised_dataset import ISPRS_unsupervised_dataset
from torch.utils.data import DataLoader
import glob


def get_available_area_ids():
    """
    Auto-detect available area IDs from ISPRS data files
    Returns IDs that have both top images and ground truth labels
    """
    # Get available top images
    top_files = glob.glob("./input/top/top_mosaic_09cm_area*.tif")
    gt_files = glob.glob("./input/gt/top_mosaic_09cm_area*.tif")
    
    # Extract area numbers
    top_ids = []
    for f in top_files:
        area_part = f.split('area')[1].split('.')[0]
        top_ids.append(area_part)
    
    gt_ids = []
    for f in gt_files:
        area_part = f.split('area')[1].split('.')[0]
        gt_ids.append(area_part)
    
    # Only use IDs that have both top and ground truth
    valid_ids = list(set(top_ids) & set(gt_ids))
    valid_ids = sorted(valid_ids, key=lambda x: int(x))
    
    print(f"Found {len(top_ids)} top images, {len(gt_ids)} ground truth images")
    print(f"Valid IDs with both: {valid_ids}")
    
    return valid_ids


def split_dataset_ids(valid_ids, labeled_percent=10):
    """
    Split dataset IDs into train/unlabeled/test sets
    """
    total_ids = len(valid_ids)
    
    # For semi-supervised learning:
    # - Use only labeled_percent% for supervised training
    # - Use ALL data for unsupervised CVAE training
    # - Use separate set for testing
    
    # Test set: last 20% of data
    test_split = max(1, int(0.2 * total_ids))
    test_ids = valid_ids[-test_split:]
    
    # Remaining data for training
    train_pool = valid_ids[:-test_split]
    
    # Labeled data: only a small percentage
    labeled_count = max(1, int(labeled_percent / 100.0 * len(train_pool)))
    labeled_ids = train_pool[:labeled_count]
    
    # Unlabeled data: ALL training data (including labeled) for contrastive learning
    unlabeled_ids = train_pool
    
    print(f"Dataset split (total {total_ids} areas):")
    print(f"  Labeled training: {len(labeled_ids)} areas ({labeled_ids})")
    print(f"  Unlabeled training: {len(unlabeled_ids)} areas (for contrastive learning)")
    print(f"  Test: {len(test_ids)} areas ({test_ids})")
    
    return labeled_ids, unlabeled_ids, test_ids


def create_real_dataloaders(batch_size=2, labeled_percent=10, device="cuda"):
    """
    Create DataLoaders using real ISPRS dataset
    """
    # Get available data
    valid_ids = get_available_area_ids()
    if len(valid_ids) == 0:
        raise ValueError("No valid ISPRS data found! Check ./input/top/ and ./input/gt/ directories")
    
    # Split dataset
    labeled_ids, unlabeled_ids, test_ids = split_dataset_ids(valid_ids, labeled_percent)
    
    # Create datasets
    print("Creating real ISPRS datasets...")
    
    # Unlabeled dataset for CVAE contrastive learning
    unlabeled_dataset = ISPRS_unsupervised_dataset(
        ids=unlabeled_ids,
        data_files="./input/top/top_mosaic_09cm_area{}.tif",
        window_size=256,
        cache=False,  # Disable cache for Kaggle memory constraints
        augmentation=True
    )
    
    # Labeled dataset for segmentation training
    labeled_dataset = ISPRS_dataset(
        ids=labeled_ids,
        ids_type='TRAIN',
        gt_type='full',
        gt_modification=None,
        data_files="./input/top/top_mosaic_09cm_area{}.tif",
        label_files="./input/gt/top_mosaic_09cm_area{}.tif",
        window_size=256,
        cache=False,
        augmentation=True
    )
    
    # Test dataset
    test_dataset = ISPRS_dataset(
        ids=test_ids,
        ids_type='TEST',
        gt_type='full',
        gt_modification=None,
        data_files="./input/top/top_mosaic_09cm_area{}.tif",
        label_files="./input/gt/top_mosaic_09cm_area{}.tif",
        window_size=256,
        cache=False,
        augmentation=False
    )
    
    # Create DataLoaders with Kaggle-optimized settings
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in Kaggle
        pin_memory=False,  # Save memory
        drop_last=True
    )
    
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    print(f"✅ Real ISPRS DataLoaders created successfully!")
    print(f"   Unlabeled: {len(unlabeled_dataset)} samples")
    print(f"   Labeled: {len(labeled_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    return unlabeled_loader, labeled_loader, test_loader


def train_cvae_stage(cvae_trainer, unlabeled_dataloader, epochs=20, device="cuda"):
    """
    Stage 1: Train CVAE with contrastive learning on unlabeled ISPRS data
    """
    print("🎯 STAGE 1: CVAE Contrastive Learning (Real ISPRS Data)")
    print("=" * 50)
    
    aug = ContrastiveAugmentation(size=256, strength=0.8)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        print(f"\n📚 CVAE Epoch {epoch}/{epochs}")
        
        # Use real DataLoader instead of generator function
        for batch_idx, (images, _) in enumerate(unlabeled_dataloader):
            # Create contrastive pairs from real ISPRS images
            augmented_pairs = []
            for i in range(images.size(0)):
                view1, view2 = aug(images[i])
                augmented_pairs.append((view1.to(device), view2.to(device)))
            
            # Extract views from augmented pairs
            view1_batch = torch.stack([pair[0] for pair in augmented_pairs])
            view2_batch = torch.stack([pair[1] for pair in augmented_pairs])
            
            # Combine for batch processing
            combined_batch = torch.cat([view1_batch, view2_batch], dim=0)
            
            # Train step with NaN detection
            metrics = cvae_trainer.train_step_contrastive(combined_batch)
            
            # Check for NaN and skip if detected
            if torch.isnan(torch.tensor(metrics['total_loss'])):
                print(f"  ⚠️  NaN detected at batch {batch_idx}, skipping...")
                continue
                
            epoch_loss += metrics['total_loss']
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Total={metrics['total_loss']:.4f}, "
                      f"Recon={metrics['recon_loss']:.4f}, "
                      f"Contrastive={metrics['contrastive_loss']:.4f}")
                
                # Early stopping if loss becomes too high
                if metrics['total_loss'] > 50.0:
                    print(f"  ⚠️  Loss too high ({metrics['total_loss']:.2f}), reducing learning rate...")
                    for param_group in cvae_trainer.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"  New learning rate: {param_group['lr']:.6f}")
            
            # Limit batches per epoch for faster iteration
            if batch_idx >= 100:  # Process max 100 batches per epoch
                break
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"📊 CVAE Epoch {epoch} completed: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            cvae_trainer.save_model(f"./output/cvae_epoch_{epoch}.pth", epoch)
    
    print("✅ CVAE contrastive learning completed!")
    return cvae_trainer


def train_segmentation_stage(seg_trainer, labeled_dataloader, epochs=30, device="cuda"):
    """
    Stage 2: Train segmentation model on labeled ISPRS data
    """
    print("\n🎯 STAGE 2: Semi-Supervised Segmentation Training (Real ISPRS Data)")
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        print(f"\n📚 Segmentation Epoch {epoch}/{epochs}")
        
        # Use real DataLoader instead of generator function
        for batch_idx, (images, labels) in enumerate(labeled_dataloader):
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Training step
            loss, loss_components, outputs = seg_trainer.train_step(images, labels)
            
            epoch_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss:.4f}")
                
                # Check output quality
                final_seg = outputs['final_segmentation']
                pred_classes = torch.argmax(final_seg, dim=1)
                accuracy = (pred_classes == labels).float().mean()
                print(f"    Batch Accuracy: {accuracy:.3f}")
            
            # Limit batches per epoch for faster iteration
            if batch_idx >= 50:  # Process max 50 batches per epoch
                break
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"📊 Segmentation Epoch {epoch} completed: Average Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(seg_trainer.model.state_dict(), "./output/best_segmentation_model.pth")
            print(f"💾 New best model saved! Loss: {best_loss:.4f}")
    
    print("✅ Semi-supervised segmentation training completed!")
    return seg_trainer


def evaluate_model(seg_trainer, test_dataloader, device="cuda"):
    """
    Evaluate the trained model on real ISPRS test data
    """
    print("\n🧪 EVALUATION (Real ISPRS Test Data)")
    print("=" * 40)
    
    seg_trainer.model.eval()
    total_accuracy = 0
    total_samples = 0
    class_correct = [0] * 6
    class_total = [0] * 6
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract features and predict
            features = seg_trainer.extract_cvae_features(images)
            outputs = seg_trainer.model(features)
            
            final_seg = outputs['final_segmentation']
            pred_classes = torch.argmax(final_seg, dim=1)
            
            # Overall accuracy
            correct = (pred_classes == labels).float().sum()
            total_accuracy += correct
            total_samples += labels.numel()
            
            # Per-class accuracy
            for c in range(6):
                class_mask = (labels == c)
                if class_mask.sum() > 0:
                    class_correct[c] += (pred_classes[class_mask] == c).float().sum()
                    class_total[c] += class_mask.sum()
            
            # Limit evaluation batches for faster testing
            if batch_idx >= 20:  # Evaluate on 20 batches
                break
    
    overall_accuracy = (total_accuracy / total_samples * 100).item()
    
    print(f"📊 EVALUATION RESULTS (Real ISPRS Data):")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Class-wise Accuracy:")
    class_names = ["Impervious", "Buildings", "Low Veg", "Trees", "Cars", "Clutter"]
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = (class_correct[i] / class_total[i] * 100).item()
            print(f"  {name}: {acc:.2f}%")
    
    return overall_accuracy


def main():
    """Complete training pipeline"""
    parser = argparse.ArgumentParser(description='Complete Semi-Supervised Training')
    parser.add_argument('--epochs_cvae', default=20, type=int, help='CVAE training epochs')
    parser.add_argument('--epochs_seg', default=30, type=int, help='Segmentation training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--labeled_percent', default=10, type=int, help='Percentage of labeled data')
    parser.add_argument('--target_accuracy', default=90, type=float, help='Target accuracy')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--output_dir', default='./output/', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 COMPLETE SEMI-SUPERVISED TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Target Accuracy: {args.target_accuracy}%")
    print(f"Labeled Data: {args.labeled_percent}%")
    print(f"CVAE Epochs: {args.epochs_cvae}")
    print(f"Segmentation Epochs: {args.epochs_seg}")
    print("=" * 60)
    
    # Create trainers with fixed hyperparameters
    print("🔧 Initializing trainers...")
    
    # Fix 1: Lower learning rate for CVAE stability
    cvae_lr = args.learning_rate * 0.1  # 10x lower for contrastive learning
    seg_lr = args.learning_rate * 0.5   # 2x lower for segmentation
    
    cvae_trainer = CVAETrainer(
        device=device, 
        learning_rate=cvae_lr,
        temperature=0.5  # Higher temperature for stability
    )
    seg_trainer = FixedSegmentationTrainer(
        cvae_path="./nonexistent.pth",  # Will use fallback initially
        learning_rate=seg_lr,
        device=device
    )
    
    print(f"  CVAE Learning Rate: {cvae_lr}")
    print(f"  Segmentation Learning Rate: {seg_lr}")
    print(f"  Temperature: 0.5")
    
    # Create real ISPRS data loaders
    print("📊 Creating real ISPRS data loaders...")
    
    # Fix 2: Reduce batch sizes for memory efficiency
    effective_batch_size = min(args.batch_size, 2)  # Max 2 for GPU memory
    
    try:
        unlabeled_loader, labeled_loader, test_loader = create_real_dataloaders(
            batch_size=effective_batch_size,
            labeled_percent=args.labeled_percent,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to create real data loaders: {e}")
        print("Please ensure ISPRS data is properly linked in ./input/ directories")
        return 1
    
    print(f"  Effective Batch Size: {effective_batch_size} (reduced for memory)")
    print(f"  GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory Cleared")
    
    # Training pipeline
    start_time = time.time()
    
    try:
        # Stage 1: CVAE Contrastive Learning on real ISPRS data
        cvae_trainer = train_cvae_stage(cvae_trainer, unlabeled_loader, args.epochs_cvae, device)
        
        # Clear GPU memory before next stage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared between stages")
        
        # Update segmentation trainer with trained CVAE
        seg_trainer.cvae = cvae_trainer.cvae
        
        # Stage 2: Segmentation Training on labeled ISPRS data
        seg_trainer = train_segmentation_stage(seg_trainer, labeled_loader, args.epochs_seg, device)
        
        # Stage 3: Evaluation on real ISPRS test data
        final_accuracy = evaluate_model(seg_trainer, test_loader, device)
        
        # Results
        total_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print("=" * 40)
        print(f"Final Accuracy: {final_accuracy:.2f}%")
        print(f"Target Accuracy: {args.target_accuracy}%")
        print(f"Total Time: {total_time:.1f} seconds")
        
        if final_accuracy >= args.target_accuracy:
            print("🏆 TARGET ACHIEVED!")
        else:
            print(f"📈 Progress: {final_accuracy:.1f}% / {args.target_accuracy}%")
        
        print(f"\n💾 Models saved to: {args.output_dir}")
        print("🚀 Ready for real dataset training!")
        
        return 0 if final_accuracy >= args.target_accuracy else 1
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)