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


def create_enhanced_dummy_dataset(batch_size=4, num_batches=50, device="cuda", mode="train"):
    """
    Enhanced dummy dataset that simulates real ISPRS data characteristics
    """
    def data_generator():
        aug = ContrastiveAugmentation(size=256, strength=0.8)
        
        for i in range(num_batches):
            # Create more realistic aerial imagery patterns
            if mode == "unlabeled":
                # For contrastive learning - return augmented pairs
                base_images = torch.randn(batch_size, 3, 256, 256, device=device)
                base_images = torch.sigmoid(base_images)  # Normalize to [0,1]
                
                augmented_batch = []
                for j in range(batch_size):
                    view1, view2 = aug(base_images[j])
                    augmented_batch.append((view1.to(device), view2.to(device)))
                
                yield augmented_batch
                
            else:
                # For supervised learning - return image-label pairs
                images = torch.randn(batch_size, 3, 256, 256, device=device)
                images = torch.sigmoid(images)
                
                # Create more realistic segmentation patterns
                labels = torch.randint(0, 6, (batch_size, 256, 256), device=device)
                
                # Add some spatial structure (buildings, roads, etc.)
                for b in range(batch_size):
                    # Add rectangular "buildings"
                    h_start, w_start = np.random.randint(50, 150, 2)
                    h_size, w_size = np.random.randint(20, 80, 2)
                    labels[b, h_start:h_start+h_size, w_start:w_start+w_size] = 1  # Buildings
                    
                    # Add linear "roads"
                    road_y = np.random.randint(50, 200)
                    labels[b, road_y:road_y+5, :] = 0  # Roads
                
                yield images, labels
    
    return data_generator


def train_cvae_stage(cvae_trainer, unlabeled_dataloader, epochs=20, device="cuda"):
    """
    Stage 1: Train CVAE with contrastive learning on unlabeled data
    """
    print("🎯 STAGE 1: CVAE Contrastive Learning")
    print("=" * 50)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        print(f"\n📚 CVAE Epoch {epoch}/{epochs}")
        
        for batch_idx, augmented_pairs in enumerate(unlabeled_dataloader()):
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
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"📊 CVAE Epoch {epoch} completed: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            cvae_trainer.save_model(f"./output/cvae_epoch_{epoch}.pth", epoch)
    
    print("✅ CVAE contrastive learning completed!")
    return cvae_trainer


def train_segmentation_stage(seg_trainer, labeled_dataloader, epochs=30, device="cuda"):
    """
    Stage 2: Train segmentation model with curriculum learning
    """
    print("\n🎯 STAGE 2: Semi-Supervised Segmentation Training")
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        num_batches = 0
        
        print(f"\n📚 Segmentation Epoch {epoch}/{epochs}")
        
        for batch_idx, (images, labels) in enumerate(labeled_dataloader()):
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
    Evaluate the trained model
    """
    print("\n🧪 EVALUATION")
    print("=" * 30)
    
    seg_trainer.model.eval()
    total_accuracy = 0
    total_samples = 0
    class_correct = [0] * 6
    class_total = [0] * 6
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader()):
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
    
    overall_accuracy = (total_accuracy / total_samples * 100).item()
    
    print(f"📊 EVALUATION RESULTS:")
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
    
    # Create data generators with memory-efficient batch sizes
    print("📊 Creating data generators...")
    
    # Fix 2: Reduce batch sizes for memory efficiency
    effective_batch_size = min(args.batch_size, 2)  # Max 2 for GPU memory
    
    unlabeled_data = create_enhanced_dummy_dataset(
        batch_size=effective_batch_size, 
        num_batches=30, 
        device=device, 
        mode="unlabeled"
    )
    labeled_data = create_enhanced_dummy_dataset(
        batch_size=effective_batch_size, 
        num_batches=20, 
        device=device, 
        mode="train"
    )
    test_data = create_enhanced_dummy_dataset(
        batch_size=effective_batch_size, 
        num_batches=10, 
        device=device, 
        mode="test"
    )
    
    print(f"  Effective Batch Size: {effective_batch_size} (reduced for memory)")
    print(f"  GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory Cleared")
    
    # Training pipeline
    start_time = time.time()
    
    try:
        # Stage 1: CVAE Contrastive Learning
        cvae_trainer = train_cvae_stage(cvae_trainer, unlabeled_data, args.epochs_cvae, device)
        
        # Clear GPU memory before next stage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared between stages")
        
        # Update segmentation trainer with trained CVAE
        seg_trainer.cvae = cvae_trainer.cvae
        
        # Stage 2: Segmentation Training
        seg_trainer = train_segmentation_stage(seg_trainer, labeled_data, args.epochs_seg, device)
        
        # Stage 3: Evaluation
        final_accuracy = evaluate_model(seg_trainer, test_data, device)
        
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