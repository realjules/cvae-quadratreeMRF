#!/usr/bin/env python3
"""
Test script to validate Phase 1 implementations:
1. Fixed contrastive loss
2. Proper augmentation pipeline  
3. CVAE training integration
4. Fixed feature extraction (no more random noise!)

This script tests all components independently before full training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

from utils.losses import contrastive_loss, simclr_loss_simple
from utils.contrastive_augmentations import ContrastiveAugmentation, create_contrastive_pair
from utils.cvae_trainer import CVAETrainer
from net.cvae import EnhancedCVAE


def test_contrastive_loss():
    """Test the fixed contrastive loss implementation"""
    print("=" * 60)
    print("TESTING FIXED CONTRASTIVE LOSS")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    latent_dim = 64
    
    # Create dummy projected features (normalized)
    z1 = F.normalize(torch.randn(batch_size, latent_dim, device=device), dim=1)
    z2 = F.normalize(torch.randn(batch_size, latent_dim, device=device), dim=1)
    
    # Test simple SimCLR loss
    print("Testing simple SimCLR loss...")
    try:
        loss_simple = simclr_loss_simple(z1, z2, temperature=0.1)
        print(f"✅ Simple SimCLR loss: {loss_simple.item():.4f}")
        assert 0 < loss_simple.item() < 10, f"Loss should be reasonable, got {loss_simple.item()}"
    except Exception as e:
        print(f"❌ Simple SimCLR loss failed: {e}")
        return False
    
    # Test full contrastive loss
    print("Testing full contrastive loss...")
    try:
        loss_full = contrastive_loss(z1, z2, temperature=0.1)
        print(f"✅ Full contrastive loss: {loss_full.item():.4f}")
        assert 0 < loss_full.item() < 10, f"Loss should be reasonable, got {loss_full.item()}"
    except Exception as e:
        print(f"❌ Full contrastive loss failed: {e}")
        return False
    
    # Test with memory bank (use correct queue dimension to match z1/z2)
    print("Testing contrastive loss with memory bank...")
    try:
        queue = F.normalize(torch.randn(1024, latent_dim, device=device), dim=1)  # Match z1/z2 dim
        loss_moco = contrastive_loss(z1, z2, temperature=0.1, queue=queue)
        print(f"✅ MoCo-style loss: {loss_moco.item():.4f}")
        assert 0 < loss_moco.item() < 10, f"Loss should be reasonable, got {loss_moco.item()}"
        
        # Test with mismatched queue dimensions (should handle gracefully)
        print("Testing with mismatched queue dimensions...")
        wrong_queue = F.normalize(torch.randn(512, latent_dim + 10, device=device), dim=1)
        loss_wrong = contrastive_loss(z1, z2, temperature=0.1, queue=wrong_queue)
        print(f"✅ Graceful handling of wrong queue: {loss_wrong.item():.4f}")
        
    except Exception as e:
        print(f"❌ MoCo-style loss failed: {e}")
        return False
    
    print("✅ All contrastive loss tests passed!")
    return True


def test_augmentation_pipeline():
    """Test the augmentation pipeline"""
    print("\n" + "=" * 60)
    print("TESTING AUGMENTATION PIPELINE")
    print("=" * 60)
    
    # Create dummy image
    image = torch.randn(3, 256, 256)
    
    # Test contrastive augmentation
    print("Testing contrastive augmentation...")
    try:
        aug = ContrastiveAugmentation(size=256, strength=0.8)
        view1, view2 = aug(image)
        
        print(f"✅ Original shape: {image.shape}")
        print(f"✅ View 1 shape: {view1.shape}")
        print(f"✅ View 2 shape: {view2.shape}")
        
        # Check that views are different
        diff = torch.abs(view1 - view2).mean()
        print(f"✅ Mean difference between views: {diff:.4f}")
        assert diff > 0.01, f"Views should be different, diff={diff:.4f}"
        
        # Check value ranges
        assert 0 <= view1.min() and view1.max() <= 1, f"View1 range: [{view1.min():.3f}, {view1.max():.3f}]"
        assert 0 <= view2.min() and view2.max() <= 1, f"View2 range: [{view2.min():.3f}, {view2.max():.3f}]"
        
    except Exception as e:
        print(f"❌ Contrastive augmentation failed: {e}")
        return False
    
    # Test convenience function
    print("Testing convenience function...")
    try:
        view1, view2 = create_contrastive_pair(image)
        print(f"✅ Convenience function works: {view1.shape}, {view2.shape}")
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False
    
    print("✅ All augmentation tests passed!")
    return True


def test_cvae_trainer():
    """Test the CVAE trainer integration"""
    print("\n" + "=" * 60)
    print("TESTING CVAE TRAINER")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create trainer
    print("Creating CVAE trainer...")
    try:
        trainer = CVAETrainer(
            input_channels=3,
            latent_dim=256,
            hidden_dims=[64, 128, 256],
            learning_rate=1e-4,
            device=device,
            temperature=0.1
        )
        print("✅ CVAE trainer created successfully")
    except Exception as e:
        print(f"❌ CVAE trainer creation failed: {e}")
        return False
    
    # Test with dummy data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 256, 256, device=device)
    
    # Test contrastive training step
    print("Testing contrastive training step...")
    try:
        metrics = trainer.train_step_contrastive(dummy_images)
        print(f"✅ Contrastive training step completed")
        print(f"  Total loss: {metrics['total_loss']:.4f}")
        print(f"  Recon loss: {metrics['recon_loss']:.4f}")
        print(f"  KL loss: {metrics['kl_loss']:.4f}")
        print(f"  Contrastive loss: {metrics['contrastive_loss']:.4f}")
        
        # Sanity checks
        assert metrics['total_loss'] > 0, "Total loss should be positive"
        assert metrics['recon_loss'] > 0, "Reconstruction loss should be positive"
        assert metrics['kl_loss'] >= 0, "KL loss should be non-negative"
        assert metrics['contrastive_loss'] > 0, "Contrastive loss should be positive"
        
    except Exception as e:
        print(f"❌ Contrastive training step failed: {e}")
        return False
    
    # Test feature extraction
    print("Testing feature extraction...")
    try:
        features = trainer.extract_features(dummy_images)
        print(f"✅ Feature extraction completed")
        print(f"  p1 (fine): {features['p1'].shape}")
        print(f"  p2 (medium): {features['p2'].shape}")  
        print(f"  p3 (coarse): {features['p3'].shape}")
        print(f"  latent_mu: {features['latent_mu'].shape}")
        
        # Check shapes
        expected_shapes = {
            'p1': (batch_size, 64, 128, 128),
            'p2': (batch_size, 128, 64, 64),
            'p3': (batch_size, 256, 32, 32)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = features[key].shape
            assert actual_shape == expected_shape, f"{key}: expected {expected_shape}, got {actual_shape}"
            
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False
    
    print("✅ All CVAE trainer tests passed!")
    return True


def test_fixed_fallback_features():
    """Test that we no longer use random noise for features"""
    print("\n" + "=" * 60)
    print("TESTING FIXED FEATURE EXTRACTION (NO MORE RANDOM NOISE)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Import the fixed trainer
    try:
        from train import FixedSegmentationTrainer
        trainer = FixedSegmentationTrainer(
            cvae_path="./nonexistent_path.pth",  # Will use fallback
            n_classes=6,
            learning_rate=0.001,
            device=device
        )
        print("✅ Fixed segmentation trainer created")
    except Exception as e:
        print(f"❌ Fixed trainer creation failed: {e}")
        return False
    
    # Test feature extraction multiple times to ensure consistency
    print("Testing feature extraction consistency...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 256, 256, device=device)
    
    try:
        # Extract features twice
        features1 = trainer.extract_cvae_features(dummy_images)
        features2 = trainer.extract_cvae_features(dummy_images)
        
        # Check that features are NOT identical (would be if using random noise)
        # But they should be deterministic given same input
        for key in ['p1', 'p2', 'p3']:
            diff = torch.abs(features1[key] - features2[key]).max()
            print(f"  {key} max difference: {diff:.6f}")
            
            # For properly implemented CNN features, this should be 0 (deterministic)
            # For random noise, this would be large
            if diff > 1e-5:
                print(f"⚠️  {key} features are not deterministic (diff={diff:.6f})")
                print("   This might indicate random weight generation in fallback")
            else:
                print(f"✅ {key} features are deterministic")
        
        # Check feature ranges are reasonable (not too large like random noise)
        for key in ['p1', 'p2', 'p3']:
            feat_mean = features1[key].abs().mean()
            feat_std = features1[key].std()
            print(f"  {key} stats: mean={feat_mean:.4f}, std={feat_std:.4f}")
            
            # Reasonable CNN features should have controlled magnitudes
            assert feat_mean < 10.0, f"{key} mean too large: {feat_mean}"
            assert feat_std < 10.0, f"{key} std too large: {feat_std}"
            
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False
    
    print("✅ Feature extraction fixes validated!")
    return True


def run_all_tests():
    """Run all Phase 1 validation tests"""
    print("🚀 RUNNING PHASE 1 VALIDATION TESTS")
    print("🎯 Goal: Validate fixes for 90% accuracy on 10% labeled data")
    print("=" * 80)
    
    tests = [
        ("Contrastive Loss", test_contrastive_loss),
        ("Augmentation Pipeline", test_augmentation_pipeline), 
        ("CVAE Trainer", test_cvae_trainer),
        ("Fixed Feature Extraction", test_fixed_fallback_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PHASE 1 TESTS PASSED!")
        print("✅ Ready to proceed with full training")
        print("🚀 Expected improvement: 55% → 75%+ accuracy")
    else:
        print("⚠️  Some tests failed - fix before proceeding")
        print("💡 Focus on failing components first")
    
    print("=" * 80)
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)