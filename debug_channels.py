#!/usr/bin/env python3
"""
Debug script to check channel dimensions and fix any remaining issues
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from train import FixedSegmentationTrainer

def debug_channel_flow():
    """Debug the channel flow through the architecture"""
    print("🔍 DEBUGGING CHANNEL DIMENSIONS")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create trainer
    try:
        trainer = FixedSegmentationTrainer(
            cvae_path="./nonexistent.pth",  # Will use fallback
            n_classes=6,
            learning_rate=0.001,
            device=device
        )
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False
    
    # Test feature extraction
    print("\n📊 Testing feature extraction...")
    test_input = torch.randn(2, 3, 256, 256, device=device)
    print(f"Input shape: {test_input.shape}")
    
    try:
        features = trainer.extract_cvae_features(test_input)
        print("✅ Feature extraction successful!")
        print(f"Features returned: {list(features.keys())}")
        for key, feat in features.items():
            if isinstance(feat, torch.Tensor):
                print(f"  {key}: {feat.shape}")
        
        # Now test the segmentation model
        print("\n🧠 Testing segmentation model...")
        
        # Check what the model expects vs what we provide
        print("\n📋 Expected vs Actual:")
        print("  process_p1 expects: (64 channels) -> 128")
        print("  process_p2 expects: (128 channels) -> 256") 
        print("  process_p3 expects: (256 channels) -> 512")
        print()
        print("  We provide:")
        print(f"  p1: {features['p1'].shape} - {features['p1'].shape[1]} channels")
        print(f"  p2: {features['p2'].shape} - {features['p2'].shape[1]} channels")
        print(f"  p3: {features['p3'].shape} - {features['p3'].shape[1]} channels")
        
        # Test forward pass
        outputs = trainer.model(features)
        print("✅ Forward pass successful!")
        print(f"Final segmentation shape: {outputs['final_segmentation'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_model_architecture():
    """Print the model architecture to understand channel flow"""
    print("\n🏗️  MODEL ARCHITECTURE ANALYSIS")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        trainer = FixedSegmentationTrainer(
            cvae_path="./nonexistent.pth",
            n_classes=6,
            device=device
        )
        
        print("📱 Segmentation Model Components:")
        print(f"  process_p1: {trainer.model.process_p1}")
        print(f"  process_p2: {trainer.model.process_p2}")
        print(f"  process_p3: {trainer.model.process_p3}")
        
        print("\n🏢 Fallback Extractor:")
        print(f"  {trainer.cvae}")
        
    except Exception as e:
        print(f"❌ Failed to analyze architecture: {e}")

if __name__ == "__main__":
    print("🚀 Starting channel dimension debugging...")
    
    success = debug_channel_flow()
    print_model_architecture()
    
    if success:
        print("\n🎉 All channel dimensions are correct!")
        print("✅ Architecture is ready for training")
    else:
        print("\n⚠️  Channel dimension issues found")
        print("💡 Check the error messages above for details")
    
    print("\n" + "=" * 50)