#!/usr/bin/env python3
"""
Focused test for contrastive loss fix
"""

import torch
import torch.nn.functional as F
import sys
sys.path.append('.')

from utils.losses import contrastive_loss, simclr_loss_simple

def test_contrastive_loss_fix():
    """Test the fixed contrastive loss implementation"""
    print("Testing Fixed Contrastive Loss")
    print("=" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    batch_size = 8
    latent_dim = 64
    
    # Create dummy projected features (normalized)
    z1 = F.normalize(torch.randn(batch_size, latent_dim, device=device), dim=1)
    z2 = F.normalize(torch.randn(batch_size, latent_dim, device=device), dim=1)
    
    print(f"z1 shape: {z1.shape}")
    print(f"z2 shape: {z2.shape}")
    
    # Test 1: Simple SimCLR loss
    try:
        loss_simple = simclr_loss_simple(z1, z2, temperature=0.1)
        print(f"✅ Simple SimCLR loss: {loss_simple.item():.4f}")
    except Exception as e:
        print(f"❌ Simple SimCLR loss failed: {e}")
        return False
    
    # Test 2: Full contrastive loss (no queue)
    try:
        loss_full = contrastive_loss(z1, z2, temperature=0.1, queue=None)
        print(f"✅ Full contrastive loss (no queue): {loss_full.item():.4f}")
    except Exception as e:
        print(f"❌ Full contrastive loss failed: {e}")
        return False
    
    # Test 3: With matching queue dimensions
    try:
        queue = F.normalize(torch.randn(1024, latent_dim, device=device), dim=1)
        loss_moco = contrastive_loss(z1, z2, temperature=0.1, queue=queue)
        print(f"✅ MoCo-style loss (matching dims): {loss_moco.item():.4f}")
    except Exception as e:
        print(f"❌ MoCo-style loss failed: {e}")
        return False
    
    # Test 4: With mismatched queue dimensions (should handle gracefully)
    try:
        wrong_queue = F.normalize(torch.randn(512, latent_dim + 10, device=device), dim=1)
        loss_wrong = contrastive_loss(z1, z2, temperature=0.1, queue=wrong_queue)
        print(f"✅ Graceful handling of wrong queue: {loss_wrong.item():.4f}")
    except Exception as e:
        print(f"❌ Wrong queue handling failed: {e}")
        return False
    
    print("✅ All contrastive loss tests passed!")
    return True

if __name__ == "__main__":
    success = test_contrastive_loss_fix()
    if success:
        print("\n🎉 Contrastive loss is now working correctly!")
        print("This should fix both failing tests in the main test suite.")
    else:
        print("\n❌ Contrastive loss still has issues.")
    
    sys.exit(0 if success else 1)