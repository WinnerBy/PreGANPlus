"""
Test script for MAMO-GAN (Migration-Aware Multi-Objective GAN)
This script tests if the enhanced models can be loaded and run correctly.

方法命名:
  - FPE-GAN (Fault Prediction Encoder GAN): 原PreGAN方法
  - TF-GAN (Transformer-based Fault GAN): 原PreGANPlus方法
  - MAMO-GAN (Migration-Aware Multi-Objective GAN): 我们的改进方法
"""

import sys
import torch
import numpy as np

# Add recovery path
sys.path.append('recovery/PreGANSrc/')

# Import models
from recovery.PreGANSrc.src.models import Gen_16_MigrationAware, Disc_16_MultiObjective
from recovery.PreGANSrc.src.constants import PROTO_DIM

def test_generator():
    """Test the migration-aware generator"""
    print("=" * 50)
    print("Testing Gen_16_MigrationAware (MAMO-GAN Generator)...")
    print("=" * 50)
    
    gen = Gen_16_MigrationAware().double()
    
    # Create dummy inputs
    embedding = torch.randn(16, PROTO_DIM, dtype=torch.double)
    schedule = torch.randn(16, 16, dtype=torch.double)
    
    # Forward pass
    try:
        new_schedule, predicted_migration_cost = gen(embedding, schedule)
        print(f"✓ Generator forward pass successful")
        print(f"  Input embedding shape: {embedding.shape}")
        print(f"  Input schedule shape: {schedule.shape}")
        print(f"  Output schedule shape: {new_schedule.shape}")
        print(f"  Predicted migration cost: {predicted_migration_cost.item():.4f}")
        print(f"  Output range: [{new_schedule.min().item():.4f}, {new_schedule.max().item():.4f}]")
        assert new_schedule.shape == (16, 16), f"Expected shape (16, 16), got {new_schedule.shape}"
        print("✓ Generator test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_discriminator():
    """Test the multi-objective discriminator"""
    print("=" * 50)
    print("Testing Disc_16_MultiObjective (MAMO-GAN Discriminator)...")
    print("=" * 50)
    
    disc = Disc_16_MultiObjective().double()
    
    # Create dummy inputs
    original_schedule = torch.randn(16, 16, dtype=torch.double)
    new_schedule = torch.randn(16, 16, dtype=torch.double)
    
    # Forward pass
    try:
        class_probs, energy_pred, response_time_pred, migration_cost_pred = disc(original_schedule, new_schedule)
        print(f"✓ Discriminator forward pass successful")
        print(f"  Input original schedule shape: {original_schedule.shape}")
        print(f"  Input new schedule shape: {new_schedule.shape}")
        print(f"  Output class_probs shape: {class_probs.shape}")
        print(f"  Output class_probs: {class_probs.tolist()}")
        print(f"  Output energy_pred: {energy_pred.item():.4f}")
        print(f"  Output response_time_pred: {response_time_pred.item():.4f}")
        print(f"  Output migration_cost_pred: {migration_cost_pred.item():.4f}")
        
        # Check outputs
        assert class_probs.shape == (2,), f"Expected class_probs shape (2,), got {class_probs.shape}"
        assert abs(class_probs.sum().item() - 1.0) < 1e-5, "Class probabilities should sum to 1"
        assert energy_pred.shape == (1,), f"Expected energy_pred shape (1,), got {energy_pred.shape}"
        assert response_time_pred.shape == (1,), f"Expected response_time_pred shape (1,), got {response_time_pred.shape}"
        assert migration_cost_pred.shape == (1,), f"Expected migration_cost_pred shape (1,), got {migration_cost_pred.shape}"
        print("✓ Discriminator test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Discriminator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_function():
    """Test the multi-objective training function"""
    print("=" * 50)
    print("Testing train_gan_multiobjective function...")
    print("=" * 50)
    
    from recovery.PreGANSrc.src.train_multiobjective import train_gan_multiobjective
    
    # Create models
    gen = Gen_16_MigrationAware().double()
    disc = Disc_16_MultiObjective().double()
    
    # Create optimizers
    gopt = torch.optim.AdamW(gen.parameters(), lr=gen.lr, weight_decay=1e-5)
    dopt = torch.optim.AdamW(disc.parameters(), lr=disc.lr, weight_decay=1e-5)
    
    # Create dummy inputs
    embedding = torch.randn(16, PROTO_DIM, dtype=torch.double)
    schedule = torch.randn(16, 16, dtype=torch.double)
    
    # Create a mock environment object
    class MockStats:
        def runSimulation(self, schedule_data):
            # Mock simulation: return random scores
            energy = torch.sum(schedule_data).item() * 0.1
            latency = torch.mean(schedule_data).item() * 0.2
            return energy, latency
    
    class MockEnv:
        def __init__(self):
            self.stats = MockStats()
    
    env = MockEnv()
    
    # Create loss function
    ganloss = torch.nn.BCELoss()
    
    try:
        # Test training function (方案6配置)
        result = train_gan_multiobjective(
            gen, disc, gopt, dopt,
            embedding, schedule, env, ganloss,
            energy_weight=0.3,
            response_time_weight=0.3,
            migration_cost_weight=0.4,
            sla_threshold=2800.0,
            migration_cost_threshold=130
        )
        
        gen_loss, disc_loss, class_loss, energy_loss, rt_loss, mc_loss, \
        new_energy, orig_energy, new_rt, orig_rt, actual_mc, pred_mc = result
        
        print(f"✓ Training function executed successfully")
        print(f"  Generator loss: {gen_loss:.4f}")
        print(f"  Discriminator loss: {disc_loss:.4f}")
        print(f"  Classification loss: {class_loss:.4f}")
        print(f"  Energy loss: {energy_loss:.4f}")
        print(f"  Response time loss: {rt_loss:.4f}")
        print(f"  Migration cost loss: {mc_loss:.4f}")
        print(f"  New energy: {new_energy:.4f}")
        print(f"  Original energy: {orig_energy:.4f}")
        print(f"  New response time: {new_rt:.4f}")
        print(f"  Original response time: {orig_rt:.4f}")
        print(f"  Actual migration count: {actual_mc}")
        print(f"  Predicted migration cost: {pred_mc:.4f}")
        print("✓ Training function test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Training function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("MAMO-GAN Implementation Test")
    print("=" * 50 + "\n")
    
    results = []
    
    # Test Generator
    results.append(test_generator())
    
    # Test Discriminator
    results.append(test_discriminator())
    
    # Test Training Function
    results.append(test_training_function())
    
    # Summary
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✓ All tests passed! MAMO-GAN implementation is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

