#!/usr/bin/env python3
"""
Simple test script for GRPO implementation.
"""

import torch
import yaml
from simple_rl.algorithms.grpo import GRPO
from simple_rl.models.policy_model import PolicyModel


def test_grpo():
    """Test basic GRPO functionality."""
    
    # Create minimal config
    config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 2,
            "kl_coef": 0.1,
            "normalize_rewards": True
        },
        "model": {
            "hf_model_name": "gpt2",  # Use smallest GPT-2
            "max_length": 128
        },
        "training": {
            "batch_size": 4,  # Must be divisible by group_size
            "learning_rate": 1e-5,
            "max_new_tokens": 20,
            "temperature": 1.0,
            "gradient_clip": 1.0
        },
        "logging": {
            "log_interval": 1
        }
    }
    
    print("Initializing GRPO...")
    
    # Initialize without wandb for testing
    grpo = GRPO(model=None, config=config, use_wandb=False)
    
    print(f"Model device: {grpo.device}")
    print(f"Model type: {type(grpo.model)}")
    print(f"Reference model type: {type(grpo.reference_model)}")
    
    # Test trajectory generation
    print("\n1. Testing trajectory generation...")
    prompts = ["The weather today is", "Machine learning is"]
    trajectories = grpo.generate_trajectories(prompts, num_samples_per_prompt=2)
    
    print(f"Generated IDs shape: {trajectories['generated_ids'].shape}")
    print(f"Attention masks shape: {trajectories['attention_masks'].shape}")
    print(f"Number of texts: {len(trajectories['texts'])}")
    print(f"Sample text: {trajectories['texts'][0][:100]}...")
    
    # Test reward computation
    print("\n2. Testing reward computation...")
    rewards = grpo.compute_rewards(prompts * 2, trajectories['texts'])
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    
    # Test reward normalization
    print("\n3. Testing reward normalization...")
    normalized = grpo.compute_relative_rewards(rewards, group_size=2)
    print(f"Normalized rewards: {normalized}")
    print(f"Mean per group should be ~0: {normalized.view(2, 2).mean(dim=1)}")
    
    # Test log prob computation
    print("\n4. Testing log prob computation...")
    with torch.no_grad():
        log_probs = grpo.compute_log_probs_for_sequences(
            trajectories['generated_ids'],
            trajectories['attention_masks'],
            trajectories['prompt_lengths'],
            model=grpo.model
        )
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Log probs: {log_probs}")
    
    # Test single training step
    print("\n5. Testing training step...")
    batch_data = {"prompts": prompts}
    metrics = grpo.train_step(batch_data)
    
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ GRPO test completed successfully!")


if __name__ == "__main__":
    test_grpo()