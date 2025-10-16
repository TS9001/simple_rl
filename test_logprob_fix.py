#!/usr/bin/env python
"""
Quick test to verify the logprob consistency fix.

This script tests that old, new, and ref logprobs are now nearly identical
when computed on the same inputs with eval() enforced.
"""

import torch
import numpy as np
from simple_rl.algorithms.grpo import GRPO


def create_test_config():
    """Create a minimal test configuration."""
    return {
        "model": {
            "model_name": "gpt2",
            "model_type": "causal",
        },
        "training": {
            "batch_size": 2,
            "group_size": 2,
            "kl_coef": 0.1,
            "clip_epsilon": 0.2,
            "learning_rate": 1e-5,
            "minibatch_size": 2,
            "rollout_batch_size": 2,
            "max_new_tokens": 20,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.95,
            "update_epochs": 1,
            "normalize_rewards": True,
            "gradient_clip": 1.0,
        },
        "optimizer": {
            "name": "adam",
            "lr": 1e-5,
        },
        "logging": {
            "log_level": "INFO",
        }
    }


def test_reward_fn(completions, answers, device, return_breakdown=False):
    """Dummy reward function."""
    rewards = torch.ones(len(completions), device=device) * 0.5
    if return_breakdown:
        format_rewards = torch.ones(len(completions), device=device) * 0.3
        correctness_rewards = torch.ones(len(completions), device=device) * 0.2
        return rewards, format_rewards, correctness_rewards
    return rewards


def main():
    print("=" * 80)
    print("Testing Log Probability Consistency Fix")
    print("=" * 80)
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create GRPO instance
    config = create_test_config()
    grpo = GRPO(config, batch_reward_fn=test_reward_fn)
    
    # Create test sequences
    batch_size = 4
    max_seq_len = 30
    max_completion_len = 20
    device = grpo.device
    
    print(f"Device: {device}")
    print(f"Model dtype: {next(grpo.policy.parameters()).dtype}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Generate random test data
    generated_ids = []
    attention_mask = []
    completion_mask = []
    
    for i in range(batch_size):
        seq_len = np.random.randint(20, max_seq_len)
        comp_len = np.random.randint(10, min(max_completion_len, seq_len - 5))
        
        seq_ids = torch.randint(10, 1000, (seq_len,), device=device)
        seq_mask = torch.ones(seq_len, device=device)
        comp_mask = torch.ones(comp_len, device=device)
        
        generated_ids.append(seq_ids)
        attention_mask.append(seq_mask)
        completion_mask.append(comp_mask)
    
    prompt_end_positions = torch.tensor([5, 6, 5, 7], device=device)
    
    # Compute all three logprob types
    print("Computing logprobs...")
    print("-" * 80)
    
    old_logprobs = grpo.compute_logprobs(
        "old", generated_ids, attention_mask, prompt_end_positions, completion_mask
    )
    print(f"✓ Old logprobs: shape={old_logprobs.shape}")
    
    ref_logprobs = grpo.compute_logprobs(
        "ref", generated_ids, attention_mask, prompt_end_positions, completion_mask
    )
    print(f"✓ Ref logprobs: shape={ref_logprobs.shape}")
    
    new_logprobs = grpo.compute_logprobs(
        "new", generated_ids, attention_mask, prompt_end_positions, completion_mask
    )
    print(f"✓ New logprobs: shape={new_logprobs.shape}, requires_grad={new_logprobs.requires_grad}")
    print()
    
    # Compute differences
    old_ref_diff = (old_logprobs - ref_logprobs).abs().max().item()
    old_new_diff = (old_logprobs - new_logprobs.detach()).abs().max().item()
    ref_new_diff = (ref_logprobs - new_logprobs.detach()).abs().max().item()
    
    old_ref_mean_diff = (old_logprobs - ref_logprobs).abs().mean().item()
    old_new_mean_diff = (old_logprobs - new_logprobs.detach()).abs().mean().item()
    
    # Display results
    print("Difference Analysis:")
    print("-" * 80)
    print(f"Old-Ref max diff:  {old_ref_diff:.6e}")
    print(f"Old-New max diff:  {old_new_diff:.6e}")
    print(f"Ref-New max diff:  {ref_new_diff:.6e}")
    print()
    print(f"Old-Ref mean diff: {old_ref_mean_diff:.6e}")
    print(f"Old-New mean diff: {old_new_mean_diff:.6e}")
    print()
    
    # Determine thresholds based on dtype
    model_dtype = next(grpo.policy.parameters()).dtype
    if model_dtype == torch.bfloat16:
        threshold = 1e-4
    elif model_dtype == torch.float16:
        threshold = 1e-5
    else:
        threshold = 1e-6
    
    print(f"Expected threshold for {model_dtype}: {threshold:.0e}")
    print()
    
    # Check results
    success = True
    
    if old_ref_diff < threshold:
        print("✅ PASS: Old-Ref difference within threshold")
    else:
        print(f"❌ FAIL: Old-Ref difference {old_ref_diff:.2e} exceeds threshold {threshold:.0e}")
        success = False
    
    if old_new_diff < threshold:
        print("✅ PASS: Old-New difference within threshold")
    else:
        print(f"❌ FAIL: Old-New difference {old_new_diff:.2e} exceeds threshold {threshold:.0e}")
        success = False
    
    if new_logprobs.requires_grad:
        print("✅ PASS: New logprobs have gradients enabled")
    else:
        print("❌ FAIL: New logprobs missing gradients")
        success = False
    
    print()
    print("=" * 80)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("The logprob consistency fix is working correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the implementation.")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

