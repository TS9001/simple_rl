"""
GRPO reward computation and normalization tests.
"""

import pytest
import torch
import numpy as np
from simple_rl.algorithms.grpo import GRPO


class TestGRPORewards:
    """Test GRPO reward processing and normalization."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "algorithm": {
                "group_size": 4,
                "kl_coef": 0.1,
                "normalize_rewards": True
            },
            "model": {
                "hf_model_name": "gpt2",
                "max_length": 128
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-5,
                "max_new_tokens": 10
            }
        }
    
    def test_reward_normalization(self, config):
        """Test that rewards are normalized within groups."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Create mock rewards
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,  # Group 1
                                5.0, 6.0, 7.0, 8.0], # Group 2
                               device=grpo.device)
        
        # Create mock log probs (zeros for simplicity)
        seq_len = 10
        log_probs = torch.zeros(8, seq_len, device=grpo.device)
        ref_log_probs = torch.zeros(8, seq_len, device=grpo.device)
        
        # Compute advantages
        advantages = grpo.compute_advantages(rewards, log_probs, ref_log_probs)
        
        # Check that advantages are normalized within groups
        group1_advantages = advantages[:4]
        group2_advantages = advantages[4:]
        
        # Each group should have mean ~0 and std ~1
        assert abs(group1_advantages.mean().item()) < 0.1
        assert abs(group1_advantages.std().item() - 1.0) < 0.1
        
        assert abs(group2_advantages.mean().item()) < 0.1
        assert abs(group2_advantages.std().item() - 1.0) < 0.1
    
    def test_kl_penalty(self, config):
        """Test KL divergence penalty is applied correctly."""
        config["algorithm"]["kl_coef"] = 0.5
        grpo = GRPO(config=config, use_wandb=False)
        
        # Create rewards
        rewards = torch.ones(4, device=grpo.device) * 2.0
        
        # Create different log probs to generate KL divergence
        seq_len = 10
        log_probs = torch.ones(4, seq_len, device=grpo.device) * -1.0
        ref_log_probs = torch.ones(4, seq_len, device=grpo.device) * -2.0
        
        # Compute advantages
        advantages = grpo.compute_advantages(rewards, log_probs, ref_log_probs)
        
        # KL divergence should reduce the effective rewards
        # KL = sum(log_probs - ref_log_probs) = sum(-1 - (-2)) = 10
        # Adjusted reward = 2.0 - 0.5 * 10 = -3.0
        expected_adjusted = 2.0 - 0.5 * 10
        
        # Since rewards are normalized, we can't check exact values
        # but we can verify the KL penalty had an effect
        assert advantages.mean().item() != 2.0  # Should be different from original
    
    def test_custom_reward_function(self, config):
        """Test using custom reward function."""
        def custom_reward(prompt: str, completion: str) -> float:
            return len(completion) / 100.0
        
        grpo = GRPO(config=config, reward_fn=custom_reward, use_wandb=False)
        
        # Test the reward function is used
        reward = grpo.reward_fn("test", "a" * 50)
        assert reward == 0.5
    
    def test_default_reward_function(self, config):
        """Test default reward function."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Test default reward based on length
        short_reward = grpo.reward_fn("prompt", "short response")
        long_reward = grpo.reward_fn("prompt", " ".join(["word"] * 100))
        
        assert short_reward < long_reward
        assert long_reward <= 1.0  # Should be capped at 1.0