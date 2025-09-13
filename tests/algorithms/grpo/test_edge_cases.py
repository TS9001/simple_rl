"""
Tests for GRPO edge cases and boundary conditions.
"""

import pytest
import torch
from simple_rl.algorithms.grpo import GRPO


class TestGRPOEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def config(self):
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 2,
                "kl_coef": 0.1,
                "normalize_rewards": True
            },
            "model": {
                "hf_model_name": "gpt2",
                "max_length": 128
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-5,
                "max_new_tokens": 10,
                "temperature": 1.0
            }
        }
    
    def test_empty_completion(self, config):
        """Test handling of sequences with empty completions (prompt = full sequence)."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 2
        seq_len = 10
        
        # All tokens are prompt, no completion
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        prompt_lengths = [seq_len, seq_len]  # Prompt is entire sequence
        
        with torch.no_grad():
            log_probs = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model
            )
        
        # Should return zeros since no completion tokens
        assert torch.allclose(log_probs, torch.zeros(batch_size))
    
    def test_single_token_completion(self, config):
        """Test with single token completions."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        prompt_lengths = [seq_len - 1, seq_len - 1]  # Only last token is completion
        
        with torch.no_grad():
            log_probs, mask = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model, return_per_token=True
            )
        
        # Check that only one position has non-zero mask
        assert mask.sum(dim=1).tolist() == [1, 1]
    
    def test_all_padding(self, config):
        """Test handling of fully padded sequences."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len)  # All padding
        prompt_lengths = [0, 0]
        
        with torch.no_grad():
            log_probs = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model
            )
        
        # Should handle gracefully without errors
        assert log_probs.shape == (batch_size,)
    
    def test_mismatched_group_size(self, config):
        """Test error handling for batch size not divisible by group size."""
        config["algorithm"]["group_size"] = 3
        config["training"]["batch_size"] = 7  # Not divisible by 3
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        rewards = torch.randn(7)
        
        with pytest.raises(ValueError, match="must be divisible"):
            grpo.compute_relative_rewards(rewards, group_size=3)
    
    def test_extreme_reward_values(self, config):
        """Test reward normalization with extreme values."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test with very large rewards
        large_rewards = torch.tensor([1e10, 1e10, 1e10, 1e10], dtype=torch.float32)
        normalized = grpo.compute_relative_rewards(large_rewards, group_size=2)
        
        # Should still have zero mean per group
        assert torch.allclose(normalized[:2].mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(normalized[2:].mean(), torch.tensor(0.0), atol=1e-5)
        
        # Test with very small rewards
        small_rewards = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10], dtype=torch.float32)
        normalized = grpo.compute_relative_rewards(small_rewards, group_size=2)
        
        # Should handle without numerical issues
        assert not torch.any(torch.isnan(normalized))
        assert not torch.any(torch.isinf(normalized))
    
    def test_zero_kl_coefficient(self, config):
        """Test GRPO with KL coefficient set to zero."""
        config["algorithm"]["kl_coef"] = 0.0
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 10
        
        log_probs = torch.randn(batch_size, seq_len, requires_grad=True) * 0.1 - 2.0
        old_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        ref_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        rewards = torch.randn(batch_size)
        
        loss, metrics = grpo.compute_policy_loss(
            log_probs, old_log_probs, ref_log_probs, rewards
        )
        
        # Total loss should equal PG loss when KL coef is 0
        assert abs(metrics["total_loss"] - metrics["pg_loss"]) < 1e-6
    
    def test_no_clip_range(self, config):
        """Test GRPO without PPO-style clipping."""
        config["algorithm"]["clip_range"] = None
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 10
        
        log_probs = torch.randn(batch_size, seq_len, requires_grad=True) * 0.1 - 2.0
        old_log_probs = log_probs.detach() + torch.randn_like(log_probs) * 0.01
        ref_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        rewards = torch.randn(batch_size)
        
        loss, metrics = grpo.compute_policy_loss(
            log_probs, old_log_probs, ref_log_probs, rewards
        )
        
        # Should compute standard policy gradient without clipping
        assert "pg_loss" in metrics
        assert isinstance(loss, torch.Tensor)
    
    def test_batch_divisibility(self, config):
        """Test that batch size must be divisible by group size."""
        config["training"]["batch_size"] = 7  # Not divisible by 4
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        rewards = torch.randn(7)
        
        with pytest.raises(ValueError, match="must be divisible"):
            grpo.compute_relative_rewards(rewards, group_size=4)
    
    def test_extreme_kl_coefficient(self, config):
        """Test GRPO with very large KL coefficient."""
        config["algorithm"]["kl_coef"] = 100.0  # Very large
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 10
        
        log_probs = torch.randn(batch_size, seq_len, requires_grad=True) * 0.1 - 2.0
        old_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        ref_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        rewards = torch.randn(batch_size)
        
        loss, metrics = grpo.compute_policy_loss(
            log_probs, old_log_probs, ref_log_probs, rewards
        )
        
        # KL should dominate the loss
        assert metrics["kl_div"] * config["algorithm"]["kl_coef"] > abs(metrics["pg_loss"])
    
    def test_negative_rewards(self, config):
        """Test handling of negative rewards."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # All negative rewards
        negative_rewards = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        normalized = grpo.compute_relative_rewards(negative_rewards, group_size=2)
        
        # Should still have zero mean per group after normalization
        assert abs(normalized[:2].mean().item()) < 1e-5
        assert abs(normalized[2:].mean().item()) < 1e-5
    
    def test_zero_rewards(self, config):
        """Test handling of all-zero rewards."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        zero_rewards = torch.zeros(4)
        normalized = grpo.compute_relative_rewards(zero_rewards, group_size=2)
        
        # Should handle zero rewards gracefully
        assert not torch.any(torch.isnan(normalized))
        # All normalized rewards should be 0 (mean=0, std=0 case)
        assert torch.allclose(normalized, torch.zeros(4))
    
    def test_temperature_effects(self, config):
        """Test different temperature settings in generation."""
        for temp in [0.1, 1.0, 2.0]:
            config["training"]["temperature"] = temp
            grpo = GRPO(model=None, config=config, use_wandb=False)
            
            # Should initialize without errors
            assert grpo.temperature == temp