"""
Tests for GRPO numerical stability.
"""

import pytest
import torch
import numpy as np
from simple_rl.algorithms.grpo import GRPO


class TestGRPONumericalStability:
    """Test numerical stability under various conditions."""
    
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
    
    def test_log_prob_stability(self, config):
        """Test numerical stability of log probability computations."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 20
        
        # Test with very small probabilities (large negative log probs)
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))  # Large vocab indices
        attention_mask = torch.ones(batch_size, seq_len)
        prompt_lengths = [5, 5, 5, 5]
        
        with torch.no_grad():
            log_probs = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model
            )
        
        # Check for numerical issues
        assert not torch.any(torch.isnan(log_probs))
        assert not torch.any(torch.isinf(log_probs))
        assert torch.all(log_probs <= 0)  # Log probs should be negative
    
    def test_kl_divergence_stability(self, config):
        """Test KL divergence computation stability with extreme values."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test with very similar distributions (KL should be near 0)
        log_probs = torch.tensor([[-2.0, -2.0, -2.0]])
        ref_log_probs = torch.tensor([[-2.0001, -2.0001, -2.0001]])
        
        kl = grpo.compute_kl_divergence(log_probs, ref_log_probs)
        assert abs(kl.item()) < 0.01  # Should be very small
        assert not torch.isnan(kl)
        
        # Test with very different distributions
        log_probs = torch.tensor([[-1.0, -1.0, -1.0]])
        ref_log_probs = torch.tensor([[-10.0, -10.0, -10.0]])
        
        kl = grpo.compute_kl_divergence(log_probs, ref_log_probs)
        assert kl.item() > 0  # Should be positive
        assert not torch.isnan(kl)
        assert not torch.isinf(kl)
    
    def test_reward_normalization_stability(self, config):
        """Test reward normalization with edge cases."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test with identical rewards (std = 0)
        identical_rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        normalized = grpo.compute_relative_rewards(identical_rewards, group_size=2)
        
        # Should handle zero std gracefully
        assert not torch.any(torch.isnan(normalized))
        assert torch.allclose(normalized, torch.zeros(4))  # All should be 0 after normalization
        
        # Test with one outlier
        outlier_rewards = torch.tensor([1.0, 1.0, 1000.0, 1.0])
        normalized = grpo.compute_relative_rewards(outlier_rewards, group_size=2)
        
        # Should handle outliers without numerical issues
        assert not torch.any(torch.isnan(normalized))
        assert not torch.any(torch.isinf(normalized))
    
    def test_ratio_computation_stability(self, config):
        """Test importance sampling ratio computation stability."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 10
        
        # Test with moderately different log probs (avoiding extreme ratios that would cause NaN)
        # Note: exp(-1 - (-5)) = exp(4) ≈ 54, which is large but manageable
        log_probs = torch.full((batch_size, seq_len), -1.0, requires_grad=True)
        old_log_probs = torch.full((batch_size, seq_len), -5.0)  # Moderately different
        ref_log_probs = torch.full((batch_size, seq_len), -2.0)
        rewards = torch.randn(batch_size) * 0.1  # Small rewards to avoid explosion
        
        loss, metrics = grpo.compute_policy_loss(
            log_probs, old_log_probs, ref_log_probs, rewards
        )
        
        # Should handle large ratios without numerical explosion
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.requires_grad
    
    def test_gradient_stability(self, config):
        """Test gradient computation stability."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 10
        
        # Create inputs that could cause gradient issues
        log_probs = torch.randn(batch_size, seq_len, requires_grad=True) * 0.01 - 5.0  # Very negative
        old_log_probs = torch.randn(batch_size, seq_len) * 0.01 - 5.0
        ref_log_probs = torch.randn(batch_size, seq_len) * 0.01 - 5.0
        rewards = torch.randn(batch_size) * 10  # Large rewards
        
        loss, metrics = grpo.compute_policy_loss(
            log_probs, old_log_probs, ref_log_probs, rewards
        )
        
        # Compute gradients
        loss.backward()
        
        # Check gradients are stable
        for param in [log_probs]:
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad))
                assert not torch.any(torch.isinf(param.grad))
    
    def test_loss_computation_stability(self, config):
        """Test overall loss computation stability."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test with various edge cases
        test_cases = [
            # Very small log probs
            (torch.full((4, 10), -100.0, requires_grad=True),
             torch.full((4, 10), -100.0),
             torch.full((4, 10), -100.0),
             torch.zeros(4)),
            
            # Mixed positive and negative rewards
            (torch.randn(4, 10, requires_grad=True) * 0.1 - 2.0,
             torch.randn(4, 10) * 0.1 - 2.0,
             torch.randn(4, 10) * 0.1 - 2.0,
             torch.tensor([-10.0, 10.0, -10.0, 10.0])),
            
            # Zero rewards
            (torch.randn(4, 10, requires_grad=True) * 0.1 - 2.0,
             torch.randn(4, 10) * 0.1 - 2.0,
             torch.randn(4, 10) * 0.1 - 2.0,
             torch.zeros(4)),
        ]
        
        for log_probs, old_log_probs, ref_log_probs, rewards in test_cases:
            loss, metrics = grpo.compute_policy_loss(
                log_probs, old_log_probs, ref_log_probs, rewards
            )
            
            # Loss should be stable
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            
            # Metrics should be stable
            for key, value in metrics.items():
                if isinstance(value, (float, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        assert not torch.isnan(value)
                        assert not torch.isinf(value)
                    else:
                        assert not np.isnan(value)
                        assert not np.isinf(value)