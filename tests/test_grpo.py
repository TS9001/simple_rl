"""
Unit tests for GRPO algorithm.
"""

import pytest
import torch
import torch.nn as nn
from simple_rl.algorithms.grpo import GRPO
from simple_rl.models.policy_model import PolicyModel


class TestGRPO:
    """Test suite for GRPO algorithm."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "algorithm": {
                "name": "grpo",
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
                "max_new_tokens": 10,
                "temperature": 1.0
            }
        }
    
    def test_initialization(self, config):
        """Test GRPO initialization."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        assert isinstance(grpo.model, PolicyModel)
        assert isinstance(grpo.reference_model, nn.Module)
        assert grpo.group_size == 4
        assert grpo.kl_coef == 0.1
        
        # Check reference model is frozen
        for param in grpo.reference_model.parameters():
            assert not param.requires_grad
    
    def test_relative_rewards(self, config):
        """Test relative reward computation."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Create test rewards
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        group_size = 4
        
        normalized = grpo.compute_relative_rewards(rewards, group_size)
        
        # Check shape
        assert normalized.shape == rewards.shape
        
        # Check normalization within groups
        group1 = normalized[:4]
        group2 = normalized[4:]
        
        # Mean should be 0 within each group
        assert abs(group1.mean().item()) < 1e-5
        assert abs(group2.mean().item()) < 1e-5
        
        # Std should be 1 within each group (if normalize_rewards=True)
        assert abs(group1.std().item() - 1.0) < 0.1
        assert abs(group2.std().item() - 1.0) < 0.1
    
    def test_kl_divergence(self, config):
        """Test KL divergence computation."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test with identical distributions (KL should be 0)
        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        ref_log_probs = torch.tensor([-1.0, -2.0, -3.0])
        
        kl = grpo.compute_kl_divergence(log_probs, ref_log_probs)
        assert abs(kl.item()) < 1e-5
        
        # Test with different distributions (KL should be > 0)
        log_probs = torch.tensor([-1.0, -2.0, -3.0])
        ref_log_probs = torch.tensor([-2.0, -3.0, -4.0])
        
        kl = grpo.compute_kl_divergence(log_probs, ref_log_probs)
        assert kl.item() > 0
    
    def test_policy_loss(self, config):
        """Test policy loss computation."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 8
        seq_len = 10
        
        # Create dummy inputs
        log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        ref_log_probs = torch.randn(batch_size, seq_len) * 0.1 - 2.0
        rewards = torch.randn(batch_size)
        
        loss, metrics = grpo.compute_policy_loss(
            log_probs, ref_log_probs, rewards
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert "pg_loss" in metrics
        assert "kl_div" in metrics
        assert "total_loss" in metrics
    
    def test_trajectory_generation_shapes(self, config):
        """Test trajectory generation produces correct shapes."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        prompts = ["Test prompt 1", "Test prompt 2"]
        num_samples = 2
        
        trajectories = grpo.generate_trajectories(prompts, num_samples)
        
        expected_batch_size = len(prompts) * num_samples
        
        assert trajectories["generated_ids"].shape[0] == expected_batch_size
        assert trajectories["attention_masks"].shape[0] == expected_batch_size
        assert len(trajectories["texts"]) == expected_batch_size
        assert len(trajectories["prompt_lengths"]) == expected_batch_size
        
        # Check all sequences have same length (padded)
        assert len(trajectories["generated_ids"].shape) == 2
        assert len(trajectories["attention_masks"].shape) == 2
    
    def test_masking_correctness(self, config):
        """Test that masking correctly separates prompt from completion."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Create a sequence where we know the prompt length
        batch_size = 2
        seq_len = 20
        prompt_lengths = [5, 8]
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # This should only compute log probs for completion tokens
        with torch.no_grad():
            log_probs = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model
            )
        
        assert log_probs.shape == (batch_size,)
        
    def test_batch_divisibility(self, config):
        """Test that batch size must be divisible by group size."""
        config["training"]["batch_size"] = 7  # Not divisible by 4
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        rewards = torch.randn(7)
        
        with pytest.raises(ValueError, match="must be divisible"):
            grpo.compute_relative_rewards(rewards, group_size=4)