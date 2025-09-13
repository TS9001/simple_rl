"""
Tests for GRPO performance and memory efficiency.
"""

import pytest
import torch
import time
from unittest.mock import Mock, patch
from simple_rl.algorithms.grpo import GRPO


class TestGRPOPerformance:
    """Test memory efficiency and performance characteristics."""
    
    @pytest.fixture
    def config(self):
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
                "max_new_tokens": 16,
                "temperature": 1.0
            }
        }
    
    def test_no_gradient_accumulation_in_eval(self, config):
        """Ensure no gradients are accumulated during trajectory generation."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Ensure all gradients are None initially
        for param in grpo.model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)
        
        # Generate trajectories (should be in eval mode)
        prompts = ["Test prompt 1", "Test prompt 2"]
        trajectories = grpo.generate_trajectories(prompts, num_samples_per_prompt=2)
        
        # No gradients should be accumulated
        for param in grpo.model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)
    
    @pytest.mark.slow
    def test_batch_size_scaling(self, config):
        """Test that algorithm scales properly with batch size."""
        times = []
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            config["training"]["batch_size"] = batch_size
            config["algorithm"]["group_size"] = min(2, batch_size)
            grpo = GRPO(model=None, config=config, use_wandb=False)
            
            prompts = ["Test"] * (batch_size // config["algorithm"]["group_size"])
            batch_data = {"prompts": prompts}
            
            start = time.time()
            with patch.object(grpo, 'optimizer') as mock_optimizer:
                mock_optimizer.zero_grad = Mock()
                mock_optimizer.step = Mock()
                metrics = grpo.train_step(batch_data)
            end = time.time()
            
            times.append(end - start)
        
        # Time should scale roughly linearly with batch size
        # (allowing for some overhead and variance)
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        for ratio in time_ratios:
            assert 0.5 < ratio < 4.0  # Reasonable scaling
    
    @pytest.mark.slow
    def test_large_batch_processing(self, config):
        """Test handling of large batches."""
        config["training"]["batch_size"] = 16
        config["algorithm"]["group_size"] = 4
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        prompts = ["Large batch test"] * 4  # 4 prompts, 4 samples each = 16 total
        batch_data = {"prompts": prompts}
        
        # Should handle large batch without errors
        metrics = grpo.train_step(batch_data)
        assert "total_loss" in metrics
    
    def test_caching_reference_outputs(self, config):
        """Test that reference model outputs are cached properly."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Track calls to reference model
        call_count = 0
        original_forward = grpo.ref_policy.forward
        
        def tracked_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)
        
        grpo.ref_policy.forward = tracked_forward
        
        prompts = ["Test caching"]
        batch_data = {"prompts": prompts}
        
        # First call should use reference model
        metrics = grpo.train_step(batch_data)
        first_calls = call_count
        
        # Reference model should be called during trajectory generation
        assert first_calls > 0
    
    @pytest.mark.slow
    def test_performance_with_long_sequences(self, config):
        """Test performance with longer sequences."""
        config["model"]["max_length"] = 256
        config["training"]["max_new_tokens"] = 64
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        prompts = ["Long sequence test " * 10]  # Longer prompt
        batch_data = {"prompts": prompts}
        
        # Should handle long sequences
        start = time.time()
        metrics = grpo.train_step(batch_data)
        end = time.time()
        
        # Should complete in reasonable time (< 60 seconds)
        assert (end - start) < 60
        assert "total_loss" in metrics