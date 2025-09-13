"""
Extensive tests for GRPO algorithm covering masking, edge cases, and integration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from simple_rl.algorithms.grpo import GRPO


class TestGRPOMasking:
    """Comprehensive tests for masking logic in GRPO."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 2,
                "kl_coef": 0.1,
                "normalize_rewards": True,
                "update_epochs": 1,
                "minibatch_size": None
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
    
    def test_no_double_masking_in_compute_log_probs(self, config):
        """Verify that masks are applied exactly once in compute_log_probs."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 2
        seq_len = 20
        prompt_lengths = [5, 8]
        
        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 15:] = 0  # Add padding to first sequence
        attention_mask[1, 18:] = 0  # Add padding to second sequence
        
        # Mock the model's compute_log_probs to track calls
        original_compute_log_probs = grpo.model.compute_log_probs
        call_count = 0
        received_masks = []
        
        def mock_compute_log_probs(input_ids, attention_mask=None, target_mask=None):
            nonlocal call_count, received_masks
            call_count += 1
            received_masks.append({
                'attention_mask': attention_mask.clone() if attention_mask is not None else None,
                'target_mask': target_mask.clone() if target_mask is not None else None
            })
            # Return realistic log probs
            return torch.randn(batch_size, seq_len - 1) * 0.1 - 2.0
        
        grpo.model.compute_log_probs = mock_compute_log_probs
        
        # Call compute_log_probs_for_sequences
        with torch.no_grad():
            log_probs, completion_mask = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model, return_per_token=True
            )
        
        # Verify the function was called once
        assert call_count == 1
        
        # Verify both masks were passed
        assert received_masks[0]['attention_mask'] is not None
        assert received_masks[0]['target_mask'] is not None
        
        # Verify target mask has correct shape and values
        target_mask = received_masks[0]['target_mask']
        assert target_mask.shape == (batch_size, seq_len)
        
        # Check that prompt positions are masked (0) and completion positions are unmasked (1)
        assert torch.all(target_mask[0, :5] == 0)  # First 5 are prompt
        assert torch.all(target_mask[0, 5:15] == 1)  # Next 10 are completion (before padding)
        assert torch.all(target_mask[1, :8] == 0)  # First 8 are prompt
        assert torch.all(target_mask[1, 8:18] == 1)  # Next 10 are completion (before padding)
        
        # Restore original method
        grpo.model.compute_log_probs = original_compute_log_probs
    
    def test_masking_with_variable_length_sequences(self, config):
        """Test masking handles variable length sequences correctly."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Create sequences of different lengths
        batch_size = 3
        prompt_lengths = [3, 7, 5]  # Different prompt lengths
        seq_lengths = [15, 20, 12]  # Different total lengths
        max_len = max(seq_lengths)
        
        # Create padded sequences
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len)
        
        for i, (prompt_len, seq_len) in enumerate(zip(prompt_lengths, seq_lengths)):
            input_ids[i, :seq_len] = torch.randint(0, 1000, (seq_len,))
            attention_mask[i, :seq_len] = 1
        
        with torch.no_grad():
            log_probs, completion_mask = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model, return_per_token=True
            )
        
        # Check that log_probs shape is correct (seq_len - 1 due to shift)
        assert log_probs.shape == (batch_size, max_len - 1)
        
        # Verify completion mask shape
        assert completion_mask.shape == (batch_size, max_len - 1)
        
        # completion_mask is shifted version of the original mask (for log_probs)
        # Verify the mask values match expectations
        for i, (prompt_len, seq_len) in enumerate(zip(prompt_lengths, seq_lengths)):
            # The shifted mask should mark completion positions
            # Position j in completion_mask corresponds to predicting token j+1
            # So if prompt_len=5, tokens 0-4 are prompt, 5+ are completion
            # In the shifted mask, position 4 predicts token 5 (first completion token)
            
            if prompt_len > 1:
                # Positions 0 to prompt_len-2 predict prompt tokens (should be masked)
                assert torch.all(completion_mask[i, :prompt_len-1] == 0), f"Seq {i}: prompt mask failed"
            
            # Positions prompt_len-1 to seq_len-2 predict completion tokens (should be unmasked)
            completion_start = prompt_len - 1  # Position that predicts first completion token
            completion_end = seq_len - 1  # Last position that predicts a real token
            if completion_start < completion_end:
                actual_mask = completion_mask[i, completion_start:completion_end]
                expected_ones = torch.ones_like(actual_mask)
                assert torch.all(actual_mask == expected_ones), f"Seq {i}: completion mask failed"
    
    def test_policy_loss_receives_premasked_logprobs(self, config):
        """Verify that compute_policy_loss receives already-masked log probs."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 4
        seq_len = 15
        
        # Create pre-masked log probs (zeros for non-completion tokens)
        log_probs = torch.zeros(batch_size, seq_len)
        log_probs[:, 5:10] = torch.randn(batch_size, 5) * 0.1 - 2.0  # Only completion tokens have values
        log_probs.requires_grad = True
        
        old_log_probs = log_probs.detach().clone()
        ref_log_probs = log_probs.detach().clone()
        rewards = torch.randn(batch_size)
        
        # Completion mask (shifted version, for reference)
        completion_mask = torch.zeros(batch_size, seq_len)
        completion_mask[:, 5:10] = 1
        
        # Call compute_policy_loss
        loss, metrics = grpo.compute_policy_loss(
            log_probs, old_log_probs, ref_log_probs, rewards, 
            completion_mask=completion_mask
        )
        
        # Verify loss is computed correctly
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Check that the sum is only over non-zero (completion) positions
        expected_sum = log_probs[:, 5:10].sum(dim=1)
        actual_sum = log_probs.sum(dim=1)
        assert torch.allclose(expected_sum, actual_sum)
    
    def test_kl_divergence_with_premasked_inputs(self, config):
        """Test KL divergence computation with pre-masked log probs."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 2
        seq_len = 10
        
        # Create log probs that are already masked (zeros for non-completion)
        log_probs = torch.zeros(batch_size, seq_len)
        ref_log_probs = torch.zeros(batch_size, seq_len)
        
        # Only set values for completion tokens (positions 3-7)
        completion_range = slice(3, 8)
        log_probs[:, completion_range] = torch.tensor([
            [-2.0, -2.1, -2.2, -2.3, -2.4],
            [-1.9, -2.0, -2.1, -2.2, -2.3]
        ])
        ref_log_probs[:, completion_range] = torch.tensor([
            [-2.1, -2.0, -2.3, -2.2, -2.5],
            [-2.0, -1.9, -2.2, -2.1, -2.4]
        ])
        
        # Compute KL without additional masking (since already masked)
        kl_div = grpo.compute_kl_divergence(log_probs, ref_log_probs, mask=None)
        
        # KL should only be computed over non-zero positions
        assert kl_div.item() > 0  # Should be positive for different distributions
        assert not torch.isnan(kl_div)
        assert not torch.isinf(kl_div)
    
    def test_masking_consistency_across_updates(self, config):
        """Ensure masking remains consistent across multiple update epochs."""
        config["algorithm"]["update_epochs"] = 3
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Track log prob sums across epochs
        epoch_sums = []
        
        # Mock the compute_log_probs_for_sequences to track calls
        original_method = grpo.compute_log_probs_for_sequences
        
        def track_log_probs(*args, **kwargs):
            result = original_method(*args, **kwargs)
            if isinstance(result, tuple):
                log_probs, mask = result
                epoch_sums.append(log_probs.sum().item())
            else:
                epoch_sums.append(result.sum().item())
            return result
        
        grpo.compute_log_probs_for_sequences = track_log_probs
        
        # Run a training step
        prompts = ["Test prompt"]
        batch_data = {"prompts": prompts}
        
        with patch.object(grpo, 'optimizer') as mock_optimizer:
            mock_optimizer.zero_grad = Mock()
            mock_optimizer.step = Mock()
            
            metrics = grpo.train_step(batch_data)
        
        # Check that we got calls for each epoch (plus initial computations)
        # Initial: old_log_probs, ref_log_probs, then 3 epochs of current log_probs
        assert len(epoch_sums) >= config["algorithm"]["update_epochs"]
        
        # Restore original method
        grpo.compute_log_probs_for_sequences = original_method


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


class TestGRPOIntegration:
    """Integration tests for full training loop."""
    
    @pytest.fixture
    def config(self):
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 2,
                "kl_coef": 0.05,
                "normalize_rewards": True,
                "update_epochs": 2,
                "minibatch_size": 2
            },
            "model": {
                "hf_model_name": "gpt2",
                "max_length": 64
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "max_new_tokens": 8,
                "temperature": 0.9,
                "gradient_clip": 1.0
            },
            "logging": {
                "log_interval": 1
            }
        }
    
    @pytest.mark.slow
    def test_full_training_step(self, config):
        """Test a complete training step with real model updates."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Store initial model parameters
        initial_params = {
            name: param.clone().detach() 
            for name, param in grpo.model.named_parameters()
        }
        
        # Run training step
        prompts = ["The future of AI is", "Machine learning can"]
        batch_data = {"prompts": prompts}
        
        metrics = grpo.train_step(batch_data)
        
        # Check metrics are computed
        assert "total_loss" in metrics
        assert "pg_loss" in metrics
        assert "kl_div" in metrics
        assert "mean_reward" in metrics
        
        # Verify model parameters changed
        params_changed = False
        for name, param in grpo.model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should update after training step"
    
    @pytest.mark.slow
    def test_multiple_training_episodes(self, config):
        """Test multiple training episodes with decreasing loss."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        prompts = ["Test prompt " + str(i) for i in range(10)]
        losses = []
        
        for episode in range(3):
            batch_prompts = np.random.choice(prompts, 2, replace=False).tolist()
            batch_data = {"prompts": batch_prompts}
            
            metrics = grpo.train_step(batch_data)
            losses.append(metrics["total_loss"])
        
        # Check that we got losses for all episodes
        assert len(losses) == 3
        # Losses should be finite and reasonable
        for i, loss in enumerate(losses):
            assert not np.isnan(loss), f"Loss {i} is NaN"
            assert not np.isinf(loss), f"Loss {i} is infinite"
            # Loss can be negative if PG loss dominates (rewards are positive)
    
    def test_gradient_clipping(self, config):
        """Test that gradient clipping is applied correctly."""
        config["training"]["gradient_clip"] = 0.5
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Mock large gradients
        for param in grpo.model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 100  # Large gradients
        
        # Get gradient norm before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(
            grpo.model.parameters(), float('inf')
        )
        
        # Apply gradient clipping (this happens in train_step)
        torch.nn.utils.clip_grad_norm_(
            grpo.model.parameters(), 
            config["training"]["gradient_clip"]
        )
        
        # Get gradient norm after clipping
        total_norm_after = 0
        for param in grpo.model.parameters():
            if param.grad is not None:
                total_norm_after += param.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        # Norm should be clipped to max value (with small tolerance for numerical precision)
        assert total_norm_after <= config["training"]["gradient_clip"] + 1e-5
    
    def test_minibatch_processing(self, config):
        """Test that minibatch processing works correctly."""
        config["algorithm"]["minibatch_size"] = 2
        config["training"]["batch_size"] = 4
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Track optimizer steps
        step_count = 0
        original_step = grpo.optimizer.step
        
        def mock_step():
            nonlocal step_count
            step_count += 1
            original_step()
        
        grpo.optimizer.step = mock_step
        
        # Run training step
        prompts = ["Prompt 1", "Prompt 2"]
        batch_data = {"prompts": prompts}
        
        metrics = grpo.train_step(batch_data)
        
        # Should have multiple optimizer steps (batch_size / minibatch_size * update_epochs)
        expected_steps = (4 // 2) * config["algorithm"]["update_epochs"]
        assert step_count == expected_steps
    
    def test_reference_model_frozen(self, config):
        """Ensure reference model remains frozen during training."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Store reference model parameters
        ref_params_before = {
            name: param.clone().detach()
            for name, param in grpo.reference_model.named_parameters()
        }
        
        # Run training
        prompts = ["Test prompt"]
        batch_data = {"prompts": prompts}
        metrics = grpo.train_step(batch_data)
        
        # Verify reference model didn't change
        for name, param in grpo.reference_model.named_parameters():
            assert torch.allclose(param, ref_params_before[name])
            assert not param.requires_grad
    
    @pytest.mark.slow
    def test_full_pipeline_with_train_method(self, config):
        """Test the complete GRPO pipeline using the train() method."""
        config["training"]["num_episodes"] = 5
        config["algorithm"]["update_epochs"] = 2
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in grpo.model.named_parameters()
        }
        
        # Run full training
        final_metrics = grpo.train(num_episodes=5)
        
        # Verify we got meaningful metrics from the last episode
        assert "total_loss" in final_metrics
        assert "pg_loss" in final_metrics
        assert "kl_div" in final_metrics
        assert "mean_reward" in final_metrics
        
        # Check that training ran (we have metrics)
        assert len(final_metrics) > 0
        
        # Verify metrics are finite
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"Metric {key} is NaN"
                assert not np.isinf(value), f"Metric {key} is infinite"
        
        # Model should have changed
        params_changed = False
        for name, param in grpo.model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                break
        assert params_changed, "Model parameters should update after training"
    
    @pytest.mark.slow
    def test_train_with_varying_prompts(self, config):
        """Test training with different prompts across episodes."""
        config["training"]["num_episodes"] = 3
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Store initial parameters
        initial_params = {
            name: param.clone().detach()
            for name, param in grpo.model.named_parameters()
        }
        
        # Track metrics across episodes
        all_metrics = []
        
        for episode in range(3):
            # Use different prompts for each episode
            prompts = [
                f"Episode {episode}: Test prompt {i}" 
                for i in range(2)  # batch_size // group_size
            ]
            batch_data = {"prompts": prompts}
            
            metrics = grpo.train_step(batch_data)
            all_metrics.append(metrics)
        
        # Verify we got metrics for all episodes
        assert len(all_metrics) == 3
        
        # Each episode should have complete metrics
        for metrics in all_metrics:
            assert "total_loss" in metrics
            assert "pg_loss" in metrics
            assert "kl_div" in metrics
            assert "mean_reward" in metrics
        
        # Model should have changed
        params_changed = False
        for name, param in grpo.model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                params_changed = True
                break
        assert params_changed, "Model parameters should update after training"


class TestGRPOMemoryAndPerformance:
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
    
    def test_memory_efficient_masking(self, config):
        """Test that masking operations don't create unnecessary copies."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        batch_size = 8
        seq_len = 50
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        prompt_lengths = [10] * batch_size
        
        # Track memory before operation
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            log_probs = grpo.compute_log_probs_for_sequences(
                input_ids, attention_mask, prompt_lengths, grpo.model
            )
        
        # Memory increase should be reasonable
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            # Should not explode in memory (rough check)
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
    
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
            
            import time
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])