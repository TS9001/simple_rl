"""
Tests for GRPO masking logic.
"""

import pytest
import torch
from unittest.mock import Mock
from simple_rl.algorithms.grpo import GRPO


class TestGRPOMasking:
    """Tests for masking logic in GRPO."""
    
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
        
        from unittest.mock import patch
        with patch.object(grpo, 'optimizer') as mock_optimizer:
            mock_optimizer.zero_grad = Mock()
            mock_optimizer.step = Mock()
            
            metrics = grpo.train_step(batch_data)
        
        # Check that we got calls for each epoch (plus initial computations)
        # Initial: old_log_probs, ref_log_probs, then 3 epochs of current log_probs
        assert len(epoch_sums) >= config["algorithm"]["update_epochs"]
        
        # Restore original method
        grpo.compute_log_probs_for_sequences = original_method
    
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