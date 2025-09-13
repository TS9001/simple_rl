"""
GRPO training and optimization tests.
"""

import pytest
import torch
from simple_rl.algorithms.grpo import GRPO


class TestGRPOTraining:
    """Test GRPO training functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "algorithm": {
                "group_size": 2,
                "kl_coef": 0.05,
                "normalize_rewards": True
            },
            "model": {
                "hf_model_name": "gpt2",
                "max_length": 64
            },
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-5,
                "max_new_tokens": 5,
                "temperature": 0.9
            }
        }
    
    @pytest.mark.slow
    def test_train_step(self, config):
        """Test single training step."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Create batch
        batch = {
            "prompts": ["What is 2+2?", "What is 3+3?"]
        }
        
        # Run train step
        metrics = grpo.train_step(batch)
        
        # Check metrics
        assert "total_loss" in metrics
        assert "reward_mean" in metrics
        assert "kl_divergence" in metrics
        assert "pg_loss" in metrics
        
        # Loss should be finite
        assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    
    @pytest.mark.slow
    def test_train_method(self, config):
        """Test train method for multiple episodes."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Train for a few episodes
        final_metrics = grpo.train(num_episodes=2)
        
        # Check final metrics
        assert "total_loss" in final_metrics
        assert grpo.episode == 1  # 0-indexed
        assert grpo.total_steps == 2
    
    def test_gradient_clipping(self, config):
        """Test gradient clipping is applied."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Create a batch that might cause large gradients
        batch = {
            "prompts": ["Test"] * config["training"]["batch_size"]
        }
        
        # Store initial params
        initial_params = [p.clone() for p in grpo.policy.parameters()]
        
        # Train step (gradient clipping is applied internally)
        metrics = grpo.train_step(batch)
        
        # Check parameters changed but not exploded
        for initial, current in zip(initial_params, grpo.policy.parameters()):
            if current.requires_grad:
                diff = (current - initial).abs().max()
                # Parameters should change but not explode
                assert diff < 1.0  # Reasonable threshold
    
    @pytest.mark.slow
    def test_evaluate_method(self, config):
        """Test evaluation method."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Evaluate
        eval_metrics = grpo.evaluate(num_episodes=1)
        
        # Check metrics
        assert "eval_reward_mean" in eval_metrics
        assert "eval_reward_std" in eval_metrics
        
        # Should be finite
        assert torch.isfinite(torch.tensor(eval_metrics["eval_reward_mean"]))
    
    def test_checkpointing(self, config, tmp_path):
        """Test save and load checkpoint."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Train for one step
        batch = {"prompts": ["Test prompt"]}
        grpo.train_step(batch)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        grpo.save_checkpoint(str(checkpoint_path))
        
        # Create new instance and load
        grpo2 = GRPO(config=config, use_wandb=False)
        grpo2.load_checkpoint(str(checkpoint_path))
        
        # Check state is restored
        assert grpo2.total_steps == grpo.total_steps
        assert grpo2.episode == grpo.episode
        
        # Check model weights are the same
        for p1, p2 in zip(grpo.policy.parameters(), grpo2.policy.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_loss_computation(self, config):
        """Test loss computation details."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Create mock data
        batch_size = 4
        seq_len = 10
        
        log_probs = torch.randn(batch_size, seq_len, device=grpo.device) * 0.1
        advantages = torch.randn(batch_size, device=grpo.device)
        ref_log_probs = torch.randn(batch_size, seq_len, device=grpo.device) * 0.1
        
        # Compute loss
        loss, metrics = grpo.compute_loss(log_probs, advantages, ref_log_probs)
        
        # Check loss is scalar and finite
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        
        # Check metrics
        assert "pg_loss" in metrics
        assert "kl_divergence" in metrics
        assert "advantages_mean" in metrics
        assert "advantages_std" in metrics