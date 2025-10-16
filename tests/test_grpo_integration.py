"""
Integration tests for GRPO training pipeline.

These tests validate the full training pipeline with realistic configurations,
ensuring that training behaviors are correct and consistent.
"""

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.algorithms import GRPO
from simple_rl.rewards import compute_math_rewards_batch


@pytest.mark.slow
class TestGRPOIntegration:
    """Integration tests for GRPO training."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @pytest.fixture
    def grpo_config(self, device):
        """Create realistic GRPO config for testing."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 4,
                "kl_coef": 0.1,
                "clip_epsilon": 0.2,
                "normalize_rewards": True,
                "store_completions": True,
            },
            "training": {
                "batch_size": 4,
                "rollout_batch_size": 4,
                "gradient_clip": 1.0,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "num_episodes": 10,
                "minibatch_size": 4,
                "update_epochs": 1,
                "top_p": 0.9,
                "entropy_coef": 0.01,
                "policy_loss_type": "token",  # Options: "sequence" or "token"
                # Clipping parameters
                "kl_clamp_min": -2.0,
                "kl_clamp_max": 2.0,
                "kl_reduction": "mean",  # Options: "mean" (average per token) or "sum" (total)
                "policy_log_ratio_clamp_min": -2.0,
                "policy_log_ratio_clamp_max": 2.0,
                "advantage_clip_min": -3.0,
                "advantage_clip_max": 3.0,
                "stop_sequences": ["</answer>"],  # Multi-token stopping for early termination
            },
            "model": {
                "max_length": 512,
                "model_name": "gpt2",
                "model_type": "fp32",
                "device": device,
                "compile": {"enabled": False}
            },
            "device_optimizations": {
                "clear_cache_on_mps": True,
            },
            "logging": {
                "log_interval": 2,
                "save_interval": 1000,
                "show_trajectory_progress": False,
            },
            "validation": {
                "enabled": False,
            },
            "wandb": {
                "enabled": False,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1e-5,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "fused": False
            },
            "optimization": {
                "mixed_precision": {
                    "enabled": False,
                    "dtype": "fp32",
                    "mode": "fp32"
                }
            },
            "timing": {
                "enabled": False,
            },
            "debug": {
                "enabled": False,
            }
        }

    @pytest.fixture
    def math_dataset(self):
        """Create a small math dataset for testing."""
        prompts = [
            "What is 5 + 3?",
            "What is 10 - 4?",
            "What is 2 * 6?",
            "What is 15 / 3?",
        ]
        answers = ["8", "6", "12", "5"]

        return {
            "prompts": prompts,
            "answers": answers
        }

    def test_training_reduces_loss(self, grpo_config, math_dataset, device):
        """Test that training reduces loss over time."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train
        results = grpo.train(
            train_data=math_dataset,
            val_data=None,
            num_episodes=10
        )

        metrics = results["training_metrics"]

        # Check that loss generally decreases (allowing for some noise)
        initial_loss = np.mean(metrics["total_loss"][:3])
        final_loss = np.mean(metrics["total_loss"][-3:])

        # Loss should decrease or stay similar (not increase significantly)
        assert final_loss <= initial_loss * 1.5, \
            f"Loss increased too much: {initial_loss:.3f} -> {final_loss:.3f}"

    def test_kl_divergence_bounded(self, grpo_config, math_dataset, device):
        """Test that KL divergence stays bounded during training."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train
        results = grpo.train(
            train_data=math_dataset,
            val_data=None,
            num_episodes=10
        )

        metrics = results["training_metrics"]

        # KL divergence should stay reasonable (not explode)
        max_kl = max(metrics["kl_divergence"])
        mean_kl = np.mean(metrics["kl_divergence"])

        # KL should be non-negative
        assert all(kl >= 0 for kl in metrics["kl_divergence"]), "KL divergence should be >= 0"

        # KL should not explode
        assert max_kl < 10.0, f"KL divergence exploded: max={max_kl}"
        assert mean_kl < 5.0, f"KL divergence too high: mean={mean_kl}"

    def test_gradient_norms_tracked(self, grpo_config, math_dataset, device):
        """Test that gradient norms are tracked correctly."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train
        results = grpo.train(
            train_data=math_dataset,
            val_data=None,
            num_episodes=5
        )

        metrics = results["training_metrics"]

        # Check gradient norms are tracked
        assert "grad_norm" in metrics, "Gradient norms should be tracked"
        assert len(metrics["grad_norm"]) == 5, "Should have gradient norm for each episode"

        # Check gradient norms are reasonable
        grad_norms = metrics["grad_norm"]
        assert all(gn >= 0 for gn in grad_norms), "Gradient norms should be >= 0"
        assert all(gn < 100 for gn in grad_norms), f"Gradient norms too large: {grad_norms}"

    def test_completions_stored_when_enabled(self, grpo_config, math_dataset, device):
        """Test that completions are stored when store_completions=True."""
        # Ensure store_completions is enabled
        grpo_config["algorithm"]["store_completions"] = True

        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Generate trajectories with store_outputs=True
        prompts = math_dataset["prompts"][:2]
        answers = math_dataset["answers"][:2]

        prompts_out, completions, rewards, completion_mask, format_rewards, correctness_rewards, generated_ids, attention_mask, prompt_end_positions = grpo.generate_trajectories(
            prompts=prompts,
            answers=answers,
            store_outputs=True
        )

        # Check completions are returned
        assert len(completions) == len(prompts) * grpo.group_size
        assert all(isinstance(c, str) for c in completions)
        assert all(len(c) > 0 for c in completions)

    def test_reward_breakdown(self, grpo_config, math_dataset, device):
        """Test that reward breakdown works correctly."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train for one episode
        results = grpo.train(
            train_data=math_dataset,
            val_data=None,
            num_episodes=1
        )

        metrics = results["training_metrics"]

        # Check that format and correctness rewards are tracked
        assert "format_reward_mean" in metrics
        assert "correctness_reward_mean" in metrics

        # Check values are reasonable
        format_reward = metrics["format_reward_mean"][0]
        correctness_reward = metrics["correctness_reward_mean"][0]

        # Format reward should be in [0, 0.3]
        assert 0.0 <= format_reward <= 0.3, \
            f"Format reward out of range: {format_reward}"

        # Correctness reward should be in [0, 2.0]
        assert 0.0 <= correctness_reward <= 2.0, \
            f"Correctness reward out of range: {correctness_reward}"

    def test_multi_epoch_updates(self, grpo_config, math_dataset, device):
        """Test that multi-epoch updates work correctly."""
        # Set update_epochs > 1
        grpo_config["training"]["update_epochs"] = 2

        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train
        results = grpo.train(
            train_data=math_dataset,
            val_data=None,
            num_episodes=3
        )

        # Should complete without errors
        assert "training_metrics" in results
        assert len(results["training_metrics"]["episode"]) == 3

    def test_different_group_sizes(self, grpo_config, math_dataset, device):
        """Test GRPO with different group sizes."""
        for group_size in [2, 4, 8]:
            grpo_config["algorithm"]["group_size"] = group_size
            grpo_config["training"]["batch_size"] = group_size  # Match batch size
            grpo_config["training"]["minibatch_size"] = group_size

            # Load model
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            # Initialize GRPO
            grpo = GRPO(
                config=grpo_config,
                batch_reward_fn=compute_math_rewards_batch,
                model=model,
                tokenizer=tokenizer,
                use_wandb=False
            )

            # Train for 2 episodes
            results = grpo.train(
                train_data=math_dataset,
                val_data=None,
                num_episodes=2
            )

            # Should complete successfully
            assert len(results["training_metrics"]["episode"]) == 2

    def test_advantage_normalization(self, grpo_config, device):
        """Test that advantage normalization works correctly."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Enable normalization
        grpo_config["algorithm"]["normalize_rewards"] = True

        # Initialize GRPO
        grpo = GRPO(
            config=grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Create mock data with clear reward differences
        device_obj = torch.device(device)
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device=device_obj)
        log_probs = torch.zeros(8, 10, device=device_obj)
        ref_log_probs = torch.zeros(8, 10, device=device_obj)

        # Compute advantages
        advantages = grpo.compute_advantages(rewards, log_probs, ref_log_probs)

        # Check normalization within groups (group_size=4)
        group1 = advantages[:4]
        group2 = advantages[4:]

        # Each group should be normalized (mean~0, std~1)
        assert abs(group1.mean().item()) < 0.2, f"Group 1 mean not normalized: {group1.mean()}"
        assert abs(group2.mean().item()) < 0.2, f"Group 2 mean not normalized: {group2.mean()}"

        # Std should be close to 1 (allowing some tolerance for small samples)
        assert 0.5 < group1.std().item() < 1.5, f"Group 1 std not normalized: {group1.std()}"
        assert 0.5 < group2.std().item() < 1.5, f"Group 2 std not normalized: {group2.std()}"


class TestTrainingBehaviors:
    """Test expected training behaviors."""

    @pytest.mark.slow
    def test_reward_improves_or_stable(self):
        """Test that rewards improve or stay stable during training."""
        device = "cpu"

        config = {
            "algorithm": {"group_size": 4, "kl_coef": 0.05, "normalize_rewards": True},
            "training": {
                "batch_size": 4,
                "rollout_batch_size": 4,
                "max_new_tokens": 50,
                "num_episodes": 10,
                "minibatch_size": 4,
                "update_epochs": 1,
                "policy_loss_type": "token",  # Options: "sequence" or "token"
                # Clipping parameters
                "kl_clamp_min": -2.0,
                "kl_clamp_max": 2.0,
                "kl_reduction": "mean",  # Options: "mean" (average per token) or "sum" (total)
                "policy_log_ratio_clamp_min": -2.0,
                "policy_log_ratio_clamp_max": 2.0,
                "advantage_clip_min": -3.0,
                "advantage_clip_max": 3.0,
                "stop_sequences": ["</answer>"],  # Multi-token stopping for early termination
            },
            "model": {"model_name": "gpt2", "device": device, "max_length": 256},
            "logging": {"log_interval": 1, "save_interval": 1000, "show_trajectory_progress": False},
            "validation": {"enabled": False},
            "wandb": {"enabled": False},
            "optimizer": {"type": "adamw", "lr": 1e-5},
            "optimization": {"mixed_precision": {"enabled": False}},
            "debug": {"enabled": False}
        }

        # Simple increasing reward function
        def increasing_reward_fn(completions, answers=None, device=None, return_breakdown=False):
            """Reward increases with length."""
            rewards = torch.tensor([min(len(c) / 50.0, 1.0) for c in completions], device=device)
            if return_breakdown:
                format_r = torch.zeros_like(rewards)
                correctness_r = rewards.clone()
                return rewards, format_r, correctness_r
            return rewards

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        grpo = GRPO(
            config=config,
            batch_reward_fn=increasing_reward_fn,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        data = {
            "prompts": ["Write a story", "Describe a sunset"] * 2,
            "answers": [None, None, None, None]
        }

        results = grpo.train(train_data=data, val_data=None, num_episodes=10)
        metrics = results["training_metrics"]

        # Rewards should improve or stay stable (not decrease significantly)
        initial_reward = np.mean(metrics["reward_mean"][:3])
        final_reward = np.mean(metrics["reward_mean"][-3:])

        # Allow some noise, but final should be >= 70% of initial
        assert final_reward >= initial_reward * 0.7, \
            f"Rewards degraded: {initial_reward:.3f} -> {final_reward:.3f}"

    @pytest.mark.slow
    def test_no_nan_or_inf_in_training(self):
        """Test that training never produces NaN or Inf values."""
        device = "cpu"

        config = {
            "algorithm": {"group_size": 2, "kl_coef": 0.1, "normalize_rewards": True},
            "training": {
                "batch_size": 2,
                "rollout_batch_size": 2,
                "max_new_tokens": 30,
                "num_episodes": 5,
                "minibatch_size": 2,
                "update_epochs": 1,
                "policy_loss_type": "token",  # Options: "sequence" or "token"
                # Clipping parameters
                "kl_clamp_min": -2.0,
                "kl_clamp_max": 2.0,
                "kl_reduction": "mean",  # Options: "mean" (average per token) or "sum" (total)
                "policy_log_ratio_clamp_min": -2.0,
                "policy_log_ratio_clamp_max": 2.0,
                "advantage_clip_min": -3.0,
                "advantage_clip_max": 3.0,
                "stop_sequences": ["</answer>"],  # Multi-token stopping for early termination
            },
            "model": {"model_name": "gpt2", "device": device, "max_length": 128},
            "logging": {"log_interval": 1, "save_interval": 1000, "show_trajectory_progress": False},
            "validation": {"enabled": False},
            "wandb": {"enabled": False},
            "optimizer": {"type": "adamw", "lr": 1e-5},
            "optimization": {"mixed_precision": {"enabled": False}},
            "debug": {"enabled": False}
        }

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        grpo = GRPO(
            config=config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        data = {
            "prompts": ["What is 1+1?", "What is 2+2?"],
            "answers": ["2", "4"]
        }

        results = grpo.train(train_data=data, val_data=None, num_episodes=5)
        metrics = results["training_metrics"]

        # Check all metrics for NaN/Inf
        for key in ["total_loss", "pg_loss", "kl_divergence", "reward_mean", "grad_norm"]:
            if key in metrics:
                values = metrics[key]
                assert not any(np.isnan(v) for v in values), f"{key} contains NaN"
                assert not any(np.isinf(v) for v in values), f"{key} contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
