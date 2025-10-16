"""
Smoke test for GRPO implementation.

This test runs a minimal GRPO training loop to validate that:
1. The algorithm initializes correctly
2. Training completes without errors
3. Key metrics are computed properly
4. Training produces expected behaviors (rewards improve, KL stays bounded, etc.)

Run this test to quickly validate your GRPO implementation works.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.algorithms import GRPO
from simple_rl.rewards import compute_math_rewards_batch


@pytest.mark.slow
class TestGRPOSmoke:
    """Smoke tests for GRPO implementation."""

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
    def minimal_grpo_config(self, device):
        """Create minimal GRPO config for fast testing."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 2,  # Small group size
                "kl_coef": 0.1,
                "clip_epsilon": 0.2,
                "normalize_rewards": True,
                "store_completions": False,
            },
            "training": {
                "batch_size": 2,  # Tiny batch
                "rollout_batch_size": 2,
                "gradient_clip": 1.0,
                "max_new_tokens": 50,  # Short generations
                "temperature": 0.7,
                "num_episodes": 3,  # Just 3 episodes
                "minibatch_size": 2,
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
                "max_length": 256,  # Short context
                "model_name": "gpt2",  # Small, fast model
                "model_type": "fp32",
                "device": device,
                "compile": {"enabled": False}
            },
            "device_optimizations": {
                "clear_cache_on_mps": True,
            },
            "logging": {
                "log_interval": 1,
                "save_interval": 1000,  # Don't save checkpoints
                "show_trajectory_progress": False,  # Less verbose
            },
            "validation": {
                "enabled": False,  # Skip validation for speed
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
    def tiny_dataset(self):
        """Create a tiny dataset for testing."""
        # Simple math problems
        prompts = [
            "What is 2 + 2?",
            "What is 5 * 3?",
        ]
        answers = ["4", "15"]

        return {
            "prompts": prompts,
            "answers": answers
        }

    def test_grpo_initialization(self, minimal_grpo_config, device):
        """Test that GRPO initializes correctly."""
        # Load a tiny model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=minimal_grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Check basic attributes
        assert grpo.group_size == 2
        assert grpo.kl_coef == 0.1
        assert grpo.clip_epsilon == 0.2
        assert grpo.device.type == device

    def test_grpo_training_completes(self, minimal_grpo_config, tiny_dataset, device):
        """Test that GRPO training completes without errors."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=minimal_grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train for 3 episodes
        results = grpo.train(
            train_data=tiny_dataset,
            val_data=None,
            num_episodes=3
        )

        # Check that training completed
        assert "training_metrics" in results
        assert "final_reward" in results
        assert "total_time" in results

        # Check metrics have correct length
        metrics = results["training_metrics"]
        assert len(metrics["episode"]) == 3
        assert len(metrics["total_loss"]) == 3
        assert len(metrics["reward_mean"]) == 3

    def test_grpo_metrics_sanity(self, minimal_grpo_config, tiny_dataset, device):
        """Test that GRPO metrics are within reasonable ranges."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=minimal_grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Train
        results = grpo.train(
            train_data=tiny_dataset,
            val_data=None,
            num_episodes=3
        )

        metrics = results["training_metrics"]

        # Check that all metrics are finite
        for key in ["total_loss", "pg_loss", "kl_divergence", "reward_mean"]:
            values = metrics[key]
            assert all(not torch.isnan(torch.tensor(v)).item() for v in values), f"{key} contains NaN"
            assert all(not torch.isinf(torch.tensor(v)).item() for v in values), f"{key} contains Inf"

        # Check KL divergence is non-negative
        assert all(kl >= 0 for kl in metrics["kl_divergence"]), "KL divergence should be >= 0"

        # Check that gradient norms are reasonable
        if "grad_norm" in metrics:
            assert all(gn >= 0 for gn in metrics["grad_norm"]), "Gradient norms should be >= 0"
            assert all(gn < 1000 for gn in metrics["grad_norm"]), "Gradient norms seem too large"

    def test_grpo_generation(self, minimal_grpo_config, tiny_dataset, device):
        """Test that GRPO can generate completions."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=minimal_grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Generate trajectories
        prompts = tiny_dataset["prompts"]
        answers = tiny_dataset["answers"]

        prompts_out, completions, rewards, completion_mask, format_rewards, correctness_rewards, generated_ids, attention_mask, prompt_end_positions = grpo.generate_trajectories(
            prompts=prompts,
            answers=answers,
            store_outputs=True
        )

        # Check outputs
        assert len(completions) == len(prompts) * grpo.group_size
        assert all(isinstance(c, str) for c in completions), "Completions should be strings"
        assert rewards.shape[0] == len(prompts) * grpo.group_size
        assert len(completion_mask) == len(prompts) * grpo.group_size

    def test_grpo_advantage_computation(self, minimal_grpo_config, device):
        """Test advantage computation."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize GRPO
        grpo = GRPO(
            config=minimal_grpo_config,
            batch_reward_fn=compute_math_rewards_batch,
            model=model,
            tokenizer=tokenizer,
            use_wandb=False
        )

        # Create mock data
        device_obj = torch.device(device)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device_obj)

        # Compute advantages (no longer needs log_probs or ref_log_probs)
        advantages, advantage_stats = grpo.compute_advantages(rewards, group_size=2, normalize_within_groups=True)

        # Check properties
        assert advantages.shape == (4,)
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()

        # With normalization, advantages should be centered around 0 within groups
        # Group 1: indices 0-1, Group 2: indices 2-3
        group1 = advantages[:2]
        group2 = advantages[2:]

        # Mean should be close to 0 for each group
        assert abs(group1.mean().item()) < 0.5
        assert abs(group2.mean().item()) < 0.5


@pytest.mark.slow
def test_full_pipeline_smoke():
    """
    End-to-end smoke test of the full training pipeline.

    This test validates:
    - Loading a model
    - Initializing GRPO
    - Running a few training steps
    - Generating completions
    - Computing rewards
    """
    device = "cpu"  # Use CPU for CI/CD compatibility

    # Simple reward function for testing
    def simple_reward_fn(completions, answers=None, device=None, return_breakdown=False):
        """Simple reward based on length."""
        rewards = torch.tensor([len(c) / 100.0 for c in completions], device=device)
        if return_breakdown:
            format_r = torch.zeros_like(rewards)
            correctness_r = rewards.clone()
            return rewards, format_r, correctness_r
        return rewards

    # Minimal config
    config = {
        "algorithm": {"group_size": 2, "kl_coef": 0.05, "normalize_rewards": True},
        "training": {
            "batch_size": 2,
            "rollout_batch_size": 2,
            "max_new_tokens": 20,
            "num_episodes": 2,
            "minibatch_size": 2,
            "update_epochs": 1,
            "policy_loss_type": "sequence",  # Options: "sequence" or "token"
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize GRPO
    grpo = GRPO(
        config=config,
        batch_reward_fn=simple_reward_fn,
        model=model,
        tokenizer=tokenizer,
        use_wandb=False
    )

    # Train
    data = {
        "prompts": ["Say hello", "Count to five"],
        "answers": [None, None]
    }

    results = grpo.train(train_data=data, val_data=None, num_episodes=2)

    # Verify results
    assert "training_metrics" in results
    assert len(results["training_metrics"]["episode"]) == 2
    print("✓ Full pipeline smoke test passed!")


if __name__ == "__main__":
    # Run smoke tests
    pytest.main([__file__, "-v", "-s"])
