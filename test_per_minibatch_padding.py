"""
Simple test to verify per-minibatch padding optimization works correctly.
"""

import torch
from simple_rl.algorithms.grpo import GRPO


def test_per_minibatch_padding():
    """Test that per-minibatch padding works correctly."""

    # Simple config for testing
    config = {
        "model": {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hf_model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "training": {
            "batch_size": 4,
            "rollout_batch_size": 2,  # Process in 2 batches to create variable-length sequences
            "minibatch_size": 2,
            "group_size": 2,
            "learning_rate": 1e-5,
            "max_new_tokens": 10,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 50,
            "clip_epsilon": 0.2,
            "kl_coef": 0.0,  # Disable KL for faster test
            "normalize_rewards": True,
            "update_epochs": 1,
            "gradient_clip": 1.0,
        },
        "device": "cpu",  # Use CPU for testing
        "logging": {
            "log_interval": 1,
            "save_interval": 100,
            "show_trajectory_progress": True,
        },
    }

    # Simple reward function
    def simple_reward_fn(completions, answers, device, return_breakdown=False):
        rewards = torch.ones(len(completions), device=device) * 0.5
        if return_breakdown:
            format_rewards = torch.ones(len(completions), device=device) * 0.3
            correctness_rewards = torch.ones(len(completions), device=device) * 0.2
            return rewards, format_rewards, correctness_rewards
        return rewards

    # Create GRPO instance
    print("Creating GRPO instance...")
    grpo = GRPO(config=config, batch_reward_fn=simple_reward_fn, use_wandb=False)

    # Create batch with different prompts (will generate variable-length completions)
    batch = {
        "prompts": [
            "What is 2+2?",
            "Calculate 5+3:",
        ],
        "answers": ["4", "8"],
    }

    print("\nRunning train_step with per-minibatch padding...")
    try:
        metrics = grpo.train_step(batch)

        print("\n✓ Training step completed successfully!")
        print(f"  Loss: {metrics['total_loss']:.4f}")
        print(f"  Reward: {metrics['reward_mean']:.4f}")
        print(f"  Tokens: {metrics['tokens_generated']:.0f}")

        # Check that metrics are valid
        assert torch.isfinite(torch.tensor(metrics['total_loss'])), "Loss is not finite"
        assert metrics['tokens_generated'] > 0, "No tokens generated"

        print("\n✓ All checks passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_per_minibatch_padding()
    exit(0 if success else 1)