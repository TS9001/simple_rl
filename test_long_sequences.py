"""
Test GRPO with longer sequences (max_new_tokens=500) to reproduce the log prob mismatch issue.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from simple_rl.algorithms import GRPO
from simple_rl.rewards import compute_math_rewards_batch


def test_long_sequences():
    """Test GRPO with max_new_tokens=500."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Config with longer sequences
    config = {
        "algorithm": {
            "name": "grpo",
            "group_size": 4,  # Group size of 4
            "kl_coef": 0.1,
            "clip_epsilon": 0.2,
            "normalize_rewards": True,
            "store_completions": False,
        },
        "training": {
            "batch_size": 8,  # 8 prompts * 4 group_size = 32 total sequences per batch
            "rollout_batch_size": 8,
            "gradient_clip": 1.0,
            "max_new_tokens": 500,  # LONG sequences - this is what triggers the bug
            "temperature": 0.7,
            "num_episodes": 1,  # Just 1 episode
            "minibatch_size": 4,  # Small minibatch
            "update_epochs": 1,
            "top_p": 0.9,
            "entropy_coef": 0.01,
            "policy_loss_type": "token",
            "kl_clamp_min": -2.0,
            "kl_clamp_max": 2.0,
            "kl_reduction": "mean",
            "policy_log_ratio_clamp_min": -2.0,
            "policy_log_ratio_clamp_max": 2.0,
            "advantage_clip_min": -3.0,
            "advantage_clip_max": 3.0,
            "stop_sequences": ["</answer>"],
        },
        "model": {
            "max_length": 1024,  # Longer context
            "model_name": "gpt2",
            "model_type": "fp32",
            "device": device,
            "compile": {"enabled": False}
        },
        "device_optimizations": {
            "clear_cache_on_mps": True,
        },
        "logging": {
            "log_interval": 1,
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

    # Simple dataset with 8 prompts
    prompts = [
        "What is 2 + 2?",
        "What is 5 * 3?",
        "What is 10 - 7?",
        "What is 15 / 3?",
        "What is 8 + 9?",
        "What is 12 * 2?",
        "What is 20 - 5?",
        "What is 18 / 6?",
    ]
    answers = ["4", "15", "3", "5", "17", "24", "15", "3"]

    dataset = {
        "prompts": prompts,
        "answers": answers
    }

    # Load model
    print(f"Loading model on device: {device}")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize GRPO
    print("Initializing GRPO...")
    grpo = GRPO(
        config=config,
        batch_reward_fn=compute_math_rewards_batch,
        model=model,
        tokenizer=tokenizer,
        use_wandb=False
    )

    # Train
    print("\nStarting training with max_new_tokens=500...")
    print("This should trigger the log prob mismatch issue if it still exists.\n")

    results = grpo.train(
        train_data=dataset,
        val_data=None,
        num_episodes=1
    )

    print("\n✓ Training completed successfully!")
    print(f"Final reward: {results['final_reward']:.4f}")
    print(f"Total time: {results['total_time']:.2f}s")


if __name__ == "__main__":
    test_long_sequences()
