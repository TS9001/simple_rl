"""
Test that GRPO correctly handles (prompt, answer) pairs.
"""

from simple_rl.algorithms.grpo import GRPO
from typing import Optional


def test_reward_fn(prompt: str, completion: str, answer: Optional[str] = None) -> float:
    """Test reward function that uses the answer."""
    if answer and answer in completion:
        return 1.0  # Reward if completion contains the answer
    return 0.0


def test_grpo_with_answers():
    """Test GRPO with prompt-answer pairs."""

    # Simple config
    config = {
        "model": {
            "hf_model_name": "gpt2",
            "max_length": 128
        },
        "algorithm": {
            "group_size": 2,
            "kl_coef": 0.1,
            "normalize_rewards": True
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-5,
            "max_new_tokens": 10,
            "temperature": 1.0
        }
    }

    print("Initializing GRPO with answer-aware reward function...")
    grpo = GRPO(config=config, reward_fn=test_reward_fn, use_wandb=False)

    # Test data with prompts and answers
    batch = {
        "prompts": ["What is 2+2?", "What is the capital of France?"],
        "answers": ["4", "Paris"]
    }

    print(f"\nTest batch:")
    print(f"  Prompts: {batch['prompts']}")
    print(f"  Answers: {batch['answers']}")

    # Run a training step
    print("\nRunning training step...")
    metrics = grpo.train_step(batch)

    print(f"\nTraining metrics:")
    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Mean reward: {metrics['reward_mean']:.4f}")
    print(f"  KL divergence: {metrics['kl_divergence']:.4f}")

    # Test without answers (should still work)
    batch_no_answers = {"prompts": ["What is 3+3?"]}

    print("\nTesting without answers...")
    metrics2 = grpo.train_step(batch_no_answers)
    print(f"  Total loss: {metrics2['total_loss']:.4f}")
    print(f"  Mean reward: {metrics2['reward_mean']:.4f}")

    print("\n✓ GRPO successfully handles (prompt, answer) pairs!")


if __name__ == "__main__":
    test_grpo_with_answers()