"""
Quick test of the new formatting architecture.
"""

from simple_rl.algorithms.grpo import GRPO


def math_formatter(prompt: str) -> str:
    """Task-specific formatter for math."""
    return f"Math Problem: {prompt}\nSolution:"


def test_formatting():
    """Test the formatting chain."""

    # Config for GPT-2 (base model, no chat template)
    config_gpt2 = {
        "model": {
            "hf_model_name": "gpt2",
            "max_length": 128,
            "use_chat_template": False
        },
        "formatting": {
            "task_formatter": math_formatter,
            "system_prompt": None  # Base models don't use system prompts
        },
        "algorithm": {"group_size": 2, "kl_coef": 0.1},
        "training": {"batch_size": 2, "learning_rate": 1e-5}
    }

    print("Testing with GPT-2 (base model)...")
    grpo = GRPO(config=config_gpt2, use_wandb=False)

    test_prompt = "What is 5 + 3?"
    formatted = grpo.format_prompt(test_prompt)
    print(f"Original: {test_prompt}")
    print(f"Formatted: {formatted}")
    print("-" * 50)

    # Test with string template instead of function
    grpo.set_task_formatter("Question: {prompt}\nAnswer:")
    formatted2 = grpo.format_prompt(test_prompt)
    print(f"With string template: {formatted2}")
    print("-" * 50)

    # Test without formatting
    formatted3 = grpo.format_prompt(test_prompt, use_formatting=False)
    print(f"Without formatting: {formatted3}")
    print("=" * 50)


if __name__ == "__main__":
    test_formatting()