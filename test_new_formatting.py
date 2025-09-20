"""
Test the new prompt formatting architecture.
"""

from simple_rl.algorithms.grpo import GRPO
from simple_rl.algorithms.supervised_learning import SupervisedLearning
import torch.nn as nn


class DummyModel(nn.Module):
    """Dummy model for testing supervised learning."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


def test_grpo_formatting():
    """Test GRPO formatting with chat templates."""

    print("=" * 60)
    print("Testing GRPO with chat template formatting")
    print("=" * 60)

    # Config for a chat model (like Qwen-Instruct)
    config_chat = {
        "model": {
            "hf_model_name": "gpt2",  # Would be "Qwen/Qwen2.5-0.5B-Instruct" in real use
            "max_length": 128,
            "use_chat_template": False  # GPT-2 doesn't have chat template, but pretend it does
        },
        "formatting": {
            "system_prompt": "You are a helpful math tutor.",
            "task_formatter": lambda p: f"Problem: {p}\nSolution:"  # Task-specific
        },
        "algorithm": {"group_size": 2, "kl_coef": 0.1},
        "training": {"batch_size": 2, "learning_rate": 1e-5}
    }

    grpo = GRPO(config=config_chat, use_wandb=False)

    # Test formatting
    test_prompt = "What is 5 + 3?"
    formatted = grpo.format_prompt(test_prompt)

    print(f"Original prompt: {test_prompt}")
    print(f"After task formatter: Problem: {test_prompt}\nSolution:")
    print(f"Final formatted: {formatted}")
    print()

    # Test with string template
    grpo.set_task_formatter("Question: {prompt}\nAnswer:")
    formatted2 = grpo.format_prompt(test_prompt)
    print(f"With string template: {formatted2}")
    print()

    # Test without formatting
    formatted3 = grpo.format_prompt(test_prompt, use_formatting=False)
    print(f"Without formatting: {formatted3}")


def test_supervised_formatting():
    """Test SupervisedLearning formatting."""

    print("\n" + "=" * 60)
    print("Testing SupervisedLearning formatting")
    print("=" * 60)

    config = {
        "formatting": {
            "system_prompt": "You are an expert classifier.",
            "task_formatter": "Classify this: {prompt}\nCategory:"
        },
        "training": {"learning_rate": 1e-3}
    }

    model = DummyModel()
    sl = SupervisedLearning(model, config, use_wandb=False)

    test_prompt = "This movie was amazing!"
    formatted = sl.format_prompt(test_prompt)

    print(f"Original prompt: {test_prompt}")
    print(f"Formatted: {formatted}")


def test_real_chat_template():
    """Test with a model that actually has chat templates."""

    print("\n" + "=" * 60)
    print("Testing with actual chat template (if model supports it)")
    print("=" * 60)

    # This would work with models like Qwen-Instruct, Llama-Chat, etc.
    config = {
        "model": {
            "hf_model_name": "gpt2",  # Change to chat model for real test
            "max_length": 128,
            "use_chat_template": True  # Enable chat template
        },
        "formatting": {
            "system_prompt": "You are a helpful assistant.",
            "task_formatter": None  # No task formatting, just chat template
        },
        "algorithm": {"group_size": 2, "kl_coef": 0.1},
        "training": {"batch_size": 2, "learning_rate": 1e-5}
    }

    try:
        grpo = GRPO(config=config, use_wandb=False)
        test_prompt = "Hello, how are you?"
        formatted = grpo.format_prompt(test_prompt)

        print(f"Original: {test_prompt}")
        print(f"With chat template: {formatted}")
        print("(Note: GPT-2 doesn't have chat template, so output is unchanged)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_grpo_formatting()
    test_supervised_formatting()
    test_real_chat_template()
    print("\n✓ All formatting tests completed!")