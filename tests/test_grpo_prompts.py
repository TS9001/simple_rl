"""
Tests for GRPO prompt customization features.
"""

import pytest
import torch
from simple_rl.algorithms.grpo import GRPO


class TestGRPOPromptCustomization:
    """Test prompt formatting and customization features."""
    
    @pytest.fixture
    def config(self):
        """Basic GRPO configuration for testing."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 2,
                "kl_coef": 0.1,
                "normalize_rewards": True,
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "max_new_tokens": 10,
                "temperature": 1.0,
            },
            "model": {
                "hidden_dim": 64,
                "vocab_size": 1000,
                "max_length": 128,
            },
            "generation": {
                "system_prompt": "You are a helpful assistant.",
                "prompt_template": "User: {prompt}\nAssistant:",
                "response_prefix": " ",
            }
        }
    
    def test_prompt_formatting(self, config):
        """Test that prompts are formatted correctly."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test with configured prompts
        raw_prompt = "What is the capital of France?"
        formatted = grpo.format_prompt(raw_prompt)
        
        expected = "You are a helpful assistant.\n\nUser: What is the capital of France?\nAssistant: "
        assert formatted == expected
    
    def test_prompt_formatting_without_config(self):
        """Test prompt formatting when no configuration is provided."""
        config = {
            "algorithm": {"name": "grpo", "group_size": 2},
            "training": {"batch_size": 4, "learning_rate": 1e-4},
            "model": {"hidden_dim": 64, "vocab_size": 1000},
        }
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        raw_prompt = "Test prompt"
        formatted = grpo.format_prompt(raw_prompt)
        
        # Without config, should return unchanged
        assert formatted == raw_prompt
    
    def test_set_generation_prompt(self, config):
        """Test dynamically setting generation prompts."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Change the prompts
        grpo.set_generation_prompt(
            system_prompt="New system prompt",
            prompt_template="[INST] {prompt} [/INST]",
            response_prefix=""
        )
        
        raw_prompt = "Hello"
        formatted = grpo.format_prompt(raw_prompt)
        
        expected = "New system prompt\n\n[INST] Hello [/INST]"
        assert formatted == expected
    
    def test_reset_generation_prompt(self, config):
        """Test resetting generation prompts."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Reset all prompts
        grpo.set_generation_prompt(reset=True)
        
        assert grpo.system_prompt is None
        assert grpo.generation_prompt_template is None
        assert grpo.response_prefix is None
        
        # Formatting should return original
        raw_prompt = "Test"
        assert grpo.format_prompt(raw_prompt) == raw_prompt
    
    def test_partial_prompt_update(self, config):
        """Test updating only some prompt components."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Update only system prompt, keeping others
        grpo.set_generation_prompt(system_prompt="Updated system")
        
        assert grpo.system_prompt == "Updated system"
        # Template should remain unchanged
        assert grpo.generation_prompt_template == "User: {prompt}\nAssistant:"
        assert grpo.response_prefix == " "
    
    def test_generate_trajectories_with_formatting(self, config):
        """Test that generate_trajectories uses formatting."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        prompts = ["Question 1", "Question 2"]
        
        # Generate with formatting (default)
        trajectories = grpo.generate_trajectories(prompts, num_samples_per_prompt=2)
        
        # Check that formatted prompts are stored
        assert "formatted_prompts" in trajectories
        assert "original_prompts" in trajectories
        
        # Check formatting was applied
        for i, orig in enumerate(trajectories["original_prompts"][:2]):
            assert orig == "Question 1"  # First two should be Question 1
            formatted = trajectories["formatted_prompts"][i]
            assert "You are a helpful assistant." in formatted
            assert "User: Question 1\nAssistant:" in formatted
    
    def test_generate_trajectories_without_formatting(self, config):
        """Test generating trajectories without formatting."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        prompts = ["Question 1"]
        
        # Generate without formatting
        trajectories = grpo.generate_trajectories(
            prompts, 
            num_samples_per_prompt=2,
            use_formatting=False
        )
        
        # Check that no formatting was applied
        for formatted in trajectories["formatted_prompts"]:
            assert formatted == "Question 1"
    
    def test_prompt_template_with_placeholder(self, config):
        """Test that prompt template correctly replaces {prompt} placeholder."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Set a complex template
        grpo.set_generation_prompt(
            system_prompt=None,
            prompt_template="### Instruction:\n{prompt}\n\n### Response:",
            response_prefix="\n"
        )
        
        raw_prompt = "Explain quantum physics"
        formatted = grpo.format_prompt(raw_prompt)
        
        expected = "### Instruction:\nExplain quantum physics\n\n### Response:\n"
        assert formatted == expected
    
    def test_multiple_prompt_formats(self, config):
        """Test different prompt format styles."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        # Test ChatML format
        grpo.set_generation_prompt(
            system_prompt="<|system|>\nYou are helpful.<|end|>",
            prompt_template="<|user|>\n{prompt}<|end|>\n<|assistant|>",
            response_prefix=""
        )
        
        formatted = grpo.format_prompt("Hello")
        assert "<|system|>" in formatted
        assert "<|user|>" in formatted
        assert "<|assistant|>" in formatted
        
        # Test Alpaca format
        grpo.set_generation_prompt(
            system_prompt="Below is an instruction that describes a task.",
            prompt_template="### Instruction:\n{prompt}\n\n### Response:",
            response_prefix=" "
        )
        
        formatted = grpo.format_prompt("Write a poem")
        assert "### Instruction:" in formatted
        assert "### Response:" in formatted