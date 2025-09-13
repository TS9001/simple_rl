"""
GRPO prompt formatting and customization tests.
"""

import pytest
from simple_rl.algorithms.grpo import GRPO


class TestGRPOPrompts:
    """Test GRPO prompt formatting capabilities."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "algorithm": {
                "group_size": 2,
                "kl_coef": 0.1
            },
            "model": {
                "hf_model_name": "gpt2",
                "max_length": 128
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-5,
                "max_new_tokens": 10
            }
        }
    
    def test_basic_prompt_formatting(self, config):
        """Test basic prompt formatting."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Initially no formatting
        prompt = "What is 2+2?"
        formatted = grpo.format_prompt(prompt)
        assert formatted == prompt
    
    def test_system_prompt(self, config):
        """Test system prompt formatting."""
        config["generation"] = {
            "system_prompt": "You are a helpful assistant."
        }
        grpo = GRPO(config=config, use_wandb=False)
        
        prompt = "What is 2+2?"
        formatted = grpo.format_prompt(prompt)
        expected = "You are a helpful assistant.\n\nWhat is 2+2?"
        assert formatted == expected
    
    def test_prompt_template(self, config):
        """Test prompt template formatting."""
        config["generation"] = {
            "prompt_template": "Question: {prompt}\nAnswer:"
        }
        grpo = GRPO(config=config, use_wandb=False)
        
        prompt = "What is 2+2?"
        formatted = grpo.format_prompt(prompt)
        expected = "Question: What is 2+2?\nAnswer:"
        assert formatted == expected
    
    def test_response_prefix(self, config):
        """Test response prefix formatting."""
        config["generation"] = {
            "response_prefix": " Let me think..."
        }
        grpo = GRPO(config=config, use_wandb=False)
        
        prompt = "What is 2+2?"
        formatted = grpo.format_prompt(prompt)
        expected = "What is 2+2? Let me think..."
        assert formatted == expected
    
    def test_combined_formatting(self, config):
        """Test all formatting options combined."""
        config["generation"] = {
            "system_prompt": "You are a math tutor.",
            "prompt_template": "Student: {prompt}",
            "response_prefix": "\nTutor:"
        }
        grpo = GRPO(config=config, use_wandb=False)
        
        prompt = "What is 2+2?"
        formatted = grpo.format_prompt(prompt)
        expected = "You are a math tutor.\n\nStudent: What is 2+2?\nTutor:"
        assert formatted == expected
    
    def test_set_generation_prompt(self, config):
        """Test dynamically setting generation prompts."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Set new prompts
        grpo.set_generation_prompt(
            system_prompt="New system",
            prompt_template="Q: {prompt}",
            response_prefix=" A:"
        )
        
        prompt = "Test"
        formatted = grpo.format_prompt(prompt)
        expected = "New system\n\nQ: Test A:"
        assert formatted == expected
    
    def test_disable_formatting(self, config):
        """Test disabling prompt formatting."""
        config["generation"] = {
            "system_prompt": "System",
            "prompt_template": "Template: {prompt}"
        }
        grpo = GRPO(config=config, use_wandb=False)
        
        prompt = "Test"
        
        # With formatting
        formatted = grpo.format_prompt(prompt, use_formatting=True)
        assert "System" in formatted
        assert "Template" in formatted
        
        # Without formatting
        unformatted = grpo.format_prompt(prompt, use_formatting=False)
        assert unformatted == prompt
    
    def test_partial_update(self, config):
        """Test partial updates to generation prompts."""
        grpo = GRPO(config=config, use_wandb=False)
        
        # Set initial prompts
        grpo.set_generation_prompt(
            system_prompt="Initial",
            prompt_template="{prompt}",
            response_prefix=" Response:"
        )
        
        # Update only system prompt
        grpo.set_generation_prompt(system_prompt="Updated")
        
        prompt = "Test"
        formatted = grpo.format_prompt(prompt)
        
        # System should be updated, others unchanged
        assert "Updated" in formatted
        assert "Response:" in formatted
        assert "Initial" not in formatted