"""
Basic GRPO functionality tests.
"""

import pytest
import torch
import torch.nn as nn
from simple_rl.algorithms.grpo import GRPO
from simple_rl.utils.huggingface_wrappers import LanguageModel


class TestGRPOBasics:
    """Test basic GRPO functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "algorithm": {
                "name": "grpo",
                "group_size": 4,
                "kl_coef": 0.1,
                "normalize_rewards": True
            },
            "model": {
                "hf_model_name": "gpt2",
                "max_length": 128
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 1e-5,
                "max_new_tokens": 10,
                "temperature": 1.0
            }
        }
    
    def test_initialization(self, config):
        """Test GRPO initialization."""
        grpo = GRPO(model=None, config=config, use_wandb=False)
        
        assert isinstance(grpo.policy, LanguageModel)
        assert isinstance(grpo.ref_policy, nn.Module)
        assert grpo.group_size == 4
        assert grpo.kl_coef == 0.1
        
        # Check reference model is frozen
        for param in grpo.ref_policy.parameters():
            assert not param.requires_grad
    
    def test_initialization_without_model(self, config):
        """Test GRPO can create its own model from config."""
        grpo = GRPO(config=config, use_wandb=False)
        assert isinstance(grpo.policy, LanguageModel)
    
    def test_initialization_with_custom_model(self, config):
        """Test GRPO with custom model."""
        custom_model = LanguageModel(config)
        grpo = GRPO(model=custom_model, config=config, use_wandb=False)
        assert grpo.policy is custom_model
    
    def test_device_placement(self, config):
        """Test model is placed on correct device."""
        grpo = GRPO(config=config, use_wandb=False)
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if model is on correct device
        for param in grpo.policy.parameters():
            assert param.device.type == expected_device.type
            break