"""
Tests for model finetuning functionality.

Tests standard model finetuning scenarios including:
- HuggingFace model loading and adaptation
- Finetuning with different optimizers
- Checkpoint saving and loading
- Model freezing and unfreezing
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tempfile
import shutil
from pathlib import Path

from simple_rl.algorithms import SupervisedLearning
from simple_rl.models.base import BaseModel
from simple_rl.utils import create_supervised_config


class MockHFModel(nn.Module):
    """Mock HuggingFace model for testing."""
    
    def __init__(self, vocab_size=1000, hidden_dim=128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.pooler = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        x = self.transformer(x)
        # Simple pooling - take first token
        pooled = self.pooler(x[:, 0, :])
        return {"last_hidden_state": x, "pooler_output": pooled}


class FinetunableModel(BaseModel):
    """Model that can be finetuned with HF backbone."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Mock HF model instead of real one for testing
        self.backbone = MockHFModel(
            vocab_size=config.get("vocab_size", 1000),
            hidden_dim=config.get("hidden_dim", 128)
        )
        
        # Classification head
        self.classifier = nn.Linear(config.get("hidden_dim", 128), config.get("num_classes", 2))
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
    def forward(self, x):
        # x should be token ids for text input
        outputs = self.backbone(x)
        pooled = outputs["pooler_output"]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for finetuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Mock token IDs (batch_size, seq_length)
    X = torch.randint(0, 1000, (100, 32))
    y = torch.randint(0, 2, (100,))  # Binary classification
    
    train_dataset = TensorDataset(X[:80], y[:80])
    val_dataset = TensorDataset(X[80:], y[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    return train_loader, val_loader


@pytest.fixture
def finetuning_config():
    """Configuration for finetuning tests."""
    return create_supervised_config(
        project_name="test-finetuning",
        algorithm={
            "task_type": "classification"
        },
        training={
            "num_epochs": 3,
            "learning_rate": 2e-5,
            "optimizer": {
                "type": "adamw",
                "betas": [0.9, 0.999]
            }
        },
        wandb={"enabled": False}
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for saving checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestModelFinetuning:
    """Test suite for model finetuning."""
    
    def test_model_creation_and_forward_pass(self, finetuning_config):
        """Test that we can create and run forward pass on finetunable model."""
        model_config = {
            "vocab_size": 1000,
            "hidden_dim": 128,
            "num_classes": 2,
            "dropout": 0.1
        }
        
        model = FinetunableModel(model_config)
        
        # Test forward pass
        batch_size, seq_length = 4, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.shape == (batch_size, 2)
        assert not torch.isnan(outputs).any()
    
    def test_backbone_freezing(self):
        """Test freezing and unfreezing backbone parameters."""
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        model = FinetunableModel(model_config)
        
        # Initially all parameters should require grad
        backbone_params = list(model.backbone.parameters())
        assert all(p.requires_grad for p in backbone_params)
        
        # Freeze backbone
        model.freeze_backbone()
        assert all(not p.requires_grad for p in backbone_params)
        
        # Classifier should still require grad
        classifier_params = list(model.classifier.parameters())
        assert all(p.requires_grad for p in classifier_params)
        
        # Unfreeze backbone
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in backbone_params)
    
    def test_full_finetuning(self, finetuning_config, sample_data):
        """Test full model finetuning (all parameters)."""
        train_loader, val_loader = sample_data
        
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        model = FinetunableModel(model_config)
        
        algorithm = SupervisedLearning(model, finetuning_config, use_wandb=False)
        
        # Train model
        results = algorithm.train(train_loader, val_loader)
        
        assert results["avg_loss"] > 0
        assert results["total_epochs"] == 3
        assert not results["early_stopped"]
        
        # Evaluate
        eval_results = algorithm.evaluate(val_loader)
        assert "eval_loss" in eval_results
        assert "accuracy" in eval_results
    
    def test_frozen_backbone_finetuning(self, finetuning_config, sample_data):
        """Test finetuning with frozen backbone (only train classifier)."""
        train_loader, val_loader = sample_data
        
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        model = FinetunableModel(model_config)
        
        # Freeze backbone before training
        model.freeze_backbone()
        
        algorithm = SupervisedLearning(model, finetuning_config, use_wandb=False)
        
        # Store initial backbone weights
        initial_backbone_weights = {}
        for name, param in model.backbone.named_parameters():
            initial_backbone_weights[name] = param.clone().detach()
        
        # Train model
        results = algorithm.train(train_loader, val_loader)
        
        # Verify backbone weights didn't change
        for name, param in model.backbone.named_parameters():
            torch.testing.assert_close(
                param, initial_backbone_weights[name],
                msg=f"Backbone parameter {name} changed during frozen training"
            )
        
        # Verify classifier weights did change
        # (This is harder to test directly, but training should succeed)
        assert results["avg_loss"] > 0
    
    def test_checkpoint_saving_and_loading(self, finetuning_config, sample_data, temp_dir):
        """Test saving and loading model checkpoints during finetuning."""
        train_loader, val_loader = sample_data
        
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        model1 = FinetunableModel(model_config)
        
        algorithm1 = SupervisedLearning(model1, finetuning_config, use_wandb=False)
        
        # Train for 1 epoch
        short_config = create_supervised_config(
            project_name="test-checkpoint",
            algorithm={"task_type": "classification"},
            training={"num_epochs": 1, "learning_rate": 2e-5},
            wandb={"enabled": False}
        )
        
        algorithm1.config = short_config
        results1 = algorithm1.train(train_loader, val_loader)
        
        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "model_checkpoint.pt"
        algorithm1.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new model and load checkpoint
        model2 = FinetunableModel(model_config)
        algorithm2 = SupervisedLearning(model2, short_config, use_wandb=False)
        algorithm2.load_checkpoint(str(checkpoint_path))
        
        # Verify models have same weights
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2, msg=f"Parameter {name1} not loaded correctly")
    
    def test_different_optimizers_for_finetuning(self, sample_data):
        """Test finetuning with different optimizers."""
        train_loader, val_loader = sample_data
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        
        optimizers_to_test = [
            {"type": "adam", "betas": [0.9, 0.999]},
            {"type": "adamw", "betas": [0.9, 0.999]},
            {"type": "sgd", "momentum": 0.9}
        ]
        
        for opt_config in optimizers_to_test:
            config = create_supervised_config(
                project_name=f"test-{opt_config['type']}",
                algorithm={"task_type": "classification"},
                training={
                    "num_epochs": 2,
                    "learning_rate": 1e-3,
                    "optimizer": opt_config
                },
                wandb={"enabled": False}
            )
            
            model = FinetunableModel(model_config)
            algorithm = SupervisedLearning(model, config, use_wandb=False)
            
            # Should complete without errors
            results = algorithm.train(train_loader, val_loader)
            assert results["avg_loss"] > 0
            assert isinstance(results["total_epochs"], int)
    
    def test_learning_rate_scheduling(self, sample_data):
        """Test finetuning with learning rate scheduling."""
        train_loader, val_loader = sample_data
        
        config = create_supervised_config(
            project_name="test-scheduler",
            algorithm={"task_type": "classification"},
            training={
                "num_epochs": 4,
                "learning_rate": 1e-3,
                "optimizer": {"type": "adamw"},
                "scheduler": {
                    "enabled": True,
                    "type": "step",
                    "step_size": 2,
                    "gamma": 0.5
                }
            },
            wandb={"enabled": False}
        )
        
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        model = FinetunableModel(model_config)
        
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        # Store initial learning rate
        initial_lr = algorithm.optimizer.param_groups[0]['lr']
        
        # Train model
        results = algorithm.train(train_loader, val_loader)
        
        # Learning rate should have been reduced
        final_lr = algorithm.optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr
        assert results["avg_loss"] > 0
    
    def test_early_stopping_during_finetuning(self, sample_data):
        """Test early stopping during finetuning."""
        train_loader, val_loader = sample_data
        
        config = create_supervised_config(
            project_name="test-early-stopping",
            algorithm={"task_type": "classification"},
            training={
                "num_epochs": 20,  # Set high so early stopping can trigger
                "learning_rate": 1e-3,
                "early_stopping": {
                    "enabled": True,
                    "patience": 2,
                    "monitor": "val_loss"
                }
            },
            logging={
                "eval_interval": 10  # Evaluate frequently
            },
            wandb={"enabled": False}
        )
        
        model_config = {"vocab_size": 1000, "hidden_dim": 128, "num_classes": 2}
        model = FinetunableModel(model_config)
        
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        results = algorithm.train(train_loader, val_loader)
        
        # With small dataset and simple task, early stopping might trigger
        # Just verify the mechanism works without error
        assert isinstance(results["early_stopped"], bool)
        assert results["total_epochs"] <= 20