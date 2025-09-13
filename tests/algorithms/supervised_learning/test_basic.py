"""
Basic tests for supervised learning algorithm.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from simple_rl.algorithms import SupervisedLearning
from simple_rl.utils import create_supervised_config


class SimpleModel(nn.Module):
    """Simple model for basic testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


@pytest.fixture
def basic_config():
    """Basic configuration for supervised learning."""
    return create_supervised_config(
        project_name="test-basic",
        algorithm={
            "task_type": "classification"
        },
        training={
            "num_epochs": 2,
            "learning_rate": 1e-3,
            "batch_size": 4
        },
        wandb={"enabled": False}
    )


@pytest.fixture
def simple_data():
    """Create simple synthetic dataset."""
    X = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    
    train_dataset = TensorDataset(X[:40], y[:40])
    val_dataset = TensorDataset(X[40:], y[40:])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    return train_loader, val_loader


class TestSupervisedLearningBasics:
    """Test basic supervised learning functionality."""
    
    def test_initialization(self, basic_config):
        """Test that supervised learning can be initialized."""
        model = SimpleModel()
        algorithm = SupervisedLearning(model, basic_config, use_wandb=False)
        
        assert algorithm.model is model
        assert algorithm.device in [torch.device("cpu"), torch.device("cuda")]
        assert algorithm.optimizer is not None
        assert algorithm.task_type == "classification"
    
    def test_simple_training(self, basic_config, simple_data):
        """Test basic training loop."""
        train_loader, val_loader = simple_data
        model = SimpleModel()
        algorithm = SupervisedLearning(model, basic_config, use_wandb=False)
        
        # Train
        results = algorithm.train(train_loader, val_loader)
        
        assert "avg_loss" in results
        assert "total_epochs" in results
        assert results["total_epochs"] == 2
        assert results["avg_loss"] > 0
    
    def test_evaluation(self, basic_config, simple_data):
        """Test model evaluation."""
        train_loader, val_loader = simple_data
        model = SimpleModel()
        algorithm = SupervisedLearning(model, basic_config, use_wandb=False)
        
        # Evaluate before training
        eval_results = algorithm.evaluate(val_loader)
        
        assert "eval_loss" in eval_results
        assert "accuracy" in eval_results
        assert 0 <= eval_results["accuracy"] <= 1
    
    def test_classification_task(self, basic_config, simple_data):
        """Test classification-specific functionality."""
        train_loader, val_loader = simple_data
        model = SimpleModel()
        
        # Ensure classification task
        basic_config["algorithm"]["task_type"] = "classification"
        algorithm = SupervisedLearning(model, basic_config, use_wandb=False)
        
        # Train and evaluate
        algorithm.train(train_loader, val_loader)
        eval_results = algorithm.evaluate(val_loader)
        
        # Should have classification metrics
        assert "accuracy" in eval_results
        assert 0 <= eval_results["accuracy"] <= 1
    
    def test_regression_task(self, basic_config):
        """Test regression-specific functionality."""
        # Create regression data
        X = torch.randn(50, 10)
        y = torch.randn(50, 1)  # Continuous targets
        
        train_dataset = TensorDataset(X[:40], y[:40])
        val_dataset = TensorDataset(X[40:], y[40:])
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        # Create model with single output
        model = SimpleModel(output_dim=1)
        
        # Configure for regression
        basic_config["algorithm"]["task_type"] = "regression"
        algorithm = SupervisedLearning(model, basic_config, use_wandb=False)
        
        # Train and evaluate
        algorithm.train(train_loader, val_loader)
        eval_results = algorithm.evaluate(val_loader)
        
        # Should have regression metrics
        assert "eval_loss" in eval_results
        assert "mse" in eval_results or "rmse" in eval_results
    
    def test_train_step(self, basic_config, simple_data):
        """Test single training step."""
        train_loader, _ = simple_data
        model = SimpleModel()
        algorithm = SupervisedLearning(model, basic_config, use_wandb=False)
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Run single train step
        loss = algorithm.train_step(batch)
        
        assert isinstance(loss, dict)
        assert "loss" in loss
        assert loss["loss"] > 0