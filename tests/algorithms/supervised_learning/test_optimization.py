"""
Tests for optimization-related features in supervised learning.

Tests different optimizers, learning rate schedulers, and
training strategies.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from simple_rl.algorithms import SupervisedLearning
from simple_rl.utils import create_supervised_config


class SimpleModel(nn.Module):
    """Simple model for optimization tests."""
    
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
def optimization_data():
    """Create sample data for optimization tests."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    train_dataset = TensorDataset(X[:80], y[:80])
    val_dataset = TensorDataset(X[80:], y[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    return train_loader, val_loader


class TestOptimization:
    """Test optimization-related features."""
    
    def test_different_optimizers(self, optimization_data):
        """Test training with different optimizers."""
        train_loader, val_loader = optimization_data
        
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
            
            model = SimpleModel()
            algorithm = SupervisedLearning(model, config, use_wandb=False)
            
            # Should complete without errors
            results = algorithm.train(train_loader, val_loader)
            assert results["avg_loss"] > 0
            assert isinstance(results["total_epochs"], int)
    
    def test_learning_rate_scheduling(self, optimization_data):
        """Test training with learning rate scheduling."""
        train_loader, val_loader = optimization_data
        
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
        
        model = SimpleModel()
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        # Store initial learning rate
        initial_lr = algorithm.optimizer.param_groups[0]['lr']
        
        # Train model
        results = algorithm.train(train_loader, val_loader)
        
        # Learning rate should have been reduced
        final_lr = algorithm.optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr
        assert results["avg_loss"] > 0
    
    def test_cosine_annealing_scheduler(self, optimization_data):
        """Test cosine annealing learning rate scheduler."""
        train_loader, val_loader = optimization_data
        
        config = create_supervised_config(
            project_name="test-cosine",
            algorithm={"task_type": "classification"},
            training={
                "num_epochs": 5,
                "learning_rate": 1e-3,
                "optimizer": {"type": "adamw"},
                "scheduler": {
                    "enabled": True,
                    "type": "cosine",
                    "T_max": 5
                }
            },
            wandb={"enabled": False}
        )
        
        model = SimpleModel()
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        # Train and check that it completes
        results = algorithm.train(train_loader, val_loader)
        assert results["total_epochs"] == 5
    
    def test_gradient_clipping(self, optimization_data):
        """Test gradient clipping during training."""
        train_loader, val_loader = optimization_data
        
        config = create_supervised_config(
            project_name="test-grad-clip",
            algorithm={"task_type": "classification"},
            training={
                "num_epochs": 2,
                "learning_rate": 1e-3,
                "gradient_clip": 1.0,
                "optimizer": {"type": "adamw"}
            },
            wandb={"enabled": False}
        )
        
        model = SimpleModel()
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        # Train with gradient clipping
        results = algorithm.train(train_loader, val_loader)
        
        # Should complete successfully
        assert results["avg_loss"] > 0
        assert results["total_epochs"] == 2
    
    def test_weight_decay(self, optimization_data):
        """Test weight decay regularization."""
        train_loader, val_loader = optimization_data
        
        config = create_supervised_config(
            project_name="test-weight-decay",
            algorithm={"task_type": "classification"},
            training={
                "num_epochs": 3,
                "learning_rate": 1e-3,
                "optimizer": {
                    "type": "adamw",
                    "weight_decay": 0.01
                }
            },
            wandb={"enabled": False}
        )
        
        model = SimpleModel()
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        # Train with weight decay
        results = algorithm.train(train_loader, val_loader)
        
        # Should complete successfully
        assert results["avg_loss"] > 0
    
    def test_early_stopping(self, optimization_data):
        """Test early stopping during training."""
        train_loader, val_loader = optimization_data
        
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
        
        model = SimpleModel()
        algorithm = SupervisedLearning(model, config, use_wandb=False)
        
        results = algorithm.train(train_loader, val_loader)
        
        # With small dataset and simple task, early stopping might trigger
        # Just verify the mechanism works without error
        assert isinstance(results["early_stopped"], bool)
        assert results["total_epochs"] <= 20
    
    def test_different_batch_sizes(self, optimization_data):
        """Test training with different batch sizes."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        train_dataset = TensorDataset(X[:80], y[:80])
        val_dataset = TensorDataset(X[80:], y[80:])
        
        batch_sizes = [4, 8, 16]
        
        for batch_size in batch_sizes:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            config = create_supervised_config(
                project_name=f"test-batch-{batch_size}",
                algorithm={"task_type": "classification"},
                training={
                    "num_epochs": 2,
                    "learning_rate": 1e-3,
                    "batch_size": batch_size
                },
                wandb={"enabled": False}
            )
            
            model = SimpleModel()
            algorithm = SupervisedLearning(model, config, use_wandb=False)
            
            results = algorithm.train(train_loader, val_loader)
            assert results["avg_loss"] > 0