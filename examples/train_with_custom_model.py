#!/usr/bin/env python3
"""
Example showing how to train any PyTorch model using SimpleRL algorithms.

This example demonstrates:
1. Creating a custom PyTorch model
2. Generating synthetic data
3. Using SupervisedLearning algorithm to train the model
4. Evaluating the trained model
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_rl.algorithms import SupervisedLearning
from simple_rl.utils import create_supervised_config, ConfigBuilder
from simple_rl.utils.logging import setup_logging, get_logger


class SimpleMLPModel(nn.Module):
    """Simple MLP model for demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def generate_synthetic_data(num_samples: int = 1000, input_dim: int = 10):
    """Generate synthetic regression data."""
    # Create random input data
    X = torch.randn(num_samples, input_dim)
    
    # Create synthetic targets (simple linear combination with noise)
    true_weights = torch.randn(input_dim, 1)
    y = X @ true_weights + 0.1 * torch.randn(num_samples, 1)
    
    return X, y


def create_dataloaders(X, y, train_ratio=0.8, batch_size=32):
    """Create train and validation dataloaders."""
    # Split data
    num_train = int(len(X) * train_ratio)
    
    X_train, X_val = X[:num_train], X[num_train:]
    y_train, y_val = y[:num_train], y[num_train:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def main():
    """Main training example."""
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("example")
    
    logger.info("Starting custom model training example")
    
    # Create configuration using config builder
    config = create_supervised_config(
        # Project settings
        project_name="simple-rl-example",
        run_name="custom-model-training",
        
        # Algorithm settings
        algorithm={
            "name": "supervised_learning",
            "task_type": "regression"
        },
        
        # Training hyperparameters
        training={
            "num_epochs": 20,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "optimizer": {
                "type": "adam",
                "betas": [0.9, 0.999]
            },
            "scheduler": {
                "enabled": True,
                "type": "step",
                "step_size": 10,
                "gamma": 0.9
            },
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "monitor": "val_loss"
            }
        },
        
        # Logging settings
        logging={
            "log_interval": 10,
            "eval_interval": 50,
            "save_interval": 200
        },
        
        # W&B settings
        wandb={
            "enabled": False  # Disabled for this example
        }
    )
    
    logger.info(f"Created config: {config.algorithm.name} with {config.training.num_epochs} epochs")
    
    # Model parameters
    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    batch_size = 32
    
    # Create model
    model = SimpleMLPModel(input_dim, hidden_dim, output_dim)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate data
    logger.info("Generating synthetic data...")
    X, y = generate_synthetic_data(num_samples=1000, input_dim=input_dim)
    train_loader, val_loader = create_dataloaders(X, y, batch_size=batch_size)
    
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Initialize algorithm (wandb setting comes from config now)
    algorithm = SupervisedLearning(
        model=model, 
        config=config, 
        use_wandb=config.wandb.enabled
    )
    
    logger.info("Initialized SupervisedLearning algorithm")
    
    # Train model with validation
    logger.info("Starting training...")
    train_results = algorithm.train(
        train_dataloader=train_loader, 
        val_dataloader=val_loader,  # Enable validation during training
        num_epochs=config.training.num_epochs
    )
    
    logger.info(f"Training completed!")
    logger.info(f"Final training loss: {train_results['avg_loss']:.4f}")
    logger.info(f"Total epochs: {train_results['total_epochs']}")
    logger.info(f"Early stopped: {train_results['early_stopped']}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = algorithm.evaluate(val_loader)
    
    logger.info(f"Final evaluation completed!")
    logger.info(f"Validation loss: {eval_results['eval_loss']:.4f}")
    
    # Save checkpoint
    checkpoint_path = "checkpoints/custom_model.pt"
    algorithm.save_checkpoint(checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")
    
    # Test inference
    logger.info("Testing inference...")
    model.eval()
    with torch.no_grad():
        sample_input = X[:5]  # Take first 5 samples
        sample_target = y[:5]
        sample_input = sample_input.to(algorithm.device)
        
        predictions = model(sample_input)
        
        logger.info("Sample predictions vs targets:")
        for i in range(5):
            pred = predictions[i].item()
            target = sample_target[i].item()
            logger.info(f"  Sample {i}: pred={pred:.4f}, target={target:.4f}, diff={abs(pred-target):.4f}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()