#!/usr/bin/env python3
"""
Example showing how to use external YAML configuration files for training.

This example demonstrates:
1. Loading configuration from YAML files
2. Using ConfigBuilder for flexible config management
3. Overriding config parameters from command line
4. Training with different algorithm configurations
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_rl.algorithms import SupervisedLearning
from simple_rl.utils import ConfigBuilder, load_config_from_file
from simple_rl.utils.logging import setup_logging, get_logger


class SimpleMLPModel(nn.Module):
    """Simple MLP model for demonstration."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def generate_synthetic_data(num_samples: int = 1000, input_dim: int = 10, task_type: str = "regression"):
    """Generate synthetic data based on task type."""
    X = torch.randn(num_samples, input_dim)
    
    if task_type == "regression":
        # Regression: linear combination with noise
        true_weights = torch.randn(input_dim, 1)
        y = X @ true_weights + 0.1 * torch.randn(num_samples, 1)
        y = y.squeeze()
    else:
        # Classification: binary classification
        true_weights = torch.randn(input_dim)
        logits = X @ true_weights
        y = (logits > 0).long()
    
    return X, y


def create_dataloaders(X, y, config, train_ratio=0.8):
    """Create train and validation dataloaders from config."""
    batch_size = config.training.batch_size
    
    # Split data
    num_train = int(len(X) * train_ratio)
    X_train, X_val = X[:num_train], X[num_train:]
    y_train, y_val = y[:num_train], y[num_train:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    data_config = config.get("data", {})
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=data_config.get("shuffle", True),
        num_workers=data_config.get("num_workers", 0),
        pin_memory=data_config.get("pin_memory", False)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get("validation", {}).get("batch_size", batch_size), 
        shuffle=False,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=data_config.get("pin_memory", False)
    )
    
    return train_loader, val_loader


def create_model_from_config(config, input_dim: int, output_dim: int):
    """Create model from configuration."""
    model_config = config.get("model", {})
    
    hidden_dims = model_config.get("hidden_dims", [64, 32])
    dropout = model_config.get("dropout", 0.0)
    
    return SimpleMLPModel(input_dim, hidden_dims, output_dim, dropout)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Config-based training example")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="simple_rl/config/algorithms/supervised_learning.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="config-based-training",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--task-type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("config_example")
    
    logger.info(f"Starting config-based training example")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Task type: {args.task_type}")
    
    # Load configuration from file
    try:
        config = load_config_from_file(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Available algorithm configs:")
        builder = ConfigBuilder()
        for algo in builder.list_available_algorithms():
            logger.info(f"  - {algo}")
        return 1
    
    # Apply command line overrides
    overrides = {}
    
    # Project settings
    overrides["project_name"] = "simple-rl-config-example"
    overrides["run_name"] = args.experiment_name
    
    # Algorithm settings
    if "algorithm" not in overrides:
        overrides["algorithm"] = {}
    overrides["algorithm"]["task_type"] = args.task_type
    
    # Training overrides
    if "training" not in overrides:
        overrides["training"] = {}
        
    if args.num_epochs is not None:
        overrides["training"]["num_epochs"] = args.num_epochs
        
    if args.learning_rate is not None:
        overrides["training"]["learning_rate"] = args.learning_rate
    
    # W&B settings
    if "wandb" not in overrides:
        overrides["wandb"] = {}
    overrides["wandb"]["enabled"] = args.enable_wandb
    
    # Apply overrides to config
    from omegaconf import OmegaConf
    override_config = OmegaConf.create(overrides)
    config = OmegaConf.merge(config, override_config)
    
    logger.info(f"Final config - Task: {config.algorithm.task_type}, Epochs: {config.training.num_epochs}, LR: {config.training.learning_rate}")
    
    # Generate data
    input_dim = config.get("model", {}).get("input_dim", 10)
    output_dim = config.get("model", {}).get("output_dim", 1)
    
    if args.task_type == "classification":
        output_dim = 2  # Binary classification
    
    logger.info("Generating synthetic data...")
    X, y = generate_synthetic_data(
        num_samples=1000, 
        input_dim=input_dim, 
        task_type=args.task_type
    )
    
    train_loader, val_loader = create_dataloaders(X, y, config)
    logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model from config
    model = create_model_from_config(config, input_dim, output_dim)
    logger.info(f"Created model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize algorithm
    algorithm = SupervisedLearning(
        model=model,
        config=config,
        use_wandb=config.wandb.enabled
    )
    
    logger.info(f"Initialized algorithm with optimizer: {type(algorithm.optimizer).__name__}")
    logger.info(f"Loss function: {type(algorithm.loss_fn).__name__}")
    if algorithm.scheduler:
        logger.info(f"Scheduler: {type(algorithm.scheduler).__name__}")
    
    # Train model
    logger.info("Starting training...")
    train_results = algorithm.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config.training.num_epochs
    )
    
    logger.info("Training completed!")
    logger.info(f"Final training loss: {train_results['avg_loss']:.4f}")
    logger.info(f"Total epochs: {train_results['total_epochs']}")
    logger.info(f"Early stopped: {train_results['early_stopped']}")
    
    # Final evaluation
    eval_results = algorithm.evaluate(val_loader)
    logger.info(f"Final validation loss: {eval_results['eval_loss']:.4f}")
    
    if args.task_type == "classification":
        logger.info(f"Final accuracy: {eval_results.get('accuracy', 0.0):.4f}")
    
    # Save final checkpoint
    checkpoint_path = f"checkpoints/{args.experiment_name}_final.pt"
    algorithm.save_checkpoint(checkpoint_path)
    logger.info(f"Final model saved to {checkpoint_path}")
    
    # Save the final configuration used
    config_save_path = f"configs/{args.experiment_name}_config.yaml"
    Path(config_save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_save_path)
    logger.info(f"Configuration saved to {config_save_path}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()