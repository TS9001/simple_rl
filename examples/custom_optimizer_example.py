#!/usr/bin/env python3
"""
Clean example showing how to use custom optimizers with configuration.

Shows two methods:
1. Register custom optimizer at runtime
2. Use import_path in config to load any optimizer
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_rl.algorithms import SupervisedLearning
from simple_rl.utils import create_supervised_config, register_optimizer
from simple_rl.utils.logging import setup_logging, get_logger


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class MyCustomOptimizer(torch.optim.Optimizer):
    """Example custom optimizer - just SGD with momentum."""
    
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = param_state['momentum_buffer']
                buf.mul_(group['momentum']).add_(p.grad.data)
                p.data.add_(buf, alpha=-group['lr'])


def main():
    setup_logging(level="INFO")
    logger = get_logger("custom_optimizer_example")
    
    # Generate simple data
    X = torch.randn(1000, 10)
    y = X.sum(dim=1) + 0.1 * torch.randn(1000)
    
    train_dataset = TensorDataset(X[:800], y[:800])
    val_dataset = TensorDataset(X[800:], y[800:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    logger.info("🔧 Method 1: Register custom optimizer at runtime")
    
    # Register our custom optimizer
    register_optimizer("my_custom", MyCustomOptimizer)
    
    # Create config using registered optimizer
    config1 = create_supervised_config(
        project_name="custom-opt-demo",
        training={
            "optimizer": {
                "type": "my_custom",  # Use registered optimizer
                "momentum": 0.95
            },
            "num_epochs": 10
        },
        wandb={"enabled": False}
    )
    
    model1 = SimpleModel()
    algorithm1 = SupervisedLearning(model1, config1, use_wandb=False)
    
    logger.info(f"✅ Using optimizer: {type(algorithm1.optimizer).__name__}")
    results1 = algorithm1.train(train_loader, val_loader)
    logger.info(f"Final loss: {results1['avg_loss']:.4f}")
    
    logger.info("\n🔧 Method 2: Import optimizer from package")
    
    # Create config using import_path (example with built-in AdamW)
    config2 = create_supervised_config(
        project_name="custom-opt-demo",
        training={
            "optimizer": {
                "type": "custom",
                "import_path": "torch.optim.AdamW",  # Any importable optimizer
                "betas": [0.9, 0.999],
                "amsgrad": True
            },
            "num_epochs": 10
        },
        wandb={"enabled": False}
    )
    
    model2 = SimpleModel()
    algorithm2 = SupervisedLearning(model2, config2, use_wandb=False)
    
    logger.info(f"✅ Using optimizer: {type(algorithm2.optimizer).__name__}")
    results2 = algorithm2.train(train_loader, val_loader)
    logger.info(f"Final loss: {results2['avg_loss']:.4f}")
    
    logger.info("\n📋 To use with external optimizers:")
    logger.info("pip install lion-pytorch")
    logger.info("Then use import_path: 'lion_pytorch.Lion'")
    
    logger.info("\n🎉 Done! Both methods work with any PyTorch optimizer.")


if __name__ == "__main__":
    main()