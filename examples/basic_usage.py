#!/usr/bin/env python3
"""
Basic usage example for Simple RL.

This example shows how to:
1. Load configuration
2. Set up logging
3. Load data (placeholder)
4. Create and use a model (placeholder)
5. Run training loop (placeholder)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_rl.utils.config import get_default_config
from simple_rl.utils.logging import setup_logging, get_logger
from simple_rl.utils.data import DatasetLoader
from simple_rl.algorithms.base import BaseAlgorithm
from simple_rl.models.base import BaseModel


def main():
    """Run basic usage example."""
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger("example")
    
    logger.info("Starting Simple RL basic usage example")
    
    # Load default configuration
    config = get_default_config()
    logger.info(f"Loaded config: {config}")
    
    # Initialize data loader
    data_loader = DatasetLoader(config)
    logger.info("Initialized data loader")
    
    # Example: Load a simple dataset (placeholder)
    logger.info("Data loading would happen here...")
    # dataset = data_loader.load_hf_dataset("some-dataset")
    
    # Initialize model (placeholder)
    logger.info("Model initialization would happen here...")
    # model = BaseModel(config.model)
    
    # Initialize algorithm (placeholder)
    logger.info("Algorithm initialization would happen here...")
    # algorithm = BaseAlgorithm(config, use_wandb=False)
    
    # Training loop (placeholder)
    logger.info("Training loop would happen here...")
    # results = algorithm.train(config.training.num_episodes)
    
    logger.info("Example completed!")


if __name__ == "__main__":
    main()