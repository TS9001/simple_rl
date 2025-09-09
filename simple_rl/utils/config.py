"""
Configuration utilities for loading and managing experiment configurations.
"""

from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any
import os


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def merge_configs(base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
    """
    Merge base configuration with override values.
    
    Args:
        base_config: Base configuration
        override_config: Override values
        
    Returns:
        Merged configuration
    """
    override_cfg = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_config, override_cfg)
    return merged


def get_default_config() -> DictConfig:
    """Get default configuration for experiments."""
    default_config = {
        "project_name": "simple-rl",
        "seed": 42,
        "device": "auto",
        "logging": {
            "level": "INFO",
            "log_interval": 100,
        },
        "training": {
            "num_episodes": 1000,
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
        "evaluation": {
            "num_episodes": 100,
            "eval_interval": 100,
        },
        "wandb": {
            "enabled": True,
            "project": "simple-rl",
        },
    }
    
    return OmegaConf.create(default_config)