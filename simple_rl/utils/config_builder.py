"""
Configuration builder and factory for creating algorithm configurations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig


class ConfigBuilder:
    """Builder for creating and managing algorithm configurations."""
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Initialize config builder.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = base_config_path
        self.config_dir = Path(__file__).parent.parent / "config"
        
    def load_algorithm_config(self, algorithm_name: str) -> DictConfig:
        """
        Load algorithm-specific configuration.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Loaded configuration
        """
        config_path = self.config_dir / "algorithms" / f"{algorithm_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Algorithm config not found: {config_path}")
            
        return OmegaConf.load(config_path)
    
    def create_config(
        self, 
        algorithm_name: str, 
        overrides: Optional[Dict[str, Any]] = None,
        base_overrides: Optional[Dict[str, Any]] = None
    ) -> DictConfig:
        """
        Create a complete configuration by merging algorithm config with overrides.
        
        Args:
            algorithm_name: Name of the algorithm
            overrides: Dictionary of override values
            base_overrides: Base configuration overrides (project settings, etc.)
            
        Returns:
            Complete configuration
        """
        # Load algorithm-specific config
        config = self.load_algorithm_config(algorithm_name)
        
        # Add base configuration if provided
        if base_overrides:
            base_config = OmegaConf.create(base_overrides)
            config = OmegaConf.merge(config, base_config)
        
        # Apply overrides
        if overrides:
            override_config = OmegaConf.create(overrides)
            config = OmegaConf.merge(config, override_config)
        
        return config
    
    def create_experiment_config(
        self,
        algorithm_name: str,
        experiment_name: str,
        project_name: str = "simple-rl",
        overrides: Optional[Dict[str, Any]] = None
    ) -> DictConfig:
        """
        Create configuration for a complete experiment.
        
        Args:
            algorithm_name: Name of the algorithm
            experiment_name: Name of the experiment/run
            project_name: Project name for W&B
            overrides: Additional configuration overrides
            
        Returns:
            Complete experiment configuration
        """
        base_overrides = {
            "project_name": project_name,
            "run_name": experiment_name,
            "wandb": {
                "enabled": True,
                "project": project_name,
                "name": experiment_name
            }
        }
        
        return self.create_config(algorithm_name, overrides, base_overrides)
    
    def save_config(self, config: DictConfig, save_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            save_path: Path to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(config, save_path)
    
    def list_available_algorithms(self) -> list[str]:
        """
        List all available algorithm configurations.
        
        Returns:
            List of algorithm names
        """
        algorithms_dir = self.config_dir / "algorithms"
        
        if not algorithms_dir.exists():
            return []
            
        algorithm_files = list(algorithms_dir.glob("*.yaml"))
        return [f.stem for f in algorithm_files if f.stem != "__init__"]


def create_supervised_config(**kwargs) -> DictConfig:
    """
    Convenience function to create supervised learning configuration.
    
    Args:
        **kwargs: Override parameters
        
    Returns:
        Configuration for supervised learning
    """
    builder = ConfigBuilder()
    return builder.create_config("supervised_learning", overrides=kwargs)


def create_rl_config(algorithm_type: str = "base_rl", **kwargs) -> DictConfig:
    """
    Convenience function to create RL configuration.
    
    Args:
        algorithm_type: Type of RL algorithm
        **kwargs: Override parameters
        
    Returns:
        Configuration for RL algorithm
    """
    builder = ConfigBuilder()
    return builder.create_config(algorithm_type, overrides=kwargs)


def load_config_from_file(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Load configuration from file with optional overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional overrides to apply
        
    Returns:
        Loaded and merged configuration
    """
    config = OmegaConf.load(config_path)
    
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)
    
    return config