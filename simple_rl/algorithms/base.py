"""
Base algorithm class for RL implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class BaseAlgorithm(ABC):
    """Base class for all RL algorithms."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], use_wandb: bool = True):
        """
        Initialize the base algorithm.
        
        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        lr = config.get("learning_rate", config.get("training", {}).get("learning_rate", 1e-4))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup loss function (can be overridden)
        self.loss_fn = nn.MSELoss()
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        
        if self.use_wandb:
            wandb.init(
                project=config.get("project_name", "simple-rl"),
                config=config,
                name=config.get("run_name", None)
            )
    
    @abstractmethod
    def train(self, num_episodes: int) -> Dict[str, float]:
        """Train the algorithm for a given number of episodes."""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate the algorithm for a given number of episodes."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to wandb if enabled."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move inputs to device (handle nested dictionaries)
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        
        if isinstance(inputs, dict):
            # Handle nested input dictionaries (e.g., for transformers)
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.total_steps += 1
        
        return {"loss": loss.item()}
    
    def evaluate_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single evaluation step.
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Move inputs to device (handle nested dictionaries)
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        
        if isinstance(inputs, dict):
            # Handle nested input dictionaries (e.g., for transformers)
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        
        return {"eval_loss": loss.item()}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_episode = checkpoint.get('current_episode', 0)
        self.total_steps = checkpoint.get('total_steps', 0)