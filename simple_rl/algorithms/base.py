"""
Base algorithm interface - just abstract methods, no logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAlgorithm(ABC):
    """Base interface for all RL algorithms."""
    
    @abstractmethod
    def train(self, num_episodes: int) -> Dict[str, float]:
        """Train the algorithm for a given number of episodes."""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate the algorithm for a given number of episodes."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        pass