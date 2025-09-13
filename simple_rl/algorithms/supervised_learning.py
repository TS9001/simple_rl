"""
Simple supervised learning - just use PyTorch directly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseAlgorithm


class SupervisedLearning(BaseAlgorithm):
    """
    Simple supervised learning algorithm.
    """
    
    def __init__(self, model: nn.Module, config: dict = None, use_wandb: bool = False):
        """
        Initialize supervised learning algorithm.
        
        Args:
            model: PyTorch model to train
            config: Configuration dict
            use_wandb: Whether to use Weights & Biases for logging
        """
        super().__init__(config, use_wandb)
        self.model = model
        self.config = config or {}
        
        # Create optimizer directly
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get("learning_rate", 1e-3)
        )
        
        # Create loss function directly
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader):
            inputs, targets = batch
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {"loss": total_loss / len(dataloader)}
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total
        }