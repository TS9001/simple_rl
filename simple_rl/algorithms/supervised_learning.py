"""
Supervised learning implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List
import wandb

from .base import BaseAlgorithm


class SupervisedLearning(BaseAlgorithm):
    """
    Supervised learning algorithm for classification and regression.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: dict = None,
        task_type: str = "classification",
        use_wandb: bool = False
    ):
        """
        Initialize supervised learning algorithm.
        
        Args:
            model: PyTorch model to train
            config: Configuration dict
            task_type: "classification" or "regression"
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.config = config or {}
        self.task_type = task_type
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training parameters
        training_config = config.get("training", {})
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.batch_size = training_config.get("batch_size", 32)
        self.num_epochs = training_config.get("num_epochs", 10)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate
        )
        
        # Create loss function based on task type
        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:  # regression
            self.criterion = nn.MSELoss()
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project=config.get("project_name", "supervised-learning"),
                config=config,
                name=config.get("run_name", None)
            )
        
        # Track best model
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
    
    def train(self, num_episodes: int = None) -> Dict[str, float]:
        """
        Train for specified number of episodes (epochs).
        Note: This expects train_dataloader to be set first via set_data()
        
        Args:
            num_episodes: Number of epochs to train (overrides config)
            
        Returns:
            Dictionary of final metrics
        """
        if not hasattr(self, 'train_dataloader'):
            raise ValueError("No training data set. Call set_data() first.")
        
        num_epochs = num_episodes or self.num_epochs
        final_metrics = {}
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase if we have validation data
            if hasattr(self, 'val_dataloader'):
                val_metrics = self._validate_epoch()
                
                # Track best model
                if self.task_type == "classification":
                    if val_metrics['accuracy'] > self.best_accuracy:
                        self.best_accuracy = val_metrics['accuracy']
                        self.save_checkpoint(f"checkpoints/best_model.pt")
                else:
                    if val_metrics['loss'] < self.best_loss:
                        self.best_loss = val_metrics['loss']
                        self.save_checkpoint(f"checkpoints/best_model.pt")
            else:
                val_metrics = {}
            
            # Log metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            
            if self.task_type == "classification" and 'accuracy' in train_metrics:
                metrics['train_accuracy'] = train_metrics['accuracy']
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Print progress
            if epoch % max(1, num_epochs // 10) == 0:
                print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}", end="")
                if self.task_type == "classification" and 'accuracy' in train_metrics:
                    print(f", Train Acc: {train_metrics['accuracy']:.4f}", end="")
                if val_metrics:
                    print(f", Val Loss: {val_metrics['loss']:.4f}", end="")
                    if self.task_type == "classification":
                        print(f", Val Acc: {val_metrics['accuracy']:.4f}", end="")
                print()
            
            final_metrics = metrics
        
        return final_metrics
    
    def evaluate(self, num_episodes: int = 1) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        Note: This expects test_dataloader to be set first via set_data()
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not hasattr(self, 'test_dataloader'):
            # If no test data, try validation data
            if hasattr(self, 'val_dataloader'):
                dataloader = self.val_dataloader
            else:
                raise ValueError("No test/validation data set. Call set_data() first.")
        else:
            dataloader = self.test_dataloader
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                if self.task_type == "classification":
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        metrics = {
            "loss": total_loss / len(dataloader),
        }
        
        if self.task_type == "classification":
            metrics["accuracy"] = correct / total
            # Could add more metrics like precision, recall, F1 here
        else:
            # For regression, could add MAE, R^2, etc.
            predictions = torch.tensor(all_predictions)
            targets = torch.tensor(all_targets)
            metrics["mae"] = torch.abs(predictions - targets).mean().item()
        
        return metrics
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary with 'inputs' and 'targets'
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        metrics = {"loss": loss.item()}
        
        # Add accuracy for classification
        if self.task_type == "classification":
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()
            metrics["accuracy"] = accuracy
        
        return metrics
    
    def set_data(
        self, 
        train_data: Optional[DataLoader] = None,
        val_data: Optional[DataLoader] = None,
        test_data: Optional[DataLoader] = None
    ):
        """
        Set data loaders for training, validation, and testing.
        
        Args:
            train_data: Training DataLoader
            val_data: Validation DataLoader
            test_data: Test DataLoader
        """
        if train_data is not None:
            self.train_dataloader = train_data
        if val_data is not None:
            self.val_dataloader = val_data
        if test_data is not None:
            self.test_dataloader = test_data
    
    def create_data_from_arrays(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None
    ):
        """
        Create DataLoaders from tensor arrays.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
        """
        # Create training dataloader
        train_dataset = TensorDataset(X_train, y_train)
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Create validation dataloader if provided
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            self.val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
        
        # Create test dataloader if provided
        if X_test is not None and y_test is not None:
            test_dataset = TensorDataset(X_test, y_test)
            self.test_dataloader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_dataloader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.task_type == "classification":
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        metrics = {"loss": total_loss / len(self.train_dataloader)}
        if self.task_type == "classification":
            metrics["accuracy"] = correct / total
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                if self.task_type == "classification":
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        metrics = {"loss": total_loss / len(self.val_dataloader)}
        if self.task_type == "classification":
            metrics["accuracy"] = correct / total
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'task_type': self.task_type,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.task_type = checkpoint.get('task_type', 'classification')
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)