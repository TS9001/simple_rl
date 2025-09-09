"""
Supervised Learning algorithm as an example of how to use BaseAlgorithm with any PyTorch model.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig

from .base import BaseAlgorithm
from ..utils.optimizer_factory import create_optimizer, create_loss_function, create_scheduler, EarlyStopping


class SupervisedLearning(BaseAlgorithm):
    """
    Simple supervised learning algorithm that can train any PyTorch model.
    This serves as an example of how to implement concrete algorithms with full config support.
    """
    
    def __init__(self, model: nn.Module, config: DictConfig, use_wandb: bool = True):
        """
        Initialize supervised learning algorithm.
        
        Args:
            model: PyTorch model to train
            config: Configuration object (OmegaConf DictConfig)
            use_wandb: Whether to use Weights & Biases for logging
        """
        # Don't call parent __init__ yet - we need to setup components first
        self.model = model
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create components from config
        self.optimizer = create_optimizer(self.model.parameters(), config)
        self.loss_fn = create_loss_function(config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.early_stopping = EarlyStopping(config)
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        
        # Initialize W&B if enabled
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.get("project_name", "simple-rl"),
                config=dict(config),
                name=config.get("run_name", None)
            )
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, num_epochs: int = None) -> Dict[str, float]:
        """
        Train the model using supervised learning.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train (overrides config if provided)
            
        Returns:
            Dictionary of training metrics
        """
        if num_epochs is None:
            num_epochs = self.config.get("training", {}).get("num_epochs", 10)
        
        # Get config parameters
        log_interval = self.config.get("logging", {}).get("log_interval", 10)
        eval_interval = self.config.get("logging", {}).get("eval_interval", 100)
        save_interval = self.config.get("logging", {}).get("save_interval", 500)
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            # Progress bar
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Prepare batch data
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]
                
                batch_data = {"inputs": inputs, "targets": targets}
                
                # Training step (using parent method that handles optimizer/loss)
                metrics = self.train_step(batch_data)
                
                # Update metrics
                batch_loss = metrics["loss"]
                if isinstance(inputs, dict):
                    # For transformer inputs, get batch size from input_ids or first tensor
                    for key, value in inputs.items():
                        if torch.is_tensor(value):
                            batch_size = value.size(0)
                            break
                    else:
                        batch_size = 1  # Fallback
                else:
                    batch_size = inputs.size(0) if torch.is_tensor(inputs) else len(inputs)
                
                epoch_loss += batch_loss * batch_size
                epoch_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})
                
                # Log to wandb
                if self.total_steps % log_interval == 0:
                    log_data = {"train_loss": batch_loss, "learning_rate": self.optimizer.param_groups[0]["lr"]}
                    self.log_metrics(log_data, self.total_steps)
                
                # Validation during training
                if val_dataloader and self.total_steps % eval_interval == 0:
                    val_metrics = self.evaluate(val_dataloader)
                    
                    # Check early stopping
                    if self.early_stopping(val_metrics):
                        print(f"Early stopping at epoch {epoch+1}, step {self.total_steps}")
                        break
                
                # Save checkpoint
                if self.total_steps % save_interval == 0:
                    checkpoint_path = f"checkpoints/step_{self.total_steps}.pt"
                    self.save_checkpoint(checkpoint_path)
            
            # Early stopping check
            if self.early_stopping.should_stop:
                break
                
            # Epoch metrics
            avg_epoch_loss = epoch_loss / epoch_samples
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            print(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Log epoch metrics
            epoch_log_data = {"epoch_loss": avg_epoch_loss, "epoch": epoch}
            if self.scheduler:
                epoch_log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            
            self.log_metrics(epoch_log_data, self.total_steps)
            
            # Update learning rate scheduler
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    # For ReduceLROnPlateau, pass validation loss
                    if val_dataloader and 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        val_metrics = self.evaluate(val_dataloader)
                        self.scheduler.step(val_metrics.get("eval_loss", avg_epoch_loss))
                    else:
                        self.scheduler.step()
            
            self.current_episode = epoch + 1
        
        # Final metrics
        avg_total_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {
            "avg_loss": avg_total_loss,
            "total_epochs": self.current_episode,
            "total_samples": total_samples,
            "early_stopped": self.early_stopping.should_stop
        }
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Prepare batch data
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]
                
                batch_data = {"inputs": inputs, "targets": targets}
                
                # Evaluation step
                metrics = self.evaluate_step(batch_data)
                
                # Update metrics
                batch_loss = metrics["eval_loss"]
                if isinstance(inputs, dict):
                    # For transformer inputs, get batch size from input_ids or first tensor
                    for key, value in inputs.items():
                        if torch.is_tensor(value):
                            batch_size = value.size(0)
                            break
                    else:
                        batch_size = 1  # Fallback
                else:
                    batch_size = inputs.size(0) if torch.is_tensor(inputs) else len(inputs)
                
                total_loss += batch_loss * batch_size
                total_samples += batch_size
                
                # Calculate accuracy for classification tasks
                task_type = self.config.get("algorithm", {}).get("task_type", "regression")
                if task_type == "classification":
                    # Move inputs to device for accuracy calculation
                    if isinstance(inputs, dict):
                        device_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    else:
                        device_inputs = inputs.to(self.device)
                    device_targets = targets.to(self.device)
                    
                    outputs = self.model(device_inputs)
                    predictions = torch.argmax(outputs, dim=1)
                    correct_predictions += (predictions == device_targets).sum().item()
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        metrics = {
            "eval_loss": avg_loss,
            "total_samples": total_samples
        }
        
        # Add accuracy for classification
        task_type = self.config.get("algorithm", {}).get("task_type", "regression")
        if task_type == "classification":
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            metrics["accuracy"] = accuracy
        
        # Log evaluation metrics (only if not called during training)
        if not self.model.training:
            self.log_metrics(metrics, self.total_steps)
        
        return metrics