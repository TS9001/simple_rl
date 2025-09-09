"""
Factory functions for creating optimizers and other training components from config.
"""

import torch.optim as optim
import torch.nn as nn
from typing import Dict, Any, Type
from omegaconf import DictConfig
import importlib
import inspect


# Global registry for custom optimizers
_CUSTOM_OPTIMIZERS = {}


def register_optimizer(name: str, optimizer_class: Type):
    """
    Register a custom optimizer.
    
    Args:
        name: Name to register optimizer under
        optimizer_class: Optimizer class (must inherit from torch.optim.Optimizer)
    """
    if not issubclass(optimizer_class, optim.Optimizer):
        raise ValueError(f"Optimizer class {optimizer_class.__name__} must inherit from torch.optim.Optimizer")
    
    _CUSTOM_OPTIMIZERS[name.lower()] = optimizer_class


def _import_optimizer_from_string(import_path: str) -> Type:
    """Import optimizer class from string path."""
    module_path, class_name = import_path.rsplit(".", 1)
    
    try:
        module = importlib.import_module(module_path)
        optimizer_class = getattr(module, class_name)
        
        if not issubclass(optimizer_class, optim.Optimizer):
            raise ValueError(f"Class {class_name} is not a PyTorch optimizer")
            
        return optimizer_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {import_path}: {e}")


def _create_custom_optimizer(optimizer_class: Type, model_parameters, config: DictConfig) -> optim.Optimizer:
    """Create custom optimizer instance with parameter filtering."""
    optimizer_config = config.get("training", {}).get("optimizer", {})
    
    # Get optimizer signature to filter valid parameters
    sig = inspect.signature(optimizer_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "params"}
    
    # Base parameters
    kwargs = {
        "lr": config.get("training", {}).get("learning_rate", 1e-3),
        "weight_decay": config.get("training", {}).get("weight_decay", 0.0)
    }
    
    # Add all valid config parameters
    for key, value in optimizer_config.items():
        if key not in ["type", "import_path"] and key in valid_params:
            kwargs[key] = value
    
    # Filter out parameters not accepted by this optimizer
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return optimizer_class(model_parameters, **filtered_kwargs)


def create_optimizer(model_parameters, config: DictConfig) -> optim.Optimizer:
    """
    Create optimizer from configuration.
    Supports built-in optimizers, registered custom optimizers, and import-based optimizers.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_config = config.get("training", {}).get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adam").lower()
    
    lr = config.get("training", {}).get("learning_rate", 1e-3)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0)
    
    # Check for custom registered optimizer first
    if optimizer_type in _CUSTOM_OPTIMIZERS:
        optimizer_class = _CUSTOM_OPTIMIZERS[optimizer_type]
        return _create_custom_optimizer(optimizer_class, model_parameters, config)
    
    # Check for import-based optimizer
    if "import_path" in optimizer_config:
        import_path = optimizer_config["import_path"]
        optimizer_class = _import_optimizer_from_string(import_path)
        return _create_custom_optimizer(optimizer_class, model_parameters, config)
    
    # Built-in optimizers
    if optimizer_type == "adam":
        return optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            eps=optimizer_config.get("eps", 1e-8)
        )
    elif optimizer_type == "sgd":
        return optim.SGD(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_config.get("momentum", 0.9),
            nesterov=optimizer_config.get("nesterov", False)
        )
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            alpha=optimizer_config.get("alpha", 0.99),
            eps=optimizer_config.get("eps", 1e-8)
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get("betas", [0.9, 0.999]),
            eps=optimizer_config.get("eps", 1e-8)
        )
    elif optimizer_type == "adagrad":
        return optim.Adagrad(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=optimizer_config.get("eps", 1e-10)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_loss_function(config: DictConfig) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss function configuration
        
    Returns:
        Configured loss function
    """
    loss_config = config.get("training", {}).get("loss", {})
    loss_type = loss_config.get("type", "auto").lower()
    task_type = config.get("algorithm", {}).get("task_type", "regression")
    
    # Auto-select loss based on task type
    if loss_type == "auto":
        if task_type == "classification":
            loss_type = "cross_entropy"
        else:
            loss_type = "mse"
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "cross_entropy":
        class_weights = loss_config.get("class_weights")
        if class_weights:
            import torch
            class_weights = torch.tensor(class_weights)
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def create_scheduler(optimizer, config: DictConfig):
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Configured scheduler or None if disabled
    """
    scheduler_config = config.get("training", {}).get("scheduler", {})
    
    if not scheduler_config.get("enabled", False):
        return None
    
    scheduler_type = scheduler_config.get("type", "step").lower()
    
    if scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1)
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 0)
        )
    elif scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.95)
        )
    elif scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
            min_lr=scheduler_config.get("min_lr", 1e-7)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, config: DictConfig):
        self.enabled = config.get("training", {}).get("early_stopping", {}).get("enabled", False)
        
        # Always initialize these attributes
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
        if not self.enabled:
            return
            
        early_stop_config = config.get("training", {}).get("early_stopping", {})
        self.patience = early_stop_config.get("patience", 5)
        self.min_delta = early_stop_config.get("min_delta", 1e-4)
        self.monitor = early_stop_config.get("monitor", "val_loss")
        
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early.
        
        Args:
            metrics: Dictionary of current metrics
            
        Returns:
            True if training should stop
        """
        if not self.enabled:
            return False
            
        if self.monitor not in metrics:
            return False
            
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            return True
            
        return False