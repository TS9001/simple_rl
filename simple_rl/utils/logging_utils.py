"""Logging and metrics utilities."""

import logging
import wandb
from pathlib import Path
from typing import Dict, Any, Optional


class WandbLogger:
    """Weights & Biases logging utility."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WandbLogger.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._initialized = False

    def init(self, **kwargs) -> None:
        """
        Initialize wandb logging.

        Args:
            **kwargs: Additional arguments for wandb.init()
        """
        if self._initialized:
            return

        init_kwargs = {
            "project": self.config.get("project_name", "grpo"),
            "config": self.config,
            "name": self.config.get("run_name", None),
        }
        init_kwargs.update(kwargs)

        wandb.init(**init_kwargs)
        self._initialized = True

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step (optional)
        """
        if not self._initialized:
            return

        log_kwargs = {"metrics": metrics}
        if step is not None:
            log_kwargs["step"] = step

        wandb.log(**log_kwargs)

    def finish(self) -> None:
        """Finish wandb logging."""
        if self._initialized:
            wandb.finish()
            self._initialized = False


class MetricsAggregator:
    """Utility for aggregating and formatting training metrics."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics = {}

    def update(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics.

        Args:
            metrics: Dictionary of metrics to update
        """
        self.metrics.update(metrics)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get metric value.

        Args:
            key: Metric key
            default: Default value if key not found

        Returns:
            Metric value or default
        """
        return self.metrics.get(key, default)

    def format_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Format metrics for printing.

        Args:
            metrics: Metrics to format (uses stored metrics if None)

        Returns:
            Formatted metrics string
        """
        if metrics is None:
            metrics = self.metrics

        formatted_parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_parts.append(f"{key}: {value:.4f}")
            else:
                formatted_parts.append(f"{key}: {value}")

        return ", ".join(formatted_parts)

    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()


class Logger:
    """Combined logging utility with wandb, console, and file output."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize logger.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.wandb_logger = WandbLogger(config) if config.get("use_wandb", False) else None
        self.metrics_aggregator = MetricsAggregator()
        self.total_steps = 0

        # Set up Python logging for console + file output
        self._setup_python_logger()

    def _setup_python_logger(self) -> None:
        """Set up Python logging with console and file handlers."""
        # Get logging configuration
        logging_config = self.config.get("logging", {})
        log_level = logging_config.get("level", "INFO")
        log_dir = logging_config.get("log_dir", "logs")
        log_file = logging_config.get("log_file", "training.log")

        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create logger (use a unique name per config to avoid conflicts)
        logger_name = f"simple_rl_{id(self)}"
        self.python_logger = logging.getLogger(logger_name)
        self.python_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Prevent duplicate handlers
        if self.python_logger.handlers:
            return

        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        simple_formatter = logging.Formatter("%(message)s")

        # Console handler (simple format)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.python_logger.addHandler(console_handler)

        # File handler (detailed format) - force unbuffered writes for real-time monitoring
        class FlushingFileHandler(logging.FileHandler):
            """File handler that flushes after EVERY write (no buffering)."""
            def emit(self, record):
                super().emit(record)
                self.flush()  # Force immediate write to disk

        file_handler = FlushingFileHandler(log_path / log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.python_logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.python_logger.propagate = False

        self.info(f"📋 GRPO internal logging to: {log_path / log_file}")
        self.info("✓ All GRPO messages will be flushed immediately (no buffering)")

    def info(self, message: str) -> None:
        """Log info message to console and file."""
        self.python_logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message to console and file."""
        self.python_logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message to console and file."""
        self.python_logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message to console and file."""
        self.python_logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical error message to console and file."""
        self.python_logger.critical(message)

    def exception(self, message: str) -> None:
        """Log exception with full traceback to console and file."""
        self.python_logger.exception(message)

    def init_wandb(self, **kwargs) -> None:
        """Initialize wandb logging."""
        if self.wandb_logger:
            self.wandb_logger.init(**kwargs)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to all configured loggers.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step
        """
        # Update aggregator
        self.metrics_aggregator.update(metrics)

        # Log to wandb
        if self.wandb_logger:
            current_step = step if step is not None else self.total_steps
            self.wandb_logger.log(metrics, current_step)

        # Update total steps
        if step is not None:
            self.total_steps = step

    def print_progress(self, episode: int, total_episodes: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Print training progress.

        Args:
            episode: Current episode
            total_episodes: Total episodes
            metrics: Metrics to include in progress message
        """
        progress_msg = f"Episode {episode}/{total_episodes}"

        if metrics:
            formatted_metrics = self.metrics_aggregator.format_metrics(metrics)
            progress_msg += f" - {formatted_metrics}"

        print(progress_msg)

    def finish(self) -> None:
        """Finish all logging."""
        if self.wandb_logger:
            self.wandb_logger.finish()


def create_logger(config: Dict[str, Any]) -> Logger:
    """
    Create logger from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Logger instance
    """
    return Logger(config)
