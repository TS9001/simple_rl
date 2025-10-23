import logging
import wandb
from pathlib import Path
from typing import Dict, Any, Optional


class WandbLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False

    def init(self, **kwargs) -> None:
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
        if not self._initialized:
            return

        log_kwargs = {"metrics": metrics}
        if step is not None:
            log_kwargs["step"] = step

        wandb.log(**log_kwargs)

    def finish(self) -> None:
        if self._initialized:
            wandb.finish()
            self._initialized = False


class MetricsAggregator:
    def __init__(self):
        self.metrics = {}

    def update(self, metrics: Dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)

    def format_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> str:
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
        self.metrics.clear()


class Logger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wandb_logger = WandbLogger(config) if config.get("use_wandb", False) else None
        self.metrics_aggregator = MetricsAggregator()
        self.total_steps = 0

        self._setup_python_logger()

    def _setup_python_logger(self) -> None:
        logging_config = self.config.get("logging", {})
        log_level = logging_config.get("level", "INFO")
        log_dir = logging_config.get("log_dir", "logs")
        log_file = logging_config.get("log_file", "training.log")

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger_name = f"simple_rl_{id(self)}"
        self.python_logger = logging.getLogger(logger_name)
        self.python_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if self.python_logger.handlers:
            return

        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        simple_formatter = logging.Formatter("%(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.python_logger.addHandler(console_handler)

        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        file_handler = FlushingFileHandler(log_path / log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.python_logger.addHandler(file_handler)

        self.python_logger.propagate = False

        self.info(f"📋 GRPO internal logging to: {log_path / log_file}")
        self.info("✓ All GRPO messages will be flushed immediately (no buffering)")

    def info(self, message: str) -> None:
        self.python_logger.info(message)

    def debug(self, message: str) -> None:
        self.python_logger.debug(message)

    def warning(self, message: str) -> None:
        self.python_logger.warning(message)

    def error(self, message: str) -> None:
        self.python_logger.error(message)

    def critical(self, message: str) -> None:
        self.python_logger.critical(message)

    def exception(self, message: str) -> None:
        self.python_logger.exception(message)

    def init_wandb(self, **kwargs) -> None:
        if self.wandb_logger:
            self.wandb_logger.init(**kwargs)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self.metrics_aggregator.update(metrics)

        if self.wandb_logger:
            current_step = step if step is not None else self.total_steps
            self.wandb_logger.log(metrics, current_step)

        if step is not None:
            self.total_steps = step

    def print_progress(self, episode: int, total_episodes: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        progress_msg = f"Episode {episode}/{total_episodes}"

        if metrics:
            formatted_metrics = self.metrics_aggregator.format_metrics(metrics)
            progress_msg += f" - {formatted_metrics}"

        print(progress_msg)

    def finish(self) -> None:
        if self.wandb_logger:
            self.wandb_logger.finish()


def create_logger(config: Dict[str, Any]) -> Logger:
    return Logger(config)
