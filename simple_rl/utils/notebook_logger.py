"""
Logging utility for notebooks and scripts with dual output (console + file).

Uses Python's built-in logging module for robust, fast logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


def setup_notebook_logger(log_dir: str = "logs", name: str = "training") -> Tuple[logging.Logger, str]:
    """
    Set up logger with both console and file handlers.

    Creates a logger that outputs to:
    - Console: Simple format (just the message)
    - File: Timestamped format (YYYY-MM-DD HH:MM:SS - message)

    Args:
        log_dir: Directory to save log files (default: "logs")
        name: Base name for log file (default: "training")

    Returns:
        tuple: (logger, log_file_path)

    Example:
        >>> logger, log_path = setup_notebook_logger()
        >>> logger.info("Training started")
        >>> logger.info(f"Log saved to: {log_path}")
    """
    # Create timestamp-based log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file_path = Path(log_dir) / f"{name}_{timestamp}.log"

    # Create logs directory if it doesn't exist
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger (use unique name with timestamp to avoid conflicts)
    logger_name = f"notebook_training_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove existing handlers

    # Prevent propagation to root logger (avoid duplicate output)
    logger.propagate = False

    # Create formatters
    # Simple format for console (just the message)
    console_formatter = logging.Formatter('%(message)s')
    # More detailed for file (with timestamp)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger, str(log_file_path)


def close_logger(logger: logging.Logger):
    """
    Close all handlers and remove them from logger.

    Args:
        logger: Logger instance to close
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
