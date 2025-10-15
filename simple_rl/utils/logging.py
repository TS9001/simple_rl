"""
Logging utilities for the simple RL package.

CRITICAL: This module ensures ALL output is captured to log file:
- Logger messages (info, warning, error)
- Print statements (stdout)
- Error messages (stderr)
- Exceptions and tracebacks
"""

import logging
import sys
import traceback
from typing import Optional, TextIO


class TeeStream:
    """
    Stream that writes to multiple destinations (e.g., console + file).

    This ensures print() statements and errors appear in both console AND log file.
    """
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str):
        """Write data to all streams."""
        for stream in self.streams:
            stream.write(data)
            stream.flush()  # Force immediate write (no buffering!)

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        """Check if any stream is a TTY."""
        return any(hasattr(s, 'isatty') and s.isatty() for s in self.streams)


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging that captures EVERYTHING.

    This function ensures that ALL output is captured to the log file:
    - Logger messages (logger.info, logger.warning, logger.error, etc.)
    - Print statements (redirected from stdout)
    - Error messages (redirected from stderr)
    - Uncaught exceptions (via sys.excepthook)

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger

    Usage:
        logger = setup_logging(level="INFO", log_file="training.log")
        logger.info("This goes to console + file")
        print("This ALSO goes to console + file!")
        raise Exception("This exception ALSO goes to console + file!")
    """
    logger = logging.getLogger("simple_rl")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter (simpler for readability)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Open log file in unbuffered mode (write immediately!)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # CRITICAL: Redirect stdout and stderr to BOTH console and file
        # This captures ALL print statements and errors
        log_file_handle = open(log_file, 'a', buffering=1)  # Line buffering

        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create Tee streams (write to both console and file)
        sys.stdout = TeeStream(original_stdout, log_file_handle)
        sys.stderr = TeeStream(original_stderr, log_file_handle)

        # Store handles for cleanup (if needed)
        logger._log_file_handle = log_file_handle
        logger._original_stdout = original_stdout
        logger._original_stderr = original_stderr

        # CRITICAL: Install exception hook to catch uncaught exceptions
        def exception_handler(exc_type, exc_value, exc_traceback):
            """Log uncaught exceptions to file."""
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts (Ctrl+C)
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Log the full exception with traceback
            logger.critical(
                "Uncaught exception:",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

            # Also print to ensure it appears in console
            print("\n" + "="*60, file=sys.stderr)
            print("UNCAUGHT EXCEPTION:", file=sys.stderr)
            print("="*60, file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
            print("="*60 + "\n", file=sys.stderr)

        sys.excepthook = exception_handler

        logger.info(f"📋 Logging to file: {log_file}")
        logger.info("✓ ALL output (logs, prints, errors) will be captured to this file")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f"simple_rl.{name}")
