import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level=logging.DEBUG):
    """Setup comprehensive logging configuration with multiple log files."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handlers for different log levels
    debug_handler = logging.FileHandler(log_dir / f"debug_{timestamp}.log", mode="w")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)

    info_handler = logging.FileHandler(log_dir / f"info_{timestamp}.log", mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(simple_formatter)

    error_handler = logging.FileHandler(log_dir / f"error_{timestamp}.log", mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add all handlers
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"LOGGING SYSTEM INITIALIZED - {timestamp}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Log directory: {log_dir}")
    logger.info("=" * 80)

    return logger


def get_logger(name):
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
