"""
Utility functions for BrainWaveDisorderNet
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        level: Logging level
        log_file: Log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger