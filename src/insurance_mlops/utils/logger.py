"""Logging utility for the Insurance Prediction MLOps project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: Optional[str] = None, log_file: Optional[str] = "logs/pipeline.log") -> logging.Logger:
    """Creates and configures a standard logger with console and file output.

    Args:
        name: Name of the logger, typically __name__. Defaults to 'insurance_mlops'.
        log_file: Path to the log file. If None, logs only to console.

    Returns:
        Configured logging.Logger instance.
    """
    logger_name = name or "insurance_mlops"
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
