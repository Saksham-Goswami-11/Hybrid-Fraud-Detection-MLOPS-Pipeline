"""
Shared utility functions for the Fraud Detection System.

Provides logging setup, seed management, and common helpers
used across training, evaluation, and API modules.
"""

import logging
import random
import sys

import numpy as np

from src.config import RANDOM_SEED


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with console output.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float) -> str:
    """Format a float as USD currency."""
    return f"${value:,.2f}"
