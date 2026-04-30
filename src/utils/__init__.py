"""Utility functions and helpers"""

from pathlib import Path


def ensure_directories(*paths):
    """Create directories if they don't exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
