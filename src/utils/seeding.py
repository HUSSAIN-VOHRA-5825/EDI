"""Reproducibility utilities for random state management.

Ensures that all stochastic operations (numpy, random, torch) yield deterministic,
reproducible results across execution runs when supplied with a seed from config.
"""

import random
from typing import Optional
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seeds for Python's random, NumPy, and PyTorch.

    Args:
        seed: The integer seed value loaded from configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Instantiate an isolated NumPy random generator.

    Using an explicit Generator prevents variations from contaminating
    or depending on hidden global NumPy state.

    Args:
        seed: Optional integer seed for this generator.

    Returns:
        np.random.Generator instance for reproducible randomness.
    """
    return np.random.default_rng(seed)
