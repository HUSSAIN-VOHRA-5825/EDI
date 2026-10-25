"""Abstract base class for all image variation generators.

Every image variation (blur, noise, lighting, etc.) extends this class and adheres
to a strict interface:
1. Controlled severity levels 1 through 5.
2. Parameter lookup driven entirely from YAML config (no hidden magic constants).
3. Explicit random number generator for reproducibility.
4. Optional facial skin mask integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class Variation(ABC):
    """Abstract base class defining the contract for controlled face variations."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize variation with its name and parameter configuration.

        Args:
            name: Identifier string matching the registry key (e.g. 'brightness_test').
            config: Configuration dictionary for this variation containing 'levels'.
        """
        self.name = name
        self.config = config or {}
        self.levels_config = self.config.get("levels", {})

    def get_level_params(self, severity: int) -> Dict[str, Any]:
        """Retrieve the parameter dictionary defined for a specific severity level.

        Args:
            severity: Integer severity level (1 to 5).

        Returns:
            Dictionary containing parameters configured for this severity level.

        Raises:
            ValueError: If severity is not an integer between 1 and 5.
        """
        if not (1 <= severity <= 5):
            raise ValueError(f"Severity must be an integer between 1 and 5, got: {severity}")

        # Support both integer and string keys from YAML
        params = self.levels_config.get(severity) or self.levels_config.get(str(severity))
        return params if params is not None else {}

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        severity: int,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Apply the variation effect to an input image at the given severity.

        Args:
            image: Input RGB image as uint8 NumPy array (H, W, 3).
            severity: Controlled severity level from 1 (mild) to 5 (extreme).
            mask: Optional binary skin mask (H, W) if effect is region-targeted.
            rng: Optional isolated NumPy random generator for deterministic noise/sampling.

        Returns:
            np.ndarray: Modified image with same shape (H, W, 3) and dtype uint8.
        """
        raise NotImplementedError("Each variation subclass must implement apply()")
