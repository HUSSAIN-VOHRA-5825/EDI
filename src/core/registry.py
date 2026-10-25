"""Registry for dynamically looking up and instantiating variation classes."""

from typing import Any, Callable, Dict, List, Optional, Type
from src.core.base import Variation

# Internal registry map
_VARIATION_REGISTRY: Dict[str, Type[Variation]] = {}


def register_variation(name: str) -> Callable[[Type[Variation]], Type[Variation]]:
    """Decorator to register a Variation class under a unique name.

    Args:
        name: Name identifier used in config files and CLI arguments.
    """
    def decorator(cls: Type[Variation]) -> Type[Variation]:
        if name in _VARIATION_REGISTRY:
            raise ValueError(f"Variation '{name}' is already registered.")
        _VARIATION_REGISTRY[name] = cls
        return cls
    return decorator


def get_variation(name: str, config: Optional[Dict[str, Any]] = None) -> Variation:
    """Retrieve and instantiate a registered variation by name.

    Args:
        name: Registered variation identifier.
        config: Configuration dictionary passed to the variation constructor.

    Returns:
        Variation: Instantiated variation object.

    Raises:
        KeyError: If variation name is not found in the registry.
    """
    # Ensure variations modules are imported so decorators execute
    import src.variations  # noqa: F401

    if name not in _VARIATION_REGISTRY:
        available = list_variations()
        raise KeyError(f"Unknown variation '{name}'. Registered variations are: {available}")

    variation_cls = _VARIATION_REGISTRY[name]
    return variation_cls(config=config)


def list_variations() -> List[str]:
    """Return a list of all registered variation names."""
    import src.variations  # noqa: F401
    return sorted(list(_VARIATION_REGISTRY.keys()))
