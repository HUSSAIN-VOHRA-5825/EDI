"""Core pipeline framework, base classes, and configuration."""

from src.core.base import Variation
from src.core.config import load_config
from src.core.pipeline import PipelineRunner
from src.core.registry import get_variation, list_variations, register_variation

__all__ = [
    "Variation",
    "load_config",
    "PipelineRunner",
    "get_variation",
    "list_variations",
    "register_variation",
]
