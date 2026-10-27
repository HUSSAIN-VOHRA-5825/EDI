"""Configuration loader and schema validator."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path] = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file.

    If loading default.yaml, also automatically detects and merges configs/chains.yaml.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If configuration cannot be parsed or lacks required fields.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration format in {path}. Expected YAML mapping.")

    # Apply defaults if missing
    config.setdefault("seed", 42)
    config.setdefault("severity_levels", [1, 2, 3, 4, 5])

    paths = config.setdefault("paths", {})
    paths.setdefault("input_dir", "data/input")
    paths.setdefault("output_dir", "data/output")
    paths.setdefault("metadata_file", "data/output/metadata.csv")

    preprocessing = config.setdefault("preprocessing", {})
    preprocessing.setdefault("target_size", 512)
    preprocessing.setdefault("interpolation", "lanczos")

    config.setdefault("preservation_thresholds", {})
    variations = config.setdefault("variations", {})

    # Automatically load and merge chains.yaml if present in the same directory
    chains_path = path.parent / "chains.yaml"
    if chains_path.is_file():
        try:
            with open(chains_path, "r", encoding="utf-8") as cf:
                chains_cfg = yaml.safe_load(cf)
            if isinstance(chains_cfg, dict):
                chains_dict = chains_cfg.get("chains", chains_cfg)
                config["chains"] = chains_dict
                for c_name, c_val in chains_dict.items():
                    full_key = c_name if c_name.startswith("chain_") else f"chain_{c_name}"
                    variations.setdefault(full_key, c_val)
        except Exception:
            pass

    # Ensure random chain is also registered under variations
    variations.setdefault("chain_random", {"name": "chain_random"})

    return config
