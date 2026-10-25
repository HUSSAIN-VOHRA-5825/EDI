"""Unit tests for core framework, test variations, and end-to-end pipeline execution."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.core.config import load_config
from src.core.pipeline import METADATA_COLUMNS, PipelineRunner
from src.core.registry import get_variation, list_variations
from src.utils.io import save_image


def test_registry_contains_dummy_variations() -> None:
    """Verify dummy test variations are properly registered."""
    variations = list_variations()
    assert "identity_test" in variations
    assert "brightness_test" in variations


def test_dummy_variations_execution() -> None:
    """Verify identity_test and brightness_test run across levels 1 to 5."""
    img = np.full((64, 64, 3), 100, dtype=np.uint8)

    id_var = get_variation("identity_test")
    bright_var = get_variation("brightness_test", config={"levels": {1: {"factor": 1.1}, 5: {"factor": 2.0}}})

    for lvl in range(1, 6):
        out_id = id_var.apply(img, severity=lvl)
        assert np.array_equal(out_id, img)

        out_br = bright_var.apply(img, severity=lvl)
        assert out_br.shape == img.shape
        assert out_br.dtype == np.uint8
        # Higher severity should have equal or higher mean brightness
        assert out_br.mean() >= img.mean()


def test_pipeline_end_to_end_and_metadata_csv(tmp_path: Path) -> None:
    """Verify pipeline runs end-to-end, writes outputs, and populates metadata CSV."""
    # Setup temporary isolated environment
    input_dir = tmp_path / "data" / "input"
    output_dir = tmp_path / "data" / "output"
    metadata_csv = output_dir / "metadata.csv"
    input_dir.mkdir(parents=True)

    # Copy or create sample image
    sample_src = Path("data/input/sample_face.png")
    test_img_path = input_dir / "astronaut.png"
    if sample_src.is_file():
        test_img_path.write_bytes(sample_src.read_bytes())
    else:
        import skimage.data
        save_image(skimage.data.astronaut(), test_img_path)

    config = {
        "seed": 42,
        "paths": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "metadata_file": str(metadata_csv),
        },
        "preprocessing": {"target_size": 256, "interpolation": "bilinear"},
        "severity_levels": [1, 2],
        "preservation_thresholds": {
            "identity_similarity_min": 0.7,
            "ssim_face_min": 0.6,
            "ssim_full_min": 0.5,
            "lpips_face_max": 0.35,
        },
        "variations": {
            "identity_test": {"levels": {1: {}, 2: {}}},
            "brightness_test": {"levels": {1: {"factor": 1.1}, 2: {"factor": 1.25}}},
        },
    }

    runner = PipelineRunner(config=config)
    df = runner.run()

    # 1. Assert metadata file existence and columns
    assert metadata_csv.is_file(), "Metadata CSV was not created."
    assert list(df.columns) == METADATA_COLUMNS
    assert len(df) == 4  # 1 image * 2 variations * 2 levels

    # 2. Check output file generation
    for _, row in df.iterrows():
        out_p = Path(row["output path"])
        if not out_p.is_file():
            out_p = tmp_path / out_p
        assert out_p.is_file(), f"Output image missing: {out_p}"


def test_pipeline_no_face_image_no_crash(tmp_path: Path) -> None:
    """Verify pipeline processes an image without a face without crashing."""
    input_dir = tmp_path / "data" / "input"
    output_dir = tmp_path / "data" / "output"
    metadata_csv = output_dir / "metadata.csv"
    input_dir.mkdir(parents=True)

    blank_path = input_dir / "blank.png"
    save_image(np.zeros((128, 128, 3), dtype=np.uint8), blank_path)

    config = {
        "seed": 42,
        "paths": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "metadata_file": str(metadata_csv),
        },
        "preprocessing": {"target_size": 128},
        "severity_levels": [1],
        "variations": {
            "identity_test": {"levels": {1: {}}},
        },
    }

    runner = PipelineRunner(config=config)
    df = runner.run()
    assert len(df) == 1
    assert "No face detected" in df.iloc[0]["notes"]
