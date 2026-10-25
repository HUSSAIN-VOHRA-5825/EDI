"""Pipeline runner for executing controlled variations and preservation checks.

Loads input images, applies pre-processing (aspect-ratio preserved resizing),
computes facial skin masks, executes configured variations across severity levels 1-5,
evaluates clinical and identity preservation metrics, and logs results to a structured CSV.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.core.config import load_config
from src.core.registry import get_variation
from src.face.mesh import get_face_skin_mask
from src.metrics.preservation import check_preservation
from src.utils.io import get_image_paths, load_image, resize_keep_aspect_ratio, save_image
from src.utils.seeding import get_rng, set_global_seed

logger = logging.getLogger(__name__)

# Expected metadata CSV header columns
METADATA_COLUMNS = [
    "original filename",
    "variation",
    "severity",
    "output path",
    "identity similarity",
    "SSIM (full)",
    "SSIM (face region)",
    "LPIPS (face region)",
    "face_clipped_fraction",
    "detail_retention",
    "delta_e_face",
    "ita_before",
    "ita_after",
    "passed",
    "notes",
]


class PipelineRunner:
    """Executes variation stress-testing workflows on batches of face images."""

    def __init__(self, config: Union[str, Path, Dict[str, Any]]) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Path to config YAML or pre-loaded config dictionary.
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        self.seed = int(self.config.get("seed", 42))
        set_global_seed(self.seed)
        self.rng = get_rng(self.seed)

        paths = self.config.get("paths", {})
        self.input_dir = Path(paths.get("input_dir", "data/input"))
        self.output_dir = Path(paths.get("output_dir", "data/output"))
        self.metadata_file = Path(paths.get("metadata_file", "data/output/metadata.csv"))

        pre = self.config.get("preprocessing", {})
        self.target_size = int(pre.get("target_size", 512))
        self.interpolation = str(pre.get("interpolation", "lanczos"))

        self.thresholds = self.config.get("preservation_thresholds", {})
        self.variations_cfg = self.config.get("variations", {})
        self.severity_levels = self.config.get("severity_levels", [1, 2, 3, 4, 5])

    def run(
        self,
        variations: Optional[List[str]] = None,
        levels: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Execute the pipeline across all input images.

        Args:
            variations: Optional list of variation names to run. Defaults to all configured.
            levels: Optional list of severity levels (1-5). Defaults to config levels.

        Returns:
            pd.DataFrame: Metadata dataframe containing all generated output records.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Determine target variations and levels
        target_variations = variations or list(self.variations_cfg.keys())
        target_levels = levels or self.severity_levels

        image_paths = get_image_paths(self.input_dir)
        if not image_paths:
            logger.warning(
                "No images found in input directory: %s. Place face images here before running.",
                self.input_dir,
            )
            # Create empty metadata dataframe if file doesn't exist
            if not self.metadata_file.is_file():
                pd.DataFrame(columns=METADATA_COLUMNS).to_csv(self.metadata_file, index=False)
            return pd.DataFrame(columns=METADATA_COLUMNS)

        records: List[Dict[str, Any]] = []

        logger.info(
            "Starting pipeline run on %d image(s) for variations: %s across levels: %s",
            len(image_paths),
            target_variations,
            target_levels,
        )

        for img_path in image_paths:
            logger.info("Processing image: %s", img_path.name)
            try:
                raw_img = load_image(img_path)
            except Exception as e:
                logger.error("Failed to load %s: %s. Skipping.", img_path.name, e)
                continue

            # Standardize resolution preserving aspect ratio
            orig_img = resize_keep_aspect_ratio(
                raw_img,
                target_longer_side=self.target_size,
                interpolation=self.interpolation,
            )

            # Compute facial skin mask once on reference image
            mask = get_face_skin_mask(orig_img)
            base_notes = []
            if mask is None:
                base_notes.append("No face detected in reference image")

            for var_name in target_variations:
                # Instantiate variation with its configured settings
                var_cfg = self.variations_cfg.get(var_name, {})
                try:
                    variation_obj = get_variation(var_name, config=var_cfg)
                except Exception as e:
                    logger.error("Failed to load variation '%s': %s. Skipping.", var_name, e)
                    continue

                for severity in target_levels:
                    # Apply variation
                    try:
                        varied_img = variation_obj.apply(
                            image=orig_img,
                            severity=int(severity),
                            mask=mask,
                            rng=self.rng,
                        )
                    except Exception as e:
                        logger.error("Error applying %s (level %s) to %s: %s", var_name, severity, img_path.name, e)
                        continue

                    # Evaluate preservation metrics against original
                    gate_metrics = var_cfg.get("gate_metrics", None)
                    preservation = check_preservation(
                        original=orig_img,
                        generated=varied_img,
                        mask=mask,
                        thresholds=self.thresholds,
                        gate_metrics=gate_metrics,
                    )

                    # Output layout: data/output/<variation_name>/level_<n>/<original_filename>.png
                    stem = img_path.stem
                    out_filename = f"{stem}.png"
                    out_abs_path = self.output_dir / var_name / f"level_{severity}" / out_filename

                    save_image(varied_img, out_abs_path)

                    try:
                        out_path_str = str(out_abs_path.relative_to(Path.cwd())).replace("\\", "/")
                    except ValueError:
                        out_path_str = str(out_abs_path).replace("\\", "/")

                    row_notes = list(base_notes)
                    if hasattr(variation_obj, "last_out_of_gamut_fraction") and variation_obj.last_out_of_gamut_fraction is not None:
                        oog = variation_obj.last_out_of_gamut_fraction
                        if oog > 0.0:
                            row_notes.append(f"out-of-gamut: {oog:.2%}")
                    if not preservation["passed"]:
                        failed_metrics = [k for k, v in preservation["passes"].items() if not v]
                        row_notes.append(f"Failed thresholds: {', '.join(failed_metrics)}")

                    record = {
                        "original filename": img_path.name,
                        "variation": var_name,
                        "severity": int(severity),
                        "output path": out_path_str,
                        "identity similarity": preservation["identity_similarity"],
                        "SSIM (full)": preservation["ssim_full"],
                        "SSIM (face region)": preservation["ssim_face"],
                        "LPIPS (face region)": preservation["lpips_face"],
                        "face_clipped_fraction": preservation["face_clipped_fraction"],
                        "detail_retention": preservation["detail_retention"],
                        "delta_e_face": preservation["delta_e_face"],
                        "ita_before": preservation.get("ita_before"),
                        "ita_after": preservation.get("ita_after"),
                        "passed": preservation["passed"],
                        "notes": "; ".join(row_notes) if row_notes else "OK",
                    }
                    records.append(record)

        df_new = pd.DataFrame(records, columns=METADATA_COLUMNS)

        # Merge or append to existing metadata CSV if present
        if self.metadata_file.is_file() and self.metadata_file.stat().st_size > 0:
            df_existing = pd.read_csv(self.metadata_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Deduplicate by (original filename, variation, severity) keeping latest
            df_combined.drop_duplicates(
                subset=["original filename", "variation", "severity"],
                keep="last",
                inplace=True,
            )
            df_combined.to_csv(self.metadata_file, index=False)
            logger.info("Updated metadata saved to: %s (%d rows)", self.metadata_file, len(df_combined))
            return df_combined
        else:
            df_new.to_csv(self.metadata_file, index=False)
            logger.info("Metadata saved to: %s (%d rows)", self.metadata_file, len(df_new))
            return df_new
