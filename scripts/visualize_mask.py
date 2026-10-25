"""CLI Entry Point: Generate and Save Face Skin Mask Overlays for Input Images.

Visualizes detected landmarks and the segmented facial skin mask (with eyes, eyebrows,
and lips excluded) overlaid onto each input image.

Usage:
    python scripts/visualize_mask.py
    python scripts/visualize_mask.py --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.face.mesh import get_default_detector, visualize_face_mask
from src.utils.io import get_image_paths, load_image, resize_keep_aspect_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_mask")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save face skin mask overlay visualizations for images in input folder."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Optional input directory override (defaults to paths.input_dir from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional destination directory for overlays (defaults to data/output/mask_visualizations)",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    input_dir = Path(args.input_dir or paths.get("input_dir", "data/input"))
    output_dir = Path(args.output_dir or Path(paths.get("output_dir", "data/output")) / "mask_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    pre = cfg.get("preprocessing", {})
    target_size = int(pre.get("target_size", 512))
    interpolation = str(pre.get("interpolation", "lanczos"))

    image_paths = get_image_paths(input_dir)
    if not image_paths:
        logger.warning("No images found in %s to visualize.", input_dir)
        print(f"No images found in {input_dir}. Place images there first.")
        return

    detector = get_default_detector()
    logger.info("Found %d image(s) in %s. Generating mask overlays...", len(image_paths), input_dir)

    for img_p in image_paths:
        try:
            raw_img = load_image(img_p)
            img = resize_keep_aspect_ratio(raw_img, target_longer_side=target_size, interpolation=interpolation)
            landmarks = detector.detect_landmarks(img)
            mask = detector.generate_skin_mask(img)

            if mask is None:
                logger.warning("No face detected in %s; skipping overlay creation.", img_p.name)
                continue

            out_path = output_dir / f"mask_{img_p.stem}.png"
            visualize_face_mask(
                image=img,
                mask=mask,
                landmarks=landmarks,
                output_path=out_path,
            )
            logger.info("Saved mask overlay: %s", out_path)
        except Exception as e:
            logger.error("Error processing %s: %s", img_p.name, e)

    print(f"\n[OK] Mask visualizations saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
