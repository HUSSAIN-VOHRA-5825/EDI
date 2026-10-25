"""CLI Entry Point: Generate a Before/After Grid Figure across Severity Levels 1-5.

Usage:
    python scripts/make_grid.py --image data/input/sample.png --variation brightness_test
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
from src.core.registry import get_variation
from src.face.mesh import get_face_skin_mask
from src.metrics.preservation import check_preservation
from src.utils.io import get_image_paths, load_image, resize_keep_aspect_ratio
from src.utils.seeding import get_rng, set_global_seed
from src.utils.visualization import create_variation_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_grid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side grid of original + levels 1-5 with preservation metrics."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input face image (or filename in data/input). If omitted, picks the first image in input folder.",
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="brightness_test",
        help="Name of the registered variation to visualize (e.g. brightness_test, identity_test).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path for grid PNG (defaults to data/output/<variation>_grid_<name>.png).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Severity levels to display (default: 1 2 3 4 5).",
    )

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = get_rng(seed)

    pre = cfg.get("preprocessing", {})
    target_size = int(pre.get("target_size", 512))
    interpolation = str(pre.get("interpolation", "lanczos"))
    thresholds = cfg.get("preservation_thresholds", {})
    var_cfg = cfg.get("variations", {}).get(args.variation, {})

    # Determine input image
    if args.image is None:
        input_dir = Path(cfg.get("paths", {}).get("input_dir", "data/input"))
        images = get_image_paths(input_dir)
        if not images:
            logger.error("No images found in %s. Please specify --image or add an image to input folder.", input_dir)
            sys.exit(1)
        image_path = images[0]
    else:
        image_path = Path(args.image)
        if not image_path.is_file():
            # Check if relative to input_dir
            input_dir = Path(cfg.get("paths", {}).get("input_dir", "data/input"))
            candidate = input_dir / args.image
            if candidate.is_file():
                image_path = candidate
            else:
                logger.error("Image file not found at %s", args.image)
                sys.exit(1)

    logger.info("Loading reference image: %s", image_path)
    raw_img = load_image(image_path)
    orig_img = resize_keep_aspect_ratio(raw_img, target_longer_side=target_size, interpolation=interpolation)

    # Compute face mask
    mask = get_face_skin_mask(orig_img)
    if mask is None:
        logger.warning("No face detected in reference image. Metrics will evaluate without mask.")

    # Instantiate variation
    variation = get_variation(args.variation, config=var_cfg)

    varied_images = []
    metrics_list = []

    logger.info("Generating severity levels %s for variation '%s'...", args.levels, args.variation)
    gate_metrics = var_cfg.get("gate_metrics", None)
    for lvl in args.levels:
        varied = variation.apply(orig_img, severity=lvl, mask=mask, rng=rng)
        metrics = check_preservation(orig_img, varied, mask=mask, thresholds=thresholds, gate_metrics=gate_metrics)
        varied_images.append(varied)
        metrics_list.append(metrics)

    # Determine output path
    if args.output is None:
        out_dir = Path(cfg.get("paths", {}).get("output_dir", "data/output"))
        out_p = out_dir / f"{args.variation}_grid_{image_path.stem}.png"
    else:
        out_p = Path(args.output)

    create_variation_grid(
        original_image=orig_img,
        varied_images=varied_images,
        metrics_list=metrics_list,
        levels=args.levels,
        variation_name=args.variation,
        mask=mask,
        output_path=out_p,
    )

    print(f"\n[OK] Grid figure saved successfully to: {out_p.resolve()}\n")


if __name__ == "__main__":
    main()
