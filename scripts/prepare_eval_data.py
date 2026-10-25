"""scripts/prepare_eval_data.py

Filter FairFace evaluation images to those with exactly one detected frontal face,
upscale to pipeline resolution (Lanczos), copy to data/eval/originals/.

Usage:
    python scripts/prepare_eval_data.py
    python scripts/prepare_eval_data.py --config configs/default.yaml --min-confidence 0.5
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import mediapipe as mp

from src.core.config import load_config
from src.utils.io import load_image, save_image, resize_keep_aspect_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_eval_data")


def count_frontal_faces(image, face_mesh, min_confidence: float) -> int:
    """Return the number of faces detected in an image by MediaPipe Face Mesh."""
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return 0
    return len(results.multi_face_landmarks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre = cfg.get("preprocessing", {})
    target_size = int(pre.get("target_size", 512))
    interpolation = str(pre.get("interpolation", "lanczos"))

    raw_dir = PROJECT_ROOT / "data" / "eval" / "raw"
    labels_path = raw_dir / "labels.csv"
    out_dir = PROJECT_ROOT / "data" / "eval" / "originals"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_path)
    logger.info("Loaded %d rows from labels.csv", len(df))

    # Images are small FairFace 224px crops — upscale to pipeline size for better
    # landmark detection and consistency with the variation pipeline.
    # Note: upscaling introduces mild interpolation artefacts but is necessary to
    # ensure MediaPipe Face Mesh runs reliably at its expected input resolution.

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,  # detect up to 2 so we can reject multi-face images
        refine_landmarks=True,
        min_detection_confidence=args.min_confidence,
    )

    kept_rows = []
    race_counts: defaultdict = defaultdict(int)
    rejected = 0

    for _, row in df.iterrows():
        img_path = raw_dir / row["filename"]
        if not img_path.is_file():
            logger.warning("File not found: %s — skipping", img_path)
            rejected += 1
            continue
        try:
            img = load_image(img_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s — skipping", img_path, e)
            rejected += 1
            continue

        # Upscale first so landmark detection matches pipeline resolution
        resized = resize_keep_aspect_ratio(img, target_longer_side=target_size, interpolation=interpolation)
        n_faces = count_frontal_faces(resized, face_mesh, args.min_confidence)
        if n_faces != 1:
            logger.debug("Rejected %s: %d faces detected", img_path.name, n_faces)
            rejected += 1
            continue

        # Save to originals/
        out_path = out_dir / row["filename"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(resized, out_path)

        kept_row = dict(row)
        kept_rows.append(kept_row)
        race_counts[row["race"]] += 1

    face_mesh.close()

    # Save filtered labels
    kept_df = pd.DataFrame(kept_rows)
    kept_df.to_csv(out_dir / "labels.csv", index=False)

    logger.info("Kept %d / %d images (%d rejected)", len(kept_rows), len(df), rejected)
    print("\nImages kept per race group:")
    for race, count in sorted(race_counts.items()):
        print(f"  {race:30s}: {count}")
    print(f"\nOriginals saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
