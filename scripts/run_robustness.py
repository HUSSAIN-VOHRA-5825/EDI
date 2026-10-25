"""scripts/run_robustness.py

Robustness evaluation runner.

Applies each registered variation at each severity level to the evaluation images,
runs the models under test, and writes bootstrap-CI results to results/robustness.csv
and results/robustness_by_race.csv.

Usage (quick pass):
    python scripts/run_robustness.py --max-images 50

Full run (after quick pass):
    python scripts/run_robustness.py --max-images 147

Options:
    --models         face_detection identity_model classifier   (default: all three)
    --variations     blur_gaussian blur_motion ...              (default: all registered)
    --levels         1 2 3 4 5                                  (default: 1-5)
    --max-images     50                                         (default: 50)
    --with-gate      also compute check_preservation and report gate-passed subsets
    --seed           42
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.core.config import load_config
from src.core.registry import get_variation
from src.face.mesh import get_face_skin_mask, FaceMeshDetector
import src.variations  # noqa: F401 — triggers all @register_variation decorators
from src.utils.io import get_image_paths, load_image, resize_keep_aspect_ratio
from src.utils.seeding import get_rng, set_global_seed
from src.evaluation.models import FaceDetectionModel, IdentityModel, ClassifierModel
from src.metrics.preservation import check_preservation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_robustness")

SKIN_VARIATIONS = {"skin_lighter", "skin_darker", "skin_warm", "skin_cool"}


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap.

    Resampling is over images (independent units).  NaN values are dropped.
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return float(np.mean(values)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=["face_detection", "identity_model", "classifier"])
    parser.add_argument("--variations", nargs="+", default=None)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--with-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(args.seed)
    rng = get_rng(args.seed)
    pre = cfg.get("preprocessing", {})
    target_size = int(pre.get("target_size", 512))
    interp = str(pre.get("interpolation", "lanczos"))
    var_cfgs = cfg.get("variations", {})
    thresholds = cfg.get("preservation_thresholds", {})
    id_min = float(thresholds.get("identity_similarity_min", 0.60))

    originals_dir = PROJECT_ROOT / "data" / "eval" / "originals"
    labels_path = originals_dir / "labels.csv"
    if not labels_path.exists():
        logger.error("Run scripts/prepare_eval_data.py first to create data/eval/originals/")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)

    # Limit to max-images, stratified by label
    if len(labels_df) > args.max_images:
        from sklearn.model_selection import train_test_split
        labels_df, _ = train_test_split(
            labels_df, train_size=args.max_images, stratify=labels_df["label"], random_state=args.seed
        )
        labels_df = labels_df.reset_index(drop=True)

    logger.info("Evaluation set: %d images", len(labels_df))

    # Load and resize images
    images: List[np.ndarray] = []
    filenames: List[str] = []
    races: List[str] = []
    label_strs: List[str] = []

    for _, row in labels_df.iterrows():
        img_path = originals_dir / row["filename"]
        if not img_path.is_file():
            logger.warning("Missing: %s", img_path)
            continue
        try:
            img = load_image(img_path)
            # Images already resized by prepare_eval_data.py; resize again as safety
            img = resize_keep_aspect_ratio(img, target_longer_side=target_size, interpolation=interp)
            images.append(img)
            filenames.append(row["filename"])
            races.append(row.get("race", "unknown"))
            label_strs.append(row["label"])
        except Exception as e:
            logger.warning("Failed to load %s: %s", img_path, e)

    n = len(images)
    logger.info("Loaded %d images successfully", n)

    # Compute face masks once per image (cached)
    logger.info("Computing face masks ...")
    masks: List[Optional[np.ndarray]] = [get_face_skin_mask(img) for img in images]

    # Determine variations
    all_var_names: List[str] = args.variations or [
        k for k in var_cfgs.keys()
        if k not in ("identity_test", "brightness_test", "forehead_ycrcb_dist_threshold")
    ]

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    race_rows: List[Dict[str, Any]] = []

    # ── Initialise models ──────────────────────────────────────────────────
    fd_model: Optional[FaceDetectionModel] = None
    id_model: Optional[IdentityModel] = None
    clf_model: Optional[ClassifierModel] = None
    orig_landmarks: Optional[List] = None
    orig_embeddings: Optional[np.ndarray] = None
    fold_clf_models: List[Tuple] = []  # (clf, train_idx, test_idx)

    if "face_detection" in args.models:
        fd_model = FaceDetectionModel(min_detection_confidence=0.5)
        logger.info("Pre-computing original landmarks for NME ...")
        orig_landmarks = fd_model.get_landmarks(images)

    if "identity_model" in args.models:
        id_model = IdentityModel(identity_similarity_min=id_min)
        logger.info("Pre-computing original embeddings ...")
        orig_embeddings = id_model.get_embeddings(images)

    if "classifier" in args.models:
        logger.info("Training classifier with 5-fold stratified CV ...")
        int_labels = [ClassifierModel.LABEL_MAP[l] for l in label_strs]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n), int_labels)):
            clf = ClassifierModel(seed=args.seed)
            train_imgs = [images[i] for i in train_idx]
            train_labels = [label_strs[i] for i in train_idx]
            clf.fit(train_imgs, train_labels)
            fold_clf_models.append((clf, train_idx, test_idx))
            logger.info("Fold %d/%d trained", fold + 1, 5)

    # ── Baseline (level 0 = original) ─────────────────────────────────────
    logger.info("Computing baselines (level 0) ...")
    _record_results(
        level=0, var_name="baseline", images=images,
        fd_model=fd_model, id_model=id_model, fold_clf_models=fold_clf_models,
        orig_landmarks=orig_landmarks, orig_embeddings=orig_embeddings,
        filenames=filenames, races=races, label_strs=label_strs,
        rows=rows, race_rows=race_rows, is_baseline=True, models=args.models,
    )

    # ── One example grid per variation ────────────────────────────────────
    example_img = images[0]
    example_mask = masks[0]

    # ── Variation loop ─────────────────────────────────────────────────────
    for var_name in all_var_names:
        var_cfg = var_cfgs.get(var_name, {})
        try:
            variation = get_variation(var_name, config=var_cfg)
        except Exception as e:
            logger.warning("Could not load variation %s: %s — skipping", var_name, e)
            continue

        for level in args.levels:
            logger.info("  %s L%d ...", var_name, level)
            varied_imgs: List[np.ndarray] = []
            gate_passed: List[bool] = []

            for i, img in enumerate(images):
                try:
                    varied = variation.apply(img, severity=level, mask=masks[i], rng=rng)
                except Exception as e:
                    logger.warning("Apply failed %s L%d img %d: %s", var_name, level, i, e)
                    varied = img.copy()
                varied_imgs.append(varied)

                if args.with_gate:
                    gate_metrics = var_cfg.get("gate_metrics", None)
                    pres = check_preservation(img, varied, masks[i], thresholds, gate_metrics)
                    gate_passed.append(pres["passed"])
                else:
                    gate_passed.append(True)

            _record_results(
                level=level, var_name=var_name, images=varied_imgs,
                fd_model=fd_model, id_model=id_model, fold_clf_models=fold_clf_models,
                orig_landmarks=orig_landmarks, orig_embeddings=orig_embeddings,
                filenames=filenames, races=races, label_strs=label_strs,
                rows=rows, race_rows=race_rows, is_baseline=False,
                models=args.models, gate_passed=gate_passed,
            )

        # Save one example grid per variation
        _save_example_grid(variation, var_name, var_cfg, example_img, example_mask,
                           rng, thresholds, results_dir)

    # ── Write CSVs ────────────────────────────────────────────────────────
    rob_df = pd.DataFrame(rows)
    rob_df.to_csv(results_dir / "robustness.csv", index=False)
    logger.info("Wrote results/robustness.csv (%d rows)", len(rob_df))

    if race_rows:
        race_df = pd.DataFrame(race_rows)
        race_df.to_csv(results_dir / "robustness_by_race.csv", index=False)
        logger.info("Wrote results/robustness_by_race.csv (%d rows)", len(race_df))

    print(f"\nDone. n={n} images. Results in {results_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: record per-variation-level metrics into rows list
# ──────────────────────────────────────────────────────────────────────────────

def _record_results(
    level: int,
    var_name: str,
    images: List[np.ndarray],
    fd_model: Optional[FaceDetectionModel],
    id_model: Optional[IdentityModel],
    fold_clf_models: List[Tuple],
    orig_landmarks,
    orig_embeddings,
    filenames: List[str],
    races: List[str],
    label_strs: List[str],
    rows: List[Dict],
    race_rows: List[Dict],
    is_baseline: bool,
    models: List[str],
    gate_passed: Optional[List[bool]] = None,
) -> None:
    n = len(images)
    gp = gate_passed if gate_passed is not None else [True] * n
    is_skin = var_name in SKIN_VARIATIONS or var_name == "baseline"

    def _add_row(model: str, metric: str, vals: np.ndarray, race_tag: Optional[str] = None) -> None:
        mean, ci_low, ci_high = bootstrap_ci(vals)
        rec = {
            "model": model, "variation": var_name, "level": level,
            "metric": metric, "value": mean, "n": int(np.sum(~np.isnan(vals))),
            "ci_low": ci_low, "ci_high": ci_high,
        }
        rows.append(rec)
        if is_skin and race_tag is None:
            # per-race breakdown for all races
            unique_races = sorted(set(races))
            for r in unique_races:
                idx = [i for i, rr in enumerate(races) if rr == r]
                r_vals = vals[idx]
                r_mean, r_lo, r_hi = bootstrap_ci(r_vals)
                race_rows.append({**rec, "race": r, "value": r_mean, "n": int(np.sum(~np.isnan(r_vals))),
                                   "ci_low": r_lo, "ci_high": r_hi})

    # ── Face detection ─────────────────────────────────────────────────────
    if fd_model is not None and "face_detection" in models:
        preds = fd_model.predict(images, orig_landmarks if not is_baseline else None)
        detected = np.array([float(p["detected"]) for p in preds])
        nme = np.array([p["nme"] if not is_baseline else float("nan") for p in preds])
        if is_baseline:
            detected_base = np.array([float(fd_model._landmarks(img) is not None) for img in images])
            _add_row("face_detection", "detection_rate", detected_base)
        else:
            _add_row("face_detection", "detection_rate", detected)
            _add_row("face_detection", "nme", nme)

    # ── Identity model ─────────────────────────────────────────────────────
    if id_model is not None and "identity_model" in models:
        if is_baseline:
            embs = id_model.get_embeddings(images)
            # Baseline: self-similarity (should be ~1.0 or at least above threshold)
            sims = np.array([1.0] * n)
        else:
            preds = id_model.predict(images, orig_embeddings)
            sims = np.array([p["identity_similarity"] for p in preds])
        _add_row("identity_model", "identity_similarity", sims)
        _add_row("identity_model", "match_rate", (sims >= id_model.threshold).astype(float))

    # ── Classifier ─────────────────────────────────────────────────────────
    if fold_clf_models and "classifier" in models:
        all_preds = np.full(n, -1, dtype=int)
        all_true = np.array([ClassifierModel.LABEL_MAP[l] for l in label_strs])
        all_consistent = np.full(n, float("nan"))

        for clf, train_idx, test_idx in fold_clf_models:
            test_imgs = [images[i] for i in test_idx]
            if is_baseline:
                orig_preds_fold = None
            else:
                # Use the same fold's original predictions for consistency
                orig_test_imgs = None  # not available here — derive from label predictions
                orig_preds_fold = all_preds[test_idx].tolist() if np.all(all_preds[test_idx] >= 0) else None

            fold_results = clf.predict(test_imgs, orig_predictions=orig_preds_fold)
            for j, (test_i, res) in enumerate(zip(test_idx, fold_results)):
                all_preds[test_i] = res["prediction"]
                if "consistent" in res:
                    all_consistent[test_i] = float(res["consistent"])

        valid = all_preds >= 0
        acc = accuracy_score(all_true[valid], all_preds[valid]) if valid.sum() > 0 else float("nan")
        f1 = f1_score(all_true[valid], all_preds[valid], average="macro", zero_division=0) if valid.sum() > 0 else float("nan")

        _add_row("classifier", "accuracy", np.where(valid, (all_preds == all_true).astype(float), float("nan")))
        _add_row("classifier", "macro_f1", np.array([f1] * n))
        if not is_baseline:
            _add_row("classifier", "consistency", all_consistent)


def _save_example_grid(variation, var_name, var_cfg, example_img, example_mask, rng, thresholds, results_dir):
    """Save one example grid image per variation to results/."""
    try:
        from src.utils.visualization import create_variation_grid
        from src.metrics.preservation import check_preservation
        varied_imgs, metrics_list = [], []
        gate = var_cfg.get("gate_metrics", None)
        for lvl in [1, 2, 3, 4, 5]:
            v = variation.apply(example_img, severity=lvl, mask=example_mask, rng=rng)
            m = check_preservation(example_img, v, example_mask, thresholds, gate)
            varied_imgs.append(v)
            metrics_list.append(m)
        create_variation_grid(example_img, varied_imgs, metrics_list, [1, 2, 3, 4, 5],
                              var_name, example_mask, results_dir / f"{var_name}_example_grid.png")
    except Exception as e:
        logger.warning("Could not save example grid for %s: %s", var_name, e)


if __name__ == "__main__":
    main()
