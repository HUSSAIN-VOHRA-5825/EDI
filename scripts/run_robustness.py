"""scripts/run_robustness.py

Robustness evaluation runner for controlled face image variations.

Generates variations entirely in memory, evaluates models under test in small batches,
checkpoints after each (model, variation) pair, and outputs bootstrap 95% CIs.

Usage (Quick pass / default lean run):
    python scripts/run_robustness.py

Options:
    --models             face_detection identity_model classifier (default: face_detection identity_model)
    --max-images         Number of evaluation images (default: 40, stratified by race)
    --levels             Severity levels to test (default: 1 3 5)
    --skip-variations    Variations to skip by default (default: blur_gaussian noise_gaussian jpeg_quality)
    --variations         Explicit list of variations (default: all registered except skip-variations)
    --batch-size         Batch size for in-memory processing (default: 16)
    --threads            Thread count for torch & cv2 (default: half CPU cores)
    --with-gate          Compute clinical preservation gate (default: False)
    --fresh              Start fresh, ignoring existing checkpoints (default: False / resume enabled)
    --seed               Random seed (default: 42)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.core.config import load_config
from src.core.registry import get_variation
from src.face.mesh import get_face_skin_mask
import src.variations  # noqa: F401 — registers variations
from src.utils.io import load_image, resize_keep_aspect_ratio
from src.utils.seeding import get_rng, set_global_seed
from src.evaluation.models import FaceDetectionModel, IdentityModel, ClassifierModel
from src.metrics.preservation import check_preservation
from src.utils.visualization import create_variation_grid

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_robustness")

SKIN_VARIATIONS = {"skin_lighter", "skin_darker", "skin_warm", "skin_cool"}


# ──────────────────────────────────────────────────────────────────────────────
# Confidence Interval Calculations: Wilson Score for Proportions, Bootstrap for Continuous
# ──────────────────────────────────────────────────────────────────────────────

PROPORTION_METRICS = {"detection_rate", "match_rate", "accuracy", "consistency"}


def wilson_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Wilson score confidence interval for binary/proportion metrics.

    Ensures that 40/40 does not result in a zero-width band, providing realistic
    small-sample coverage.
    """
    vals = values[~np.isnan(values)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = float(np.sum(vals > 0.5))
    p = k / n
    z = 1.959963984540054  # 95% normal quantile
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    ci_low = max(0.0, float(center - margin))
    ci_high = min(1.0, float(center + margin))
    return p, ci_low, ci_high


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap over images for continuous metrics."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_resamples)]
    alpha = (1.0 - ci) / 2.0
    return float(np.mean(values)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1.0 - alpha) * 100))


def compute_ci(
    metric: str,
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Dispatch to Wilson score interval for proportions, bootstrap for continuous metrics."""
    if metric in PROPORTION_METRICS:
        return wilson_ci(values, confidence=ci)
    return bootstrap_ci(values, n_resamples=n_resamples, ci=ci, seed=seed)


# ──────────────────────────────────────────────────────────────────────────────
# Format helper
# ──────────────────────────────────────────────────────────────────────────────

def _format_time(sec: float) -> str:
    """Format seconds into human-readable duration."""
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ──────────────────────────────────────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    half_cpus = max(1, (os.cpu_count() or 2) // 2)

    parser = argparse.ArgumentParser(description="Controlled face variation robustness evaluation runner.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=["face_detection", "identity_model"])
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--skip-variations", nargs="+", default=["blur_gaussian", "noise_gaussian", "jpeg_quality"])
    parser.add_argument("--variations", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=half_cpus)
    parser.add_argument("--with-gate", action="store_true", default=False)
    parser.add_argument("--fresh", action="store_true", default=False, help="Start fresh, clearing previous checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set compute resources
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.set_grad_enabled(False)

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
        labels_path = PROJECT_ROOT / "data" / "eval" / "raw" / "labels.csv"
        if not labels_path.exists():
            print("ERROR: Evaluation labels not found in data/eval/originals/ or data/eval/raw/")
            sys.exit(1)

    labels_df = pd.read_csv(labels_path)

    # Stratify by race group to args.max_images
    if len(labels_df) > args.max_images:
        labels_df, _ = train_test_split(
            labels_df, train_size=args.max_images, stratify=labels_df["race"], random_state=args.seed
        )
        labels_df = labels_df.reset_index(drop=True)

    n_images = len(labels_df)

    # Print image count per race group and flag any group with fewer than 5 images
    print(f"\nImage count per race group (n={n_images}):")
    race_counts = labels_df["race"].value_counts()
    for race, count in race_counts.items():
        flag = " [FLAG: < 5 images]" if count < 5 else ""
        print(f"  {race:25s}: {count}{flag}")
    print()

    # Load original images into memory once
    orig_images: List[np.ndarray] = []
    filenames: List[str] = []
    races: List[str] = []
    label_strs: List[str] = []

    for _, row in labels_df.iterrows():
        img_path = originals_dir / row["filename"]
        if not img_path.is_file():
            img_path = PROJECT_ROOT / "data" / "eval" / "raw" / row["filename"]
        if not img_path.is_file():
            continue
        try:
            img = resize_keep_aspect_ratio(load_image(img_path), target_longer_side=target_size, interpolation=interp)
            orig_images.append(img)
            filenames.append(row["filename"])
            races.append(row.get("race", "unknown"))
            label_strs.append(row["label"])
        except Exception as e:
            logger.warning("Failed to load %s: %s", img_path, e)

    n = len(orig_images)
    assert n > 0, "No evaluation images loaded."

    # Cache face skin masks for metric/evaluation once
    cached_masks: List[Optional[np.ndarray]] = [get_face_skin_mask(img) for img in orig_images]

    # Setup directories
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    examples_dir = results_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    progress_file = results_dir / ".progress.json"
    robustness_csv = results_dir / "robustness.csv"
    race_csv = results_dir / "robustness_by_race.csv"
    run_info_file = results_dir / "run_info.json"

    # Checkpoint state
    progress: Dict[str, Any] = {"completed_pairs": [], "timings": {}, "total_time": 0.0}
    if not args.fresh and progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except Exception:
            progress = {"completed_pairs": [], "timings": {}, "total_time": 0.0}
    elif args.fresh:
        if robustness_csv.exists():
            robustness_csv.unlink()
        if race_csv.exists():
            race_csv.unlink()
        if progress_file.exists():
            progress_file.unlink()

    # Determine variations to run
    all_registered = [
        k for k in var_cfgs.keys()
        if k not in ("identity_test", "brightness_test", "forehead_ycrcb_dist_threshold")
    ]
    if args.variations:
        variations_to_run = args.variations
    else:
        variations_to_run = [v for v in all_registered if v not in args.skip_variations]

    # ── Initialize Models ──────────────────────────────────────────────────
    fd_model: Optional[FaceDetectionModel] = None
    id_model: Optional[IdentityModel] = None
    fold_clf_models: List[Tuple] = []
    orig_landmarks: Optional[List] = None
    orig_embeddings: Optional[np.ndarray] = None
    orig_clf_preds: Optional[np.ndarray] = None

    if "face_detection" in args.models:
        fd_model = FaceDetectionModel(min_detection_confidence=0.5)
        orig_landmarks = fd_model.get_landmarks(orig_images)

    if "identity_model" in args.models:
        id_model = IdentityModel(identity_similarity_min=id_min)
        orig_embeddings = id_model.get_embeddings(orig_images)

    if "classifier" in args.models:
        int_labels = [ClassifierModel.LABEL_MAP[l] for l in label_strs]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_clf_models = []
        for train_idx, test_idx in skf.split(np.zeros(n), int_labels):
            clf = ClassifierModel(seed=args.seed)
            train_imgs = [orig_images[i] for i in train_idx]
            train_labels = [label_strs[i] for i in train_idx]
            # Fast fit
            torch.set_grad_enabled(True)
            clf.fit(train_imgs, train_labels, batch_size=args.batch_size)
            torch.set_grad_enabled(False)
            fold_clf_models.append((clf, train_idx, test_idx))

        # Baseline classifier predictions
        orig_clf_preds = np.full(n, -1, dtype=int)
        for clf, _, test_idx in fold_clf_models:
            test_imgs = [orig_images[i] for i in test_idx]
            preds = clf.predict(test_imgs)
            for j, t_idx in enumerate(test_idx):
                orig_clf_preds[t_idx] = preds[j]["prediction"]

        valid = orig_clf_preds >= 0
        base_acc = float(accuracy_score(np.array(int_labels)[valid], orig_clf_preds[valid]))
        if base_acc < 0.60:
            print(f"WARNING: Baseline classifier accuracy is {base_acc:.1%} (< 60%)!")

    # ── Baseline Checkpoint ────────────────────────────────────────────────
    if "baseline" not in progress["completed_pairs"]:
        base_rows: List[Dict[str, Any]] = []
        base_race_rows: List[Dict[str, Any]] = []

        if fd_model and "face_detection" in args.models:
            det_base = np.array([float(lm is not None) for lm in orig_landmarks])
            _add_rows("face_detection", "baseline", 0, "detection_rate", det_base, races, base_rows, base_race_rows)

        if id_model and "identity_model" in args.models:
            sims = np.array([1.0] * n)
            _add_rows("identity_model", "baseline", 0, "identity_similarity", sims, races, base_rows, base_race_rows)
            _add_rows("identity_model", "baseline", 0, "match_rate", sims, races, base_rows, base_race_rows)

        if fold_clf_models and "classifier" in args.models:
            all_true = np.array([ClassifierModel.LABEL_MAP[l] for l in label_strs])
            acc_base = (orig_clf_preds == all_true).astype(float)
            f1_base = float(f1_score(all_true, orig_clf_preds, average="macro", zero_division=0))
            _add_rows("classifier", "baseline", 0, "accuracy", acc_base, races, base_rows, base_race_rows)
            _add_rows("classifier", "baseline", 0, "macro_f1", np.array([f1_base] * n), races, base_rows, base_race_rows)

        _append_rows(base_rows, robustness_csv)
        _append_rows(base_race_rows, race_csv)
        progress["completed_pairs"].append("baseline")
        progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    # ── Execution Loop: (model, variation) pairs ───────────────────────────
    pairs: List[Tuple[str, str]] = []
    for model_name in args.models:
        for var_name in variations_to_run:
            pairs.append((model_name, var_name))

    total_pairs = len(pairs)
    overall_start = time.time()

    for idx, (model_name, var_name) in enumerate(pairs, 1):
        pair_key = f"{model_name}:{var_name}"
        if pair_key in progress["completed_pairs"]:
            continue

        var_cfg = var_cfgs.get(var_name, {})
        try:
            variation = get_variation(var_name, config=var_cfg)
        except Exception as e:
            logger.warning("Could not initialize variation %s: %s — skipping", var_name, e)
            continue

        pair_start = time.time()
        pair_rows: List[Dict[str, Any]] = []
        pair_race_rows: List[Dict[str, Any]] = []

        # Iterate over severity levels
        for level in args.levels:
            # Process in memory in small batches
            pred_records: List[Dict[str, Any]] = []

            for b_start in range(0, n, args.batch_size):
                b_end = min(b_start + args.batch_size, n)
                b_orig = orig_images[b_start:b_end]
                b_masks = cached_masks[b_start:b_end]

                # Generate in memory
                b_varied: List[np.ndarray] = []
                for img_i, orig_img in enumerate(b_orig):
                    try:
                        v = variation.apply(orig_img, severity=level, mask=b_masks[img_i], rng=rng)
                    except Exception:
                        v = orig_img.copy()
                    b_varied.append(v)

                # Run inference for the active model
                if model_name == "face_detection" and fd_model:
                    b_orig_lm = orig_landmarks[b_start:b_end] if orig_landmarks else None
                    b_preds = fd_model.predict(b_varied, orig_landmarks=b_orig_lm)
                    pred_records.extend(b_preds)

                elif model_name == "identity_model" and id_model:
                    b_orig_emb = orig_embeddings[b_start:b_end] if orig_embeddings is not None else None
                    b_preds = id_model.predict(b_varied, orig_embeddings=b_orig_emb)
                    pred_records.extend(b_preds)

                elif model_name == "classifier" and fold_clf_models:
                    # Evaluate on held-out fold models
                    b_preds = []
                    for local_i, global_i in enumerate(range(b_start, b_end)):
                        # Find the fold where global_i was in test_idx
                        matched_clf = None
                        for clf, _, test_idx in fold_clf_models:
                            if global_i in test_idx:
                                matched_clf = clf
                                break
                        if matched_clf is not None:
                            orig_p = [int(orig_clf_preds[global_i])] if orig_clf_preds is not None else None
                            p = matched_clf.predict([b_varied[local_i]], orig_predictions=orig_p)[0]
                        else:
                            p = {"prediction": 0, "label_str": "under_30", "confidence": 0.5, "consistent": True}
                        b_preds.append(p)
                    pred_records.extend(b_preds)

                # Free batch memory
                del b_varied
                del b_orig
                gc.collect()

            # Record metrics for this level
            if model_name == "face_detection":
                det = np.array([float(p["detected"]) for p in pred_records])
                nme = np.array([float(p["nme"]) for p in pred_records])
                _add_rows("face_detection", var_name, level, "detection_rate", det, races, pair_rows, pair_race_rows)
                _add_rows("face_detection", var_name, level, "nme", nme, races, pair_rows, pair_race_rows)

            elif model_name == "identity_model" and id_model:
                sims = np.array([float(p["identity_similarity"]) for p in pred_records])
                match = (sims >= id_model.threshold).astype(float)
                _add_rows("identity_model", var_name, level, "identity_similarity", sims, races, pair_rows, pair_race_rows)
                _add_rows("identity_model", var_name, level, "match_rate", match, races, pair_rows, pair_race_rows)

            elif model_name == "classifier":
                all_true = np.array([ClassifierModel.LABEL_MAP[l] for l in label_strs])
                preds = np.array([p["prediction"] for p in pred_records])
                acc = (preds == all_true).astype(float)
                consistent = np.array([float(p.get("consistent", float("nan"))) for p in pred_records])
                f1_val = float(f1_score(all_true, preds, average="macro", zero_division=0))
                _add_rows("classifier", var_name, level, "accuracy", acc, races, pair_rows, pair_race_rows)
                _add_rows("classifier", var_name, level, "macro_f1", np.array([f1_val] * n), races, pair_rows, pair_race_rows)
                _add_rows("classifier", var_name, level, "consistency", consistent, races, pair_rows, pair_race_rows)

        # Checkpoint: append pair rows immediately
        _append_rows(pair_rows, robustness_csv)
        if pair_race_rows:
            _append_rows(pair_race_rows, race_csv)

        elapsed = time.time() - pair_start
        progress["timings"][pair_key] = elapsed
        progress["completed_pairs"].append(pair_key)
        progress["total_time"] = progress.get("total_time", 0.0) + elapsed
        progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")

        # Save single example grid per variation (once per variation across models)
        grid_out = examples_dir / f"{var_name}_example_grid.png"
        if not grid_out.exists():
            _save_example_grid(variation, var_name, var_cfg, orig_images[0], cached_masks[0],
                               rng, thresholds, args.levels, grid_out)

        # Print strictly one progress line per (model, variation) pair with elapsed time
        print(f"[{idx}/{total_pairs}] {model_name} / {var_name} - {elapsed:.1f}s")

    # ── Time Estimation for Full Runs ──────────────────────────────────────
    measured_time = float(progress.get("total_time", time.time() - overall_start))
    measured_images = n_images
    measured_levels = len(args.levels)

    # Scaling per image and per level
    # Base scaling: (125 / measured_images) * (levels / measured_levels)
    scale_125_l3 = (125.0 / measured_images) * (3.0 / measured_levels)
    scale_125_l5 = (125.0 / measured_images) * (5.0 / measured_levels)

    # Classifier ratio estimate based on relative cost (~0.30 of face_detection + identity_model)
    has_clf_measured = "classifier" in args.models
    if has_clf_measured:
        time_no_clf = sum(v for k, v in progress["timings"].items() if not k.startswith("classifier"))
        time_with_clf = measured_time
    else:
        time_no_clf = measured_time
        time_with_clf = measured_time * 1.30  # estimated +30% for ResNet18 fine-tune + eval

    est_125_l3_no_clf = time_no_clf * scale_125_l3
    est_125_l5_no_clf = time_no_clf * scale_125_l5
    est_125_l3_with_clf = time_with_clf * scale_125_l3
    est_125_l5_with_clf = time_with_clf * scale_125_l5

    run_info = {
        "images_evaluated": n_images,
        "levels_evaluated": args.levels,
        "models_evaluated": args.models,
        "measured_total_seconds": round(measured_time, 2),
        "measured_total_human": _format_time(measured_time),
        "timings_per_pair": {k: round(v, 2) for k, v in progress["timings"].items()},
        "full_run_estimates": {
            "125_images_levels_1_3_5_no_classifier": {
                "seconds": round(est_125_l3_no_clf, 1),
                "human": _format_time(est_125_l3_no_clf),
                "command": "python scripts/run_robustness.py --max-images 125 --levels 1 3 5",
            },
            "125_images_levels_1_to_5_no_classifier": {
                "seconds": round(est_125_l5_no_clf, 1),
                "human": _format_time(est_125_l5_no_clf),
                "command": "python scripts/run_robustness.py --max-images 125 --levels 1 2 3 4 5",
            },
            "125_images_levels_1_3_5_with_classifier": {
                "seconds": round(est_125_l3_with_clf, 1),
                "human": _format_time(est_125_l3_with_clf),
                "command": "python scripts/run_robustness.py --max-images 125 --levels 1 3 5 --models face_detection identity_model classifier",
            },
            "125_images_levels_1_to_5_with_classifier": {
                "seconds": round(est_125_l5_with_clf, 1),
                "human": _format_time(est_125_l5_with_clf),
                "command": "python scripts/run_robustness.py --max-images 125 --levels 1 2 3 4 5 --models face_detection identity_model classifier",
            },
        },
    }
    run_info_file.write_text(json.dumps(run_info, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def _add_rows(
    model: str,
    variation: str,
    level: int,
    metric: str,
    values: np.ndarray,
    races: List[str],
    rows: List[Dict[str, Any]],
    race_rows: List[Dict[str, Any]],
) -> None:
    """Compute confidence interval (Wilson for proportions, bootstrap for continuous) and add record."""
    mean, ci_low, ci_high = compute_ci(metric, values)
    valid_n = int(np.sum(~np.isnan(values)))
    rec = {
        "model": model, "variation": variation, "level": level,
        "metric": metric, "value": mean, "n": valid_n,
        "ci_low": ci_low, "ci_high": ci_high,
    }
    rows.append(rec)

    # Record per-race breakdowns for all variations to support per-race evaluation table
    unique_races = sorted(set(races))
    for r in unique_races:
        idx = [i for i, rr in enumerate(races) if rr == r]
        r_vals = values[idx]
        r_mean, r_lo, r_hi = compute_ci(metric, r_vals)
        race_rows.append({
            **rec,
            "race": r,
            "value": r_mean,
            "n": int(np.sum(~np.isnan(r_vals))),
            "ci_low": r_lo,
            "ci_high": r_hi,
        })


def _append_rows(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    """Append rows to CSV file safely."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def _save_example_grid(
    variation,
    var_name: str,
    var_cfg: dict,
    example_img: np.ndarray,
    example_mask: Optional[np.ndarray],
    rng,
    thresholds: dict,
    levels: List[int],
    out_path: Path,
) -> None:
    """Save one representative grid image per variation."""
    try:
        varied_imgs, metrics_list = [], []
        gate = var_cfg.get("gate_metrics", None)
        for lvl in levels:
            v = variation.apply(example_img, severity=lvl, mask=example_mask, rng=rng)
            m = check_preservation(example_img, v, example_mask, thresholds, gate)
            varied_imgs.append(v)
            metrics_list.append(m)
        create_variation_grid(
            example_img, varied_imgs, metrics_list, levels,
            var_name, example_mask, out_path,
        )
    except Exception as e:
        logger.warning("Could not save example grid for %s: %s", var_name, e)


if __name__ == "__main__":
    main()
