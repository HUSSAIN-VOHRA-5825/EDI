"""scripts/plot_robustness.py

Generate robustness visualizations and summary markdown from results/robustness.csv.

Outputs:
  - results/<model>_<metric>_small_multiples.png
  - results/<model>_<metric>_heatmap.png
  - results/robustness_summary.md

Usage:
    python scripts/plot_robustness.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = {
    "blur": ["blur_gaussian", "blur_motion", "blur_defocus"],
    "noise": ["noise_gaussian", "noise_sensor"],
    "jpeg": ["jpeg_quality", "jpeg_lowres"],
    "lighting": ["light_lowlight", "light_overexposed", "light_directional", "light_colortemp"],
    "skin_tone": ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"],
}

COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def _get_majority_class_rate() -> float:
    """Compute share of the larger label in labels.csv."""
    for p in ["data/eval/originals/labels.csv", "data/eval/raw/labels.csv"]:
        f = PROJECT_ROOT / p
        if f.is_file():
            try:
                ldf = pd.read_csv(f)
                if "label" in ldf.columns:
                    counts = ldf["label"].value_counts(normalize=True)
                    return float(counts.max())
            except Exception:
                pass
    return 0.584


def _ci_overlap(low1: float, high1: float, low2: float, high2: float) -> bool:
    """Return True if two confidence intervals [low1, high1] and [low2, high2] overlap."""
    if any(np.isnan([low1, high1, low2, high2])):
        return False
    return max(low1, low2) <= min(high1, high2)


def _format_metric_title(model: str, metric: str) -> str:
    """Format plot title, highlighting headline metrics."""
    if model == "identity_model":
        if metric == "identity_similarity":
            return "Identity Model — Cosine Similarity (Continuous Feature Retention, Headline)"
        elif metric == "match_rate":
            return "Identity Model — Match Rate (Thresholded Proportion, Secondary)"
    elif model == "classifier":
        if metric == "consistency":
            return "Classifier — Prediction Consistency vs Original (Headline)"
        elif metric == "accuracy":
            return "Classifier — Accuracy (5-Fold CV)"
        elif metric == "macro_f1":
            return "Classifier — Macro F1-Score"
    elif model == "face_detection":
        if metric == "detection_rate":
            return "Face Detection — Detection Rate (Headline)"
    return f"{model.replace('_', ' ').title()} — {metric.replace('_', ' ').title()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/robustness.csv")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    csv_path = PROJECT_ROOT / args.results
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist. Run scripts/run_robustness.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("ERROR: robustness.csv is empty.")
        sys.exit(1)

    # Delete old top-level *.png files in results/ before generating new ones (do not touch subdirectories like examples/)
    for old_png in results_dir.glob("*.png"):
        if old_png.is_file():
            old_png.unlink()

    models = df["model"].unique().tolist()
    # Headline metrics: identity_similarity for identity, consistency for classifier
    metrics_per_model = {
        "face_detection": ["detection_rate", "nme"],
        "identity_model": ["identity_similarity", "match_rate"],
        "classifier": ["consistency", "accuracy", "macro_f1"],
    }

    # ── 1. Small-multiples: one panel per family with CI bands ─────────────
    for model in models:
        mdf = df[df["model"] == model]
        metrics = [m for m in metrics_per_model.get(model, []) if m in mdf["metric"].values]
        for metric in metrics:
            mmd = mdf[mdf["metric"] == metric]
            families = [f for f in FAMILY if any(v in mmd["variation"].values for v in FAMILY[f])]
            if not families:
                continue

            fig, axes = plt.subplots(1, len(families), figsize=(max(4, 3.5 * len(families)), 4), sharey=False)
            if len(families) == 1:
                axes = [axes]

            for ax, fam in zip(axes, families):
                vars_in_fam = [v for v in FAMILY[fam] if v in mmd["variation"].values]
                for ci, var in enumerate(vars_in_fam):
                    vd = mmd[mmd["variation"] == var].sort_values("level")
                    if vd.empty:
                        continue
                    color = COLORS[ci % len(COLORS)]
                    is_headline = (model == "identity_model" and metric == "identity_similarity") or (model == "classifier" and metric == "consistency")
                    ax.plot(vd["level"], vd["value"], marker="o", lw=2.2 if is_headline else 1.8, color=color, label=var)
                    ax.fill_between(vd["level"], vd["ci_low"], vd["ci_high"], alpha=0.18, color=color)

                # Baseline reference line (level 0)
                base = df[(df["model"] == model) & (df["metric"] == metric) &
                          (df["variation"] == "baseline") & (df["level"] == 0)]
                if not base.empty:
                    base_val = float(base["value"].iloc[0])
                    ax.axhline(base_val, ls="--", lw=1.2, color="gray", alpha=0.6, label="baseline")

                ax.set_title(fam.replace("_", " ").title(), fontsize=10, fontweight="bold")
                ax.set_xlabel("Severity Level", fontsize=9)
                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.legend(fontsize=7, loc="best", framealpha=0.7)

            fig.suptitle(_format_metric_title(model, metric), fontsize=11, fontweight="bold", y=1.02)
            fig.tight_layout()
            out_path = results_dir / f"{model}_{metric}_small_multiples.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    # ── 2. Heatmaps: variation × level showing drop vs baseline ────────────
    for model in models:
        mdf = df[df["model"] == model]
        metrics = [m for m in metrics_per_model.get(model, []) if m in mdf["metric"].values]
        for metric in metrics:
            mmd = mdf[mdf["metric"] == metric]
            base_row = mmd[(mmd["variation"] == "baseline") & (mmd["level"] == 0)]
            if base_row.empty:
                continue
            base_val = float(base_row["value"].iloc[0])
            levels = sorted([lvl for lvl in mmd["level"].unique() if lvl > 0])
            var_names = sorted([v for v in mmd["variation"].unique() if v != "baseline"])
            if not levels or not var_names:
                continue

            mat = np.full((len(var_names), len(levels)), float("nan"))
            for vi, var in enumerate(var_names):
                for li, lvl in enumerate(levels):
                    cell = mmd[(mmd["variation"] == var) & (mmd["level"] == lvl)]
                    if not cell.empty:
                        val = float(cell["value"].iloc[0])
                        # For NME: higher = worse (drop is inverted)
                        if metric == "nme":
                            mat[vi, li] = -(val - base_val)
                        else:
                            mat[vi, li] = val - base_val     # negative = drop

            fig, ax = plt.subplots(figsize=(max(5, len(levels) * 1.2), max(4, len(var_names) * 0.45)))
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.30, vmax=0.05)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([f"L{l}" for l in levels], fontsize=9)
            ax.set_yticks(range(len(var_names)))
            ax.set_yticklabels(var_names, fontsize=8)
            ax.set_xlabel("Severity Level", fontsize=9)
            ax.set_title(_format_metric_title(model, metric) + " (Drop vs Baseline)", fontsize=10, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

            # Annotate delta values in cells
            for vi in range(len(var_names)):
                for li in range(len(levels)):
                    v = mat[vi, li]
                    if not np.isnan(v):
                        ax.text(li, vi, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                                color="black" if abs(v) < 0.20 else "white")

            fig.tight_layout()
            out_hm = results_dir / f"{model}_{metric}_heatmap.png"
            fig.savefig(out_hm, dpi=120, bbox_inches="tight")
            plt.close(fig)

    # ── 3. Summary Markdown ────────────────────────────────────────────────
    _write_summary(df, results_dir, metrics_per_model)
    print("Visualizations and summary generated in", results_dir.resolve())


def _write_summary(df: pd.DataFrame, results_dir: Path, metrics_per_model: dict) -> None:
    """Generate results/robustness_summary.md (one page)."""
    n_images = int(df["n"].max()) if not df.empty else 0
    majority_rate = _get_majority_class_rate()

    lines: list[str] = [
        "# Robustness Evaluation Summary\n",
        f"**Sample size:** $n = {n_images}$ images (stratified evaluation set).\n",
        "**Definition of Drop:** Throughout this report, **Drop** is defined strictly as "
        "*(level-0 baseline of the same model on the same images) minus (value at level 5)*. "
        "For metrics where higher values indicate degradation (NME), drop is defined as *(value at level 5) minus baseline*.\n",
    ]

    # Baseline performance overview
    lines.append("## Baseline Performance (Level 0 Reference)\n")
    for m in df["model"].unique():
        m_base = df[(df["model"] == m) & (df["variation"] == "baseline") & (df["level"] == 0)]
        for _, r in m_base.iterrows():
            metric = r["metric"]
            val = float(r["value"])
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            if m == "classifier" and metric == "accuracy":
                lines.append(
                    f"- **{m} / {metric}**: {val:.3f} (95% CI: [{lo:.3f}, {hi:.3f}]) | "
                    f"**Majority-class rate**: {majority_rate:.1%} (chance = 50.0%)"
                )
            elif m == "classifier" and metric == "consistency":
                lines.append(f"- **{m} / {metric}**: {val:.3f} (headline; by definition, reference image is 100% consistent with itself)")
            else:
                lines.append(f"- **{m} / {metric}**: {val:.3f} (95% CI: [{lo:.3f}, {hi:.3f}])")

    # CI Warning
    wide_ci = df[(df["ci_high"] - df["ci_low"]) > 0.15]
    if not wide_ci.empty:
        lines.append(
            f"\n> ⚠️ **Confidence Interval Warning**: {len(wide_ci)} measurements have wide 95% CIs "
            f"(span > 0.15), indicating higher uncertainty at small sample size (n={n_images}).\n"
        )

    models = [m for m in df["model"].unique() if m != "baseline"]

    # ── Level 5 Drop Rankings & Baseline Comparison Tables ─────────────────
    lines.append("\n## Variation Rankings by Level-5 Drop\n")
    lines.append("Variations ranked by Level-5 drop per metric. "
                 "If a variation's Level-5 95% CI overlaps the baseline 95% CI, it is flagged as *'not distinguishable from baseline at this n'*.\n")

    not_distinguishable_count = 0
    total_l5_evaluations = 0

    for model in models:
        mdf = df[df["model"] == model]
        metrics = [m for m in metrics_per_model.get(model, []) if m in mdf["metric"].values]
        for metric in metrics:
            base_row = df[(df["model"] == model) & (df["metric"] == metric) &
                          (df["variation"] == "baseline") & (df["level"] == 0)]
            if base_row.empty:
                continue
            base_val = float(base_row["value"].iloc[0])
            base_lo = float(base_row["ci_low"].iloc[0])
            base_hi = float(base_row["ci_high"].iloc[0])

            headline_tag = " (Headline)" if ((model == "identity_model" and metric == "identity_similarity") or (model == "classifier" and metric == "consistency")) else ""
            lines.append(f"\n### `{model}` — `{metric}`{headline_tag} (Baseline: {base_val:.3f} [{base_lo:.3f}, {base_hi:.3f}])\n")
            lines.append("| Variation | Level 1 (95% CI) | Level 3 (95% CI) | Level 5 (95% CI) | Drop at Level 5 | Distinction vs Baseline |")
            lines.append("|---|---|---|---|---|---|")

            var_records = []
            vars_present = sorted([v for v in mdf["variation"].unique() if v != "baseline"])
            for var in vars_present:
                sub = mdf[(mdf["variation"] == var) & (mdf["metric"] == metric)]
                r1 = sub[sub["level"] == 1]
                r3 = sub[sub["level"] == 3]
                r5 = sub[sub["level"] == 5]

                val1_str = f"{float(r1['value'].iloc[0]):.3f} [{float(r1['ci_low'].iloc[0]):.3f}, {float(r1['ci_high'].iloc[0]):.3f}]" if not r1.empty else "—"
                val3_str = f"{float(r3['value'].iloc[0]):.3f} [{float(r3['ci_low'].iloc[0]):.3f}, {float(r3['ci_high'].iloc[0]):.3f}]" if not r3.empty else "—"

                if not r5.empty:
                    val5 = float(r5["value"].iloc[0])
                    lo5 = float(r5["ci_low"].iloc[0])
                    hi5 = float(r5["ci_high"].iloc[0])
                    val5_str = f"{val5:.3f} [{lo5:.3f}, {hi5:.3f}]"
                    drop = (val5 - base_val) if metric == "nme" else (base_val - val5)
                    overlap = _ci_overlap(base_lo, base_hi, lo5, hi5)
                    total_l5_evaluations += 1
                    if overlap:
                        not_distinguishable_count += 1
                        dist_str = "not distinguishable from baseline at this n"
                    else:
                        dist_str = "distinguishable"
                else:
                    drop = -999.0
                    val5_str = "—"
                    dist_str = "—"

                var_records.append({
                    "var": var,
                    "val1": val1_str,
                    "val3": val3_str,
                    "val5": val5_str,
                    "drop": drop,
                    "dist": dist_str,
                })

            # Rank by level-5 drop descending
            var_records.sort(key=lambda x: x["drop"], reverse=True)
            for rec in var_records:
                drop_str = f"{rec['drop']:+.3f}" if rec["drop"] != -999.0 else "—"
                lines.append(f"| `{rec['var']}` | {rec['val1']} | {rec['val3']} | {rec['val5']} | **{drop_str}** | {rec['dist']} |")

    # ── 3 Most Damaging Variations per Model ────────────────────────────────
    lines.append("\n## 3 Most Damaging Variations Per Model\n")
    lines.append("Ranked strictly by Level-5 drop on the model's headline metric:\n")
    max_level = int(df["level"].max()) if not df.empty else 5

    headline_metric_for_model = {
        "face_detection": "detection_rate",
        "identity_model": "identity_similarity",
        "classifier": "consistency",
    }

    for model in models:
        mdf = df[(df["model"] == model) & (df["level"] == max_level) & (df["variation"] != "baseline")]
        base_rows = df[(df["model"] == model) & (df["level"] == 0) & (df["variation"] == "baseline")]

        metric = headline_metric_for_model.get(model, metrics_per_model.get(model, [""])[0])
        base_r = base_rows[base_rows["metric"] == metric]
        if base_r.empty:
            continue
        base_v = float(base_r["value"].iloc[0])

        damage_scores: dict[str, float] = {}
        for var in mdf[mdf["metric"] == metric]["variation"].unique():
            sub = mdf[(mdf["metric"] == metric) & (mdf["variation"] == var)]
            if not sub.empty:
                val = float(sub["value"].iloc[0])
                drop = (val - base_v) if metric == "nme" else (base_v - val)
                damage_scores[var] = drop

        top3 = sorted(damage_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top3:
            lines.append(f"- **{model}** (headline `{metric}`, Level {max_level}): " +
                         ", ".join([f"`{v}` ({d:+.3f} drop)" for v, d in top3]))
        else:
            lines.append(f"- **{model}**: Insufficient variation data.")

    # ── Breaking-Point Table ───────────────────────────────────────────────
    lines.append("\n## Breaking-Point Table\n")
    lines.append("First severity level where the metric drops **>5 points (>0.05)** or **>10 points (>0.10)** below baseline:\n")
    lines.append("| Model | Variation | Metric | Drop >5 pts | Drop >10 pts |")
    lines.append("|---|---|---|---|---|")

    for model in models:
        mdf = df[df["model"] == model]
        metrics = [m for m in metrics_per_model.get(model, []) if m in mdf["metric"].values]
        for metric in metrics:
            base_row = df[(df["model"] == model) & (df["metric"] == metric) &
                          (df["variation"] == "baseline") & (df["level"] == 0)]
            if base_row.empty:
                continue
            base_val = float(base_row["value"].iloc[0])
            vars_present = sorted([v for v in mdf["variation"].unique() if v != "baseline"])

            for var in vars_present:
                vd = mdf[(mdf["variation"] == var) & (mdf["metric"] == metric)].sort_values("level")
                bp5, bp10 = "—", "—"
                for _, r in vd.iterrows():
                    val = float(r["value"])
                    drop = (val - base_val) if metric == "nme" else (base_val - val)
                    if bp5 == "—" and drop > 0.05:
                        bp5 = f"Level {int(r['level'])}"
                    if bp10 == "—" and drop > 0.10:
                        bp10 = f"Level {int(r['level'])}"
                lines.append(f"| `{model}` | `{var}` | `{metric}` | {bp5} | {bp10} |")

    # ── Per-Race Baseline Differences (Level 0) ───────────────────────────
    race_path = results_dir / "robustness_by_race.csv"
    if race_path.exists():
        rdf = pd.read_csv(race_path)
        base_race = rdf[rdf["level"] == 0]
        if not base_race.empty:
            lines.append("\n## Per-Race Baseline Differences (Level 0)\n")
            lines.append("*Note: Identity metrics at level 0 are 1.0 by construction (an image compared with itself), "
                         "so a per-race baseline gap is only meaningful for detection and classifier metrics.*\n")
            for model in models:
                for metric in metrics_per_model.get(model, []):
                    sub = base_race[(base_race["model"] == model) & (base_race["metric"] == metric)]
                    if len(sub) > 1:
                        best = sub.loc[sub["value"].idxmax()]
                        worst = sub.loc[sub["value"].idxmin()]
                        gap = float(best["value"]) - float(worst["value"])
                        if gap < 1e-4:
                            lines.append(f"- **{model} / {metric}**: no difference across race groups (all {float(best['value']):.3f}).")
                        else:
                            lines.append(
                                f"- **{model} / {metric}**: highest `{best['race']}` ({float(best['value']):.3f}), "
                                f"lowest `{worst['race']}` ({float(worst['value']):.3f}), race disparity gap = {gap:.3f}."
                            )

    # ── Per-Race Variation Effects Table (exploratory) ────────────────────
    if race_path.exists():
        rdf = pd.read_csv(race_path)
        max_lvl = int(rdf["level"].max()) if not rdf.empty else 5
        base_rdf = rdf[rdf["level"] == 0]
        eval_rdf = rdf[rdf["level"] == max_lvl]

        if not eval_rdf.empty and not base_rdf.empty:
            lines.append(f"\n## Per-Race Variation Effects (Level {max_lvl})\n")

            race_n: dict[str, int] = {}
            for _, r in base_rdf.drop_duplicates(subset=["race"]).iterrows():
                race_n[str(r["race"])] = int(r["n"])
            races_sorted = sorted(race_n.keys())

            # Check if any group has n < 30 or fewer than 5 detections
            min_race_n = min(race_n.values()) if race_n else 0
            if min_race_n < 30:
                lines.append("> ⚠️ **Notice**: exploratory, groups too small for conclusions (groups have n < 30).\n")

            target_pairs = [
                ("face_detection", "detection_rate"),
                ("identity_model", "identity_similarity"),
                ("classifier", "consistency"),
            ]

            headers = ["Model", "Variation", "Metric"] + [f"{r} (n={race_n[r]})" for r in races_sorted]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")

            for model_name, metric_name in target_pairs:
                sub_eval = eval_rdf[(eval_rdf["model"] == model_name) & (eval_rdf["metric"] == metric_name)]
                vars_in_sub = sorted(sub_eval["variation"].unique())

                for v in vars_in_sub:
                    row_cells = [f"`{model_name}`", f"`{v}`", f"`{metric_name}`"]
                    for r in races_sorted:
                        b_val_series = base_rdf[(base_rdf["model"] == model_name) &
                                                (base_rdf["metric"] == metric_name) &
                                                (base_rdf["race"] == r)]["value"]
                        e_val_series = sub_eval[(sub_eval["variation"] == v) &
                                                (sub_eval["race"] == r)]["value"]
                        if not b_val_series.empty and not e_val_series.empty:
                            b_v = float(b_val_series.iloc[0])
                            e_v = float(e_val_series.iloc[0])
                            drop = (e_v - b_v) if metric_name == "nme" else (b_v - e_v)
                            row_cells.append(f"{drop:+.3f}")
                        else:
                            row_cells.append("—")
                    lines.append("| " + " | ".join(row_cells) + " |")

    lines.append(f"\n*Summary metric count: {not_distinguishable_count}/{total_l5_evaluations} variation evaluations at Level 5 are not distinguishable from baseline at this n.*\n")

    (results_dir / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
