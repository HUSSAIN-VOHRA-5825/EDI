"""scripts/plot_robustness.py

Generate robustness plots and the summary markdown from results/robustness.csv.

Usage:
    python scripts/plot_robustness.py
    python scripts/plot_robustness.py --results results/robustness.csv
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
import matplotlib.patches as mpatches

FAMILY = {
    "blur": ["blur_gaussian", "blur_motion", "blur_defocus"],
    "noise": ["noise_gaussian", "noise_sensor"],
    "jpeg": ["jpeg_quality", "jpeg_lowres"],
    "lighting": ["light_lowlight", "light_overexposed", "light_directional", "light_colortemp"],
    "skin_tone": ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"],
}

COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/robustness.csv")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    df = pd.read_csv(PROJECT_ROOT / args.results)

    models = df["model"].unique().tolist()
    metrics_per_model = {
        "face_detection": ["detection_rate", "nme"],
        "identity_model": ["identity_similarity", "match_rate"],
        "classifier": ["accuracy", "macro_f1", "consistency"],
    }

    # ── Small-multiples: one panel per family ──────────────────────────────
    for model in models:
        mdf = df[df["model"] == model]
        metric_names = [m for m in metrics_per_model.get(model, []) if m in mdf["metric"].values]
        for metric in metric_names:
            mmd = mdf[mdf["metric"] == metric]
            families = [f for f in FAMILY if any(v in mmd["variation"].values for v in FAMILY[f])]
            if not families:
                continue
            fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 4), sharey=False)
            if len(families) == 1:
                axes = [axes]
            for ax, fam in zip(axes, families):
                vars_in_fam = [v for v in FAMILY[fam] if v in mmd["variation"].values]
                for ci, var in enumerate(vars_in_fam):
                    vd = mmd[mmd["variation"] == var].sort_values("level")
                    ax.plot(vd["level"], vd["value"], marker="o", color=COLORS[ci % 10], label=var)
                    ax.fill_between(vd["level"], vd["ci_low"], vd["ci_high"],
                                    alpha=0.18, color=COLORS[ci % 10])
                    # Baseline (level 0)
                    base = df[(df["model"] == model) & (df["metric"] == metric) &
                              (df["variation"] == "baseline") & (df["level"] == 0)]
                    if not base.empty:
                        ax.axhline(float(base["value"].iloc[0]), ls="--", lw=1, color="grey", alpha=0.5)
                ax.set_title(fam, fontsize=10)
                ax.set_xlabel("Severity")
                ax.set_ylabel(metric)
                ax.legend(fontsize=7, loc="lower left")
            fig.suptitle(f"{model} — {metric}", y=1.01, fontsize=12)
            fig.tight_layout()
            out = results_dir / f"{model}_{metric}_small_multiples.png"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)

    # ── Heatmaps: variation × level, showing drop vs baseline ─────────────
    for model in models:
        mdf = df[df["model"] == model]
        metric_names = [m for m in metrics_per_model.get(model, []) if m in mdf["metric"].values]
        for metric in metric_names:
            mmd = mdf[mdf["metric"] == metric]
            base_row = mmd[(mmd["variation"] == "baseline") & (mmd["level"] == 0)]
            if base_row.empty:
                continue
            base_val = float(base_row["value"].iloc[0])
            levels = sorted(mmd["level"].unique())
            levels = [l for l in levels if l > 0]
            var_names = sorted([v for v in mmd["variation"].unique() if v != "baseline"])
            mat = np.full((len(var_names), len(levels)), float("nan"))
            for vi, var in enumerate(var_names):
                for li, lvl in enumerate(levels):
                    cell = mmd[(mmd["variation"] == var) & (mmd["level"] == lvl)]
                    if not cell.empty:
                        mat[vi, li] = float(cell["value"].iloc[0]) - base_val
            fig, ax = plt.subplots(figsize=(max(6, len(levels) * 1.2), max(4, len(var_names) * 0.5)))
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
            ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels)
            ax.set_yticks(range(len(var_names))); ax.set_yticklabels(var_names, fontsize=8)
            ax.set_xlabel("Severity"); ax.set_title(f"{model} — {metric} drop vs baseline")
            plt.colorbar(im, ax=ax, fraction=0.03)
            fig.tight_layout()
            fig.savefig(results_dir / f"{model}_{metric}_heatmap.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    # ── Summary markdown ───────────────────────────────────────────────────
    _write_summary(df, results_dir, metrics_per_model)
    print("Plots and summary written to", results_dir.resolve())


def _write_summary(df: pd.DataFrame, results_dir: Path, metrics_per_model: dict) -> None:
    """Write results/robustness_summary.md."""
    n_images = int(df["n"].max()) if not df.empty else 0
    lines: list[str] = []
    lines.append("# Robustness Evaluation Summary\n")
    lines.append(f"**n = {n_images} images** (evaluation set).\n")
    wide_ci_warn = df[(df["ci_high"] - df["ci_low"]) > 0.15]
    if not wide_ci_warn.empty:
        lines.append(
            f"> ⚠️ **Wide CIs (>0.15)**: {len(wide_ci_warn)} metric/variation/level combinations "
            f"have wide confidence intervals — interpret with caution; n is small.\n"
        )

    models = df["model"].unique().tolist()

    # Breaking-point table
    lines.append("\n## Breaking-Point Table\n")
    lines.append("First level where each metric falls **>5 pts** / **>10 pts** below baseline.\n")
    lines.append("| Model | Variation | Metric | >5 pts | >10 pts |")
    lines.append("|---|---|---|---|---|")
    for model in models:
        mdf = df[df["model"] == model]
        for metric in metrics_per_model.get(model, []):
            mmd = mdf[mdf["metric"] == metric]
            base_row = mmd[(mmd["variation"] == "baseline") & (mmd["level"] == 0)]
            if base_row.empty:
                continue
            base_val = float(base_row["value"].iloc[0])
            var_names = sorted([v for v in mmd["variation"].unique() if v != "baseline"])
            for var in var_names:
                vd = mmd[mmd["variation"] == var].sort_values("level")
                bp5 = bp10 = "—"
                for _, r in vd.iterrows():
                    drop = (base_val - float(r["value"])) * 100
                    if bp5 == "—" and drop > 5:
                        bp5 = str(int(r["level"]))
                    if bp10 == "—" and drop > 10:
                        bp10 = str(int(r["level"]))
                lines.append(f"| {model} | {var} | {metric} | {bp5} | {bp10} |")

    # Most damaging variations per model
    lines.append("\n## 3 Most Damaging Variations Per Model\n")
    for model in models:
        mdf = df[(df["model"] == model) & (df["level"] == 5) & (df["variation"] != "baseline")]
        base_rows = df[(df["model"] == model) & (df["level"] == 0) & (df["variation"] == "baseline")]
        drops: dict[str, float] = {}
        for metric in metrics_per_model.get(model, []):
            base_r = base_rows[base_rows["metric"] == metric]
            if base_r.empty:
                continue
            base_v = float(base_r["value"].iloc[0])
            for var in mdf[mdf["metric"] == metric]["variation"].unique():
                cell = mdf[(mdf["metric"] == metric) & (mdf["variation"] == var)]
                if not cell.empty:
                    drop = (base_v - float(cell["value"].iloc[0])) * 100
                    drops[var] = drops.get(var, 0) + max(0, drop)
        top3 = sorted(drops, key=drops.get, reverse=True)[:3]  # type: ignore[arg-type]
        lines.append(f"**{model}**: {', '.join(top3) if top3 else '(not enough data)'}\n")

    # Per-race baseline paragraph
    race_df_path = results_dir / "robustness_by_race.csv"
    if race_df_path.exists():
        rdf = pd.read_csv(race_df_path)
        base_race = rdf[rdf["level"] == 0]
        if not base_race.empty:
            lines.append("\n## Per-Race Baseline Differences\n")
            for model in models:
                for metric in metrics_per_model.get(model, []):
                    sub = base_race[(base_race["model"] == model) & (base_race["metric"] == metric)]
                    if sub.empty:
                        continue
                    best = sub.loc[sub["value"].idxmax()]
                    worst = sub.loc[sub["value"].idxmin()]
                    gap = float(best["value"]) - float(worst["value"])
                    if gap > 0.01:
                        lines.append(
                            f"- **{model}/{metric}**: highest `{best['race']}` "
                            f"({float(best['value']):.3f}), lowest `{worst['race']}` "
                            f"({float(worst['value']):.3f}), gap = {gap:.3f}.\n"
                        )

    (results_dir / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
