"""Visualization helpers for before/after grids and metric inspection."""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from src.metrics.preservation import compute_face_clipped_fraction


def create_variation_grid(
    original_image: np.ndarray,
    varied_images: List[np.ndarray],
    metrics_list: List[dict],
    levels: List[int],
    variation_name: str,
    mask: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Generate a clean side-by-side figure showing the original and levels 1-5 with metrics.

    Args:
        original_image: Original RGB image (H, W, 3).
        varied_images: List of 5 generated RGB images (levels 1 through 5).
        metrics_list: List of 5 metrics dictionaries from check_preservation.
        levels: List of severity levels (typically [1, 2, 3, 4, 5]).
        variation_name: Name of the variation being visualized.
        mask: Optional binary facial skin mask (H, W) for reference clipping calculation.
        output_path: Optional path to save the generated PNG figure.
        dpi: Resolution for saved figure.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    total_cols = len(varied_images) + 1
    # Figure height 6.4 inches gives ample vertical room for image, column title, and 7 metric lines
    fig, axes = plt.subplots(1, total_cols, figsize=(3.8 * total_cols, 6.4), squeeze=False)
    axes = axes[0]

    # Baseline clipping fraction on the reference face region
    orig_clip = compute_face_clipped_fraction(original_image, mask=mask)

    # Column 0: Original reference image
    axes[0].imshow(original_image)
    axes[0].set_title("Original (Reference)", fontsize=13, fontweight="bold", pad=12)
    orig_caption = (
        f"Baseline\n"
        f"ID: 1.00\n"
        f"SSIM (face): 1.00\n"
        f"LPIPS (face): 0.00\n"
        f"Detail: 1.00\n"
        f"ΔE (face): 0.00\n"
        f"Clip (face): {orig_clip:.3f}"
    )
    axes[0].set_xlabel(orig_caption, fontsize=9.0, color="#1a202c", labelpad=8, linespacing=1.3)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#4a5568")
        spine.set_linewidth(1.5)

    # Columns 1 to 5: Severity levels
    for idx, (img, metrics, lvl) in enumerate(zip(varied_images, metrics_list, levels), start=1):
        ax = axes[idx]
        ax.imshow(img)

        # Title with severity level — pad=12 ensures breathing room below suptitle
        ax.set_title(f"Level {lvl}", fontsize=13, fontweight="bold", pad=12)

        # Face-region preservation metrics
        id_val = metrics.get("identity_similarity", 0.0)
        ssim_face_val = metrics.get("ssim_face", 0.0)
        lpips_face_val = metrics.get("lpips_face", 0.0)
        detail_val = metrics.get("detail_retention", 0.0)
        de_val = metrics.get("delta_e_face", 0.0)
        clip_val = metrics.get("face_clipped_fraction", 0.0)
        passed = metrics.get("passed", False)

        status_text = "PASS" if passed else "FAIL"
        status_color = "#16a34a" if passed else "#dc2626"

        ita_after = metrics.get("ita_after")
        ita_before = metrics.get("ita_before")
        ita_str = ""
        if ita_after is not None:
            if ita_before is not None:
                ita_str = f"ITA: {ita_before:+.1f}° → {ita_after:+.1f}°\n"
            else:
                ita_str = f"ITA: {ita_after:+.1f}°\n"

        caption = (
            f"ID: {id_val:.2f}\n"
            f"SSIM (face): {ssim_face_val:.2f}\n"
            f"LPIPS (face): {lpips_face_val:.2f}\n"
            f"Detail: {detail_val:.2f}\n"
            f"ΔE (face): {de_val:.2f}\n"
            f"{ita_str}"
            f"Clip (face): {clip_val:.3f}\n"
            f"Status: {status_text}"
        )
        ax.set_xlabel(
            caption,
            fontsize=9.0,
            color=status_color if not passed else "#1a202c",
            labelpad=8,
            linespacing=1.3,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor(status_color)
            spine.set_linewidth(2.0 if not passed else 1.0)

    # Suptitle well above the column titles; rect reserves ~12% for it
    fig.suptitle(
        f"Controlled Variation Benchmark: {variation_name}",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    # rect=[left, bottom, right, top] reserves the top 10% exclusively for suptitle,
    # completely eliminating overlap with column titles (Level 2, Level 3, etc.)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig
