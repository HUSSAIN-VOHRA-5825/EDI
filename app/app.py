"""app/app.py

Controlled Face Image Variation Tool - Streamlit Interactive Demo.

Usage:
    streamlit run app/app.py
"""

import inspect
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Backward/forward-compatible image rendering helper
_ST_IMAGE_ACCEPTS_CONTAINER_WIDTH = "use_container_width" in inspect.signature(st.image).parameters


def show_image(image, caption=None, width=None, **kwargs):
    """Render image adapting to Streamlit version compatibility (<1.40 uses use_column_width, >=1.40 uses use_container_width)."""
    if width is not None:
        return st.image(image, caption=caption, width=width, **kwargs)
    elif _ST_IMAGE_ACCEPTS_CONTAINER_WIDTH:
        return st.image(image, caption=caption, use_container_width=True, **kwargs)
    else:
        return st.image(image, caption=caption, use_column_width=True, **kwargs)


try:
    from app.helpers import (
        create_batch_zip,
        format_metrics_table,
        get_failed_check_explanations,
        get_no_face_message,
        get_run_info_caption,
        overlay_mask_on_image,
    )
except (ImportError, ModuleNotFoundError):
    from helpers import (
        create_batch_zip,
        format_metrics_table,
        get_failed_check_explanations,
        get_no_face_message,
        get_run_info_caption,
        overlay_mask_on_image,
    )
from src.core.config import load_config
from src.core.registry import get_variation, list_variations
import src.variations  # noqa: F401 - ensure all variations including facial_aging are registered
from src.face.mesh import get_face_skin_mask
from src.metrics.preservation import check_preservation
from src.utils.io import resize_keep_aspect_ratio

# Page configuration
try:
    st.set_page_config(
        page_title="Face Variation Pipeline - Clinical Stress-Testing Demo",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# Variation Definitions & Families
# ---------------------------------------------------------------------------

VARIATION_FAMILIES = {
    "Aging Progression": ["facial_aging"],
    "Blur": ["blur_gaussian", "blur_motion", "blur_defocus"],
    "Noise": ["noise_gaussian", "noise_sensor"],
    "JPEG Compression": ["jpeg_quality", "jpeg_lowres"],
    "Lighting": ["light_lowlight", "light_overexposed", "light_directional", "light_colortemp"],
    "Skin Tone": ["skin_lighter", "skin_darker", "skin_warm", "skin_cool"],
}

CHAIN_PRESETS = [
    "chain_dim_clinic_phone",
    "chain_shaky_handheld",
    "chain_harsh_exam_lamp",
    "chain_remote_upload",
    "chain_skin_and_light",
    "chain_random",
]

PLOT_CAPTIONS = {
    "face_detection_detection_rate_heatmap.png": "Face Detection Rate across Variations and Severities (RetinaFace)",
    "face_detection_detection_rate_small_multiples.png": "Face Detection Rate Breakdown across Severity Levels 1–5",
    "face_detection_nme_small_multiples.png": "Normalized Mean Error (NME) in Landmark Localization across Severities",
    "identity_model_identity_similarity_heatmap.png": "FaceNet Identity Cosine Similarity Heatmap across Variations",
    "identity_model_identity_similarity_small_multiples.png": "Mean Identity Cosine Similarity Curves with 95% Bootstrap CIs",
    "identity_model_match_rate_heatmap.png": "Verification Match Rate Heatmap at Cosine Similarity Threshold 0.60",
    "identity_model_match_rate_small_multiples.png": "Verification Match Rate Curves with Wilson 95% Confidence Intervals",
}


# ---------------------------------------------------------------------------
# Caching Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_cached_config():
    """Load and cache pipeline configuration."""
    return load_config("configs/default.yaml")


@st.cache_data(show_spinner=False)
def load_sample_image() -> np.ndarray:
    """Load default reference image from data/input/."""
    sample_path = PROJECT_ROOT / "data" / "input" / "person_1.jpg"
    if not sample_path.exists():
        # Fallback to first image in data/input
        input_dir = PROJECT_ROOT / "data" / "input"
        imgs = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        if imgs:
            sample_path = imgs[0]
        else:
            raise FileNotFoundError("No reference image found in data/input/")
    with Image.open(sample_path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def load_sample_image_and_mask() -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load default reference image from data/input/, resize to 512, and compute skin mask."""
    raw = load_sample_image()
    resized = resize_keep_aspect_ratio(raw, target_longer_side=512)
    mask = get_face_skin_mask(resized)
    return resized, mask



@st.cache_data(show_spinner=False)
def process_uploaded_image_bytes(file_bytes: bytes, target_size: int = 512) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Decode, standardize to target_size, and compute face skin mask (cached by file bytes)."""
    with Image.open(io.BytesIO(file_bytes)) as img:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    resized = resize_keep_aspect_ratio(rgb, target_longer_side=target_size)
    mask = get_face_skin_mask(resized)
    return resized, mask


@st.cache_data(show_spinner=False)
def load_plot_bytes(file_path_str: str, mtime: float) -> bytes:
    """Load plot image bytes keyed on modification time to dynamically reflect newly plotted files."""
    with open(file_path_str, "rb") as f:
        return f.read()


def image_to_png_bytes(img_arr: np.ndarray) -> bytes:
    """Convert an RGB NumPy array to downloadable PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🔬 Variation Settings")
    st.sidebar.caption("Control degradation parameters and inspection overlays")

    # 1. Image Source
    img_source = st.sidebar.radio(
        "Image Source",
        ["Use sample image", "Upload image (PNG/JPG)"],
        index=0,
    )

    uploaded_file = None
    if img_source == "Upload image (PNG/JPG)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Face Image",
            type=["png", "jpg", "jpeg"],
            help="Image will be standardized to 512 px on its longer side",
        )

    st.sidebar.markdown("---")

    # 2. Mode Selector
    mode = st.sidebar.selectbox(
        "Execution Mode",
        ["Single variation", "Facial aging progression", "Chain preset", "Compare all severities"],
        index=0,
    )

    # 3. Variation / Chain Selection
    selected_variation = None
    if mode == "Single variation":
        family = st.sidebar.selectbox("Variation Family", list(VARIATION_FAMILIES.keys()))
        selected_variation = st.sidebar.selectbox("Variation", VARIATION_FAMILIES[family])
    elif mode == "Facial aging progression":
        selected_variation = "facial_aging"
    elif mode == "Chain preset":
        selected_variation = st.sidebar.selectbox("Chain Preset", CHAIN_PRESETS)
    else:  # Compare all severities
        compare_type = st.sidebar.radio("Compare Type", ["Single variation", "Facial aging progression", "Chain preset"])
        if compare_type == "Single variation":
            family = st.sidebar.selectbox("Variation Family", list(VARIATION_FAMILIES.keys()))
            selected_variation = st.sidebar.selectbox("Variation", VARIATION_FAMILIES[family])
        elif compare_type == "Facial aging progression":
            selected_variation = "facial_aging"
        else:
            selected_variation = st.sidebar.selectbox("Chain Preset", CHAIN_PRESETS)

    # 4. Aging Progression or Severity Level Controls
    is_aging = (selected_variation == "facial_aging")
    source_age = 25
    progression = 30
    target_age = 55

    if is_aging:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⏳ Age Progression Controls")
        source_age = st.sidebar.number_input(
            "Estimated Source Age (years)",
            min_value=10,
            max_value=85,
            value=25,
            step=1,
            help="Approximate current age of the subject pictured",
        )
        if mode != "Compare all severities":
            progression = st.sidebar.select_slider(
                "Age Progression",
                options=[10, 20, 30, 40, 50],
                value=30,
                format_func=lambda x: f"+{x} years",
                help="Select how many years forward to age the face",
            )
            severity = max(1, min(5, progression // 10))
            target_age = min(95, source_age + progression)
            st.sidebar.success(f"🎯 Target Age: **~{target_age} years** (`+{progression} yrs`)")
        else:
            severity = None
            st.sidebar.info(f"Comparative strip will show: +10, +20, +30, +40, and +50 years from age {source_age}.")
    else:
        if mode in ["Single variation", "Chain preset"]:
            severity = st.sidebar.slider(
                "Severity Level",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="Level 1 = barely noticeable degradation; Level 5 = severe real-world stress",
            )
        else:
            severity = None

    # 5. Reproducibility Seed
    seed = st.sidebar.number_input("Random Seed", value=42, step=1, min_value=0)

    # 6. Face Mask Overlay Toggle
    show_mask = st.sidebar.checkbox(
        "Show face mask overlay",
        value=False,
        help="Highlight the anatomical skin region used for clinical preservation gating",
    )

    return {
        "img_source": img_source,
        "uploaded_file": uploaded_file,
        "mode": mode,
        "variation": selected_variation,
        "severity": severity,
        "seed": int(seed),
        "show_mask": show_mask,
        "source_age": int(source_age),
        "progression": int(progression),
        "target_age": int(target_age),
        "is_aging": is_aging,
    }


# ---------------------------------------------------------------------------
# Tab 1: Generate
# ---------------------------------------------------------------------------

def render_generate_tab(controls: dict, config: dict):
    st.subheader("Interactive Face Variation & Preservation Verification")

    # Acquire base image and mask
    with st.spinner("Preparing image and computing facial skin landmarks..."):
        try:
            if controls["img_source"] == "Use sample image":
                orig_img, face_mask = load_sample_image_and_mask()
            else:
                if controls["uploaded_file"] is None:
                    st.info("👈 Please upload an image in the sidebar to begin.")
                    return
                orig_img, face_mask = process_uploaded_image_bytes(controls["uploaded_file"].getvalue(), 512)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

    # Check for face detection
    if face_mask is None:
        st.warning(f"⚠️ {get_no_face_message()}")
        show_image(orig_img, caption="Original Input (No Face Detected)", width=350)
        return

    thresholds = config.get("preservation_thresholds", {})
    var_cfgs = config.get("variations", {})
    var_name = controls["variation"]
    var_cfg = var_cfgs.get(var_name, {})
    gate_metrics = var_cfg.get("gate_metrics", None)

    # 1. Compare All Severities Mode
    if controls["mode"] == "Compare all severities":
        is_aging = controls.get("is_aging", False)
        if is_aging:
            st.markdown(f"#### Comparative Aging Strip: +10 to +50 Years Progression (Base Age ~{controls['source_age']})")
        else:
            st.markdown(f"#### Comparative Strip: `{var_name}` across Levels 1–5")
        st.caption("First run may take a moment while models load into memory.")

        variation_obj = get_variation(var_name, config=var_cfg)
        run_rng = np.random.default_rng(controls["seed"])

        cols = st.columns(6)

        # Original column
        with cols[0]:
            disp_orig = overlay_mask_on_image(orig_img, face_mask) if controls["show_mask"] else orig_img
            orig_caption = f"Original (Age ~{controls['source_age']})" if is_aging else "Original (Reference)"
            show_image(disp_orig, caption=orig_caption)
            st.markdown("**Baseline**<br>ID: 1.000<br>SSIM: 1.000<br>LPIPS: 0.000", unsafe_allow_html=True)

        strip_records = []
        spinner_msg = "Generating age progressions (+10 to +50 yrs) and evaluating metrics..." if is_aging else "Generating severities 1 to 5 and evaluating metrics..."
        with st.spinner(spinner_msg):
            for s in [1, 2, 3, 4, 5]:
                if is_aging:
                    delta_yrs = s * 10
                    tgt_s = min(95, controls["source_age"] + delta_yrs)
                    varied_s = variation_obj.apply(
                        orig_img,
                        severity=s,
                        mask=face_mask,
                        rng=run_rng,
                        source_age=controls["source_age"],
                        target_age=tgt_s,
                    )
                else:
                    varied_s = variation_obj.apply(orig_img, severity=s, mask=face_mask, rng=run_rng)

                pres_s = check_preservation(
                    original=orig_img,
                    generated=varied_s,
                    mask=face_mask,
                    thresholds=thresholds,
                    gate_metrics=gate_metrics,
                )
                strip_records.append((s, varied_s, pres_s))

        for idx, (s, varied_s, pres_s) in enumerate(strip_records, start=1):
            with cols[idx]:
                disp_varied = overlay_mask_on_image(varied_s, face_mask) if controls["show_mask"] else varied_s
                if is_aging:
                    delta_yrs = s * 10
                    tgt_s = min(95, controls["source_age"] + delta_yrs)
                    col_cap = f"+{delta_yrs}y (~{tgt_s}y)"
                else:
                    col_cap = f"Level {s}"
                show_image(disp_varied, caption=col_cap)
                badge = "🟢 Within" if pres_s["passed"] else "🟠 Beyond"
                st.markdown(
                    f"**{badge}**<br>"
                    f"ID: {pres_s['identity_similarity']:.3f}<br>"
                    f"SSIM: {pres_s['ssim_face']:.3f}<br>"
                    f"LPIPS: {pres_s['lpips_face']:.3f}",
                    unsafe_allow_html=True,
                )

        return

    # 2. Single Variation or Chain Preset Mode
    severity = controls["severity"]
    is_aging = controls.get("is_aging", False)

    if is_aging:
        st.markdown(f"#### Facial Aging Progression: Current Age ~{controls['source_age']} ➔ Target Age ~{controls['target_age']} (+{controls['progression']} years)")
    else:
        st.markdown(f"#### Degradation: `{var_name}` (Severity Level {severity})")

    spinner_text = "Applying facial aging progression and running preservation check..." if is_aging else "Applying variation and running clinical preservation check..."
    with st.spinner(spinner_text):
        variation_obj = get_variation(var_name, config=var_cfg)
        run_rng = np.random.default_rng(controls["seed"])
        if is_aging:
            varied_img = variation_obj.apply(
                image=orig_img,
                severity=severity,
                mask=face_mask,
                rng=run_rng,
                source_age=controls["source_age"],
                target_age=controls["target_age"],
            )
        else:
            varied_img = variation_obj.apply(
                image=orig_img,
                severity=severity,
                mask=face_mask,
                rng=run_rng,
            )
        preservation = check_preservation(
            original=orig_img,
            generated=varied_img,
            mask=face_mask,
            thresholds=thresholds,
            gate_metrics=gate_metrics,
        )

    # Side-by-side Images
    col_orig, col_res = st.columns(2)
    disp_orig = overlay_mask_on_image(orig_img, face_mask) if controls["show_mask"] else orig_img
    disp_res = overlay_mask_on_image(varied_img, face_mask) if controls["show_mask"] else varied_img

    with col_orig:
        if is_aging:
            st.markdown(f"**Original Reference Image (Age ~{controls['source_age']})**")
        else:
            st.markdown("**Original Reference Image**")
        show_image(disp_orig)

    with col_res:
        if is_aging:
            st.markdown(f"**Re-Aged Synthesized Face (Target ~{controls['target_age']}, +{controls['progression']} yrs)**")
        else:
            st.markdown(f"**Synthesized Output (Level {severity})**")
        show_image(disp_res)

    # Notes for Chains
    if hasattr(variation_obj, "last_notes") and variation_obj.last_notes:
        st.info(f"📋 **Applied Steps:** `{variation_obj.last_notes}`")

    # Pass/Fail Banner
    st.markdown("---")
    if preservation["passed"]:
        st.success("### ✅ Within preservation limits\nAll active clinical preservation thresholds were maintained.")
    else:
        st.warning("### ⚠️ Beyond preservation limits\nOne or more clinical quality thresholds were violated.")
        failures = get_failed_check_explanations(preservation["passes"])
        for fail in failures:
            st.markdown(f"- ⚠️ **{fail}**")

    st.caption("Beyond-limit images are still valid stress-test inputs. They show where models break, but they are no longer a close match to the original face.")

    # Metrics Table
    st.markdown("#### Quantitative Preservation Metrics")
    df_metrics = format_metrics_table(preservation, var_name, thresholds=thresholds)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Downloads
    col_dl1, col_dl2 = st.columns(2)
    dl_prefix = f"aged_plus_{controls['progression']}_years" if is_aging else f"{var_name}_level_{severity}"
    with col_dl1:
        png_bytes = image_to_png_bytes(varied_img)
        st.download_button(
            label="📥 Download Result Image (PNG)",
            data=png_bytes,
            file_name=f"{dl_prefix}.png",
            mime="image/png",
        )
    with col_dl2:
        csv_bytes = df_metrics.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Download Metrics (CSV)",
            data=csv_bytes,
            file_name=f"{dl_prefix}_metrics.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Tab 2: Batch Processing
# ---------------------------------------------------------------------------

def render_batch_tab(config: dict):
    st.subheader("Batch Variation Generation & Export")
    st.markdown(
        "Upload up to **10 face images**, select target variations and severity levels, "
        "and download a bundled ZIP package containing generated variations and `metadata.csv`."
    )

    uploaded_files = st.file_uploader(
        "Upload Face Images (Max 10)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="All images will be resized to 512px on their longer side",
    )

    if uploaded_files and len(uploaded_files) > 10:
        st.warning("⚠️ Maximum 10 images allowed for interactive batch processing. Using the first 10 files.")
        uploaded_files = uploaded_files[:10]

    all_registered = [v for v in list_variations() if not v.endswith("_test")]
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        chosen_variations = st.multiselect(
            "Select Variations to Apply",
            all_registered,
            default=["blur_motion", "noise_sensor", "chain_dim_clinic_phone"],
        )
    with col_b2:
        chosen_levels = st.multiselect(
            "Select Severity Levels",
            [1, 2, 3, 4, 5],
            default=[1, 3, 5],
        )

    if not uploaded_files or not chosen_variations or not chosen_levels:
        st.info("Upload images and select at least one variation and severity level to run batch generation.")
        return

    if st.button("🚀 Run Batch Generation", type="primary"):
        thresholds = config.get("preservation_thresholds", {})
        var_cfgs = config.get("variations", {})
        records: List[dict] = []
        generated_images: Dict[str, np.ndarray] = {}

        total_runs = len(uploaded_files) * len(chosen_variations) * len(chosen_levels)
        progress_bar = st.progress(0)
        status_text = st.empty()

        completed = 0
        for up_file in uploaded_files:
            file_bytes = up_file.getvalue()
            stem = Path(up_file.name).stem
            orig_img, face_mask = process_uploaded_image_bytes(file_bytes, 512)

            if face_mask is None:
                st.warning(f"Skipping {up_file.name}: no face detected.")
                completed += len(chosen_variations) * len(chosen_levels)
                progress_bar.progress(min(1.0, completed / max(1, total_runs)))
                continue

            for v_name in chosen_variations:
                v_cfg = var_cfgs.get(v_name, {})
                var_obj = get_variation(v_name, config=v_cfg)
                gate = v_cfg.get("gate_metrics", None)

                for lvl in chosen_levels:
                    status_text.text(f"Processing: {up_file.name} | {v_name} | Level {lvl}")
                    rng = np.random.default_rng(42)
                    varied = var_obj.apply(orig_img, severity=lvl, mask=face_mask, rng=rng)
                    pres = check_preservation(orig_img, varied, mask=face_mask, thresholds=thresholds, gate_metrics=gate)

                    rel_path = f"{v_name}/level_{lvl}/{stem}.png"
                    generated_images[rel_path] = varied

                    notes_list = []
                    if hasattr(var_obj, "last_notes") and var_obj.last_notes:
                        notes_list.append(var_obj.last_notes)
                    if not pres["passed"]:
                        failed_keys = [k for k, v in pres["passes"].items() if not v]
                        notes_list.append(f"Failed: {', '.join(failed_keys)}")

                    record = {
                        "original_filename": up_file.name,
                        "variation": v_name,
                        "severity": lvl,
                        "output_path": rel_path,
                        "identity_similarity": pres["identity_similarity"],
                        "ssim_face": pres["ssim_face"],
                        "ssim_full": pres["ssim_full"],
                        "lpips_face": pres["lpips_face"],
                        "detail_retention": pres["detail_retention"],
                        "delta_e_face": pres["delta_e_face"],
                        "face_clipped_fraction": pres["face_clipped_fraction"],
                        "passed": pres["passed"],
                        "notes": "; ".join(notes_list) if notes_list else "OK",
                    }
                    records.append(record)

                    completed += 1
                    progress_bar.progress(min(1.0, completed / max(1, total_runs)))

        progress_bar.progress(1.0)
        status_text.text("Batch generation completed!")

        if records:
            df_results = pd.DataFrame(records)
            st.success(f"Generated {len(records)} image variations in memory.")
            st.dataframe(df_results, use_container_width=True)

            zip_bytes = create_batch_zip(records, generated_images)
            st.download_button(
                label="📦 Download Batch ZIP (Images + metadata.csv)",
                data=zip_bytes,
                file_name="batch_variations.zip",
                mime="application/zip",
            )


# ---------------------------------------------------------------------------
# Tab 3: Robustness Results
# ---------------------------------------------------------------------------

def render_robustness_tab():
    st.subheader("Model Robustness Evaluation Reports & Curves")

    results_dir = PROJECT_ROOT / "results"
    summary_path = results_dir / "robustness_summary.md"

    if not results_dir.exists() or not any(results_dir.iterdir()):
        st.info("ℹ️ No robustness evaluation results found. Run `python scripts/run_robustness.py` to generate reports.")
        return

    # Display summary markdown if available
    if summary_path.is_file():
        st.markdown("### Summary Report")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_content = f.read()
        st.markdown(summary_content)

    st.markdown("---")
    st.markdown("### Robustness Curves & Degradation Heatmaps")

    # Run info caption
    caption_text = get_run_info_caption(results_dir)
    st.caption(f"ℹ️ {caption_text}")

    png_plots = sorted(list(results_dir.glob("*.png")))
    if not png_plots:
        st.info("No plot PNGs found in results/.")
        return

    for plot_path in png_plots:
        caption = PLOT_CAPTIONS.get(plot_path.name, plot_path.stem.replace("_", " ").title())
        st.markdown(f"#### {caption}")
        mtime = plot_path.stat().st_mtime
        img_bytes = load_plot_bytes(str(plot_path), mtime)
        show_image(img_bytes)


# ---------------------------------------------------------------------------
# Tab 4: About & Limitations
# ---------------------------------------------------------------------------

def render_about_tab():
    st.subheader("About the Controlled Face Image Variation Tool")

    st.markdown("""
    ### 1. How It Works: 5-Step Pipeline
    1. **Reference Face Ingestion & Standardized Resizing**:
       Reference images are loaded and standardized to 512 px on their longer side using Lanczos interpolation.
    2. **High-Precision Anatomical Landmark & Mask Extraction**:
       MediaPipe Face Mesh (468 landmarks) detects facial geometry, and the Multiclass Selfie Segmenter isolates dermal skin regions while excluding hair, eyes, and background.
    3. **Controlled Physical Degradation & Aging**:
       Degradation variations (blur, sensor noise, compression, exposure, skin tone shifts, realistic chains, or FRAN U-Net facial aging) are applied in linear/CIE-LAB physical space across severity levels 1–5 (with aging providing +10 to +50 years progression).
    4. **Multi-Metric Clinical Preservation Evaluation**:
       Preservation metrics quantify degradation against the original image (FaceNet identity similarity, face-region SSIM, perceptual LPIPS, Laplacian texture detail, delta-E, and clipping).
    5. **Pass/Fail Quality Gate Enforcement**:
       Each variation is evaluated against clinical tolerance thresholds. Failed images are explicitly flagged in metadata and retained for AI vulnerability analysis.

    ---

    ### 2. The Metrics & What "Within" / "Beyond Preservation Limits" Means
    - **Within preservation limits**: The image retains facial identity, structure, and clinical quality tolerances relative to the baseline portrait.
    - **Beyond preservation limits**: Degradation exceeded clinical tolerances; the image is no longer a close match to the original face.
    - **Valid stress-test inputs**: Beyond-limit images remain essential for robustness benchmarking to discover where medical AI models break.

    ---

    ### 3. Known Limitations
    - **Facial Aging Dynamics**: As subjects age over multi-decade spans (+30 to +50 years), identity similarity naturally drops below standard verification thresholds due to wrinkle formation, soft-tissue loss, and epidermal changes, while maintaining anatomical recognizability.
    - **Skin colour carries diagnostic information**: Pathological conditions (e.g., cyanosis, erythema, jaundice, pallor) manifest as skin tone changes; artificial skin tone shifts must be calibrated carefully.
    - **Generated images are synthetic test data**: These variations serve as controlled stress-test benchmarks, not verified clinical diagnostic photographs.
    - **Sample calibration**: Acceptance thresholds are calibrated on a sample benchmark reference set.
    - **Evaluation dataset context**: The evaluation subset (FairFace) is a general benchmark dataset, not a clinical medical imaging cohort.
    - **Detail retention sensitivity to noise**: High-frequency sensor or Gaussian noise increases Laplacian variance, which can artificially inflate the detail retention metric.
    - **ITA scope**: Individual Typology Angle (ITA) is only meaningful for evaluating skin tone shifts and is omitted for non-skin variations.
    """)


# ---------------------------------------------------------------------------
# Main App Controller
# ---------------------------------------------------------------------------

def main():
    st.title("Controlled Face Image Variation Tool")
    st.caption("Realistic synthetic stress-testing and clinical preservation benchmarking for medical AI")

    # Load config
    config = get_cached_config()

    # Render sidebar
    controls = render_sidebar()

    # Main page navigation tabs
    tab_gen, tab_batch, tab_robust, tab_about = st.tabs([
        "🔬 Generate",
        "📦 Batch",
        "📈 Robustness Results",
        "ℹ️ About",
    ])

    with tab_gen:
        render_generate_tab(controls, config)

    with tab_batch:
        render_batch_tab(config)

    with tab_robust:
        render_robustness_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
