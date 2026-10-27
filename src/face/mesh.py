"""MediaPipe Face Mesh landmark extraction and facial skin mask generation.

In clinical dermatological AI, the primary region of interest is the facial skin.
Non-skin structures such as hair, eyes, eyebrows, lips, and oral cavity introduce
high-contrast features that can artificially skew image similarity metrics or clinical lesion analysis.
This module extracts the facial boundary while carving out facial orifices and hair,
yielding a clean, smoothed binary skin mask.

Edit-mask vs. metric-mask distinction
--------------------------------------
generate_skin_mask()      → METRIC mask: used only for computing preservation metrics.
                            Conservative; does NOT include the forehead extension or neck.
generate_skin_edit_mask() → EDIT mask: used by skin-tone variations to decide which pixels
                            to colour-shift. Extends over the forehead, includes the neck
                            box, has tighter eye exclusion (no halo), and returns an optional
                            lip mask for 50 % blending.  Changing this function does NOT
                            affect the calibrated preservation thresholds.
"""

import logging
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# Configure module logger for visible warning outputs
logger = logging.getLogger(__name__)

# Model download constants for MediaPipe multiclass selfie segmenter
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_PATH = _MODEL_DIR / "selfie_multiclass_256x256.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)


def _ensure_segmenter_model() -> Optional[Path]:
    """Ensure multiclass selfie segmenter model exists locally, downloading if necessary."""
    if _MODEL_PATH.is_file():
        return _MODEL_PATH
    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe multiclass selfie segmenter model to %s...", _MODEL_PATH)
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        if _MODEL_PATH.is_file() and _MODEL_PATH.stat().st_size > 1000:
            return _MODEL_PATH
    except Exception as e:
        logger.warning(
            "Could not download MediaPipe multiclass selfie segmenter model: %s. Falling back to YCrCb.", e
        )
    return None


def _build_ordered_loops(connections: frozenset) -> List[List[int]]:
    """Convert an unordered set of connection edges into ordered contour index loops.

    MediaPipe provides connections as pairs (start_idx, end_idx). To draw a filled polygon
    or extract a continuous perimeter, we trace adjacent nodes into a sequence of closed loops.
    """
    adj = defaultdict(list)
    for u, v in connections:
        adj[u].append(v)
        adj[v].append(u)

    unvisited = set(adj.keys())
    loops = []
    while unvisited:
        start = next(iter(unvisited))
        loop = [start]
        unvisited.remove(start)
        if not adj[start]:
            continue
        curr = adj[start][0]
        while curr != start and curr in unvisited:
            loop.append(curr)
            unvisited.remove(curr)
            candidates = [n for n in adj[curr] if n != (loop[-2] if len(loop) > 1 else None)]
            if not candidates:
                break
            curr = candidates[0]
        loops.append(loop)
    return loops


class FaceMeshDetector:
    """Reusable MediaPipe Face Mesh model and selfie segmenter wrapper.

    Reusing the detector across pipeline calls avoids the substantial latency and memory
    overhead of re-initializing the underlying TensorFlow Lite graph for every image.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """Initialize MediaPipe FaceMesh model and multiclass selfie segmenter.

        Args:
            static_image_mode: Treat each image independently (optimal for batches of still photos).
            max_num_faces: Maximum number of faces to detect (clinical benchmarks evaluate primary face).
            refine_landmarks: Includes detailed iris and lip contour landmarks.
            min_detection_confidence: Confidence threshold for face detection.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
        )

        # Precompute ordered contour indices for efficiency
        self.oval_loops = _build_ordered_loops(self.mp_face_mesh.FACEMESH_FACE_OVAL)
        self.left_eye_indices = list({idx for pair in self.mp_face_mesh.FACEMESH_LEFT_EYE for idx in pair})
        self.right_eye_indices = list({idx for pair in self.mp_face_mesh.FACEMESH_RIGHT_EYE for idx in pair})
        self.left_eyebrow_indices = list(
            {idx for pair in self.mp_face_mesh.FACEMESH_LEFT_EYEBROW for idx in pair}
        )
        self.right_eyebrow_indices = list(
            {idx for pair in self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW for idx in pair}
        )
        self.lips_indices = list({idx for pair in self.mp_face_mesh.FACEMESH_LIPS for idx in pair})

        # Initialize multiclass selfie segmenter if available
        self.segmenter = None
        try:
            model_path = _ensure_segmenter_model()
            if model_path is not None and model_path.is_file():
                # Read model bytes directly to avoid Windows absolute path prefixing issues in MediaPipe C++ runtime
                model_bytes = model_path.read_bytes()
                base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
                options = mp_vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
                self.segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        except Exception as e:
            logger.warning(
                "Failed to initialize MediaPipe multiclass ImageSegmenter: %s. Using YCrCb fallback.", e
            )
            self.segmenter = None

    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and return facial landmark coordinates for an image.

        Args:
            image: RGB image as uint8 NumPy array (H, W, 3).

        Returns:
            np.ndarray of shape (N, 2) containing pixel (x, y) coordinates for landmarks,
            or None if no face was detected.
        """
        h, w = image.shape[:2]
        # MediaPipe expects RGB; input is guaranteed RGB
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            logger.warning("No face detected by MediaPipe Face Mesh.")
            return None

        # Extract primary face landmarks
        first_face = results.multi_face_landmarks[0]
        landmarks_px = np.array(
            [[lm.x * w, lm.y * h] for lm in first_face.landmark],
            dtype=np.float32,
        )
        return landmarks_px

    def _run_segmenter(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe multiclass segmenter and return the full category mask (HxW int array)."""
        if self.segmenter is None:
            return None
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
            result = self.segmenter.segment(mp_img)
            cat_mask = result.category_mask.numpy_view().copy()
            if cat_mask.shape[:2] != image.shape[:2]:
                cat_mask = cv2.resize(
                    cat_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            return cat_mask
        except Exception as e:
            logger.warning("MediaPipe multiclass segmentation inference failed: %s", e)
            return None

    def _segment_skin_multiclass(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return binary mask for face-skin (class 3) at image resolution."""
        cat_mask = self._run_segmenter(image)
        if cat_mask is None:
            return None
        skin_mask = (cat_mask == 3).astype(np.uint8) * 255
        if np.count_nonzero(skin_mask) > 0:
            return skin_mask
        return None

    def _fallback_ycrcb_skin_mask(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        polygon_mask: np.ndarray,
        dist_threshold: float = 25.0,
        extra_sample_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Fallback skin segmentation: YCrCb colour distance from median cheek colour + largest connected region.

        Args:
            dist_threshold: Euclidean distance in Cr-Cb space below which a pixel is considered skin.
                            Default 25 works for most Fitzpatrick types; lower = stricter (less hair included).
        """
        h, w = image.shape[:2]
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        # Sample cheek and nose-bridge landmarks for a representative skin colour reference
        # 50/280: cheeks; 205/425: zygomatic; 4: nose tip (stable skin area)
        sample_indices = extra_sample_indices if extra_sample_indices else [50, 280, 205, 425, 4]
        samples = []
        for idx in sample_indices:
            if idx < len(landmarks):
                cx, cy = int(round(landmarks[idx][0])), int(round(landmarks[idx][1]))
                r = 4
                y1, y2 = max(0, cy - r), min(h, cy + r + 1)
                x1, x2 = max(0, cx - r), min(w, cx + r + 1)
                patch = ycrcb[y1:y2, x1:x2]
                if patch.size > 0:
                    samples.append(patch.reshape(-1, 3))

        if samples:
            all_samples = np.vstack(samples)
            med_cr = float(np.median(all_samples[:, 1]))
            med_cb = float(np.median(all_samples[:, 2]))
        else:
            # Canonical skin Cr/Cb medians as last resort
            med_cr = 150.0
            med_cb = 100.0

        # Calculate Euclidean colour distance in Cr-Cb space (illumination-invariant chrominance)
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)
        dist = np.sqrt((cr - med_cr) ** 2 + (cb - med_cb) ** 2)

        # Retain candidate pixels within dist_threshold units colour distance
        candidate_skin = (dist <= dist_threshold).astype(np.uint8) * 255
        combined = cv2.bitwise_and(polygon_mask, candidate_skin)

        # Select largest connected component to filter out stray hair / noise
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = int(np.argmax(areas)) + 1
            largest_comp = (labels == max_label).astype(np.uint8) * 255
            return largest_comp
        return combined

    # ------------------------------------------------------------------
    # METRIC mask (conservative — do not change without recalibrating thresholds)
    # ------------------------------------------------------------------

    def generate_skin_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Generate a binary facial skin mask (uint8, 0/255) excluding hair, eyes, eyebrows, and lips.

        THIS MASK IS USED FOR METRICS ONLY.  Do not alter it without re-calibrating
        the preservation thresholds in configs/default.yaml.

        Algorithm:
        1. Extract face perimeter using MediaPipe Face Mesh face oval contour.
        2. Fill the facial silhouette with white (255) and carve out eyes, eyebrows, and lips.
        3. Remove hair: use MediaPipe multiclass selfie segmenter (face-skin class 3) to keep only
           landmark-polygon pixels classified as face-skin. If unavailable, fall back to YCrCb colour
           distance from median cheek colour + largest connected region.
        4. Erode the mask by ~2% of face width (bounding-box width of the landmark polygon).
        5. Apply morphological closing and slight edge smoothing.

        Args:
            image: RGB image as uint8 NumPy array (H, W, 3).

        Returns:
            np.ndarray: Binary mask with shape (H, W) and dtype uint8 (values 0 and 255),
            or None if no face is detected.
        """
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            return None

        h, w = image.shape[:2]
        polygon_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Fill base face oval region
        for loop in self.oval_loops:
            poly_pts = landmarks[loop].astype(np.int32)
            cv2.fillPoly(polygon_mask, [poly_pts], 255)

        # Helper to carve out excluded facial features using their convex hull
        def carve_feature(indices: List[int], dilation_px: int = 2, target: Optional[np.ndarray] = None) -> None:
            if target is None:
                target = polygon_mask
            feature_pts = landmarks[indices].astype(np.int32)
            hull = cv2.convexHull(feature_pts)
            ex_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(ex_mask, hull, 255)
            if dilation_px > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
                ex_mask = cv2.dilate(ex_mask, kernel)
            target[ex_mask > 0] = 0

        # 2. Exclude left/right eyes, eyebrows, lips
        carve_feature(self.left_eye_indices, dilation_px=3)
        carve_feature(self.right_eye_indices, dilation_px=3)
        carve_feature(self.left_eyebrow_indices, dilation_px=4)
        carve_feature(self.right_eyebrow_indices, dilation_px=4)
        carve_feature(self.lips_indices, dilation_px=3)

        # 3. Remove hair via MediaPipe multiclass selfie segmenter, with YCrCb fallback
        skin_segmented = self._segment_skin_multiclass(image)
        if skin_segmented is not None:
            face_skin_mask = cv2.bitwise_and(polygon_mask, skin_segmented)
            # Ensure intersection captured sufficient skin area; if degenerate, use fallback
            if np.count_nonzero(face_skin_mask) < 0.10 * np.count_nonzero(polygon_mask):
                face_skin_mask = self._fallback_ycrcb_skin_mask(image, landmarks, polygon_mask)
        else:
            face_skin_mask = self._fallback_ycrcb_skin_mask(image, landmarks, polygon_mask)

        # 4. Scale mask erosion by FACE width (landmark polygon bounding-box width, about 2 percent)
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        face_w = max(10.0, float(x_max - x_min))

        erode_px = max(1, int(round(0.02 * face_w)))
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        eroded_mask = cv2.erode(face_skin_mask, erode_kernel)

        target_mask = eroded_mask if np.count_nonzero(eroded_mask) > 0 else face_skin_mask

        # 5. Smooth the mask boundary with morphological close then open (elliptical kernel, ~2-3% of face width)
        smooth_k = max(1, int(round(0.025 * face_w)))
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_k * 2 + 1, smooth_k * 2 + 1))
        mask_closed = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel_smooth)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_smooth)

        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = int(np.argmax(areas)) + 1
            cleaned_mask = (labels == max_label).astype(np.uint8) * 255
        else:
            cleaned_mask = mask_opened

        # Keep the eye, brow, and lip holes by re-carving them so morphology operations don't bridge them
        carve_feature(self.left_eye_indices, dilation_px=3, target=cleaned_mask)
        carve_feature(self.right_eye_indices, dilation_px=3, target=cleaned_mask)
        carve_feature(self.left_eyebrow_indices, dilation_px=4, target=cleaned_mask)
        carve_feature(self.right_eyebrow_indices, dilation_px=4, target=cleaned_mask)
        carve_feature(self.lips_indices, dilation_px=3, target=cleaned_mask)

        # Smooth edges slightly
        blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), sigmaX=1.5)
        _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        return final_mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # EDIT mask (for skin-tone colour shifting — includes forehead + neck)
    # ------------------------------------------------------------------

    def _build_extended_polygon(
        self,
        landmarks: np.ndarray,
        face_h: float,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Return a filled mask for the face oval extended upward over the forehead.

        We lift the topmost landmark points of the oval by 0.25 × face_height
        (distance from brow line to chin) to cover the forehead, then keep the
        side points (temples) at their original positions and smooth the result
        via convex hull.  This naturally includes the forehead skin that sits
        above the brow ridge but below where the hair starts.

        Args:
            landmarks: (N, 2) pixel coordinates of all face landmarks.
            face_h: Vertical extent of the face bounding box (used to scale the lift).
            h, w: Image height and width.

        Returns:
            Binary mask (H, W) uint8 covering the extended polygon.
        """
        # Collect all oval landmark points
        oval_pts: List[List[float]] = []
        for loop in self.oval_loops:
            oval_pts.extend(landmarks[loop].tolist())
        oval_arr = np.array(oval_pts, dtype=np.float32)

        # Identify the top portion of the oval (top 20 % of y range)
        y_min_oval = float(oval_arr[:, 1].min())
        y_max_oval = float(oval_arr[:, 1].max())
        top_thresh = y_min_oval + 0.20 * (y_max_oval - y_min_oval)

        # Lift the topmost points upward by 0.25 × face_h; clamp to image boundary
        shifted = oval_arr.copy()
        is_top = shifted[:, 1] <= top_thresh
        lift = 0.25 * face_h
        shifted[is_top, 1] = np.clip(shifted[is_top, 1] - lift, 0, h - 1)

        # Convex hull over shifted points gives a smooth, gap-free polygon
        hull = cv2.convexHull(shifted.astype(np.int32))
        ext_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(ext_mask, hull, 255)
        return ext_mask

    def generate_skin_edit_mask(
        self,
        image: np.ndarray,
        return_lip_mask: bool = False,
        forehead_ycrcb_dist_threshold: float = 22.0,
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Generate the binary skin EDIT mask and optional lip mask for skin-tone colour shifting.

        The edit mask is DISTINCT from the metric mask.  Changing this function does NOT
        affect the calibrated preservation thresholds.

        Regions included:
          - Face skin (MediaPipe class 3) intersected with the extended face polygon.
          - Forehead: union of (a) segmenter face-skin inside extended polygon, and
            (b) pixels whose YCrCb Cr-Cb distance from the median cheek colour is within
            forehead_ycrcb_dist_threshold (default 22, configurable in default.yaml under
            variations.forehead_ycrcb_dist_threshold).  This recovers forehead pixels that
            the segmenter mis-classifies as hair at the scalp boundary.
          - Neck body-skin (class 2) inside a box 0.6 face-heights below chin, feathered
            into the face region.

        Regions excluded:
          - Hair (class 1) dilated by 0.5 % of face width — removes hair-leak pixels at
            the forehead boundary.  Where fringe genuinely covers the forehead those
            pixels remain unedited (correct behaviour).
          - Eyebrows (dilated slightly).
          - Eye openings with dilation 0 (tight exclusion, no light halo).
          - Lips carved out and returned separately for 50 % blending.

        Args:
            image: RGB uint8 image (H, W, 3).
            return_lip_mask: If True, return (edit_mask, lip_mask) tuple.
            forehead_ycrcb_dist_threshold: YCrCb Cr-Cb distance threshold for forehead skin
                recovery (see comment in configs/default.yaml → variations.forehead_ycrcb_dist_threshold).

        Returns:
            edit_mask (H, W) uint8 binary, or (edit_mask, lip_mask) if return_lip_mask=True.
        """
        # ---- 1. Obtain base metric mask and landmarks ----
        face_mask = self.generate_skin_mask(image)
        if face_mask is None:
            return (None, None) if return_lip_mask else None

        h, w = image.shape[:2]
        landmarks = self.detect_landmarks(image)
        if landmarks is None:
            return (face_mask, None) if return_lip_mask else face_mask

        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        face_w = max(10.0, float(x_max - x_min))
        face_h = max(10.0, float(y_max - y_min))
        cx = 0.5 * (x_min + x_max)
        chin_y = float(landmarks[152][1]) if len(landmarks) > 152 else float(y_max)

        # ---- 2. Run segmenter once; extract hair, face-skin, body-skin ----
        cat_mask = self._run_segmenter(image)
        if cat_mask is not None:
            hair_bin = (cat_mask == 1).astype(np.uint8) * 255
            body_skin = (cat_mask == 2).astype(np.uint8) * 255
            seg_face_skin = (cat_mask == 3).astype(np.uint8) * 255
        else:
            hair_bin = np.zeros((h, w), dtype=np.uint8)
            body_skin = np.zeros((h, w), dtype=np.uint8)
            seg_face_skin = None

        # ---- 3. Build extended polygon (covers forehead) ----
        ext_polygon = self._build_extended_polygon(landmarks, face_h, h, w)

        # ---- 4. Forehead mask: segmenter face-skin ∪ YCrCb match, inside extended polygon ----
        # Hair dilated by 0.5 % of face width to remove boundary fringe pixels
        hair_k = max(1, int(round(0.005 * face_w)))
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hair_k * 2 + 1, hair_k * 2 + 1))
        hair_dilated = cv2.dilate(hair_bin, hair_kernel)

        # (a) segmenter face-skin inside extended polygon
        if seg_face_skin is not None:
            seg_part = cv2.bitwise_and(ext_polygon, seg_face_skin)
        else:
            seg_part = np.zeros((h, w), dtype=np.uint8)

        # (b) YCrCb forehead recovery: pixels close to median cheek/nose skin colour
        # forehead_ycrcb_dist_threshold is set in configs/default.yaml → variations.forehead_ycrcb_dist_threshold
        # Pixels are sampled from stable mid-face landmarks (cheeks + nose tip) to get a robust skin reference.
        ycrcb_part = self._fallback_ycrcb_skin_mask(
            image, landmarks, ext_polygon, dist_threshold=forehead_ycrcb_dist_threshold
        )

        # Union of both methods, intersected with extended polygon, minus dilated hair
        forehead_skin = cv2.bitwise_or(seg_part, ycrcb_part)
        forehead_skin = cv2.bitwise_and(forehead_skin, ext_polygon)
        forehead_skin[hair_dilated > 0] = 0  # Remove pixels covered by (dilated) hair

        # ---- Eyebrow hair extraction (3a): exclude only the brow hairs (dark, high-texture
        #      pixels inside brow region, dilated slightly), NOT the surrounding skin.
        #      The skin around the brows receives the full shift, eliminating the halo.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        sample_indices = [50, 280, 205, 425, 4]
        skin_samples = []
        for s_idx in sample_indices:
            if s_idx < len(landmarks):
                px, py = int(round(landmarks[s_idx][0])), int(round(landmarks[s_idx][1]))
                if 0 <= py < h and 0 <= px < w:
                    skin_samples.append(float(gray[py, px]))
        med_skin_lum = float(np.median(skin_samples)) if skin_samples else 150.0

        brow_hair_mask = np.zeros((h, w), dtype=np.uint8)
        for brow_indices in [self.left_eyebrow_indices, self.right_eyebrow_indices]:
            brow_pts = landmarks[brow_indices].astype(np.int32)
            brow_hull = cv2.convexHull(brow_pts)
            brow_hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(brow_hull_mask, brow_hull, 255)
            # Dilate search area slightly (~0.8% face_w, ~2-3px) to enclose hair tips
            search_k = max(1, int(round(0.008 * face_w)))
            search_region = cv2.dilate(
                brow_hull_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_k * 2 + 1, search_k * 2 + 1)),
            )
            in_search = search_region > 0

            # Hair criteria: dark pixels OR high-texture hair edges OR segmenter hair
            dark_hair = gray < (med_skin_lum - 18.0)
            textured_hair = (lap > 15.0) & (gray < (med_skin_lum - 5.0))
            is_hair = (dark_hair | textured_hair) & in_search
            if hair_dilated is not None:
                is_hair |= (hair_dilated > 0) & in_search

            hair_bin = is_hair.astype(np.uint8) * 255
            # Dilate slightly (~0.5% face width, ~1-2px)
            dil_k = max(1, int(round(0.005 * face_w)))
            hair_dil = cv2.dilate(
                hair_bin,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k * 2 + 1, dil_k * 2 + 1)),
            )
            brow_hair_mask = cv2.bitwise_or(brow_hair_mask, hair_dil)

        # Subtract brow hair from forehead
        forehead_skin[brow_hair_mask > 0] = 0

        # Keep only the largest connected component in the forehead region to avoid isolated patches
        num_labels, lbl_map, stats, _ = cv2.connectedComponentsWithStats(forehead_skin)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = int(np.argmax(areas)) + 1
            forehead_skin = (lbl_map == best).astype(np.uint8) * 255

        # Morphological closing to fill small holes above brows, then re-open to remove thin bridges
        close_k = max(1, int(round(0.03 * face_w)))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k * 2 + 1, close_k * 2 + 1))
        forehead_skin = cv2.morphologyEx(forehead_skin, cv2.MORPH_CLOSE, close_kernel)

        # ---- 5. Neck region (3b): widen smoothly downward from jaw width, feather edges ----
        neck_mask = np.zeros((h, w), dtype=np.uint8)
        if np.any(body_skin):
            # Determine lower-face / jaw boundary from face oval landmarks
            oval_pts = landmarks[self.oval_loops[0]]
            lower_jaw = oval_pts[oval_pts[:, 1] >= (chin_y - 0.25 * face_h)]
            if len(lower_jaw) > 0:
                jaw_x_min = float(np.min(lower_jaw[:, 0]))
                jaw_x_max = float(np.max(lower_jaw[:, 0]))
            else:
                jaw_x_min = cx - 0.45 * face_w
                jaw_x_max = cx + 0.45 * face_w
            jaw_w = max(jaw_x_max - jaw_x_min, 0.70 * face_w)

            # Meeting point with the jaw (slightly above chin to ensure gap-free overlap)
            y_meet = max(0, int(chin_y - 0.08 * face_h))
            y_bot = min(h, int(chin_y + 0.65 * face_h))

            # Top width: at least as wide as the jaw where they meet (with slight outward margin)
            neck_top_x1 = max(0, int(min(jaw_x_min, cx - 0.5 * jaw_w) - 0.05 * face_w))
            neck_top_x2 = min(w, int(max(jaw_x_max, cx + 0.5 * jaw_w) + 0.05 * face_w))

            # Bottom width: widens smoothly downward (flaring outward ~20% of face_w on each side)
            neck_bot_x1 = max(0, int(neck_top_x1 - 0.20 * face_w))
            neck_bot_x2 = min(w, int(neck_top_x2 + 0.20 * face_w))

            # Construct smooth trapezoidal neck polygon
            poly = np.array(
                [[neck_top_x1, y_meet], [neck_top_x2, y_meet],
                 [neck_bot_x2, y_bot], [neck_bot_x1, y_bot]],
                dtype=np.int32,
            )
            neck_poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(neck_poly_mask, [poly], 255)

            # Restrict body skin to the flared trapezoid
            neck_candidate = cv2.bitwise_and(body_skin, neck_poly_mask)

            # Morphologically close to eliminate small notches or gaps
            neck_close_k = max(1, int(round(0.02 * face_w)))
            neck_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (neck_close_k * 2 + 1, neck_close_k * 2 + 1)
            )
            neck_mask = cv2.morphologyEx(neck_candidate, cv2.MORPH_CLOSE, neck_kernel)

        # ---- 6. Combine: metric face mask ∪ forehead ∪ neck ----
        edit_mask = cv2.bitwise_or(face_mask, forehead_skin)
        edit_mask = cv2.bitwise_or(edit_mask, neck_mask)
        # Remove any remaining hair pixels
        edit_mask[hair_dilated > 0] = 0

        # ---- 7. Carve facial features (0d tight eyes, 3a brow hairs, lips separate) ----
        def carve_edit_feature(indices: List[int], dilation_px: int = 0) -> None:
            feature_pts = landmarks[indices].astype(np.int32)
            hull = cv2.convexHull(feature_pts)
            ex_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(ex_mask, hull, 255)
            if dilation_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
                )
                ex_mask = cv2.dilate(ex_mask, kernel)
            edit_mask[ex_mask > 0] = 0

        # 0d: Exclude only visible eye opening (dilation 0) — no light halo around eyelids
        carve_edit_feature(self.left_eye_indices, dilation_px=0)
        carve_edit_feature(self.right_eye_indices, dilation_px=0)

        # 3a: Exclude only brow hairs; skin around the brows receives the full shift
        edit_mask[brow_hair_mask > 0] = 0

        # 0c: Extract lip region and carve from edit_mask (skin_tone.py blends at 50 %)
        lip_pts = landmarks[self.lips_indices].astype(np.int32)
        lip_hull = cv2.convexHull(lip_pts)
        lip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(lip_mask, lip_hull, 255)
        carve_edit_feature(self.lips_indices, dilation_px=0)

        # ---- 8. Feather the edit mask boundary (0.5 % of image width sigma) ----
        # This removes visible seams between edited and unedited skin areas.
        # NOTE: the feathering is intentionally NOT applied to the underlying binary mask
        # that is returned; the caller (skin_tone.py) performs Gaussian feathering itself.
        # We do one final morphological pass to ensure the mask is a single solid region.
        close_k2 = max(1, int(round(0.02 * face_w)))
        ck2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k2 * 2 + 1, close_k2 * 2 + 1))
        edit_mask = cv2.morphologyEx(edit_mask, cv2.MORPH_CLOSE, ck2)

        if return_lip_mask:
            return edit_mask.astype(np.uint8), lip_mask.astype(np.uint8)
        return edit_mask.astype(np.uint8)

    def close(self) -> None:
        """Release underlying MediaPipe native resources."""
        if hasattr(self, "face_mesh") and self.face_mesh is not None:
            self.face_mesh.close()
        if hasattr(self, "segmenter") and self.segmenter is not None:
            self.segmenter.close()


# Module-level cached instance for simple function calls
_DEFAULT_DETECTOR: Optional[FaceMeshDetector] = None


def get_default_detector() -> FaceMeshDetector:
    """Retrieve or create the singleton FaceMeshDetector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = FaceMeshDetector()
    return _DEFAULT_DETECTOR


def get_face_skin_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """Convenience wrapper to generate face skin mask using default detector.

    Args:
        image: RGB image (H, W, 3).

    Returns:
        Binary mask (H, W) uint8 (0/255) or None.
    """
    detector = get_default_detector()
    return detector.generate_skin_mask(image)


def get_skin_edit_mask(
    image: np.ndarray,
    forehead_ycrcb_dist_threshold: float = 22.0,
) -> Optional[np.ndarray]:
    """Convenience wrapper to generate skin edit mask (face skin + forehead + body skin, excluding orifices & hair).

    Args:
        image: RGB image (H, W, 3).
        forehead_ycrcb_dist_threshold: Passed to generate_skin_edit_mask.

    Returns:
        Binary mask (H, W) uint8 (0/255) or None.
    """
    detector = get_default_detector()
    return detector.generate_skin_edit_mask(
        image, forehead_ycrcb_dist_threshold=forehead_ycrcb_dist_threshold
    )


def visualize_face_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    landmarks: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    overlay_color: Tuple[int, int, int] = (0, 230, 115),  # Fresh medical emerald green
    alpha: float = 0.35,
) -> np.ndarray:
    """Generate and optionally save a clean semi-transparent overlay showing the skin mask without landmark dots.

    Args:
        image: Original RGB image (H, W, 3).
        mask: Optional binary skin mask (H, W). If None, will be computed.
        landmarks: Optional landmarks (N, 2). Kept for signature compatibility.
        output_path: Optional path to save the composite image.
        overlay_color: RGB tuple for the mask tint.
        alpha: Blending transparency for the mask overlay.

    Returns:
        np.ndarray: Composite RGB visualization image without landmark dots.
    """
    composite = image.copy()
    detector = get_default_detector()

    if mask is None:
        mask = detector.generate_skin_mask(image)

    # 1. Overlay the skin mask as a clean semi-transparent layer
    if mask is not None:
        colored_mask = np.zeros_like(composite)
        colored_mask[:] = overlay_color
        # Blend only where mask is positive (255)
        skin_region = mask > 0
        composite[skin_region] = cv2.addWeighted(
            composite[skin_region],
            1.0 - alpha,
            colored_mask[skin_region],
            alpha,
            0.0,
        )
        # Draw contour border in vibrant green
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(composite, contours, -1, (0, 255, 100), thickness=2)

    # Note: Landmark dots intentionally omitted for clean clinical overlay visualization

    # 2. Save if output path is specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for OpenCV imwrite
        cv2.imwrite(str(output_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    return composite


def save_skin_edit_overlay(
    image: np.ndarray,
    output_path: Union[str, Path],
    forehead_ycrcb_dist_threshold: float = 22.0,
    overlay_color: Tuple[int, int, int] = (255, 140, 0),  # Orange to distinguish from metric mask
    alpha: float = 0.40,
) -> np.ndarray:
    """Compute and save a semi-transparent overlay of the SKIN EDIT mask.

    Orange tint (vs. green for the metric mask) makes it easy to distinguish
    which pixels will be colour-shifted by the skin-tone variations.

    Args:
        image: Original RGB image (H, W, 3).
        output_path: Where to save the PNG.
        forehead_ycrcb_dist_threshold: Passed through to generate_skin_edit_mask.
        overlay_color: RGB tuple for the edit-mask tint.
        alpha: Blending transparency.

    Returns:
        np.ndarray: Composite RGB visualization.
    """
    detector = get_default_detector()
    edit_mask, _ = detector.generate_skin_edit_mask(
        image,
        return_lip_mask=True,
        forehead_ycrcb_dist_threshold=forehead_ycrcb_dist_threshold,
    )

    composite = image.copy()
    if edit_mask is not None:
        colored = np.zeros_like(composite)
        colored[:] = overlay_color
        region = edit_mask > 0
        composite[region] = cv2.addWeighted(
            composite[region], 1.0 - alpha, colored[region], alpha, 0.0
        )
        contours, _ = cv2.findContours(edit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(composite, contours, -1, (255, 180, 0), thickness=2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    logger.info("Skin edit mask overlay saved to %s", out)
    return composite
