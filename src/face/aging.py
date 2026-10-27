"""FRAN-style ONNX Face Re-aging Inference Engine for CPU execution."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import onnxruntime as ort
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "reaging"
ONNX_MODEL_PATH = MODEL_DIR / "unet.onnx"
MASK_512_PATH = MODEL_DIR / "mask512.jpg"
MASK_1024_PATH = MODEL_DIR / "mask1024.jpg"

_CACHED_MODEL: Optional["OnnxUNet"] = None


class OnnxUNet:
    """Wraps the ONNX Runtime session for FRAN U-Net execution."""

    def __init__(self, onnx_path: Path):
        if not onnx_path.exists():
            try:
                from scripts.download_reaging_model import ensure_reaging_assets
                ensure_reaging_assets()
            except Exception as e:
                raise FileNotFoundError(
                    f"ONNX model file not found at {onnx_path}. "
                    f"Run 'python scripts/download_reaging_model.py' to download. (Error: {e})"
                )
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on a (B, 5, 512, 512) tensor."""
        x_np = x.cpu().numpy().astype(np.float32)
        out_np = self.session.run([self.output_name], {self.input_name: x_np})[0]
        return torch.from_numpy(out_np)


def get_reaging_model() -> OnnxUNet:
    """Return a cached singleton instance of the OnnxUNet session."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = OnnxUNet(ONNX_MODEL_PATH)
    return _CACHED_MODEL


def _load_mask(path: Path, size: int) -> torch.Tensor:
    """Load blend mask or fall back to an all-ones tensor."""
    if path.exists():
        try:
            return torch.from_numpy(np.array(Image.open(path).convert("L"))).float() / 255.0
        except Exception:
            pass
    return torch.ones((size, size), dtype=torch.float32)


def detect_face_bbox(image_rgb: np.ndarray) -> Tuple[int, int, int, int]:
    """Detect face bounding box using MediaPipe Face Detection.

    Returns:
        (top, right, bottom, left) in pixel coordinates.
    """
    H, W = image_rgb.shape[:2]
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as fd:
        res = fd.process(image_rgb)
        if res.detections:
            b = res.detections[0].location_data.relative_bounding_box
            top = max(0, int(b.ymin * H))
            left = max(0, int(b.xmin * W))
            bottom = min(H, int((b.ymin + b.height) * H))
            right = min(W, int((b.xmin + b.width) * W))
            return (top, right, bottom, left)
    return (0, W, H, 0)


def sliding_window_onnx(
    model: OnnxUNet,
    input_tensor: torch.Tensor,
    window_size: int = 512,
    stride: int = 256,
) -> torch.Tensor:
    """Apply the UNet via a sliding window with smooth mask blending."""
    n, c, h, w = input_tensor.shape
    mask = _load_mask(MASK_1024_PATH, max(h, w))[:h, :w]
    small_mask = _load_mask(MASK_512_PATH, window_size)

    output_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32)
    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32)

    add = 2 if window_size % stride != 0 else 1

    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y : y + window_size, x : x + window_size]
            with torch.no_grad():
                out = model(window)

            output_tensor[:, :, y : y + window_size, x : x + window_size] += out * small_mask
            count_tensor[:, :, y : y + window_size, x : x + window_size] += small_mask

    count_tensor = torch.clamp(count_tensor, min=1.0)
    output_tensor /= count_tensor
    output_tensor *= mask
    return output_tensor


def reage_face(
    image: np.ndarray,
    source_age: int = 25,
    target_age: int = 55,
    window_size: int = 512,
    stride: int = 256,
    model: Optional[OnnxUNet] = None,
) -> np.ndarray:
    """Re-age a face image from source_age to target_age using FRAN U-Net.

    Args:
        image: RGB input image as uint8 NumPy array (H, W, 3).
        source_age: Estimated or approximate source age in years (10–90).
        target_age: Target age in years (10–95).
        window_size: Sliding window size (default 512).
        stride: Sliding window stride (default 256).
        model: Optional pre-loaded OnnxUNet instance.

    Returns:
        Aged RGB image as uint8 NumPy array (H, W, 3).
    """
    if model is None:
        model = get_reaging_model()

    fl = detect_face_bbox(image)

    # Margins matching FRAN reference implementation
    margin_y_t = int((fl[2] - fl[0]) * 0.63 * 0.85)
    margin_y_b = int((fl[2] - fl[0]) * 0.37 * 0.85)
    margin_x = int((fl[1] - fl[3]) // (2 / 0.85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b

    l_y = max(fl[0] - margin_y_t, 0)
    r_y = min(fl[2] + margin_y_b, image.shape[0])
    l_x = max(fl[3] - margin_x, 0)
    r_x = min(fl[1] + margin_x, image.shape[1])

    cropped = image[l_y:r_y, l_x:r_x]
    orig_crop_size = cropped.shape[:2]
    input_size = (1024, 1024)

    img_tensor = T.ToTensor()(Image.fromarray(image))
    crop_tensor = T.ToTensor()(Image.fromarray(cropped))
    crop_resized = T.Resize(input_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)(crop_tensor)

    # 5-channel conditioning input: [R, G, B, source_age/100, target_age/100]
    src_ch = torch.full_like(crop_resized[:1], source_age / 100.0)
    tgt_ch = torch.full_like(crop_resized[:1], target_age / 100.0)
    inp = torch.cat([crop_resized, src_ch, tgt_ch], dim=0).unsqueeze(0)

    # Predict residual delta map
    aged_crop = sliding_window_onnx(model, inp, window_size, stride)

    # Resize residual back to original crop bounds
    aged_crop_resized = T.Resize(orig_crop_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)(aged_crop)

    # Additive blend onto original face region and clamp
    img_tensor[:, l_y:r_y, l_x:r_x] += aged_crop_resized.squeeze(0)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    out_pil = T.functional.to_pil_image(img_tensor)
    return np.array(out_pil, dtype=np.uint8)
