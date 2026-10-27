"""scripts/download_reaging_model.py

Automated script to download pretrained FRAN face re-aging checkpoint from Hugging Face
and export it to ONNX (models/reaging/unet.onnx) for local CPU inference.
"""

from pathlib import Path
import sys
import urllib.request
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REAGING_DIR = PROJECT_ROOT / "models" / "reaging"
PTH_PATH = REAGING_DIR / "best_unet_model.pth"
ONNX_PATH = REAGING_DIR / "unet.onnx"
MASK_512 = REAGING_DIR / "mask512.jpg"
MASK_1024 = REAGING_DIR / "mask1024.jpg"

PTH_URL = "https://huggingface.co/timroelofs123/face_re-aging/resolve/main/best_unet_model.pth"
MASK_512_URL = "https://raw.githubusercontent.com/Glat0s/face_reaging-onnx/main/assets/mask512.jpg"
MASK_1024_URL = "https://raw.githubusercontent.com/Glat0s/face_reaging-onnx/main/assets/mask1024.jpg"


def download_file(url: str, dst: Path, name: str) -> None:
    """Download a file with progress reporting."""
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"✓ {name} already exists ({dst.stat().st_size:,} bytes).")
        return
    print(f"Downloading {name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                print(f"  Downloaded {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({downloaded / total * 100:.1f}%)")
    print(f"✓ Successfully downloaded {name}: {dst.stat().st_size:,} bytes.")


def export_pth_to_onnx(pth_path: Path, onnx_path: Path) -> None:
    """Export PyTorch checkpoint to ONNX model format."""
    import torch
    import torch.nn as nn
    try:
        import antialiased_cnns
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "antialiased_cnns"])
        import antialiased_cnns

    class DownLayer(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1),
                antialiased_cnns.BlurPool(in_channels, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )

        def forward(self, x):
            return self.layer(x)

    class UpLayer(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.blur_upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                antialiased_cnns.BlurPool(out_channels, stride=1)
            )
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )

        def forward(self, x, skip):
            x = self.blur_upsample(x)
            x = torch.cat([x, skip], dim=1)
            return self.layer(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.init_conv = nn.Sequential(
                nn.Conv2d(5, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )
            self.down1 = DownLayer(64, 128)
            self.down2 = DownLayer(128, 256)
            self.down3 = DownLayer(256, 512)
            self.down4 = DownLayer(512, 1024)
            self.up1 = UpLayer(1024, 512)
            self.up2 = UpLayer(512, 256)
            self.up3 = UpLayer(256, 128)
            self.up4 = UpLayer(128, 64)
            self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

        def forward(self, x):
            x0 = self.init_conv(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x4 = self.down4(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x, x0)
            return self.final_conv(x)

    print(f"Loading PyTorch checkpoint from {pth_path}...")
    device = torch.device("cpu")
    model = UNet().to(device)
    state_dict = torch.load(pth_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()

    print(f"Exporting to ONNX at {onnx_path}...")
    dummy_input = torch.randn(1, 5, 512, 512, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        dynamo=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"✓ ONNX export complete: {onnx_path.stat().st_size:,} bytes.")


def ensure_reaging_assets() -> Path:
    """Download required assets and build unet.onnx if not present."""
    REAGING_DIR.mkdir(parents=True, exist_ok=True)
    download_file(MASK_512_URL, MASK_512, "mask512.jpg")
    download_file(MASK_1024_URL, MASK_1024, "mask1024.jpg")

    if not ONNX_PATH.exists() or ONNX_PATH.stat().st_size < 1000000:
        download_file(PTH_URL, PTH_PATH, "best_unet_model.pth")
        export_pth_to_onnx(PTH_PATH, ONNX_PATH)
    else:
        print(f"✓ ONNX model exists: {ONNX_PATH} ({ONNX_PATH.stat().st_size:,} bytes).")

    return ONNX_PATH


if __name__ == "__main__":
    ensure_reaging_assets()
