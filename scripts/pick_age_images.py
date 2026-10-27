"""Script to build age_input/ with padding-1.25 full-head FairFace images for the aging pilot."""

import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    out_dir = PROJECT_ROOT / "age_input"
    zip_path = PROJECT_ROOT / "age_input.zip"
    preview_path = PROJECT_ROOT / "age_input_preview.png"
    demo_img_path = PROJECT_ROOT / "data" / "input" / "person_1.jpg"

    # Step 1: Clean old files
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        zip_path.unlink()

    # Step 2: Stream FairFace 1.25
    dataset_id = "HuggingFaceM4/FairFace"
    dataset_config = "1.25"
    print(f"Connecting to {dataset_id} (config={dataset_config}, streaming=True)...")

    ds = load_dataset(dataset_id, dataset_config, split="train", streaming=True)
    race_feat = ds.features["race"]
    age_feat = ds.features["age"]

    # Target: 3 images per race group (7 groups)
    race_names = race_feat.names
    target_per_race = 3
    counts: dict[str, int] = {r: 0 for r in race_names}
    collected: list[dict] = []

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as detector:
        for idx, sample in enumerate(ds):
            raw_race = sample["race"]
            race_str = race_feat.int2str(raw_race) if hasattr(race_feat, "int2str") else str(raw_race)

            if counts[race_str] >= target_per_race:
                # Check if all completed
                if all(c >= target_per_race for c in counts.values()):
                    break
                continue

            raw_age = sample["age"]
            age_str = age_feat.int2str(raw_age) if hasattr(age_feat, "int2str") else str(raw_age)
            pil_img = sample["image"].convert("RGB")

            # Check MediaPipe face detection
            img_arr = np.array(pil_img)
            res = detector.process(img_arr)
            if not res.detections:
                continue

            counts[race_str] += 1
            cur_idx = counts[race_str]
            # Standard filename: <race>_<index>.png
            slug = race_str.lower().replace(" ", "_")
            filename = f"{slug}_{cur_idx}.png"

            # Upscale 2x with Lanczos
            upscaled = pil_img.resize((pil_img.width * 2, pil_img.height * 2), Image.Resampling.LANCZOS)
            save_path = out_dir / filename
            upscaled.save(save_path, "PNG")

            collected.append({
                "filename": filename,
                "race": race_str,
                "age": age_str,
                "path": save_path,
                "size": upscaled.size,
            })

            print(f"[{len(collected)}/21] Saved {filename} (race={race_str}, age={age_str}, size={upscaled.size})")

            if all(c >= target_per_race for c in counts.values()):
                break

    # Step 3: Copy demo image
    if demo_img_path.exists():
        demo_dst = out_dir / demo_img_path.name
        shutil.copy2(demo_img_path, demo_dst)
        collected.append({
            "filename": demo_img_path.name,
            "race": "demo",
            "age": "unknown",
            "path": demo_dst,
            "size": Image.open(demo_dst).size,
        })
        print(f"Copied {demo_img_path.name} to {out_dir}")

    # Step 4: Write labels.csv
    labels_df = pd.DataFrame([{
        "filename": item["filename"],
        "race": item["race"],
        "age": item["age"]
    } for item in collected])
    labels_df.to_csv(out_dir / "labels.csv", index=False)
    print(f"Wrote {len(labels_df)} entries to {out_dir / 'labels.csv'}")

    # Step 5: Create contact sheet preview (age_input_preview.png)
    # 22 images: arrange in grid of 6 cols x 4 rows
    cols = 6
    rows = (len(collected) + cols - 1) // cols
    cell_w, cell_h = 240, 240
    text_h = 28
    grid_w = cols * cell_w
    grid_h = rows * (cell_h + text_h)

    preview_img = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(preview_img)
    font = ImageFont.load_default()

    for i, item in enumerate(collected):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = row * (cell_h + text_h)

        img = Image.open(item["path"]).convert("RGB")
        img.thumbnail((cell_w - 8, cell_h - 8), Image.Resampling.LANCZOS)
        # Center in cell area
        img_x = x + (cell_w - img.width) // 2
        img_y = y + (cell_h - img.height) // 2
        preview_img.paste(img, (img_x, img_y))

        # Text label
        label_text = f"{item['filename']} ({item['race']})"
        draw.rectangle([x, y + cell_h, x + cell_w, y + cell_h + text_h], fill=(45, 45, 45))
        draw.text((x + 6, y + cell_h + 6), label_text, fill=(240, 240, 240), font=font)

    preview_img.save(preview_path, "PNG")
    print(f"Created preview contact sheet: {preview_path} ({preview_img.size})")

    # Step 6: Zip folder to age_input.zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.iterdir():
            if f.is_file():
                zf.write(f, arcname=f.name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        total_in_zip = len(zf.namelist())
    print(f"Created {zip_path.name} with {total_in_zip} items.")


if __name__ == "__main__":
    main()
