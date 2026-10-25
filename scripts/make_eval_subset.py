"""Extract a stratified evaluation subset from the FairFace archive.

Reads directly from data/eval/fairface-img-margin025-trainval.zip without extracting
the whole archive to disk. Samples ~150 images stratified equally across races,
writes them to data/eval/raw/, and saves data/eval/raw/labels.csv.
"""

from pathlib import Path
import sys
import zipfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "eval" / "fairface-img-margin025-trainval.zip"
CSV_PATH = PROJECT_ROOT / "data" / "eval" / "fairface_label_train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "raw"
LABELS_CSV_PATH = OUTPUT_DIR / "labels.csv"

AGE_30_PLUS = {"30-39", "40-49", "50-59", "60-69", "more than 70"}


def get_key(path_str: str) -> str:
    """Normalize path to last two components (e.g. 'train/123.jpg')."""
    parts = Path(str(path_str).replace("\\", "/")).parts
    return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]


def make_eval_subset() -> None:
    # 1. Skip if data/eval/raw already has at least 100 images and labels.csv
    if OUTPUT_DIR.is_dir() and LABELS_CSV_PATH.is_file():
        existing_images = [
            p for p in OUTPUT_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if len(existing_images) >= 100:
            print(f"Skipping: {OUTPUT_DIR} already contains {len(existing_images)} images and labels.csv.")
            return

    if not ZIP_PATH.is_file():
        print(f"Error: Archive not found at {ZIP_PATH}", file=sys.stderr)
        return

    if not CSV_PATH.is_file():
        print(f"Error: Label CSV not found at {CSV_PATH}", file=sys.stderr)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load labels CSV and perform stratified sampling across race groups
    df = pd.read_csv(CSV_PATH)
    races = df["race"].unique()
    num_races = len(races)
    # Target ~150 rows stratified equally (21 per race * 7 = 147 rows)
    n_per_group = max(1, round(150 / num_races))

    sampled_df = (
        df.groupby("race", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), n_per_group), random_state=42))
        .reset_index(drop=True)
    )

    # 3. Read zip and index members by last two path components
    records = []
    copied_count = 0
    race_counts = {r: 0 for r in sorted(races)}

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        zip_lookup = {get_key(n): n for n in namelist if not n.endswith("/")}

        for _, row in sampled_df.iterrows():
            key = get_key(row["file"])
            member_name = zip_lookup.get(key)
            if member_name is None:
                continue

            target_filename = Path(row["file"]).name
            target_path = OUTPUT_DIR / target_filename

            # Read directly from zip stream and write to destination
            with zf.open(member_name) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

            age_val = str(row["age"]).strip()
            label_val = "30_plus" if age_val in AGE_30_PLUS else "under_30"
            race_val = str(row["race"]).strip()

            records.append({
                "filename": target_filename,
                "label": label_val,
                "race": race_val,
                "age": age_val,
            })

            copied_count += 1
            race_counts[race_val] = race_counts.get(race_val, 0) + 1

    # 4. Handle mismatch or write labels.csv
    if copied_count == 0:
        print("Error: Zero images were copied. Debugging mismatch:")
        print("First 3 CSV file values:", df["file"].head(3).tolist())
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            print("First 3 Zip member names:", [n for n in zf.namelist() if not n.endswith("/")][:3])
        return

    labels_df = pd.DataFrame(records, columns=["filename", "label", "race", "age"])
    labels_df.to_csv(LABELS_CSV_PATH, index=False)

    # 5. Print results
    print(f"Copied {copied_count} images to {OUTPUT_DIR}")
    print("Count per race group:")
    for race, count in sorted(race_counts.items()):
        print(f"  {race}: {count}")


if __name__ == "__main__":
    make_eval_subset()
