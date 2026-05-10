"""Run this script locally before uploading to Google Drive.

Converts the raw BDD100K dataset into filtered YOLO and COCO splits and
auto-generates configs/dataset.yaml and configs/conditions.json from the
discovered (weather, timeofday) combinations.

Usage:
    python prepare_data.py
"""

import json
from pathlib import Path
from src.data_utils import create_splits, generate_dataset_yaml

DATA_ROOT   = Path("data")           # raw BDD100K: data/100k/ and data/labels/
OUT_ROOT    = Path("data_prepared")  # output written here
CONFIGS_DIR = Path("configs")

if __name__ == "__main__":
    print("BDD100K data root:", DATA_ROOT.resolve())
    print("Output root:      ", OUT_ROOT.resolve())
    print()

    if not (DATA_ROOT / "labels" / "bdd100k_labels_images_train.json").exists():
        raise FileNotFoundError(
            f"Could not find label JSON under {DATA_ROOT / 'labels'}. "
            "Make sure the script is run from the project root."
        )

    adverse_conditions = create_splits(
        data_root=DATA_ROOT,
        yolo_output=OUT_ROOT / "yolo",
        coco_output=OUT_ROOT / "coco",
    )

    # Write dataset.yaml with all discovered test splits
    generate_dataset_yaml(
        yolo_root="/content/data_prepared/yolo",
        adverse_conditions=adverse_conditions,
        output_path=CONFIGS_DIR / "dataset.yaml",
    )
    print(f"\nWrote configs/dataset.yaml with {len(adverse_conditions)} adverse splits.")

    # Write conditions.json so notebooks can load the split list without hardcoding
    conditions = {"baseline": "clear_day", "adverse": adverse_conditions}
    (CONFIGS_DIR / "conditions.json").write_text(json.dumps(conditions, indent=2))
    print("Wrote configs/conditions.json.")

    print("\nDone. Upload data_prepared/ and configs/ to Google Drive.")