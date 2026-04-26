"""Run this script locally (Windows) before uploading to Google Drive.

Converts the raw BDD100K dataset into filtered YOLO and COCO splits,
so only the ~18k images actually needed for training/evaluation are
uploaded — instead of all 80k raw images.

Usage:
    python prepare_data.py

Output (written next to this script):
    data_prepared/
      yolo/
        clear_day/train/images/   clear_day/train/labels/   (~12 400 images)
        clear_day/val/images/     clear_day/val/labels/     (~3 500 images)
        rainy_day/images/         rainy_day/labels/         (~396 images)
        snowy_day/images/         snowy_day/labels/         (~422 images)
        night_clear/images/       night_clear/labels/       (~3 274 images)
        overcast_day/images/      overcast_day/labels/      (~1 039 images)
        partly_cloudy_day/images/ partly_cloudy_day/labels/ (~638 images)
        dawn_dusk_clear/images/   dawn_dusk_clear/labels/   (~307 images)
      coco/
        (same split names as yolo/)

Upload `data_prepared/` to Google Drive when done.
On Colab, set BDD100K_ROOT to point at the mounted Drive path.
"""

from pathlib import Path
from src.data_utils import create_splits

DATA_ROOT = Path("data")           # raw BDD100K: data/100k/ and data/labels/
OUT_ROOT  = Path("data_prepared")  # output written here

if __name__ == "__main__":
    print("BDD100K data root:", DATA_ROOT.resolve())
    print("Output root:      ", OUT_ROOT.resolve())
    print()

    if not (DATA_ROOT / "labels" / "bdd100k_labels_images_train.json").exists():
        raise FileNotFoundError(
            f"Could not find label JSON under {DATA_ROOT / 'labels'}. "
            "Make sure the script is run from the project root."
        )

    create_splits(
        data_root=DATA_ROOT,
        yolo_output=OUT_ROOT / "yolo",
        coco_output=OUT_ROOT / "coco",
    )

    print("\nDone. Verify counts above, then upload data_prepared/ to Google Drive.")
    print("Expected total: ~21 500 images across 8 splits.")