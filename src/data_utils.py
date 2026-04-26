"""Utilities for loading, filtering, and converting BDD100K data.

Actual BDD100K label format (two big JSON files, one per split):
    [
        {
            "name": "b1c66a42-6f7d68ca.jpg",
            "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
            "labels": [
                {"category": "car", "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}},
                ...
            ]
        },
        ...
    ]

Expected raw layout on disk:
    data/
      100k/train/    100k/val/    100k/test/     <- images (test has no labels)
      labels/
        bdd100k_labels_images_train.json
        bdd100k_labels_images_val.json

Output layout produced by :func:`create_splits`:
    <output_root>/
      clear_day/train/images/   clear_day/train/labels/
      clear_day/val/images/     clear_day/val/labels/
      rainy/images/             rainy/labels/
      snowy/images/             snowy/labels/
      night/images/             night/labels/
      overcast/images/          overcast/labels/
      partly_cloudy/images/     partly_cloudy/labels/
      dawn_dusk/images/         dawn_dusk/labels/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

# Canonical BDD100K detection categories (index = YOLO class id).
# "lane" and "drivable area" exist in the raw labels but are segmentation
# tasks — they are excluded here.
CATEGORIES: dict[str, int] = {
    "car":           0,
    "person":        1,
    "traffic sign":  2,
    "traffic light": 3,
    "truck":         4,
    "bus":           5,
    "bike":          6,
    "rider":         7,
    "motor":         8,
    "train":         9,
}
CLASS_NAMES: list[str] = [k for k, _ in sorted(CATEGORIES.items(), key=lambda x: x[1])]

IMG_W, IMG_H = 1280, 720

# Condition filter functions keyed by the output split name.
# All adverse splits are drawn from the val set (test set has no public labels).
CONDITION_FILTERS: dict[str, Callable[[dict], bool]] = {
    "clear_day":    lambda a: a.get("weather") == "clear" and a.get("timeofday") == "daytime",
    "rainy":        lambda a: a.get("weather") == "rainy",
    "snowy":        lambda a: a.get("weather") == "snowy",
    "night":        lambda a: a.get("timeofday") == "night",
    "overcast":     lambda a: a.get("weather") == "overcast",
    "partly_cloudy": lambda a: a.get("weather") == "partly cloudy",
    "dawn_dusk":    lambda a: a.get("timeofday") == "dawn/dusk",
}


def _entry_to_yolo_lines(entry: dict) -> list[str]:
    """Convert one BDD100K JSON entry to YOLO label lines.

    Args:
        entry: One element of the top-level JSON list, containing
               ``"name"``, ``"attributes"``, and ``"labels"`` keys.

    Returns:
        List of YOLO-format strings ``"<cls> <cx> <cy> <w> <h>"``.
        Empty list if the entry has no valid detection boxes.
    """
    lines: list[str] = []
    for lbl in (entry.get("labels") or []):
        cat = lbl.get("category")
        if cat not in CATEGORIES:
            continue
        box = lbl.get("box2d")
        if box is None:
            continue

        x1 = max(0.0, min(float(box["x1"]), IMG_W))
        y1 = max(0.0, min(float(box["y1"]), IMG_H))
        x2 = max(0.0, min(float(box["x2"]), IMG_W))
        y2 = max(0.0, min(float(box["y2"]), IMG_H))

        w = (x2 - x1) / IMG_W
        h = (y2 - y1) / IMG_H
        if w <= 0 or h <= 0:
            continue

        cx = (x1 + x2) / 2 / IMG_W
        cy = (y1 + y2) / 2 / IMG_H
        lines.append(f"{CATEGORIES[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_to_yolo(
    labels_json: Path,
    images_dir: Path,
    dst_dir: Path,
    filter_fn: Callable[[dict], bool],
) -> int:
    """Convert and filter one BDD100K split to YOLO format.

    Reads the big label JSON, applies ``filter_fn`` on each entry's
    ``attributes``, and writes matching images + ``.txt`` labels into
    ``dst_dir/images/`` and ``dst_dir/labels/``.

    Args:
        labels_json: Path to ``bdd100k_labels_images_{train|val}.json``.
        images_dir: Directory containing the source ``.jpg`` images
                    (e.g. ``data/100k/train/``).
        dst_dir: Destination root directory for this split.
        filter_fn: Returns ``True`` for entries whose attributes match.

    Returns:
        Number of images written.
    """
    dst_img = dst_dir / "images"
    dst_lbl = dst_dir / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    with open(labels_json) as f:
        entries: list[dict] = json.load(f)

    count = 0
    for entry in entries:
        if not filter_fn(entry.get("attributes", {})):
            continue

        name: str = entry["name"]
        src_img = images_dir / name
        if not src_img.exists():
            continue

        lines = _entry_to_yolo_lines(entry)

        dst_img_path = dst_img / name
        if not dst_img_path.exists():
            shutil.copy2(src_img, dst_img_path)

        (dst_lbl / (Path(name).stem + ".txt")).write_text("\n".join(lines))
        count += 1

    return count


def convert_to_coco(
    labels_json: Path,
    images_dir: Path,
    dst_dir: Path,
    filter_fn: Callable[[dict], bool],
) -> int:
    """Convert and filter one BDD100K split to COCO JSON format.

    Writes images into ``dst_dir/`` and creates ``dst_dir/annotations.json``.
    Used by RF-DETR which requires COCO input.

    Args:
        labels_json: Path to ``bdd100k_labels_images_{train|val}.json``.
        images_dir: Directory containing the source ``.jpg`` images.
        dst_dir: Destination root directory for this split.
        filter_fn: Returns ``True`` for entries whose attributes match.

    Returns:
        Number of images written.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": v, "name": k}
        for k, v in sorted(CATEGORIES.items(), key=lambda x: x[1])
    ]
    coco: dict = {
        "info": {"description": "BDD100K subset — COCO format"},
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    with open(labels_json) as f:
        entries: list[dict] = json.load(f)

    img_id = ann_id = count = 0
    for entry in entries:
        if not filter_fn(entry.get("attributes", {})):
            continue

        name: str = entry["name"]
        src_img = images_dir / name
        if not src_img.exists():
            continue

        dst_img_path = dst_dir / name
        if not dst_img_path.exists():
            shutil.copy2(src_img, dst_img_path)

        coco["images"].append({
            "id": img_id, "file_name": name,
            "width": IMG_W, "height": IMG_H,
        })

        for lbl in (entry.get("labels") or []):
            cat = lbl.get("category")
            if cat not in CATEGORIES:
                continue
            box = lbl.get("box2d")
            if box is None:
                continue
            x1 = max(0.0, min(float(box["x1"]), IMG_W))
            y1 = max(0.0, min(float(box["y1"]), IMG_H))
            x2 = max(0.0, min(float(box["x2"]), IMG_W))
            y2 = max(0.0, min(float(box["y2"]), IMG_H))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": CATEGORIES[cat],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1
        count += 1

    (dst_dir / "annotations.json").write_text(json.dumps(coco))
    return count


def create_splits(
    data_root: Path,
    yolo_output: Path | None = None,
    coco_output: Path | None = None,
) -> None:
    """Create all train/val/test splits in YOLO and/or COCO formats.

    Expects the standard BDD100K layout under ``data_root``::

        data_root/
          100k/train/   100k/val/
          labels/
            bdd100k_labels_images_train.json
            bdd100k_labels_images_val.json

    The test set (``100k/test/``) has no public labels and is not used.
    Adverse condition splits are drawn from the val set.

    Splits produced:
        clear_day/train  — clear+daytime from train JSON     (~12 k images)
        clear_day/val    — clear+daytime from val JSON       (~3.5 k images)
        rainy            — rainy weather from val JSON       (~740 images)
        snowy            — snowy weather from val JSON       (~770 images)
        night            — night timeofday from val JSON     (~3.9 k images)
        overcast         — overcast weather from val JSON    (~1.2 k images)
        partly_cloudy    — partly cloudy weather from val    (~740 images)
        dawn_dusk        — dawn/dusk timeofday from val      (~780 images)

    Args:
        data_root: Root of the raw BDD100K dataset directory.
        yolo_output: Destination for YOLO-format splits. Pass ``None`` to skip.
        coco_output: Destination for COCO-format splits. Pass ``None`` to skip.
    """
    train_json = data_root / "labels" / "bdd100k_labels_images_train.json"
    val_json   = data_root / "labels" / "bdd100k_labels_images_val.json"
    train_imgs = data_root / "100k" / "train"
    val_imgs   = data_root / "100k" / "val"

    jobs: list[tuple[Path, Path, str, str]] = [
        # (label_json, img_dir, output_split_name, condition_key)
        (train_json, train_imgs, "clear_day/train", "clear_day"),
        (val_json,   val_imgs,   "clear_day/val",   "clear_day"),
        (val_json,   val_imgs,   "rainy",           "rainy"),
        (val_json,   val_imgs,   "snowy",           "snowy"),
        (val_json,   val_imgs,   "night",           "night"),
        (val_json,   val_imgs,   "overcast",        "overcast"),
        (val_json,   val_imgs,   "partly_cloudy",   "partly_cloudy"),
        (val_json,   val_imgs,   "dawn_dusk",       "dawn_dusk"),
    ]

    for lbl_json, img_dir, split_name, condition_key in jobs:
        filter_fn = CONDITION_FILTERS[condition_key]
        if yolo_output:
            n = convert_to_yolo(lbl_json, img_dir, yolo_output / split_name, filter_fn)
            print(f"[YOLO/{split_name}] {n} images")
        if coco_output:
            n = convert_to_coco(lbl_json, img_dir, coco_output / split_name, filter_fn)
            print(f"[COCO/{split_name}] {n} images")