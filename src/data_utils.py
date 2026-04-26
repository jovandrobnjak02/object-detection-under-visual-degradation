"""Utilities for loading, filtering, and converting BDD100K data.

BDD100K label format (per-image JSON):
    {
        "attributes": {"weather": "clear", "timeofday": "daytime", ...},
        "frames": [{"objects": [{"category": "car", "box2d": {...}}, ...]}]
    }

Directory layout expected on disk:
    data/
      images/train/  images/val/  images/test/
      labels/train/  labels/val/  labels/test/   <- one .json per image

Output layout produced by :func:`create_splits`:
    data/
      clear_day/train/images/  clear_day/train/labels/
      clear_day/val/images/    clear_day/val/labels/
      rainy/images/            rainy/labels/
      snowy/images/            snowy/labels/
      night/images/            night/labels/
      overcast/images/         overcast/labels/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

# Canonical BDD100K detection categories (index = YOLO class id)
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

# Condition filter functions keyed by the split name used in output paths
CONDITION_FILTERS: dict[str, Callable[[dict], bool]] = {
    "clear_day": lambda a: a.get("weather") == "clear" and a.get("timeofday") == "daytime",
    "rainy":     lambda a: a.get("weather") == "rainy",
    "snowy":     lambda a: a.get("weather") == "snowy",
    "night":     lambda a: a.get("timeofday") == "night",
    "overcast":  lambda a: a.get("weather") == "overcast",
}


def _parse_label(json_path: Path) -> tuple[list[str], dict]:
    """Read one per-image BDD100K JSON and return (yolo_lines, attributes).

    Args:
        json_path: Path to a single per-image label JSON file.

    Returns:
        A tuple of:
        - list of YOLO-format strings ``"<cls> <cx> <cy> <w> <h>"``
        - the top-level ``attributes`` dict (weather, timeofday, scene)
    """
    with open(json_path) as f:
        data = json.load(f)

    attrs = data.get("attributes", {})
    lines: list[str] = []

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            cat = obj.get("category")
            if cat not in CATEGORIES:
                continue
            box = obj.get("box2d")
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

    return lines, attrs


def convert_to_yolo(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_dir: Path,
    filter_fn: Callable[[dict], bool],
) -> int:
    """Convert and filter one BDD100K split to YOLO format.

    Reads per-image JSON labels from ``src_lbl_dir``, applies ``filter_fn``
    on the image attributes, and writes matching images + ``.txt`` labels into
    ``dst_dir/images/`` and ``dst_dir/labels/``.

    Args:
        src_img_dir: Directory containing source ``.jpg`` images.
        src_lbl_dir: Directory containing per-image ``.json`` label files.
        dst_dir: Destination directory (will be created if absent).
        filter_fn: Returns ``True`` for images whose attributes should be kept.

    Returns:
        Number of images written.
    """
    dst_img = dst_dir / "images"
    dst_lbl = dst_dir / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_file in sorted(src_lbl_dir.glob("*.json")):
        stem = json_file.stem
        img_file = src_img_dir / f"{stem}.jpg"
        if not img_file.exists():
            continue

        lines, attrs = _parse_label(json_file)
        if not filter_fn(attrs):
            continue

        dst_img_path = dst_img / f"{stem}.jpg"
        if not dst_img_path.exists():
            shutil.copy2(img_file, dst_img_path)

        (dst_lbl / f"{stem}.txt").write_text("\n".join(lines))
        count += 1

    return count


def convert_to_coco(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_dir: Path,
    filter_fn: Callable[[dict], bool],
) -> int:
    """Convert and filter one BDD100K split to COCO JSON format.

    Writes images into ``dst_dir/`` and creates ``dst_dir/annotations.json``
    in COCO object detection format. Used by RF-DETR which requires COCO input.

    Args:
        src_img_dir: Directory containing source ``.jpg`` images.
        src_lbl_dir: Directory containing per-image ``.json`` label files.
        dst_dir: Destination directory (will be created if absent).
        filter_fn: Returns ``True`` for images whose attributes should be kept.

    Returns:
        Number of images written.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": v, "name": k} for k, v in sorted(CATEGORIES.items(), key=lambda x: x[1])]
    coco: dict = {
        "info": {"description": "BDD100K subset — COCO format"},
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    img_id = ann_id = count = 0
    for json_file in sorted(src_lbl_dir.glob("*.json")):
        stem = json_file.stem
        img_file = src_img_dir / f"{stem}.jpg"
        if not img_file.exists():
            continue

        lines, attrs = _parse_label(json_file)
        if not filter_fn(attrs):
            continue

        dst_img_path = dst_dir / f"{stem}.jpg"
        if not dst_img_path.exists():
            shutil.copy2(img_file, dst_img_path)

        coco["images"].append({
            "id": img_id, "file_name": f"{stem}.jpg",
            "width": IMG_W, "height": IMG_H,
        })

        # Re-parse to get absolute pixel coords for COCO bbox
        with open(json_file) as f:
            raw = json.load(f)
        for frame in raw.get("frames", []):
            for obj in frame.get("objects", []):
                cat = obj.get("category")
                if cat not in CATEGORIES:
                    continue
                box = obj.get("box2d")
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

    Expects the standard BDD100K directory layout under ``data_root``::

        data_root/images/train/   data_root/labels/train/
        data_root/images/val/     data_root/labels/val/
        data_root/images/test/    data_root/labels/test/

    Training and validation use **clear + daytime** only.
    Test splits cover: rainy, snowy, night, overcast.

    Args:
        data_root: Root of the raw BDD100K dataset directory.
        yolo_output: Destination for YOLO-format splits. Pass ``None`` to skip.
        coco_output: Destination for COCO-format splits. Pass ``None`` to skip.
    """
    images = data_root / "images"
    labels = data_root / "labels"

    jobs: list[tuple[Path, Path, str, str]] = [
        # (img_src, lbl_src, split_output_name, condition_key)
        (images / "train", labels / "train", "clear_day/train", "clear_day"),
        (images / "val",   labels / "val",   "clear_day/val",   "clear_day"),
        (images / "test",  labels / "test",  "rainy",           "rainy"),
        (images / "test",  labels / "test",  "snowy",           "snowy"),
        (images / "test",  labels / "test",  "night",           "night"),
        (images / "test",  labels / "test",  "overcast",        "overcast"),
    ]

    for img_src, lbl_src, split_name, condition_key in jobs:
        filter_fn = CONDITION_FILTERS[condition_key]
        if yolo_output:
            n = convert_to_yolo(img_src, lbl_src, yolo_output / split_name, filter_fn)
            print(f"[YOLO/{split_name}] {n} images")
        if coco_output:
            n = convert_to_coco(img_src, lbl_src, coco_output / split_name, filter_fn)
            print(f"[COCO/{split_name}] {n} images")