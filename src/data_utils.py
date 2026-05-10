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
      <weather>_<timeofday>/images/   <weather>_<timeofday>/labels/   (one per discovered combo)
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable

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

# Minimum images required for a split to be included.
MIN_SPLIT_SIZE = 50


def _split_name(weather: str, timeofday: str) -> str:
    w = weather.replace(" ", "_").replace("/", "_")
    t = timeofday.replace(" ", "_").replace("/", "_")
    return f"{w}_{t}"


def _make_filter(weather: str, timeofday: str) -> Callable[[dict], bool]:
    return lambda a, w=weather, t=timeofday: a.get("weather") == w and a.get("timeofday") == t


def _build_image_index(images_dir: Path) -> dict[str, Path]:
    """Return a filename→path mapping, searching recursively."""
    return {p.name: p for p in images_dir.rglob("*.jpg")}


def discover_adverse_splits(
    val_json: Path,
    min_size: int = MIN_SPLIT_SIZE,
) -> dict[str, Callable[[dict], bool]]:
    """Scan the val JSON and return a filter for every (weather, timeofday) combo
    that has at least ``min_size`` images, excluding the clear-day baseline and
    entries with undefined attributes.

    All adverse splits are drawn from the val set (test set has no public labels).
    """
    with open(val_json) as f:
        entries: list[dict] = json.load(f)

    counts: Counter = Counter()
    for entry in entries:
        a = entry.get("attributes", {})
        w = a.get("weather", "undefined")
        t = a.get("timeofday", "undefined")
        if "undefined" in (w, t):
            continue
        if w == "clear" and t == "daytime":
            continue
        counts[(w, t)] += 1

    return {
        _split_name(w, t): _make_filter(w, t)
        for (w, t), count in sorted(counts.items())
        if count >= min_size
    }


def generate_dataset_yaml(
    yolo_root: str,
    adverse_conditions: list[str],
    output_path: Path,
) -> None:
    """Write a YOLO dataset.yaml covering all splits."""
    lines = [
        f"path: {yolo_root}",
        "",
        "train: clear_day/train/images",
        "val:   clear_day/val/images",
        "",
    ]
    for cond in adverse_conditions:
        lines.append(f"test_{cond}: {cond}/images")
    lines += [
        "",
        f"nc: {len(CATEGORIES)}",
        "",
        "names:",
    ]
    for name in CLASS_NAMES:
        lines.append(f"  {CATEGORIES[name]}: {name}")

    output_path.write_text("\n".join(lines) + "\n")


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

    img_index = _build_image_index(images_dir)
    count = 0
    for entry in entries:
        if not filter_fn(entry.get("attributes", {})):
            continue

        name: str = entry["name"]
        src_img = img_index.get(name)
        if src_img is None:
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

    ann_path = dst_dir / "_annotations.coco.json"
    if ann_path.exists():
        existing = json.loads(ann_path.read_text())
        coco["images"] = existing["images"]
        coco["annotations"] = existing["annotations"]
    img_id = len(coco["images"])
    ann_id = len(coco["annotations"])

    img_index = _build_image_index(images_dir)
    count = 0
    for entry in entries:
        if not filter_fn(entry.get("attributes", {})):
            continue

        name: str = entry["name"]
        src_img = img_index.get(name)
        if src_img is None:
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

    (dst_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    return count


def create_splits(
    data_root: Path,
    yolo_output: Path | None = None,
    coco_output: Path | None = None,
    val_only_splits: frozenset[str] = frozenset({"clear_night"}),
) -> list[str]:
    """Create all train/val/test splits in YOLO and/or COCO formats.

    Expects the standard BDD100K layout under ``data_root``::

        data_root/
          100k/train/   100k/val/
          labels/
            bdd100k_labels_images_train.json
            bdd100k_labels_images_val.json

    The test set (``100k/test/``) has no public labels and is not used.
    Adverse condition splits are discovered automatically from the val JSON —
    every (weather, timeofday) combination with >= MIN_SPLIT_SIZE images is
    included, excluding the clear-day baseline.

    Splits listed in ``val_only_splits`` are built from the val JSON only.
    All other adverse splits merge val + train (models only train on
    clear+daytime, so non-clear_day train images are unseen at eval time).

    Args:
        data_root: Root of the raw BDD100K dataset directory.
        yolo_output: Destination for YOLO-format splits. Pass ``None`` to skip.
        coco_output: Destination for COCO-format splits. Pass ``None`` to skip.
        val_only_splits: Adverse split names to build from val JSON only.

    Returns:
        List of discovered adverse condition split names.
    """
    train_json = data_root / "labels" / "bdd100k_labels_images_train.json"
    val_json   = data_root / "labels" / "bdd100k_labels_images_val.json"
    train_imgs = data_root / "100k" / "train"
    val_imgs   = data_root / "100k" / "val"

    clear_filter = _make_filter("clear", "daytime")

    # clear_day train and val — COCO val is "valid/" per RF-DETR's expected format
    for yolo_split, coco_split, lbl_json, img_dir in [
        ("clear_day/train", "clear_day/train", train_json, train_imgs),
        ("clear_day/val",   "clear_day/valid", val_json,   val_imgs),
    ]:
        if yolo_output:
            n = convert_to_yolo(lbl_json, img_dir, yolo_output / yolo_split, clear_filter)
            print(f"[YOLO/{yolo_split}] {n} images")
        if coco_output:
            n = convert_to_coco(lbl_json, img_dir, coco_output / coco_split, clear_filter)
            print(f"[COCO/{coco_split}] {n} images")

    adverse_splits = discover_adverse_splits(val_json)
    for split_name, filter_fn in adverse_splits.items():
        sources = [(val_json, val_imgs)]
        if split_name not in val_only_splits:
            sources.append((train_json, train_imgs))
        n_total = 0
        for lbl_json, img_dir in sources:
            if yolo_output:
                n_total += convert_to_yolo(lbl_json, img_dir, yolo_output / split_name, filter_fn)
        if yolo_output:
            print(f"[YOLO/{split_name}] {n_total} images")
        n_total = 0
        for lbl_json, img_dir in sources:
            if coco_output:
                n_total += convert_to_coco(lbl_json, img_dir, coco_output / split_name, filter_fn)
        if coco_output:
            print(f"[COCO/{split_name}] {n_total} images")

    return list(adverse_splits.keys())
