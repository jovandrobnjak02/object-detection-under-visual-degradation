"""Evaluation utilities: mAP, per-class AP, robustness metrics, and summary DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


ModelName = str
Condition = str


def compute_map(
    results: Any,
    iou_threshold: float | None = None,
) -> dict[str, float]:
    """Extract mAP@50 and mAP@50-95 from an Ultralytics results object.

    For RF-DETR, pass a dict with keys ``"map50"`` and ``"map50_95"`` instead.

    Args:
        results: Ultralytics ``Results`` object returned by ``model.val()``,
                 or a plain dict with ``"map50"`` / ``"map50_95"`` keys.
        iou_threshold: Unused — kept for API symmetry. IoU thresholds are
                       fixed at 0.50 and 0.50:0.95 per COCO convention.

    Returns:
        Dict with keys ``"map50"`` and ``"map50_95"``.
    """
    if isinstance(results, dict):
        return {"map50": float(results["map50"]), "map50_95": float(results["map50_95"])}

    box = results.box
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def compute_per_class_ap(
    results: Any,
    class_names: list[str],
) -> dict[str, float]:
    """Extract per-class AP@50 values from an Ultralytics results object.

    Args:
        results: Ultralytics ``Results`` object from ``model.val()``.
        class_names: Ordered list of class name strings.

    Returns:
        Dict mapping class name to AP@50 (0.0 for classes absent from eval).
    """
    box = results.box
    ap_by_class = {name: 0.0 for name in class_names}
    for idx, ap in zip(box.ap_class_index, box.ap50):
        name = class_names[int(idx)]
        ap_by_class[name] = round(float(ap), 4)
    return ap_by_class


def compute_precision_recall(results: Any) -> dict[str, float]:
    """Extract mean precision and recall from an Ultralytics results object.

    Args:
        results: Ultralytics ``Results`` object from ``model.val()``.

    Returns:
        Dict with keys ``"precision"`` and ``"recall"``.
    """
    box = results.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def compute_robustness_metrics(
    clear_map: float,
    adverse_map: float,
) -> dict[str, float]:
    """Compute absolute and relative mAP degradation from clear to adverse conditions.

    Args:
        clear_map: mAP@50 on the clear test set.
        adverse_map: mAP@50 on the adverse (rain or nighttime) test set.

    Returns:
        Dict with:
        - ``"map_drop"``: absolute drop (clear_map - adverse_map)
        - ``"relative_degradation_pct"``: drop as percentage of clear_map
        - ``"retention_pct"``: adverse_map / clear_map as percentage
    """
    drop = clear_map - adverse_map
    relative = (drop / clear_map * 100) if clear_map > 0 else float("nan")
    retention = (adverse_map / clear_map * 100) if clear_map > 0 else float("nan")
    return {
        "map_drop": round(drop, 4),
        "relative_degradation_pct": round(relative, 2),
        "retention_pct": round(retention, 2),
    }


def evaluate_rfdetr(
    model,
    coco_data_dir: Path,
    split: str = "valid",
    threshold: float = 0.001,
) -> dict[str, float]:
    """Run RF-DETR inference on a COCO split and compute mAP via pycocotools.

    RF-DETR has no built-in .val() method, so evaluation is done by running
    model.predict() on each image and comparing against COCO ground truth.

    Args:
        model: RFDETRMedium instance with loaded weights.
        coco_data_dir: Root COCO data directory (contains the split subfolder).
        split: Subfolder name with images and _annotations.coco.json.
        threshold: Confidence threshold — keep low (0.001) so COCOeval can
                   sweep all predictions across IoU thresholds.

    Returns:
        Dict with ``"map50"`` and ``"map50_95"``.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    split_dir = Path(coco_data_dir) / split
    coco_gt = COCO(str(split_dir / "_annotations.coco.json"))

    coco_dt: list[dict] = []
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.imgs[img_id]
        img_path = split_dir / img_info["file_name"]
        detections = model.predict(str(img_path), threshold=threshold)
        if len(detections) == 0:
            continue
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            coco_dt.append({
                "image_id": img_id,
                "category_id": int(detections.class_id[i]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(detections.confidence[i]),
            })

    if not coco_dt:
        per_class_ap = {coco_gt.cats[cid]["name"]: 0.0 for cid in coco_gt.getCatIds()}
        return {"map50": 0.0, "map50_95": 0.0, "per_class_ap": per_class_ap}

    coco_res = coco_gt.loadRes(coco_dt)
    coco_eval = COCOeval(coco_gt, coco_res, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Per-class AP@50: precision shape is [T, R, K, A, M]
    # T=0 → IoU=0.50, A=0 → all areas, M=2 → maxDets=100
    precision = coco_eval.eval["precision"]
    per_class_ap: dict[str, float] = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p = precision[0, :, k, 0, 2]
        valid = p[p > -1]
        per_class_ap[coco_gt.cats[cat_id]["name"]] = round(float(np.mean(valid)), 4) if len(valid) else 0.0

    return {
        "map50": float(coco_eval.stats[1]),
        "map50_95": float(coco_eval.stats[0]),
        "per_class_ap": per_class_ap,
    }


def compute_per_class_robustness(
    clear_per_class: dict[str, float],
    adverse_per_class: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute per-class AP drop from clear to adverse conditions.

    Args:
        clear_per_class: ``{class_name: ap50}`` for the clear baseline.
        adverse_per_class: ``{class_name: ap50}`` for the adverse condition.

    Returns:
        ``{class_name: {"ap_drop": float, "relative_drop_pct": float}}``
    """
    result: dict[str, dict[str, float]] = {}
    for cls, clear_ap in clear_per_class.items():
        adv_ap = adverse_per_class.get(cls, 0.0)
        drop = clear_ap - adv_ap
        rel = (drop / clear_ap * 100) if clear_ap > 0 else float("nan")
        result[cls] = {"ap_drop": round(drop, 4), "relative_drop_pct": round(rel, 2)}
    return result


def build_comparison_df(
    scores: dict[ModelName, dict[Condition, dict[str, float]]],
) -> pd.DataFrame:
    """Build a flat comparison DataFrame from nested score dicts.

    Args:
        scores: Nested dict of the form
                ``{model_name: {condition: {metric: value}}}``.
                Example::

                    {
                        "yolov11": {
                            "clear":     {"map50": 0.45, "map50_95": 0.30},
                            "rain":      {"map50": 0.38, "map50_95": 0.25},
                            "nighttime": {"map50": 0.35, "map50_95": 0.22},
                        },
                        ...
                    }

    Returns:
        DataFrame with columns:
        ``model``, ``condition``, ``map50``, ``map50_95``,
        ``map_drop``, ``relative_degradation_pct``, ``retention_pct``.
        Robustness columns are only populated for non-clear conditions.
    """
    rows: list[dict] = []
    for model, cond_scores in scores.items():
        clear_map50 = cond_scores.get("clear_day", {}).get("map50", float("nan"))
        for condition, metrics in cond_scores.items():
            row = {"model": model, "condition": condition, **metrics}
            if condition != "clear_day":
                rob = compute_robustness_metrics(clear_map50, metrics.get("map50", 0.0))
                row.update(rob)
            rows.append(row)
    return pd.DataFrame(rows)
