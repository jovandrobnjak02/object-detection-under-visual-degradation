"""Evaluation utilities: mAP, per-class AP, robustness metrics, and summary DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


ModelName = str
Condition = str  # "clear" | "rain" | "nighttime"


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

    # Ultralytics Results object
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
        Dict mapping class name to AP@50 value.
    """
    # TODO: Ultralytics stores per-class AP in results.box.ap_class_index and
    #       results.box.ap. Implement mapping once API version is confirmed.
    raise NotImplementedError("implement per-class AP extraction for the confirmed Ultralytics version")


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
        clear_map50 = cond_scores.get("clear", {}).get("map50", float("nan"))
        for condition, metrics in cond_scores.items():
            row = {"model": model, "condition": condition, **metrics}
            if condition != "clear":
                rob = compute_robustness_metrics(clear_map50, metrics.get("map50", 0.0))
                row.update(rob)
            rows.append(row)
    return pd.DataFrame(rows)
