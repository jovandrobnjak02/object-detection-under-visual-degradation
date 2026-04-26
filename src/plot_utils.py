"""Plotting utilities for thesis result figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

MODEL_COLORS: dict[str, str] = {
    "yolov11": "#4C72B0",
    "yolov12": "#DD8452",
    "rtdetr":  "#55A868",
    "rfdetr":  "#C44E52",
}
CONDITIONS = [
    "clear_day", "rainy_day", "snowy_day", "night_clear",
    "overcast_day", "partly_cloudy_day", "dawn_dusk_clear",
]
CONDITION_LABELS = {
    "clear_day":         "Clear Day",
    "rainy_day":         "Rain (day)",
    "snowy_day":         "Snow (day)",
    "night_clear":       "Night (clear)",
    "overcast_day":      "Overcast (day)",
    "partly_cloudy_day": "Partly Cloudy (day)",
    "dawn_dusk_clear":   "Dawn/Dusk (clear)",
}


def plot_map_comparison(
    df: pd.DataFrame,
    metric: str = "map50",
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing mAP across all 4 models and 3 conditions.

    Args:
        df: DataFrame from :func:`~src.eval_utils.build_comparison_df`.
        metric: Column name to plot (``"map50"`` or ``"map50_95"``).
        output_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    models = df["model"].unique().tolist()
    x = np.arange(len(CONDITIONS))
    width = 0.2
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, model in zip(offsets, models):
        vals = [
            df.loc[(df["model"] == model) & (df["condition"] == c), metric].values
            for c in CONDITIONS
        ]
        heights = [v[0] if len(v) else 0.0 for v in vals]
        ax.bar(x + offset, heights, width, label=model, color=MODEL_COLORS.get(model))

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by Model and Condition")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(title="Model")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_degradation_curves(
    df: pd.DataFrame,
    metric: str = "retention_pct",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line chart showing mAP retention (%) from clear to adverse conditions.

    Args:
        df: DataFrame from :func:`~src.eval_utils.build_comparison_df`.
        metric: Column to plot on the y-axis (default: ``"retention_pct"``).
        output_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    adverse_conditions = ["rainy_day", "snowy_day", "night_clear", "overcast_day", "partly_cloudy_day", "dawn_dusk_clear"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        ys = [
            sub.loc[sub["condition"] == c, metric].values
            for c in adverse_conditions
        ]
        heights = [y[0] if len(y) else np.nan for y in ys]
        ax.plot(
            [CONDITION_LABELS[c] for c in adverse_conditions],
            heights,
            marker="o",
            label=model,
            color=MODEL_COLORS.get(model),
        )

    ax.axhline(100, linestyle="--", color="gray", linewidth=0.8, label="Baseline (clear)")
    ax.set_ylabel("mAP Retention (%)")
    ax.set_title("Robustness: mAP Retention vs. Clear Baseline")
    ax.legend(title="Model")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_efficiency_scatter(
    df: pd.DataFrame,
    hw_df: pd.DataFrame,
    x_metric: str = "gflops",
    y_metric: str = "retention_pct",
    condition: str = "rain",
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of efficiency (x) vs robustness retention (y).

    Args:
        df: Evaluation DataFrame from :func:`~src.eval_utils.build_comparison_df`.
        hw_df: Hardware metrics DataFrame with columns ``model``, ``gflops``,
               ``params_m``, ``ms_per_frame``, ``peak_vram_mb``.
        x_metric: Hardware column to use on the x-axis.
        y_metric: Evaluation column to use on the y-axis.
        condition: Adverse condition to pull robustness values from.
        output_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    merged = df[df["condition"] == condition].merge(hw_df, on="model")
    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in merged.iterrows():
        model = row["model"]
        ax.scatter(row[x_metric], row[y_metric], s=120, color=MODEL_COLORS.get(model), zorder=3)
        ax.annotate(model, (row[x_metric], row[y_metric]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.set_xlabel(x_metric.upper().replace("_", " "))
    ax.set_ylabel(f"{y_metric} — {CONDITION_LABELS.get(condition, condition)}")
    ax.set_title(f"Efficiency vs. Robustness ({CONDITION_LABELS.get(condition, condition)})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_per_class_heatmap(
    per_class_data: dict[str, dict[str, dict[str, float]]],
    class_names: list[str],
    condition: str = "clear",
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of per-class AP@50 for all models under a given condition.

    Args:
        per_class_data: Nested dict ``{model: {condition: {class_name: ap}}}``.
        class_names: Ordered list of class name strings.
        condition: Which condition to visualise.
        output_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    models = list(per_class_data.keys())
    matrix = np.array([
        [per_class_data[m].get(condition, {}).get(cls, np.nan) for cls in class_names]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=class_names,
        yticklabels=models,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "AP@50"},
    )
    ax.set_title(f"Per-Class AP@50 — {CONDITION_LABELS.get(condition, condition)}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig
