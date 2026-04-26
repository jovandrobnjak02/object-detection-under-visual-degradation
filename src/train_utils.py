"""Shared training helpers: checkpoint I/O and logging setup."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_dir: Path,
    filename: str | None = None,
    is_best: bool = False,
) -> Path:
    """Save a model checkpoint to disk.

    Args:
        model: The model whose state_dict will be saved.
        epoch: Current epoch number (included in the saved payload).
        metrics: Dict of metric names to values recorded at this epoch.
        checkpoint_dir: Directory where checkpoints are written.
        filename: Override the auto-generated filename. If ``None``, the
                  file is named ``epoch_{epoch:04d}.pt``.
        is_best: If ``True``, also copies the file to ``best.pt`` in the
                 same directory.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"epoch_{epoch:04d}.pt"
    path = checkpoint_dir / fname

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, path)

    if is_best:
        shutil.copy(path, checkpoint_dir / "best.pt")

    return path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint` into a model.

    Args:
        model: The model to load weights into (modified in-place).
        checkpoint_path: Path to a ``.pt`` checkpoint file.
        device: Device to map tensors to when loading.

    Returns:
        The full checkpoint payload dict (``epoch``, ``metrics``, etc.).
    """
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def setup_logging(
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the root logger with console (and optional file) handlers.

    Args:
        log_file: If provided, also writes logs to this file path.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        Configured root logger.
    """
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
