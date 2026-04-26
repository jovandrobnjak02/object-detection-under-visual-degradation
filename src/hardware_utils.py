"""Hardware profiling: VRAM, FLOPs, parameters, and inference speed."""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn


def measure_vram(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | str = "cuda",
    n_warmup: int = 3,
) -> dict[str, float]:
    """Measure peak GPU VRAM consumption during a forward pass.

    Runs ``n_warmup`` passes to load CUDA kernels, then records peak
    allocated memory over a single forward pass.

    Args:
        model: The model to profile (will be set to eval mode).
        input_tensor: A single batched input tensor on the correct device.
        device: CUDA device to profile.
        n_warmup: Number of warmup forward passes before measurement.

    Returns:
        Dict with ``"peak_vram_mb"`` and ``"peak_vram_gb"``.
    """
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)

        torch.cuda.reset_peak_memory_stats(device)
        _ = model(input_tensor)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / 1024 ** 2
    return {"peak_vram_mb": round(peak_mb, 2), "peak_vram_gb": round(peak_mb / 1024, 4)}


def count_flops_and_params(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, float]:
    """Count FLOPs and trainable parameters using fvcore.

    Args:
        model: The model to analyse (will be set to eval mode).
        input_tensor: A single batched input tensor (on CPU is fine).

    Returns:
        Dict with ``"gflops"`` (giga-FLOPs) and ``"params_m"`` (millions).
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError as e:
        raise ImportError("Install fvcore: pip install fvcore") from e

    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    total_flops = flops.total()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "gflops": round(total_flops / 1e9, 3),
        "params_m": round(total_params / 1e6, 3),
    }


def measure_inference_speed(
    infer_fn: Callable[[torch.Tensor], object],
    input_tensor: torch.Tensor,
    device: torch.device | str = "cuda",
    n_warmup: int = 10,
    n_frames: int = 100,
) -> dict[str, float]:
    """Measure average per-frame inference latency and FPS.

    Uses CUDA events for GPU timing (sub-millisecond accuracy). Falls back
    to ``time.perf_counter`` on CPU.

    Args:
        infer_fn: Callable that accepts a batched tensor and returns model
                  output. Should already have ``torch.no_grad()`` applied
                  (or be the model's ``__call__`` — this function wraps it).
        input_tensor: A single batched input tensor.
        device: Device used for timing (set to ``"cpu"`` to skip CUDA sync).
        n_warmup: Number of warmup calls before timing.
        n_frames: Number of timed calls to average over.

    Returns:
        Dict with ``"ms_per_frame"`` and ``"fps"``.
    """
    input_tensor = input_tensor.to(device)
    use_cuda = str(device) != "cpu" and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(n_warmup):
            infer_fn(input_tensor)
        if use_cuda:
            torch.cuda.synchronize(device)

        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(n_frames):
                infer_fn(input_tensor)
            ender.record()
            torch.cuda.synchronize(device)
            elapsed_ms = starter.elapsed_time(ender)
        else:
            t0 = time.perf_counter()
            for _ in range(n_frames):
                infer_fn(input_tensor)
            elapsed_ms = (time.perf_counter() - t0) * 1000

    ms_per_frame = elapsed_ms / n_frames
    fps = 1000.0 / ms_per_frame
    return {"ms_per_frame": round(ms_per_frame, 3), "fps": round(fps, 2)}
