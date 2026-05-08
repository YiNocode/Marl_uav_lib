"""Helpers for moving NumPy arrays to torch tensors efficiently."""

from __future__ import annotations

import torch


def tensor_from_numpy_on_device(
    arr,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CPU tensor from NumPy, then pin_memory + non_blocking H2D when CUDA."""
    t = torch.as_tensor(arr, dtype=dtype)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)
