"""Helpers for moving NumPy arrays to torch tensors efficiently."""

from __future__ import annotations

import torch


def tensor_from_numpy_on_device(
    arr,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move array-like data to ``device`` without relying on torch's default device.

    Genesis may configure PyTorch so tensors created without an explicit device
    land on CUDA.  PPO batches are CPU/NumPy data, so always create a CPU staging
    tensor before optionally pinning it for an asynchronous CUDA copy.
    """
    t = torch.as_tensor(arr, dtype=dtype, device=torch.device("cpu"))
    if device.type == "cuda":
        try:
            return t.pin_memory().to(device, non_blocking=True)
        except RuntimeError:
            return t.to(device)
    return t.to(device)
