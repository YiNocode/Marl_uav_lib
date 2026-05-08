"""Limit PyTorch CPU threads to reduce contention with VecEnv worker processes.

When ``num_envs > 1``, default to single-threaded intra-op unless overridden via:

- ``train_cfg["torch_num_threads"]`` / ``train_num_interop_threads``
- Environment variables ``TORCH_NUM_THREADS``, ``TORCH_NUM_INTEROP_THREADS``

NUMA / affinity: prefer binding the **training process** (and optionally workers) with
``taskset`` or ``numactl`` on Linux, e.g. ``taskset -c 0-7 python scripts/train.py ...``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping


def configure_torch_threads(*, num_envs: int, train_cfg: Mapping[str, Any] | None = None) -> None:
    """Call early in ``train.main`` after loading YAML (before rollout / heavy matmul)."""
    try:
        import torch
    except ImportError:
        return

    cfg = dict(train_cfg or {})
    env_nt = os.environ.get("TORCH_NUM_THREADS", "").strip()
    env_ni = os.environ.get("TORCH_NUM_INTEROP_THREADS", "").strip()

    cfg_nt = cfg.get("torch_num_threads")
    cfg_ni = cfg.get("torch_num_interop_threads")

    if env_nt:
        n = max(1, int(env_nt))
        torch.set_num_threads(n)
    elif cfg_nt is not None:
        torch.set_num_threads(max(1, int(cfg_nt)))
    elif num_envs > 1:
        torch.set_num_threads(1)

    if env_ni:
        torch.set_num_interop_threads(max(1, int(env_ni)))
    elif cfg_ni is not None:
        torch.set_num_interop_threads(max(1, int(cfg_ni)))
    elif num_envs > 1:
        torch.set_num_interop_threads(1)
