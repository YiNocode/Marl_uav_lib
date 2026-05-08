"""Training device resolution (CUDA / CPU / MPS)."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def resolve_train_device(train_cfg: Mapping[str, Any] | None = None) -> torch.device:
    """Pick ``torch.device`` from top-level train YAML.

    - ``device: auto`` (default): ``cuda`` if available, else ``mps`` if available, else ``cpu``.
    - ``device: cuda`` / ``cuda:0`` / ``cpu`` / ``mps``: passed to ``torch.device``.
    """
    cfg = dict(train_cfg or {})
    raw = cfg.get("device", "auto")
    s = str(raw).strip().lower() if raw is not None else "auto"
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(raw))
