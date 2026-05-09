"""Training device resolution (CUDA / CPU / MPS)."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _debug_agent_log(*, hypothesis_id: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    try:
        log_path = _repo_root() / "debug-c143b2.log"
        payload = {
            "sessionId": "c143b2",
            "hypothesisId": hypothesis_id,
            "location": "marl_uav/utils/device.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def _cuda_probe_ok(device_str: str = "cuda") -> bool:
    """True if a minimal op runs on CUDA (catches sm_* mismatch / no kernel image)."""
    try:
        x = torch.zeros(2, 2, device=device_str, dtype=torch.float32)
        _ = float((x + 1.0).sum().item())
        return True
    except RuntimeError:
        return False


def resolve_train_device(train_cfg: Mapping[str, Any] | None = None) -> torch.device:
    """Pick ``torch.device`` from top-level train YAML.

    - ``device: auto`` (default): tries CUDA (with runtime probe), then MPS, then CPU.
    - ``device: cuda`` / ``cuda:0`` / ``cpu`` / ``mps``: passed to ``torch.device``, except
      CUDA is **probed**; on failure (e.g. GPU newer than PyTorch's bundled arch), falls back
      to CPU with a warning (Blackwell sm_120 vs wheels built only to sm_90).
    """
    cfg = dict(train_cfg or {})
    raw = cfg.get("device", "auto")
    s = str(raw).strip().lower() if raw is not None else "auto"

    if s == "auto":
        if torch.cuda.is_available():
            if _cuda_probe_ok("cuda"):
                return torch.device("cuda")
            cap = None
            try:
                cap = torch.cuda.get_device_capability(0)
            except Exception:
                pass
            _debug_agent_log(
                hypothesis_id="H1",
                message="cuda_probe_failed_fallback",
                data={
                    "requested": "auto->cuda",
                    "cuda_capability": cap,
                    "fallback": "cpu",
                },
            )
            warnings.warn(
                "CUDA is visible but a minimal GPU op failed (often: GPU architecture newer than "
                "this PyTorch build, e.g. sm_120 Blackwell vs wheels built to sm_90). "
                "Falling back to CPU. Install a PyTorch build that supports your GPU "
                "(see https://pytorch.org/get-started/locally/).",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = torch.device(str(raw))
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else 0
        cuda_str = f"cuda:{idx}"
        if _cuda_probe_ok(cuda_str):
            return dev
        cap = None
        try:
            cap = torch.cuda.get_device_capability(idx)
        except Exception:
            pass
        _debug_agent_log(
            hypothesis_id="H1",
            message="cuda_probe_failed_fallback",
            data={
                "requested": str(raw),
                "cuda_capability": cap,
                "fallback": "cpu",
            },
        )
        warnings.warn(
            f"Requested device {raw!r} but CUDA probe failed; using CPU. "
            "Upgrade PyTorch for your GPU architecture.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return dev
