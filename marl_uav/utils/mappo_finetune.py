"""Helpers for MAPPO fine-tuning after behavior cloning."""

from __future__ import annotations

from typing import Any


def resolve_mappo_finetune_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge ``mappo_finetune`` with optional overrides under ``bc_warmstart``."""
    finetune = dict(train_cfg.get("mappo_finetune") or {})
    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    if bc_cfg.get("mappo_finetune"):
        finetune = {**finetune, **dict(bc_cfg["mappo_finetune"])}
    return finetune


def deterministic_rollout_for_epoch(finetune_cfg: dict[str, Any], epoch: int) -> bool:
    until = int(finetune_cfg.get("deterministic_rollout_epochs", 0))
    return until > 0 and epoch < until


def entropy_coef_for_epoch(
    learner: Any,
    finetune_cfg: dict[str, Any],
    epoch: int,
    default_entropy_coef: float,
) -> float:
    """Linearly ramp entropy after BC if ``entropy_coef_ramp_epochs`` > 0."""
    start = finetune_cfg.get("entropy_coef_start")
    if start is None:
        return float(default_entropy_coef)
    start_val = float(start)
    target = float(finetune_cfg.get("entropy_coef_end", default_entropy_coef))
    ramp_epochs = int(finetune_cfg.get("entropy_coef_ramp_epochs", 0))
    if ramp_epochs <= 0:
        return start_val if epoch < int(finetune_cfg.get("deterministic_rollout_epochs", 0)) else target
    frac = min(max(epoch / ramp_epochs, 0.0), 1.0)
    return start_val + (target - start_val) * frac
