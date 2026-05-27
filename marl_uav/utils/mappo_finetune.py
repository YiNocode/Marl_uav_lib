"""Helpers for MAPPO fine-tuning after behavior cloning."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


def resolve_mappo_finetune_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge ``mappo_finetune`` with optional overrides under ``bc_warmstart``."""
    finetune = dict(train_cfg.get("mappo_finetune") or {})
    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    if bc_cfg.get("mappo_finetune"):
        finetune = {**finetune, **dict(bc_cfg["mappo_finetune"])}
    return finetune


def _protected_epochs_limit(finetune_cfg: dict[str, Any]) -> int:
    return int(finetune_cfg.get("protected_epochs", 0) or 0)


def deterministic_rollout_until_epoch(finetune_cfg: dict[str, Any]) -> int:
    """Last epoch (exclusive) that uses deterministic (mean) actions in rollout."""
    prot = _protected_epochs_limit(finetune_cfg)
    det = int(finetune_cfg.get("deterministic_rollout_epochs", 0) or 0)
    return max(prot, det)


def deterministic_rollout_for_epoch(finetune_cfg: dict[str, Any], epoch: int) -> bool:
    until = deterministic_rollout_until_epoch(finetune_cfg)
    return until > 0 and epoch < until


def freeze_actor_until_epoch(finetune_cfg: dict[str, Any]) -> int:
    prot = _protected_epochs_limit(finetune_cfg)
    freeze = int(finetune_cfg.get("freeze_actor_epochs", 0) or 0)
    return max(prot, freeze)


def entropy_coef_for_epoch(
    learner: Any,
    finetune_cfg: dict[str, Any],
    epoch: int,
    default_entropy_coef: float,
) -> float:
    """Entropy schedule; ``protected_epochs`` keeps entropy at ``entropy_coef_start``."""
    start = finetune_cfg.get("entropy_coef_start")
    if start is None:
        return float(default_entropy_coef)
    start_val = float(start)
    target = float(finetune_cfg.get("entropy_coef_end", default_entropy_coef))
    ramp_epochs = int(finetune_cfg.get("entropy_coef_ramp_epochs", 0) or 0)
    ramp_start = _protected_epochs_limit(finetune_cfg)
    if ramp_start <= 0:
        ramp_start = deterministic_rollout_until_epoch(finetune_cfg)

    if ramp_start > 0 and epoch < ramp_start:
        return start_val
    if ramp_epochs <= 0:
        return target
    frac = min(max((epoch - ramp_start) / ramp_epochs, 0.0), 1.0)
    return start_val + (target - start_val) * frac


def freeze_actor_for_epoch(finetune_cfg: dict[str, Any], epoch: int) -> bool:
    """True when only the critic should be updated this epoch."""
    until = freeze_actor_until_epoch(finetune_cfg)
    return until > 0 and epoch < until


def bc_kl_coef_for_epoch(finetune_cfg: dict[str, Any], epoch: int) -> float:
    """BC KL penalty weight; optional linear decay via ``bc_kl_coef_ramp_epochs``."""
    coef = float(finetune_cfg.get("bc_kl_coef", 0.0) or 0.0)
    if coef <= 0.0:
        return 0.0
    ramp_epochs = int(finetune_cfg.get("bc_kl_coef_ramp_epochs", 0))
    if ramp_epochs <= 0:
        return coef
    end_coef = float(finetune_cfg.get("bc_kl_coef_end", 0.0) or 0.0)
    frac = min(max(epoch / ramp_epochs, 0.0), 1.0)
    return coef + (end_coef - coef) * frac


def actor_lr_scale_for_epoch(finetune_cfg: dict[str, Any], epoch: int) -> float:
    """Scale actor LR after unfreeze; 0 while actor is frozen."""
    if freeze_actor_for_epoch(finetune_cfg, epoch):
        return 0.0
    end_scale = float(finetune_cfg.get("actor_lr_scale_end", 1.0) or 1.0)
    start_scale = float(finetune_cfg.get("actor_lr_scale_start", end_scale) or end_scale)
    if start_scale >= end_scale:
        return end_scale
    ramp_epochs = int(finetune_cfg.get("actor_lr_ramp_epochs", 0) or 0)
    freeze_until = freeze_actor_until_epoch(finetune_cfg)
    if ramp_epochs <= 0:
        return start_scale
    t = max(epoch - freeze_until, 0)
    frac = min(max(t / ramp_epochs, 0.0), 1.0)
    return start_scale + (end_scale - start_scale) * frac


def ppo_inner_epochs_for_epoch(
    finetune_cfg: dict[str, Any], epoch: int, default_epochs: int
) -> int:
    """Fewer PPO passes per batch during the protected window."""
    prot = _protected_epochs_limit(finetune_cfg)
    until = prot if prot > 0 else int(finetune_cfg.get("ppo_epochs_protected_until", 0) or 0)
    if until > 0 and epoch < until:
        return max(1, int(finetune_cfg.get("ppo_epochs_during_protection", 1) or 1))
    return max(1, int(default_epochs))


def uses_bc_finetune(finetune_cfg: dict[str, Any]) -> bool:
    """Whether BC-specific MAPPO fine-tune knobs are active."""
    return (
        _protected_epochs_limit(finetune_cfg) > 0
        or freeze_actor_for_epoch(finetune_cfg, 0)
        or bc_kl_coef_for_epoch(finetune_cfg, 0) > 0.0
    )


def iter_critic_modules(policy: nn.Module) -> Iterable[nn.Module]:
    for name in ("critic_encoder", "value_head"):
        mod = getattr(policy, name, None)
        if isinstance(mod, nn.Module):
            yield mod


def iter_actor_modules(policy: nn.Module) -> Iterable[nn.Module]:
    """Actor submodules for centralized / actor-critic / DREAM policies."""
    for name in ("actor_encoder", "policy_head", "actor_head", "dream_actor_head"):
        mod = getattr(policy, name, None)
        if isinstance(mod, nn.Module):
            yield mod


def set_policy_actor_trainable(policy: nn.Module, trainable: bool) -> None:
    """Enable or disable gradients on actor submodules only."""
    for mod in iter_actor_modules(policy):
        for param in mod.parameters():
            param.requires_grad_(trainable)


def create_bc_policy_anchor(policy: nn.Module, device: torch.device | None = None) -> nn.Module:
    """Deep-copy ``policy`` as a frozen BC reference (caller must load BC weights first)."""
    anchor = copy.deepcopy(policy)
    if device is not None:
        anchor = anchor.to(device)
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)
    return anchor


def attach_bc_anchor_to_learner(
    learner: Any,
    *,
    policy: nn.Module,
    bc_ckpt_path: Path | None,
    device: torch.device | None = None,
    log_std_after_bc: float | None = None,
) -> None:
    """Install a frozen BC policy copy on a MAPPO learner (no-op if unsupported).

    Uses a deep copy of the live policy (already BC-loaded + log_std applied).
    Do not reload the raw checkpoint here: it may omit ``log_std_after_bc`` and
    would desync the anchor from rollout actions.
    """
    if not hasattr(learner, "set_bc_policy_anchor"):
        return
    del bc_ckpt_path
    anchor = create_bc_policy_anchor(policy, device=device)
    if log_std_after_bc is not None:
        from marl_uav.runners.bc_pretrainer import set_policy_log_std

        set_policy_log_std(anchor, float(log_std_after_bc))
    learner.set_bc_policy_anchor(anchor)


def apply_learner_finetune_epoch(learner: Any, finetune_cfg: dict[str, Any], epoch: int) -> None:
    """Per-epoch BC fine-tune state (freeze actor, KL coef). Safe no-op for plain MAPPO."""
    if not hasattr(learner, "apply_finetune_epoch"):
        return
    learner.apply_finetune_epoch(finetune_cfg, epoch)


def adaptive_bc_kl_coef(
    finetune_cfg: dict[str, Any],
    *,
    base_coef: float,
    rolling_capture: float | None,
    peak_capture: float | None,
) -> float:
    """Raise BC KL when capture regresses; relax only at new peaks (mappo_bc only)."""
    if not finetune_cfg.get("bc_kl_adaptive"):
        return base_coef
    if rolling_capture is None:
        return base_coef

    baseline_cfg = finetune_cfg.get("bc_kl_capture_baseline")
    if baseline_cfg in (None, "auto", "peak"):
        baseline = float(peak_capture or rolling_capture)
    else:
        baseline = float(baseline_cfg)
    if baseline <= 0.0:
        return base_coef

    tol = float(finetune_cfg.get("bc_kl_regression_tolerance", 0.08))
    boost = float(finetune_cfg.get("bc_kl_regression_boost", 2.5))
    relax = float(finetune_cfg.get("bc_kl_relax_when_stable", 0.6))
    max_coef = float(finetune_cfg.get("bc_kl_coef_max", base_coef * 4.0))
    min_coef = float(finetune_cfg.get("bc_kl_coef_min", base_coef * 0.5))
    peak = float(peak_capture or rolling_capture)

    if rolling_capture < baseline * (1.0 - tol):
        return min(base_coef * boost, max_coef)
    if peak > 0.0 and rolling_capture >= peak * 0.98:
        return max(base_coef * relax, min_coef)
    return base_coef


def apply_capture_adaptive_bc_kl(
    learner: Any,
    finetune_cfg: dict[str, Any],
    *,
    rolling_capture: float | None,
    peak_capture: float | None,
) -> None:
    if not hasattr(learner, "apply_capture_adaptive_bc_kl"):
        return
    learner.apply_capture_adaptive_bc_kl(
        finetune_cfg,
        rolling_capture=rolling_capture,
        peak_capture=peak_capture,
    )
