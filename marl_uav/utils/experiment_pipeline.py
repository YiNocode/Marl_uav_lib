"""Resolve experiment / pipeline / capture_protection blocks for E2 BC-MAPPO runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def resolve_experiment_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("experiment") or {})


def resolve_pipeline_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("pipeline") or {})


def resolve_capture_protection_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    cp = dict(train_cfg.get("capture_protection") or {})
    finetune = dict(train_cfg.get("mappo_finetune") or {})
    # Legacy: if only mappo_finetune.protected_epochs set, treat as enabled bc_kl_guard
    if not cp and int(finetune.get("protected_epochs", 0) or 0) > 0:
        cp = {
            "enabled": True,
            "mode": "bc_kl_guard",
        }
    if "enabled" not in cp:
        cp.setdefault("enabled", bool(finetune.get("protected_epochs", 0)))
    cp.setdefault("mode", "bc_kl_guard")
    return cp


def _merge_mappo_finetune_yaml(train_cfg: dict[str, Any]) -> dict[str, Any]:
    finetune = dict(train_cfg.get("mappo_finetune") or {})
    bc_cfg = dict(train_cfg.get("bc_warmstart") or {})
    if bc_cfg.get("mappo_finetune"):
        finetune = {**finetune, **dict(bc_cfg["mappo_finetune"])}
    return finetune


def resolve_mappo_finetune_with_protection(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge ``mappo_finetune``, ``bc_warmstart.mappo_finetune``, and capture_protection."""
    finetune = _merge_mappo_finetune_yaml(train_cfg)
    cp = resolve_capture_protection_cfg(train_cfg)

    if not bool(cp.get("enabled", False)):
        # Ablation: no protection — zero out guard knobs
        finetune = deepcopy(finetune)
        finetune["protected_epochs"] = 0
        finetune["freeze_actor_epochs"] = 0
        finetune["deterministic_rollout_epochs"] = 0
        finetune["bc_kl_coef"] = 0.0
        finetune["bc_kl_adaptive"] = False
        finetune["_capture_protection_mode"] = "disabled"
        finetune["_capture_protection_enabled"] = False
        return finetune

    mode = str(cp.get("mode", "bc_kl_guard"))
    finetune = deepcopy(finetune)
    finetune["_capture_protection_mode"] = mode
    finetune["_capture_protection_enabled"] = True
    finetune["_capture_protection"] = cp

    if mode == "bc_kl_guard":
        # Keep existing protected_epochs / bc_kl / freeze_actor from YAML
        pass
    elif mode in ("action_guard", "mixed_action"):
        finetune.setdefault("protected_epochs", int(cp.get("protection_decay_steps", 0) or 0) // max(
            int(train_cfg.get("rollout_steps", 1024)), 1
        ))
    return finetune


def should_attach_bc_anchor(train_cfg: dict[str, Any]) -> bool:
    """False for no_bc_anchor ablation."""
    pipeline = resolve_pipeline_cfg(train_cfg)
    if pipeline.get("attach_bc_anchor") is False:
        return False
    finetune = dict(train_cfg.get("mappo_finetune") or {})
    if float(finetune.get("bc_kl_coef", 0) or 0) > 0:
        return True
    cp = resolve_capture_protection_cfg(train_cfg)
    return bool(cp.get("enabled", False)) and str(cp.get("mode", "bc_kl_guard")) == "bc_kl_guard"


def pipeline_stage(train_cfg: dict[str, Any]) -> str:
    return str(resolve_pipeline_cfg(train_cfg).get("stage", "finetune_capture_protected"))


def skip_mappo_training(train_cfg: dict[str, Any]) -> bool:
    """True for bc_only / bc_eval stages."""
    stage = pipeline_stage(train_cfg)
    return stage in ("bc_only", "bc_eval") or int(train_cfg.get("num_epochs", 1)) <= 0
