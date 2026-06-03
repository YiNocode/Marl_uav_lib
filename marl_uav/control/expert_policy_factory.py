"""Factory helpers for geometric expert policies used in BC warm-start."""

from __future__ import annotations

from typing import Any, Callable

from marl_uav.control.fixed_ring_pursuit import make_fixed_ring_get_actions_fn
from marl_uav.control.geometric_pursuit_baselines import (
    make_hungarian_slot_get_actions_fn,
    make_oracle_slot_get_actions_fn,
    make_ot_slot_get_actions_fn,
    make_pure_pursuit_get_actions_fn,
)
from marl_uav.control.obstacle_aware_sce_baselines import (
    make_sce_cached_path_slot_get_actions_fn,
    make_sce_turn_radius_slot_get_actions_fn,
)

ExpertGetActionsFn = Callable[[Any, Any, Any], Any]


def make_expert_get_actions_fn(
    env: Any,
    expert: str,
    expert_cfg: dict[str, Any] | None = None,
) -> ExpertGetActionsFn:
    """Build a RolloutWorker-compatible ``get_actions_fn`` for BC data collection."""
    name = str(expert).strip().lower().replace("-", "_")
    cfg = dict(expert_cfg or {})

    if name in ("pure_pursuit", "purepursuit"):
        return make_pure_pursuit_get_actions_fn(env, **cfg)
    if name in ("fixed_ring", "fixedring"):
        return make_fixed_ring_get_actions_fn(env, **cfg)
    if name in ("oracle_slot", "oracleslot"):
        return make_oracle_slot_get_actions_fn(env, **cfg)
    if name in ("hungarian_slot", "hungarian"):
        return make_hungarian_slot_get_actions_fn(env, **cfg)
    if name in ("ot_slot", "ot"):
        return make_ot_slot_get_actions_fn(env, **cfg)
    if name in ("sce_cached_path_slot", "sce_cached_path", "cached_path_slot"):
        reach = dict(cfg.pop("reachability", {}) or {})
        rates = dict(cfg.pop("runtime_rates", {}) or {})
        return make_sce_cached_path_slot_get_actions_fn(
            env, reachability=reach, runtime_rates=rates, **cfg
        )
    if name in ("sce_turn_radius_slot", "sce_turn_radius", "turn_radius_slot"):
        reach = dict(cfg.pop("reachability", {}) or {})
        rates = dict(cfg.pop("runtime_rates", {}) or {})
        planner = dict(cfg.pop("turn_radius_planner", {}) or {})
        return make_sce_turn_radius_slot_get_actions_fn(
            env, reachability=reach, runtime_rates=rates, turn_radius_planner=planner, **cfg
        )
    raise ValueError(
        f"Unsupported BC expert={expert!r}. "
        "Choose one of: pure_pursuit, fixed_ring, oracle_slot, hungarian_slot, ot_slot, "
        "sce_cached_path_slot, sce_turn_radius_slot."
    )
