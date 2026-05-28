"""Visualization capability profiles keyed by execution method / controller."""

from __future__ import annotations

from typing import Any

import numpy as np

_VIZ_DEFAULTS: dict[str, bool] = {
    "manifold_curve": False,
    "slot_targets": False,
    "role_allocation": False,
    "dream_manifold": False,
    "structure_metrics": False,
    "ot_details": False,
    "pursuit_targets": False,
    "fixed_ring_targets": False,
    "fixed_ring_curve": False,
    "obstacles": True,
}


def _profile(method: str, **flags: bool) -> dict[str, Any]:
    out = dict(_VIZ_DEFAULTS)
    out.update(flags)
    out["method"] = method
    return out


def resolve_viz_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map experiment YAML to frontend/backend visualization capabilities."""
    if "sce" in cfg:
        return _profile(
            "sce",
            manifold_curve=True,
            slot_targets=True,
            role_allocation=True,
            ot_details=True,
            structure_metrics=True,
        )
    if "oracle_slot" in cfg:
        return _profile(
            "oracle_slot",
            slot_targets=True,
            role_allocation=True,
        )
    if "fixed_ring" in cfg:
        return _profile(
            "fixed_ring",
            fixed_ring_targets=True,
            fixed_ring_curve=True,
        )
    if "pure_pursuit" in cfg:
        return _profile("pure_pursuit", pursuit_targets=True)
    if "algo" in cfg:
        model_ref = str(cfg.get("model", "")).lower()
        if "dream_mappo" in model_ref:
            return _profile(
                "dream_mappo_full",
                manifold_curve=True,
                slot_targets=True,
                role_allocation=True,
                dream_manifold=True,
                structure_metrics=True,
            )
        return _profile("mappo")

    method = str((cfg.get("benchmark") or {}).get("method", "")).strip().lower()
    if method in ("sce",):
        return resolve_viz_profile({**cfg, "sce": {}})
    if method in ("oracle_slot", "oracleslot"):
        return resolve_viz_profile({**cfg, "oracle_slot": {}})
    if method in ("fixed_ring", "fixedring"):
        return resolve_viz_profile({**cfg, "fixed_ring": {}})
    if method in ("pure_pursuit", "purepursuit"):
        return resolve_viz_profile({**cfg, "pure_pursuit": {}})
    if method in ("dream_mappo_full", "dream_mappo"):
        return resolve_viz_profile({**cfg, "algo": "x", "model": "dream_mappo"})
    if method in ("mappo", "mappo_bc"):
        return resolve_viz_profile({**cfg, "algo": "x", "model": "mappo"})

    return _profile("generic")


def _active_viz_profile(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from marl_uav.utils.debug_browser import get_debug_browser_hub

    hub = get_debug_browser_hub()
    if hub is not None and isinstance(hub._meta.get("viz"), dict):
        return dict(hub._meta["viz"])
    if extra and isinstance(extra.get("viz"), dict):
        return dict(extra["viz"])
    return _profile("generic")


def build_controller_targets(
    env: Any,
    *,
    viz: dict[str, Any],
    controller_cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Optional pursuer goal markers derived from the active controller (not task manifold)."""
    task_state = getattr(env, "task_state", None)
    backend_state = getattr(env, "prev_backend_state", None)
    if task_state is None or backend_state is None:
        return None

    lin_pos = np.asarray(backend_state.states[:, 3, :], dtype=np.float32)
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    pursuer_pos = lin_pos[pursuer_ids]
    evader_pos = lin_pos[int(task_state.evader_id)]

    if viz.get("pursuit_targets"):
        targets = np.repeat(evader_pos[None, :], 3, axis=0)
        return {
            "kind": "evader",
            "targets": targets.tolist(),
        }

    if viz.get("fixed_ring_targets"):
        cfg = dict(controller_cfg or {})
        ring_radius = float(cfg.get("ring_radius", 1.6))
        phase = float(cfg.get("phase", 0.0))
        assignment = str(cfg.get("assignment", "fixed"))
        angles = phase + np.arange(3, dtype=np.float32) * (2.0 * np.pi / 3.0)
        targets = np.zeros((3, 3), dtype=np.float32)
        targets[:, 0] = evader_pos[0] + ring_radius * np.cos(angles)
        targets[:, 1] = evader_pos[1] + ring_radius * np.sin(angles)
        targets[:, 2] = evader_pos[2]
        if assignment == "angle_order":
            rel = pursuer_pos[:, :2] - evader_pos[None, :2]
            pursuer_order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
            assigned = np.zeros_like(targets)
            assigned[pursuer_order] = targets[np.arange(3)]
            targets = assigned
        ring_curve = []
        if viz.get("fixed_ring_curve"):
            theta = np.linspace(0.0, 2.0 * np.pi, 91, endpoint=True, dtype=np.float32)
            ring_curve = [
                [
                    float(evader_pos[0] + ring_radius * np.cos(t)),
                    float(evader_pos[1] + ring_radius * np.sin(t)),
                    float(evader_pos[2]),
                ]
                for t in theta
            ]
        return {
            "kind": "fixed_ring",
            "targets": targets.tolist(),
            "ring_radius": ring_radius,
            "ring_curve": ring_curve or None,
        }

    return None


def filter_algorithm_fields(algo: dict[str, Any], viz: dict[str, Any]) -> dict[str, Any]:
    """Keep sidebar algorithm fields that match active visualization capabilities."""
    if not algo:
        return {}
    always = {
        "method": algo.get("method"),
        "task_name": algo.get("task_name"),
        "capture_dist": algo.get("capture_dist"),
    }
    if algo.get("pursuer_speed_xy") is not None:
        always["pursuer_speed_xy"] = algo.get("pursuer_speed_xy")
    if algo.get("evader_speed_xy") is not None:
        always["evader_speed_xy"] = algo.get("evader_speed_xy")
    out = {k: v for k, v in always.items() if v is not None}
    out["method"] = algo.get("method") or viz.get("method")

    if viz.get("structure_metrics"):
        out.update(
            {
                "target_radius_xy": algo.get("target_radius_xy"),
                "mean_radius_xy": algo.get("mean_radius_xy"),
                "structure_hold_steps": algo.get("structure_hold_steps"),
            }
        )
    if viz.get("manifold_curve") or viz.get("slot_targets"):
        out.update(
            {
                "manifold_target_phase": algo.get("manifold_target_phase"),
                "manifold_target_radius_scale": algo.get("manifold_target_radius_scale"),
                "manifold_contraction_rate": algo.get("manifold_contraction_rate"),
                "manifold_structure_gate_scale": algo.get("manifold_structure_gate_scale"),
            }
        )
    if viz.get("role_allocation"):
        out["role_assignment_mode"] = algo.get("role_assignment_mode")
    if viz.get("ot_details"):
        out.update(
            {
                "ot_epsilon": algo.get("ot_epsilon"),
                "ot_epsilon_scale": algo.get("ot_epsilon_scale"),
                "ot_sinkhorn_iterations": algo.get("ot_sinkhorn_iterations"),
                "assignment_inertia_margin": algo.get("assignment_inertia_margin"),
            }
        )
    return out
