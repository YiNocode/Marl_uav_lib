"""Structure-first slot selection on the reference manifold (optional radius expansion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.reachability.structure_assignment_cost import (
    StructureAssignmentConfig,
    select_structure_assignment,
)


@dataclass
class StructureSlotSelectionConfig:
    """Pick 3 manifold slots maximizing encirclement structure; expand radius if needed."""

    enabled: bool = True
    radius_scales: tuple[float, ...] = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
    max_radius_scale: float = 2.0
    w_radius_penalty: float = 0.02
    require_reachable_assignment: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> StructureSlotSelectionConfig:
        d = dict(raw or {})
        scales_raw = d.get("radius_scales", (1.0, 1.15, 1.3, 1.5, 1.75, 2.0))
        scales = tuple(float(x) for x in scales_raw)
        return cls(
            enabled=bool(d.get("enabled", True)),
            radius_scales=scales,
            max_radius_scale=float(d.get("max_radius_scale", 2.0)),
            w_radius_penalty=float(d.get("w_radius_penalty", 0.02)),
            require_reachable_assignment=bool(d.get("require_reachable_assignment", True)),
        )


def _manifold_geometry_at_scale(
    task: Any,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    task_state: Any | None,
    radius_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (3 slot targets, dense curve, rho_base_used)."""
    pursuer_pos = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    evader_pos = np.asarray(evader_pos, dtype=np.float32).reshape(3)
    scale = max(float(radius_scale), 0.05)
    rho_base = float(task._compute_target_radius_xy(pursuer_pos, evader_pos, task_state=task_state)) * scale

    ang = np.float32(task.manifold_target_phase) + (2.0 * np.pi / 3.0) * np.arange(3, dtype=np.float32)
    if hasattr(task, "_obstacle_aware_radius"):
        rho_slots = task._obstacle_aware_radius(ang, rho_base, evader_pos, task_state=task_state)
    else:
        rho_slots = np.full(3, np.float32(rho_base), dtype=np.float32)

    targets = np.zeros((3, 3), dtype=np.float32)
    targets[:, 0] = evader_pos[0] + rho_slots * np.cos(ang)
    targets[:, 1] = evader_pos[1] + rho_slots * np.sin(ang)
    targets[:, 2] = evader_pos[2]

    n = max(int(getattr(task, "manifold_curve_num_samples", 121)), 16)
    theta = (
        np.linspace(0.0, 2.0 * np.pi, n, endpoint=True, dtype=np.float32)
        + np.float32(task.manifold_target_phase)
    )
    if hasattr(task, "_obstacle_aware_radius"):
        rho_curve = task._obstacle_aware_radius(theta, rho_base, evader_pos, task_state=task_state)
    else:
        rho_curve = np.full(theta.shape[0], np.float32(rho_base), dtype=np.float32)

    curve = np.zeros((theta.shape[0], 3), dtype=np.float32)
    curve[:, 0] = evader_pos[0] + rho_curve * np.cos(theta)
    curve[:, 1] = evader_pos[1] + rho_curve * np.sin(theta)
    curve[:, 2] = evader_pos[2]
    return targets, curve, rho_base


def select_structure_manifold_slots(
    task: Any,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list,
    task_state: Any | None,
    slot_cfg: StructureSlotSelectionConfig,
    struct_cfg: StructureAssignmentConfig,
    *,
    previous_assignment: np.ndarray | None,
    safety_margin: float,
    uav_radius: float,
    switch_penalty: float,
    path_cache: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Search manifold radius scales; pick slots with best joint structure + reachability.

    Structure dominates; ``w_radius_penalty`` mildly prefers tighter rings when scores tie.
    """
    scales = [s for s in slot_cfg.radius_scales if s <= slot_cfg.max_radius_scale + 1e-9]
    if not scales:
        scales = [1.0]

    best_targets: np.ndarray | None = None
    best_curve: np.ndarray | None = None
    best_cost = float("inf")
    best_diag: dict[str, Any] = {}

    for scale in scales:
        targets, curve, rho_base = _manifold_geometry_at_scale(
            task, pursuer_pos, evader_pos, task_state, scale,
        )
        assign, diag = select_structure_assignment(
            pursuer_pos, targets, evader_pos, obstacles, previous_assignment, struct_cfg,
            safety_margin=safety_margin, uav_radius=uav_radius,
            switch_penalty=switch_penalty, path_cache=path_cache,
            exclude_unreachable=slot_cfg.require_reachable_assignment,
        )
        reachable = diag.get("pair_reachable_matrix")
        if slot_cfg.require_reachable_assignment and reachable is not None:
            if not all(bool(reachable[i, int(assign[i])]) for i in range(3)):
                continue

        score = float(diag.get("assigned_structure_score", 0.0))
        cost = -score + slot_cfg.w_radius_penalty * float(scale)
        if cost < best_cost:
            best_cost = cost
            best_targets = targets
            best_curve = curve
            best_diag = {
                **diag,
                "manifold_radius_scale": float(scale),
                "manifold_rho_base": float(rho_base),
                "structure_slot_cost": float(cost),
            }

    if best_targets is None:
        targets, curve, rho_base = _manifold_geometry_at_scale(
            task, pursuer_pos, evader_pos, task_state, 1.0,
        )
        if hasattr(task, "_reference_manifold_targets"):
            targets = task._reference_manifold_targets(
                pursuer_pos, evader_pos, task_state=task_state,
            )
        if hasattr(task, "_reference_manifold_curve"):
            curve = task._reference_manifold_curve(
                pursuer_pos, evader_pos, task_state=task_state,
            )
        best_diag = {
            "manifold_radius_scale": 1.0,
            "manifold_rho_base": float(rho_base),
            "structure_slot_fallback": True,
        }
        return np.asarray(targets, dtype=np.float32), np.asarray(curve, dtype=np.float32), best_diag

    return (
        np.asarray(best_targets, dtype=np.float32),
        np.asarray(best_curve, dtype=np.float32),
        best_diag,
    )
