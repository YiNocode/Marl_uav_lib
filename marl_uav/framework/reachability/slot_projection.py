"""Slot reachability projection on the reference manifold (structure-first)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task import compute_pursuit_structure_metrics_3v1
from marl_uav.framework.geometry.obstacle_geometry import Obstacle
from marl_uav.framework.reachability.structure_assignment_cost import (
    StructureAssignmentConfig,
    formation_structure_score,
    pair_reachable,
)


@dataclass
class SlotProjectionConfig:
    enabled: bool = False
    angular_window_deg: float = 45.0
    fallback_angular_window_deg: float = 90.0
    w_geometry: float = 0.5
    w_structure: float = 2.0
    w_clearance: float = 1.5
    min_obstacle_clearance: float = 0.35
    replan_trigger_threshold: float = 0.05
    trigger_path_replan: bool = True
    structure: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SlotProjectionConfig:
        d = dict(raw or {})
        return cls(
            enabled=bool(d.get("enabled", False)),
            angular_window_deg=float(d.get("angular_window_deg", 45.0)),
            fallback_angular_window_deg=float(d.get("fallback_angular_window_deg", 90.0)),
            w_geometry=float(d.get("w_geometry", 0.5)),
            w_structure=float(d.get("w_structure", 2.0)),
            w_clearance=float(d.get("w_clearance", 1.5)),
            min_obstacle_clearance=float(d.get("min_obstacle_clearance", 0.35)),
            replan_trigger_threshold=float(d.get("replan_trigger_threshold", 0.05)),
            trigger_path_replan=bool(d.get("trigger_path_replan", True)),
            structure=dict(d.get("structure") or {}),
        )

    def structure_cfg(self) -> StructureAssignmentConfig:
        return StructureAssignmentConfig.from_dict(self.structure)


def _wrap_angle_diff(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))


def _point_clearance(xy: np.ndarray, obstacles: list[Obstacle]) -> float:
    p = np.asarray(xy, dtype=np.float64).reshape(2)
    if not obstacles:
        return float("inf")
    return min(
        float(np.linalg.norm(p - np.asarray(o.center).reshape(2)) - float(o.radius))
        for o in obstacles
    )


def _subsample_curve(curve: np.ndarray, max_points: int = 48) -> np.ndarray:
    c = np.asarray(curve, dtype=np.float64).reshape(-1, 3)
    if c.shape[0] <= max_points:
        return c
    idx = np.linspace(0, c.shape[0] - 1, max_points, dtype=np.int64)
    return c[idx]


def _curve_angles(curve: np.ndarray, evader_xy: np.ndarray) -> np.ndarray:
    rel = curve[:, :2] - np.asarray(evader_xy, dtype=np.float64).reshape(1, 2)
    return np.arctan2(rel[:, 1], rel[:, 0])


def project_slot_on_manifold(
    slot_idx: int,
    nominal_targets: np.ndarray,
    manifold_curve: np.ndarray,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list[Obstacle],
    cfg: SlotProjectionConfig,
    *,
    safety_margin: float,
    uav_radius: float,
    angular_window_deg: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Slide slot along manifold to improve structure + reachability."""
    struct_cfg = cfg.structure_cfg()
    nominal = np.asarray(nominal_targets, dtype=np.float64).reshape(3, 3)[int(slot_idx)].copy()
    evader = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    curve = _subsample_curve(manifold_curve)
    theta_nom = float(np.arctan2(nominal[1] - evader[1], nominal[0] - evader[0]))
    curve_theta = _curve_angles(curve, evader[:2])
    win = np.deg2rad(float(angular_window_deg if angular_window_deg is not None else cfg.angular_window_deg))
    delta = np.array([_wrap_angle_diff(t, theta_nom) for t in curve_theta], dtype=np.float64)
    mask = delta <= win
    if not np.any(mask):
        mask = np.ones(curve.shape[0], dtype=bool)

    best = nominal.copy()
    best_cost = float("inf")
    best_theta = theta_nom
    for idx in np.where(mask)[0]:
        cand = curve[int(idx)].copy()
        hyp = p.copy()
        hyp[int(slot_idx), :2] = cand[:2]
        struct_gain = formation_structure_score(hyp, evader, struct_cfg) - formation_structure_score(p, evader, struct_cfg)
        geom = _wrap_angle_diff(float(curve_theta[int(idx)]), theta_nom)
        clearance = _point_clearance(cand[:2], obstacles)
        clear_pen = 0.0 if clearance >= cfg.min_obstacle_clearance else (
            cfg.w_clearance * (cfg.min_obstacle_clearance - clearance)
        )
        reachable = any(
            pair_reachable(
                hyp[i, :2], cand[:2], obstacles,
                safety_margin=safety_margin, uav_radius=uav_radius,
            )
            for i in range(3)
        )
        total = cfg.w_geometry * geom - cfg.w_structure * struct_gain + clear_pen
        if not reachable:
            total += 50.0
        if total < best_cost:
            best_cost = total
            best = cand
            best_theta = float(curve_theta[int(idx)])

    return best.astype(np.float32), {
        "slot_idx": int(slot_idx),
        "moved": float(np.linalg.norm(best[:2] - nominal[:2])) > 1e-3,
        "shift_m": float(np.linalg.norm(best[:2] - nominal[:2])),
        "theta_shift_deg": float(np.degrees(_wrap_angle_diff(best_theta, theta_nom))),
        "projection_cost": float(best_cost),
        "reason": "obstacle_slide",
        "nominal_xy": nominal[:2].astype(float).tolist(),
        "projected_xy": best[:2].astype(float).tolist(),
    }


def slot_reachable_from_any_pursuer(
    slot_xy: np.ndarray,
    pursuer_pos: np.ndarray,
    obstacles: list[Obstacle],
    *,
    safety_margin: float,
    uav_radius: float,
) -> bool:
    pxy = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)[:, :2]
    g = np.asarray(slot_xy, dtype=np.float64).reshape(2)
    return any(
        pair_reachable(
            pxy[i], g, obstacles,
            safety_margin=safety_margin, uav_radius=uav_radius,
            path_cache=None,
        )
        for i in range(3)
    )


def detect_slot_projection_moves(
    nominal_targets: np.ndarray,
    effective_targets: np.ndarray,
    previous_effective: np.ndarray | None,
    *,
    threshold: float,
) -> np.ndarray:
    """Return bool mask of slots whose effective position shifted (obstacle slide signal)."""
    nominal = np.asarray(nominal_targets, dtype=np.float64).reshape(3, 3)
    effective = np.asarray(effective_targets, dtype=np.float64).reshape(3, 3)
    prev = None if previous_effective is None else np.asarray(previous_effective, dtype=np.float64).reshape(3, 3)
    moved = np.zeros(3, dtype=bool)
    thr = max(float(threshold), 1e-4)
    for j in range(3):
        from_nominal = float(np.linalg.norm(effective[j, :2] - nominal[j, :2]))
        from_prev = 0.0 if prev is None else float(np.linalg.norm(effective[j, :2] - prev[j, :2]))
        if from_nominal > thr or from_prev > thr:
            moved[j] = True
    return moved


def project_manifold_slots(
    nominal_targets: np.ndarray,
    manifold_curve: np.ndarray,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list[Obstacle],
    cfg: SlotProjectionConfig,
    *,
    safety_margin: float,
    uav_radius: float,
    angular_window_deg: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not cfg.enabled or manifold_curve is None or len(manifold_curve) < 2:
        return np.asarray(nominal_targets, dtype=np.float32).copy(), []

    out = np.asarray(nominal_targets, dtype=np.float32).copy()
    diags: list[dict[str, Any]] = []

    for j in range(3):
        if slot_reachable_from_any_pursuer(
            out[j, :2], pursuer_pos, obstacles,
            safety_margin=safety_margin, uav_radius=uav_radius,
        ):
            diags.append({
                "slot_idx": j,
                "moved": False,
                "shift_m": 0.0,
                "theta_shift_deg": 0.0,
                "reason": "reachable",
            })
            continue
        out[j], diag = project_slot_on_manifold(
            j, nominal_targets, manifold_curve, pursuer_pos, evader_pos, obstacles, cfg,
            safety_margin=safety_margin, uav_radius=uav_radius,
            angular_window_deg=angular_window_deg,
        )
        diags.append(diag)
    return out, diags
