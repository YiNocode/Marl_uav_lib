"""Unified adapter for manifold generators.

The suite first tries to route through the repository's lightweight manifold
curve builder. If that import or call fails, it records the reason and uses the
local fallback generator. The fallback is only infrastructure scaffolding, not
the research method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _SyntheticTaskState:
    elapsed_steps: int = 0
    initial_mean_radius_xy: float = 3.0


class _SyntheticTask:
    """Tiny task shim matching the repo manifold builder's expected methods."""

    def __init__(self, config: dict[str, Any], obstacles: list[dict]) -> None:
        self.base_radius = float(config.get("base_radius", 3.0))
        self.safe_margin = float(config.get("safe_margin", 0.4))
        self.obstacle_weight = float(config.get("obstacle_weight", 1.0))
        self.manifold_target_phase = 0.0
        self.manifold_curve_num_samples = int(config.get("K", 256))
        self.manifold_target_rho_min = max(0.2 * self.base_radius, 0.1)
        self.manifold_target_radius_scale = 1.0
        self.manifold_contraction_rate = 0.0
        self._obstacles = list(obstacles or [])

    def _compute_target_radius_xy(self, pursuers: np.ndarray, evader: np.ndarray, task_state: Any | None = None) -> float:
        del pursuers, evader, task_state
        return self.base_radius

    def _obstacle_aware_radius(
        self,
        theta: np.ndarray,
        rho_base: np.float32,
        evader_pos: np.ndarray,
        task_state: Any | None = None,
    ) -> np.ndarray:
        del task_state
        theta64 = np.asarray(theta, dtype=np.float64).reshape(-1)
        evader_xy = np.asarray(evader_pos, dtype=np.float64).reshape(-1)[:2]
        rho = np.full(theta64.shape, float(rho_base), dtype=np.float64)
        for obs in self._obstacles:
            c = np.asarray(obs["center"], dtype=np.float64).reshape(2)
            radius = float(obs["radius"])
            rel = c - evader_xy
            dist = max(float(np.linalg.norm(rel)), 1e-6)
            ang = float(np.arctan2(rel[1], rel[0]))
            desired = dist + radius + self.safe_margin
            if desired <= float(rho_base):
                desired = float(rho_base) + self.obstacle_weight * max(0.0, radius + self.safe_margin)
            delta = _angle_delta(theta64, ang)
            sigma = max(0.18, np.arcsin(min(0.95, (radius + self.safe_margin) / dist)) * 1.8)
            bump = np.exp(-0.5 * (delta / sigma) ** 2)
            rho = np.maximum(rho, float(rho_base) + (desired - float(rho_base)) * self.obstacle_weight * bump)
        return rho.astype(np.float32)


def _angle_delta(theta: np.ndarray, center: float) -> np.ndarray:
    return (np.asarray(theta, dtype=np.float64) - float(center) + np.pi) % (2.0 * np.pi) - np.pi


def _evader_xy(evader_state: Any) -> np.ndarray:
    if isinstance(evader_state, dict):
        if "pos" in evader_state:
            return np.asarray(evader_state["pos"], dtype=np.float64).reshape(-1)[:2]
        if "position" in evader_state:
            return np.asarray(evader_state["position"], dtype=np.float64).reshape(-1)[:2]
    return np.asarray(evader_state, dtype=np.float64).reshape(-1)[:2]


def _closed_circle_points(center: np.ndarray, radius: np.ndarray | float, K: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(K), endpoint=True, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    if r.ndim == 0:
        r = np.full(theta.shape, float(r), dtype=np.float64)
    pts = np.column_stack((center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)))
    return pts.astype(np.float64)


def _inside_obstacle_or_too_close(evader_xy: np.ndarray, obstacles: list[dict], safe_margin: float) -> tuple[bool, str | None]:
    for i, obs in enumerate(obstacles or []):
        c = np.asarray(obs["center"], dtype=np.float64).reshape(2)
        d = float(np.linalg.norm(c - evader_xy))
        if d <= float(obs["radius"]) + float(safe_margin):
            return True, f"evader within obstacle safety disk {i}"
    return False, None


def _fallback_generate(
    evader_state: Any,
    obstacles: list[dict],
    boundary: dict,
    config: dict[str, Any],
    prev_manifold: np.ndarray | None = None,
) -> dict:
    center = _evader_xy(evader_state)
    K = max(int(config.get("K", 256)), 16)
    base_radius = float(config.get("base_radius", 3.0))
    safe_margin = float(config.get("safe_margin", 0.4))
    obstacle_weight = float(config.get("obstacle_weight", 1.0))
    smooth_lambda = float(np.clip(config.get("smooth_lambda", 0.3), 0.0, 1.0))

    infeasible, reason = _inside_obstacle_or_too_close(center, obstacles, safe_margin)
    if infeasible:
        return {"status": "INFEASIBLE", "points": None, "meta": {"source": "fallback", "reason": reason}}

    theta = np.linspace(0.0, 2.0 * np.pi, K, endpoint=True, dtype=np.float64)
    rho = np.full(K, base_radius, dtype=np.float64)

    for obs in obstacles or []:
        c = np.asarray(obs["center"], dtype=np.float64).reshape(2)
        radius = float(obs["radius"])
        rel = c - center
        dist = max(float(np.linalg.norm(rel)), 1e-6)
        ang = float(np.arctan2(rel[1], rel[0]))
        delta = _angle_delta(theta, ang)
        sigma = max(0.16, np.arcsin(min(0.95, (radius + safe_margin) / dist)) * 1.8)
        desired = dist + radius + safe_margin
        bump_height = max(0.0, desired - base_radius)
        if abs(dist - base_radius) <= radius + safe_margin:
            bump_height = max(bump_height, radius + safe_margin)
        rho += obstacle_weight * bump_height * np.exp(-0.5 * (delta / sigma) ** 2)

    # Boundary deformation shrinks only the samples that would leave the arena.
    # This deliberately does not clip points, so later metrics can still expose
    # any remaining boundary violation.
    dirs = np.column_stack((np.cos(theta), np.sin(theta)))
    max_r = np.full(K, np.inf, dtype=np.float64)
    for i, d in enumerate(dirs):
        if d[0] > 1e-9:
            max_r[i] = min(max_r[i], (float(boundary["xmax"]) - safe_margin - center[0]) / d[0])
        elif d[0] < -1e-9:
            max_r[i] = min(max_r[i], (float(boundary["xmin"]) + safe_margin - center[0]) / d[0])
        if d[1] > 1e-9:
            max_r[i] = min(max_r[i], (float(boundary["ymax"]) - safe_margin - center[1]) / d[1])
        elif d[1] < -1e-9:
            max_r[i] = min(max_r[i], (float(boundary["ymin"]) + safe_margin - center[1]) / d[1])

    if np.nanmin(max_r) <= safe_margin:
        return {
            "status": "INFEASIBLE",
            "points": None,
            "meta": {"source": "fallback", "reason": "boundary leaves insufficient radius around evader"},
        }
    rho = np.minimum(rho, np.maximum(max_r, safe_margin))

    # A small circular convolution smooths abrupt radial bumps while preserving
    # closed periodic indexing.
    kernel = np.array([0.06, 0.18, 0.52, 0.18, 0.06], dtype=np.float64)
    for _ in range(3):
        rho = sum(kernel[j] * np.roll(rho, j - 2) for j in range(kernel.size))

    points = _closed_circle_points(center, rho, K)
    if prev_manifold is not None:
        prev = np.asarray(prev_manifold, dtype=np.float64)
        if prev.shape == points.shape:
            points = (1.0 - smooth_lambda) * points + smooth_lambda * prev

    return {"status": "OK", "points": points, "meta": {"source": "fallback"}}


def _try_existing_generator(
    evader_state: Any,
    obstacles: list[dict],
    config: dict[str, Any],
) -> dict:
    from marl_uav.control.manifold_generator import build_shared_manifold_curve

    center = _evader_xy(evader_state)
    K = max(int(config.get("K", 256)), 16)
    task = _SyntheticTask(config, obstacles)
    z = float(config.get("z", 1.0))
    pursuer_angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=np.float64)
    pursuers = np.zeros((3, 3), dtype=np.float32)
    pursuers[:, 0] = center[0] + float(config.get("base_radius", 3.0)) * np.cos(pursuer_angles)
    pursuers[:, 1] = center[1] + float(config.get("base_radius", 3.0)) * np.sin(pursuer_angles)
    pursuers[:, 2] = z
    evader_pos = np.array([center[0], center[1], z], dtype=np.float32)
    curve, meta = build_shared_manifold_curve(
        task,
        pursuers,
        evader_pos,
        _SyntheticTaskState(initial_mean_radius_xy=float(config.get("base_radius", 3.0))),
        num_samples=K,
    )
    return {
        "status": "OK",
        "points": np.asarray(curve, dtype=np.float64).reshape(-1, 3)[:, :2],
        "meta": {"source": "marl_uav.control.manifold_generator.build_shared_manifold_curve", **dict(meta)},
    }


def generate_manifold(
    evader_state: Any,
    obstacles: list[dict],
    boundary: dict,
    config: dict[str, Any],
    prev_manifold: np.ndarray | None = None,
) -> dict:
    """Generate a manifold with a stable debug-suite return contract."""
    cfg = dict(config or {})
    center = _evader_xy(evader_state)
    infeasible, reason = _inside_obstacle_or_too_close(center, obstacles, float(cfg.get("safe_margin", 0.4)))
    if infeasible:
        return {"status": "INFEASIBLE", "points": None, "meta": {"source": "adapter_feasibility_check", "reason": reason}}
    if not bool(cfg.get("force_fallback", False)):
        try:
            out = _try_existing_generator(evader_state, obstacles, cfg)
            if prev_manifold is not None and out.get("status") == "OK":
                smooth_lambda = float(np.clip(cfg.get("smooth_lambda", 0.3), 0.0, 1.0))
                pts = np.asarray(out["points"], dtype=np.float64)
                prev = np.asarray(prev_manifold, dtype=np.float64)
                if pts.shape == prev.shape:
                    out["points"] = (1.0 - smooth_lambda) * pts + smooth_lambda * prev
                    out["meta"]["temporal_smoothing_applied"] = True
            return out
        except Exception as exc:  # noqa: BLE001 - diagnostic adapter must not crash the suite.
            fallback = _fallback_generate(evader_state, obstacles, boundary, cfg, prev_manifold)
            fallback.setdefault("meta", {})["existing_generator_error"] = repr(exc)
            return fallback
    return _fallback_generate(evader_state, obstacles, boundary, cfg, prev_manifold)
