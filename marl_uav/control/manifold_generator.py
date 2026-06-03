"""Debug manifold generator for E2 trajectory planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ManifoldGeneratorConfig:
    curve_num_samples: int | None = None
    pursuer_path_samples: int = 32

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ManifoldGeneratorConfig":
        d = dict(raw or {})
        samples = d.get("curve_num_samples", None)
        return cls(
            curve_num_samples=None if samples is None else int(samples),
            pursuer_path_samples=max(int(d.get("pursuer_path_samples", cls.pursuer_path_samples)), 4),
        )


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _shortest_angle_delta(a0: float, a1: float) -> float:
    return _wrap_to_pi(float(a1) - float(a0))


def _dedupe_polyline(points: np.ndarray, *, tol: float = 1e-4) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts
    keep = [0]
    for i in range(1, pts.shape[0]):
        if float(np.linalg.norm(pts[i, :2] - pts[keep[-1], :2])) > tol:
            keep.append(i)
    if keep[-1] != pts.shape[0] - 1:
        if float(np.linalg.norm(pts[-1, :2] - pts[keep[-1], :2])) > tol:
            keep.append(pts.shape[0] - 1)
        else:
            keep[-1] = pts.shape[0] - 1
    return pts[np.asarray(keep, dtype=np.int64)]


def _pin_rho_through_pursuers(
    theta: np.ndarray,
    rho: np.ndarray,
    pursuer_pos: np.ndarray,
    evader_xy: np.ndarray,
) -> np.ndarray:
    """Raise rho at each pursuer bearing so the shared curve passes through all pursuers."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    rho_out = np.asarray(rho, dtype=np.float64).reshape(-1).copy()
    pursuers = np.asarray(pursuer_pos, dtype=np.float64).reshape(-1, 2)
    e = np.asarray(evader_xy, dtype=np.float64).reshape(2)
    for i in range(pursuers.shape[0]):
        rel = pursuers[i] - e
        ang = float(np.arctan2(rel[1], rel[0]))
        r_tgt = max(float(np.linalg.norm(rel)), 1e-6)
        dtheta = np.array([_shortest_angle_delta(float(t), ang) for t in theta], dtype=np.float64)
        idx = int(np.argmin(np.abs(dtheta)))
        rho_out[idx] = max(float(rho_out[idx]), r_tgt)
    return rho_out


def _radius_at_angle(theta: np.ndarray, rho: np.ndarray, ang: float) -> float:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    dtheta = np.array([_shortest_angle_delta(float(t), float(ang)) for t in theta], dtype=np.float64)
    idx = int(np.argmin(np.abs(dtheta)))
    return float(rho[idx])


def build_shared_manifold_curve(
    task: Any,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    task_state: Any | None,
    *,
    num_samples: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build the single team encirclement manifold around the evader.

    One closed curve for all pursuers. Base radius comes from
    ``task._compute_target_radius_xy`` (exponential contraction over elapsed steps);
    radial bumps from ``task._obstacle_aware_radius`` clear obstacles. Each pursuer
    bearing is pinned so the curve passes through that pursuer's current position.
    """
    if not hasattr(task, "_obstacle_aware_radius") or not hasattr(task, "_compute_target_radius_xy"):
        raise TypeError("task must provide _obstacle_aware_radius and _compute_target_radius_xy")

    pursuers = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float32).reshape(3)

    rho_base = float(task._compute_target_radius_xy(pursuers, e, task_state=task_state))
    elapsed = 0 if task_state is None else int(getattr(task_state, "elapsed_steps", 0))
    rho_min = float(getattr(task, "manifold_target_rho_min", 0.0))
    rho0 = float(getattr(task_state, "initial_mean_radius_xy", rho_base)) if task_state is not None else rho_base
    rho_max = max(rho0 * float(getattr(task, "manifold_target_radius_scale", 1.0)), rho_min)
    rate = float(getattr(task, "manifold_contraction_rate", 0.0))
    decay = float(np.exp(-rate * float(elapsed)))

    n = int(getattr(task, "manifold_curve_num_samples", 121))
    if num_samples is not None:
        n = max(int(num_samples), 16)
    phase = float(getattr(task, "manifold_target_phase", 0.0))
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True, dtype=np.float64) + phase
    rho = np.asarray(
        task._obstacle_aware_radius(
            theta.astype(np.float32),
            np.float32(rho_base),
            e,
            task_state=task_state,
        ),
        dtype=np.float64,
    )
    rho = _pin_rho_through_pursuers(theta, rho, pursuers[:, :2], e[:2])

    curve = np.zeros((theta.shape[0], 3), dtype=np.float64)
    curve[:, 0] = e[0] + rho * np.cos(theta)
    curve[:, 1] = e[1] + rho * np.sin(theta)
    curve[:, 2] = e[2]

    meta = {
        "rho_base": rho_base,
        "rho_max": float(rho_max),
        "rho_min": rho_min,
        "elapsed_steps": float(elapsed),
        "contraction_decay": decay,
        "manifold_contraction_rate": rate,
    }
    return curve, meta


@dataclass(frozen=True)
class ManifoldSignature:
    rho_base: float
    contraction_decay: float
    max_curve_displacement: float = 0.0


def manifold_max_displacement(
    old_curve: np.ndarray | None,
    new_curve: np.ndarray,
) -> float:
    if old_curve is None:
        return float("inf")
    old_pts = np.asarray(old_curve, dtype=np.float64).reshape(-1, 3)[:, :2]
    new_pts = np.asarray(new_curve, dtype=np.float64).reshape(-1, 3)[:, :2]
    n = min(old_pts.shape[0], new_pts.shape[0])
    if n <= 0:
        return float("inf")
    if old_pts.shape[0] != new_pts.shape[0]:
        old_pts = old_pts[:n]
        new_pts = new_pts[:n]
    return float(np.max(np.linalg.norm(new_pts - old_pts, axis=1)))


def build_manifold_signature(
    old_curve: np.ndarray | None,
    new_curve: np.ndarray,
    curve_meta: dict[str, float],
) -> ManifoldSignature:
    return ManifoldSignature(
        rho_base=float(curve_meta.get("rho_base", 0.0)),
        contraction_decay=float(curve_meta.get("contraction_decay", 1.0)),
        max_curve_displacement=manifold_max_displacement(old_curve, new_curve),
    )


def should_replan_manifold_paths(
    prev_sig: ManifoldSignature | None,
    new_sig: ManifoldSignature,
    prev_assignment: np.ndarray | None,
    new_assignment: np.ndarray,
    *,
    curve_tol: float = 0.05,
    rho_tol: float = 1e-3,
) -> bool:
    if prev_sig is None or prev_assignment is None:
        return True
    if not np.array_equal(np.asarray(prev_assignment, dtype=np.int64).reshape(3), np.asarray(new_assignment, dtype=np.int64).reshape(3)):
        return True
    if abs(float(new_sig.rho_base) - float(prev_sig.rho_base)) > float(rho_tol):
        return True
    if float(new_sig.max_curve_displacement) > float(curve_tol):
        return True
    return False


def _curve_point_at_angle(
    curve: np.ndarray,
    evader_xy: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Pick the manifold sample whose bearing from evader best matches ``angle``."""
    pts = np.asarray(curve, dtype=np.float64).reshape(-1, 3)
    e = np.asarray(evader_xy, dtype=np.float64).reshape(2)
    rel = pts[:, :2] - e[None, :]
    curve_angles = np.arctan2(rel[:, 1], rel[:, 0])
    diffs = np.abs(np.asarray([_wrap_to_pi(float(a) - float(angle)) for a in curve_angles]))
    idx = int(np.argmin(diffs))
    return pts[idx].copy()


def build_pursuer_manifold_path(
    pursuer_xy: np.ndarray,
    slot_xy: np.ndarray,
    manifold_curve: np.ndarray,
    *,
    evader_xy: np.ndarray,
    evader_z: float = 1.0,
    num_samples: int = 32,
) -> np.ndarray:
    """Build a pursuer flight path along the shared encirclement manifold."""
    p = np.asarray(pursuer_xy, dtype=np.float64).reshape(2)
    s = np.asarray(slot_xy, dtype=np.float64).reshape(3)
    curve = np.asarray(manifold_curve, dtype=np.float64).reshape(-1, 3)
    e = np.asarray(evader_xy, dtype=np.float64).reshape(2)
    z = float(s[2]) if s.shape[0] >= 3 else float(evader_z)
    n = max(int(num_samples), 4)

    if curve.shape[0] < 2:
        return np.array([[p[0], p[1], z], [s[0], s[1], z]], dtype=np.float64)

    ang_p = float(np.arctan2(p[1] - e[1], p[0] - e[0]))
    ang_s = float(np.arctan2(s[1] - e[1], s[0] - e[0]))
    delta = _shortest_angle_delta(ang_p, ang_s)

    ts = np.linspace(0.0, 1.0, n, dtype=np.float64)
    points = np.zeros((n, 3), dtype=np.float64)
    for k, t in enumerate(ts):
        if k == 0:
            points[k] = np.array([p[0], p[1], z], dtype=np.float64)
        elif k == n - 1:
            points[k] = np.array([s[0], s[1], z], dtype=np.float64)
        else:
            ang = ang_p + float(t) * delta
            on_curve = _curve_point_at_angle(curve, e, ang)
            points[k, :2] = on_curve[:2]
            points[k, 2] = z
    return _dedupe_polyline(points)


def build_pursuer_manifold_paths(
    pursuer_pos: np.ndarray,
    assigned_targets: np.ndarray,
    manifold_curve: np.ndarray,
    *,
    evader_xy: np.ndarray,
    num_samples: int = 32,
) -> list[np.ndarray]:
    """Build one tracking path per pursuer on the shared manifold."""
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    targets = np.asarray(assigned_targets, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_xy, dtype=np.float64).reshape(2)
    curve = np.asarray(manifold_curve, dtype=np.float64).reshape(-1, 3)
    return [
        build_pursuer_manifold_path(
            p[i, :2],
            targets[i],
            curve,
            evader_xy=e,
            evader_z=float(targets[i, 2]),
            num_samples=num_samples,
        )
        for i in range(3)
    ]


# Backward-compatible wrapper (returns curve only).
def build_anchored_manifold_curve(
    task: Any,
    pursuer_xy: np.ndarray,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    task_state: Any | None,
    *,
    num_samples: int | None = None,
) -> np.ndarray:
    del pursuer_xy
    curve, _meta = build_shared_manifold_curve(
        task, pursuer_pos, evader_pos, task_state, num_samples=num_samples
    )
    return curve


class ManifoldGenerator:
    """Single shared encirclement manifold + per-pursuer arc paths."""

    def __init__(self, cfg: dict[str, Any] | ManifoldGeneratorConfig | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, ManifoldGeneratorConfig) else ManifoldGeneratorConfig.from_dict(cfg)
        self._last_manifold_curve: np.ndarray | None = None

    def generate(
        self,
        task: Any,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        task_state: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        if not hasattr(task, "_reference_manifold_targets"):
            raise TypeError("trajectory_planner requires a pursuit task with reference manifold targets")

        p = np.asarray(pursuer_pos, dtype=np.float32).reshape(3, 3)
        e = np.asarray(evader_pos, dtype=np.float32).reshape(3)
        targets = np.asarray(
            task._reference_manifold_targets(p, e, task_state=task_state),
            dtype=np.float32,
        ).reshape(3, 3)

        curve, curve_meta = build_shared_manifold_curve(
            task,
            p,
            e,
            task_state,
            num_samples=self.cfg.curve_num_samples,
        )
        self._last_manifold_curve = curve

        radii = np.linalg.norm(targets[:, :2] - e[None, :2], axis=1)
        diag = {
            "target_radius_xy_mean": float(np.mean(radii)),
            "target_radius_xy_min": float(np.min(radii)),
            "target_radius_xy_max": float(np.max(radii)),
            "curve_num_samples": int(curve.shape[0]),
            **curve_meta,
        }
        return targets, curve.astype(np.float32), diag
    def generate_pursuer_paths(
        self,
        pursuer_pos: np.ndarray,
        assigned_targets: np.ndarray,
        global_curve: np.ndarray | None,
        *,
        evader_pos: np.ndarray | None = None,
        task: Any | None = None,
        task_state: Any | None = None,
    ) -> list[np.ndarray]:
        del task, task_state
        if evader_pos is None:
            raise ValueError("evader_pos is required to build pursuer manifold paths")

        e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
        num_samples = int(self.cfg.pursuer_path_samples)
        curve = self._last_manifold_curve
        if curve is None and global_curve is not None:
            curve = np.asarray(global_curve, dtype=np.float64).reshape(-1, 3)
        if curve is None:
            p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
            t = np.asarray(assigned_targets, dtype=np.float64).reshape(3, 3)
            return [
                np.array([[p[i, 0], p[i, 1], t[i, 2]], [t[i, 0], t[i, 1], t[i, 2]]], dtype=np.float64)
                for i in range(3)
            ]

        return build_pursuer_manifold_paths(
            pursuer_pos,
            assigned_targets,
            curve,
            evader_xy=e[:2],
            num_samples=num_samples,
        )

