"""Geometry metrics for closed 2D encirclement manifolds."""

from __future__ import annotations

from typing import Any

import numpy as np


METRIC_FIELDS = [
    "closure_error",
    "polygon_area_signed",
    "polygon_area",
    "target_inside",
    "winding_number_about_evader",
    "self_intersection_count",
    "boundary_violation_rate",
    "min_boundary_margin",
    "min_obstacle_clearance",
    "obstacle_penetration_count",
    "arc_length_cv",
    "curvature_mean",
    "curvature_max",
    "curvature_p95",
    "pointwise_shift_mean",
    "pointwise_shift_p95",
    "hausdorff_shift",
    "area_change_rate",
    "curvature_change_mean",
]


def _nan_metrics() -> dict[str, float]:
    out = {name: float("nan") for name in METRIC_FIELDS}
    out["target_inside"] = 0.0
    out["self_intersection_count"] = float("nan")
    out["boundary_violation_rate"] = float("nan")
    out["obstacle_penetration_count"] = float("nan")
    return out


def validate_points(points: Any) -> tuple[np.ndarray | None, str | None]:
    if points is None:
        return None, "points is None"
    try:
        pts = np.asarray(points, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - caller needs diagnostic text.
        return None, f"points cannot be converted to array: {exc!r}"
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None, f"points must have shape [K, 2], got {pts.shape}"
    if pts.shape[0] < 4:
        return None, f"points must contain at least 4 samples, got {pts.shape[0]}"
    if not np.all(np.isfinite(pts)):
        return None, "points contain NaN or inf"
    return pts, None


def polygon_area_signed(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    return float(0.5 * np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]))


def cyclic_unique_points(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] > 1 and float(np.linalg.norm(pts[0] - pts[-1])) <= float(tol):
        return pts[:-1]
    return pts


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test for a closed cyclic sample list.

    A horizontal ray is fired from ``point`` to +x. Every segment crossing the
    ray toggles the inside flag. The small epsilon in the denominator protects
    horizontal edges without moving or clipping the input geometry.
    """
    p = np.asarray(point, dtype=np.float64).reshape(2)
    poly = np.asarray(polygon, dtype=np.float64)
    x, y = float(p[0]), float(p[1])
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x0, y0 = float(poly[i, 0]), float(poly[i, 1])
        x1, y1 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-15) + x0):
            inside = not inside
    return bool(inside)


def winding_number_about_point(point: np.ndarray, polygon: np.ndarray) -> float:
    """Approximate winding number from summed wrapped bearing changes.

    For each cyclic edge, the bearing from the query point to the next vertex is
    compared to the previous bearing. Wrapping each delta to [-pi, pi] prevents
    a branch-cut jump from looking like a geometric discontinuity.
    """
    p = np.asarray(point, dtype=np.float64).reshape(2)
    rel = np.asarray(polygon, dtype=np.float64) - p[None, :]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    next_angles = np.roll(angles, -1)
    deltas = (next_angles - angles + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sum(deltas) / (2.0 * np.pi))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-10) -> bool:
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
    )


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Robust-enough segment intersection with collinear endpoint handling."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    eps = 1e-10
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and _on_segment(a, c, b):
        return True
    if abs(o2) <= eps and _on_segment(a, d, b):
        return True
    if abs(o3) <= eps and _on_segment(c, a, d):
        return True
    if abs(o4) <= eps and _on_segment(c, b, d):
        return True
    return False


def self_intersection_count(points: np.ndarray) -> int:
    """Count intersections between non-adjacent cyclic polygon segments."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    count = 0
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c = pts[j]
            d = pts[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                count += 1
    return int(count)


def boundary_metrics(points: np.ndarray, boundary: dict) -> tuple[float, float]:
    pts = np.asarray(points, dtype=np.float64)
    xmin, xmax = float(boundary["xmin"]), float(boundary["xmax"])
    ymin, ymax = float(boundary["ymin"]), float(boundary["ymax"])
    margins = np.column_stack((pts[:, 0] - xmin, xmax - pts[:, 0], pts[:, 1] - ymin, ymax - pts[:, 1]))
    min_margin_each = np.min(margins, axis=1)
    return float(np.mean(min_margin_each < 0.0)), float(np.min(min_margin_each))


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-15:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def obstacle_clearance_metrics(points: np.ndarray, obstacles: list[dict]) -> tuple[float, int]:
    """Minimum circle clearance using vertices and cyclic segments.

    For every circular obstacle, the distance from its center to all manifold
    vertices and all manifold segments is checked. The obstacle radius is then
    subtracted, so negative values are penetrations.
    """
    if not obstacles:
        return float("inf"), 0
    pts = np.asarray(points, dtype=np.float64)
    clearances: list[float] = []
    n = pts.shape[0]
    for obs in obstacles:
        c = np.asarray(obs["center"], dtype=np.float64).reshape(2)
        radius = float(obs["radius"])
        vertex_min = float(np.min(np.linalg.norm(pts - c[None, :], axis=1)))
        segment_min = min(_point_segment_distance(c, pts[i], pts[(i + 1) % n]) for i in range(n))
        clearances.append(min(vertex_min, segment_min) - radius)
    arr = np.asarray(clearances, dtype=np.float64)
    return float(np.min(arr)), int(np.sum(arr < 0.0))


def arc_length_cv(points: np.ndarray) -> float:
    seg = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    mean = float(np.mean(seg))
    if mean <= 1e-15:
        return float("nan")
    return float(np.std(seg) / mean)


def discrete_curvature(points: np.ndarray) -> np.ndarray:
    """Estimate curvature from cyclic turning angle over local arc length.

    At each sample, two neighboring segment vectors define a signed turn angle.
    Dividing its magnitude by the average adjacent segment length yields a
    stable discrete curvature proxy; degenerate zero-length neighborhoods are
    assigned zero curvature to avoid divide-by-zero explosions.
    """
    pts = np.asarray(points, dtype=np.float64)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    v0 = pts - prev_pts
    v1 = next_pts - pts
    l0 = np.linalg.norm(v0, axis=1)
    l1 = np.linalg.norm(v1, axis=1)
    cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    dot = np.sum(v0 * v1, axis=1)
    turn = np.abs(np.arctan2(cross, dot))
    denom = 0.5 * (l0 + l1)
    curv = np.zeros_like(turn)
    mask = denom > 1e-12
    curv[mask] = turn[mask] / denom[mask]
    return curv


def _resample_by_index(points: np.ndarray, target_n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == target_n:
        return pts
    src = np.linspace(0.0, 1.0, pts.shape[0], endpoint=False)
    dst = np.linspace(0.0, 1.0, target_n, endpoint=False)
    src_ext = np.r_[src, 1.0]
    x_ext = np.r_[pts[:, 0], pts[0, 0]]
    y_ext = np.r_[pts[:, 1], pts[0, 1]]
    return np.column_stack((np.interp(dst, src_ext, x_ext), np.interp(dst, src_ext, y_ext)))


def _resample_scalar_by_index(values: np.ndarray, target_n: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.shape[0] == target_n:
        return vals
    src = np.linspace(0.0, 1.0, vals.shape[0], endpoint=False)
    dst = np.linspace(0.0, 1.0, target_n, endpoint=False)
    src_ext = np.r_[src, 1.0]
    vals_ext = np.r_[vals, vals[0]]
    return np.interp(dst, src_ext, vals_ext)


def temporal_shift_metrics(points: np.ndarray, prev_points: np.ndarray | None) -> tuple[float, float, float]:
    """Corresponding-index and approximate symmetric Hausdorff shifts."""
    if prev_points is None:
        return float("nan"), float("nan"), float("nan")
    prev = _resample_by_index(np.asarray(prev_points, dtype=np.float64), points.shape[0])
    d = np.linalg.norm(points - prev, axis=1)
    # For debugging scale, the dense O(K^2) distance matrix is simple and fine.
    dist = np.linalg.norm(points[:, None, :] - prev[None, :, :], axis=2)
    haus = max(float(np.max(np.min(dist, axis=1))), float(np.max(np.min(dist, axis=0))))
    return float(np.mean(d)), float(np.percentile(d, 95)), haus


def compute_metrics(
    status: str,
    points: Any,
    evader_state: Any,
    obstacles: list[dict],
    boundary: dict,
    *,
    prev_points: np.ndarray | None = None,
    prev_area: float | None = None,
    prev_curvature: np.ndarray | None = None,
    dt: float = 0.1,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None, str | None]:
    metrics = _nan_metrics()
    if status != "OK":
        return metrics, None, None, None

    pts, invalid_reason = validate_points(points)
    if pts is None:
        return metrics, None, None, invalid_reason

    closure = float(np.linalg.norm(pts[0] - pts[-1]))
    geom_pts = cyclic_unique_points(pts)
    evader = np.asarray(evader_state, dtype=np.float64).reshape(-1)[:2]
    area_signed = polygon_area_signed(geom_pts)
    curv = discrete_curvature(geom_pts)
    shift_mean, shift_p95, haus = temporal_shift_metrics(pts, prev_points)
    violation_rate, min_margin = boundary_metrics(pts, boundary)
    min_clearance, penetration_count = obstacle_clearance_metrics(geom_pts, obstacles)

    metrics.update(
        {
            "closure_error": closure,
            "polygon_area_signed": area_signed,
            "polygon_area": abs(area_signed),
            "target_inside": 1.0 if point_in_polygon(evader, geom_pts) else 0.0,
            "winding_number_about_evader": winding_number_about_point(evader, geom_pts),
            "self_intersection_count": float(self_intersection_count(geom_pts)),
            "boundary_violation_rate": violation_rate,
            "min_boundary_margin": min_margin,
            "min_obstacle_clearance": min_clearance,
            "obstacle_penetration_count": float(penetration_count),
            "arc_length_cv": arc_length_cv(pts),
            "curvature_mean": float(np.mean(curv)),
            "curvature_max": float(np.max(curv)),
            "curvature_p95": float(np.percentile(curv, 95)),
            "pointwise_shift_mean": shift_mean,
            "pointwise_shift_p95": shift_p95,
            "hausdorff_shift": haus,
            "area_change_rate": float(abs(abs(area_signed) - prev_area) / max(float(dt), 1e-12))
            if prev_area is not None
            else float("nan"),
            "curvature_change_mean": float(
                np.mean(np.abs(curv - _resample_scalar_by_index(prev_curvature, curv.shape[0])))
            )
            if prev_curvature is not None
            else float("nan"),
        }
    )
    return metrics, pts, curv, None


DEFAULT_THRESHOLDS = {
    "closure_error_mean_max": 1e-3,
    "self_intersection_rate_max": 0.0,
    "boundary_violation_rate_max": 0.0,
    "obstacle_penetration_rate_max": 0.0,
    "target_inside_rate_min": 0.99,
    "winding_number_abs_error_max": 0.05,
    "invalid_output_rate_max": 0.0,
    "pointwise_shift_p95_failure": 1.0,
}


def failure_reasons(row: dict, thresholds: dict | None = None) -> list[str]:
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    reasons: list[str] = []
    status = str(row.get("status", ""))
    if status == "INVALID":
        reasons.append("generator returned INVALID")
    if status == "INFEASIBLE":
        reasons.append("generator returned INFEASIBLE")
    if _finite_lt(row.get("min_obstacle_clearance"), 0.0):
        reasons.append("min_obstacle_clearance < 0")
    if _finite_gt(row.get("boundary_violation_rate"), 0.0):
        reasons.append("boundary_violation_rate > 0")
    if _finite_gt(row.get("self_intersection_count"), 0.0):
        reasons.append("self_intersection_count > 0")
    if status == "OK" and float(row.get("target_inside", 0.0) or 0.0) < 0.5:
        reasons.append("target_inside is False while status is OK")
    if _finite_gt(row.get("pointwise_shift_p95"), float(th["pointwise_shift_p95_failure"])):
        reasons.append("pointwise_shift_p95 exceeds threshold")
    if row.get("invalid_geometry_reason"):
        reasons.append(str(row["invalid_geometry_reason"]))
    return reasons


def _finite_lt(value: Any, threshold: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v < threshold)


def _finite_gt(value: Any, threshold: float) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > threshold)
