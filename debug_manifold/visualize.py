"""Matplotlib visualizations for manifold debug runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_PLOTS = [
    "min_obstacle_clearance",
    "min_boundary_margin",
    "closure_error",
    "self_intersection_count",
    "target_inside",
    "pointwise_shift_p95",
    "hausdorff_shift",
    "curvature_p95",
]


def _draw_boundary(ax, boundary: dict) -> None:
    xmin, xmax = float(boundary["xmin"]), float(boundary["xmax"])
    ymin, ymax = float(boundary["ymin"]), float(boundary["ymax"])
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color="black", linewidth=1.2)
    ax.set_xlim(xmin - 0.75, xmax + 0.75)
    ax.set_ylim(ymin - 0.75, ymax + 0.75)
    ax.set_aspect("equal", adjustable="box")


def _draw_obstacles(ax, obstacles: list[dict], *, alpha: float = 0.35) -> None:
    for obs in obstacles or []:
        c = np.asarray(obs["center"], dtype=np.float64).reshape(2)
        r = float(obs["radius"])
        patch = plt.Circle((c[0], c[1]), r, facecolor="#d55e00", edgecolor="#8b2f00", alpha=alpha, linewidth=1.0)
        ax.add_patch(patch)


def plot_case_overlay(case_id: str, frames: list[dict], out_path: Path, stride: int | None = None) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if stride is None:
        stride = max(1, len(frames) // 20)
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=140)
    _draw_boundary(ax, frames[0]["boundary"])
    ev = np.asarray([f["evader_state"] for f in frames], dtype=np.float64)
    ax.plot(ev[:, 0], ev[:, 1], color="#0072b2", linewidth=1.8, label="evader trajectory")
    _draw_obstacles(ax, frames[-1].get("obstacles", []), alpha=0.30)

    for idx in range(0, len(frames), stride):
        pts = frames[idx].get("points")
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        ax.plot(pts[:, 0], pts[:, 1], color="#999999", linewidth=0.7, alpha=0.35)

    final = frames[-1].get("points")
    if final is not None:
        final = np.asarray(final, dtype=np.float64)
        ax.plot(final[:, 0], final[:, 1], color="#009e73", linewidth=2.0, label="final manifold")
    ax.scatter(ev[-1, 0], ev[-1, 1], color="#0072b2", s=30, zorder=4)
    ax.set_title(f"{case_id} overlay")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_timeseries(case_id: str, rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray([float(r["timestep"]) for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(4, 2, figsize=(10, 9), dpi=140, sharex=True)
    axes = axes.reshape(-1)
    for ax, name in zip(axes, METRIC_PLOTS):
        vals = np.asarray([_to_float(r.get(name)) for r in rows], dtype=np.float64)
        ax.plot(t, vals, color="#005f73", linewidth=1.2)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("timestep")
    axes[-2].set_xlabel("timestep")
    fig.suptitle(f"{case_id} metric time series", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_failure_snapshot(case_id: str, frame: dict, row: dict, reasons: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=140)
    _draw_boundary(ax, frame["boundary"])
    _draw_obstacles(ax, frame.get("obstacles", []), alpha=0.40)
    pts = frame.get("points")
    if pts is not None:
        pts = np.asarray(pts, dtype=np.float64)
        ax.plot(pts[:, 0], pts[:, 1], color="#009e73", linewidth=2.0)
    ev = np.asarray(frame["evader_state"], dtype=np.float64).reshape(2)
    ax.scatter(ev[0], ev[1], color="#0072b2", s=36, zorder=4)
    reason_text = "; ".join(reasons[:3])
    metric_text = (
        f"case={case_id}\n"
        f"timestep={row.get('timestep')}\n"
        f"status={row.get('status')}\n"
        f"reason={reason_text}\n"
        f"clearance={_fmt(row.get('min_obstacle_clearance'))}\n"
        f"boundary_margin={_fmt(row.get('min_boundary_margin'))}\n"
        f"self_intersections={_fmt(row.get('self_intersection_count'))}\n"
        f"shift_p95={_fmt(row.get('pointwise_shift_p95'))}"
    )
    ax.text(
        0.02,
        0.98,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#555555", "alpha": 0.88},
    )
    ax.set_title(f"{case_id} failure frame")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value) -> str:
    v = _to_float(value)
    if np.isfinite(v):
        return f"{v:.4g}"
    return "nan"

