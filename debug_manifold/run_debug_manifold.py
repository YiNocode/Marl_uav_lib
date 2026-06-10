"""CLI entry point for the lightweight manifold-generation debug suite."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from debug_manifold.cases import DEFAULT_BOUNDARY, list_case_ids, make_case, resolve_case_id  # noqa: E402
from debug_manifold.manifold_adapter import generate_manifold  # noqa: E402
from debug_manifold.metrics import DEFAULT_THRESHOLDS, compute_metrics, failure_reasons  # noqa: E402
from debug_manifold.report import SUMMARY_FIELDS, summarize_case, write_csv, write_report  # noqa: E402
from debug_manifold.visualize import plot_case_overlay, plot_metric_timeseries, save_failure_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug closed encirclement manifold generation.")
    parser.add_argument("--case", required=True, help="Case id, short alias g0..g5, or all.")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--base-radius", type=float, default=3.0)
    parser.add_argument("--safe-margin", type=float, default=0.4)
    parser.add_argument("--obstacle-weight", type=float, default=1.0)
    parser.add_argument("--smooth-lambda", type=float, default=0.3)
    parser.add_argument("--output-dir", default="debug_manifold/outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    figures_dir = run_dir / "figures"
    csv_dir = run_dir / "csv"
    failures_dir = run_dir / "failure_cases"
    for path in (figures_dir, csv_dir, failures_dir):
        path.mkdir(parents=True, exist_ok=True)

    case_ids = list_case_ids() if str(args.case).lower() == "all" else [resolve_case_id(args.case)]
    config = {
        "K": int(args.K),
        "dt": float(args.dt),
        "base_radius": float(args.base_radius),
        "safe_margin": float(args.safe_margin),
        "obstacle_weight": float(args.obstacle_weight),
        "smooth_lambda": float(args.smooth_lambda),
        "seed": int(args.seed),
    }

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    case_infos: dict[str, dict] = {}

    for case_id in case_ids:
        case = make_case(case_id, args.num_steps, seed=args.seed, boundary=DEFAULT_BOUNDARY)
        print(f"[debug_manifold] running {case.case_id} ({len(case.steps)} steps)")
        rows, frames = run_case(case, config, run_dir, figures_dir, failures_dir, fail_fast=bool(args.fail_fast))
        write_csv(csv_dir / f"{case.case_id}_metrics.csv", rows)
        plot_case_overlay(case.case_id, frames, figures_dir / f"{case.case_id}_overlay.png")
        plot_metric_timeseries(case.case_id, rows, figures_dir / f"{case.case_id}_metrics.png")

        all_rows.extend(rows)
        summary = summarize_case(case.case_id, rows)
        summary_rows.append(summary)
        case_infos[case.case_id] = {
            "num_steps": len(case.steps),
            "description": case.description,
            "expected": case.expected,
        }

    write_csv(run_dir / "metrics_timeseries.csv", all_rows)
    write_csv(run_dir / "summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    write_report(run_dir / "report.md", case_infos, all_rows, summary_rows, DEFAULT_THRESHOLDS)
    write_run_config(run_dir / "run_config.json", args, config, case_ids)
    print(f"[debug_manifold] outputs saved to {run_dir}")
    return 0


def run_case(case, config: dict[str, Any], run_dir: Path, figures_dir: Path, failures_dir: Path, *, fail_fast: bool) -> tuple[list[dict], list[dict]]:
    del run_dir, figures_dir
    rows: list[dict] = []
    frames: list[dict] = []
    prev_points: np.ndarray | None = None
    prev_area: float | None = None
    prev_curvature: np.ndarray | None = None
    failure_count = 0

    for t, step in enumerate(case.steps):
        result = generate_manifold(
            step["evader_state"],
            step["obstacles"],
            step["boundary"],
            config,
            prev_manifold=prev_points,
        )
        raw_status = str(result.get("status", "INVALID"))
        metrics, valid_points, curv, invalid_reason = compute_metrics(
            raw_status,
            result.get("points"),
            step["evader_state"],
            step["obstacles"],
            step["boundary"],
            prev_points=prev_points,
            prev_area=prev_area,
            prev_curvature=prev_curvature,
            dt=float(config.get("dt", 0.1)),
        )
        status = raw_status
        if raw_status == "OK" and invalid_reason:
            status = "INVALID"
        meta = dict(result.get("meta") or {})
        row = {
            "case_id": case.case_id,
            "timestep": int(t),
            "time": float(t) * float(config.get("dt", 0.1)),
            "status": status,
            "generator_source": meta.get("source", ""),
            "generator_reason": meta.get("reason", ""),
            "existing_generator_error": meta.get("existing_generator_error", ""),
            "invalid_geometry_reason": invalid_reason or "",
            **metrics,
        }
        reasons = failure_reasons(row)
        row["failure_reasons"] = "; ".join(reasons)
        rows.append(row)

        frame = {
            "case_id": case.case_id,
            "timestep": int(t),
            "evader_state": np.asarray(step["evader_state"], dtype=np.float64).copy(),
            "obstacles": step["obstacles"],
            "boundary": step["boundary"],
            "points": None if result.get("points") is None else np.asarray(result.get("points"), dtype=np.float64).copy(),
            "status": status,
        }
        frames.append(frame)

        if reasons:
            failure_count += 1
            snapshot = failures_dir / f"{case.case_id}_t{t:04d}_{_safe_reason(reasons[0])}.png"
            save_failure_snapshot(case.case_id, frame, row, reasons, snapshot)
            if fail_fast:
                raise RuntimeError(f"{case.case_id} timestep {t} failed: {'; '.join(reasons)}")

        if status == "OK" and valid_points is not None:
            prev_points = valid_points.copy()
            prev_area = float(metrics["polygon_area"])
            prev_curvature = None if curv is None else curv.copy()

    print(f"[debug_manifold] {case.case_id}: {failure_count} failure frames")
    return rows, frames


def write_run_config(path: Path, args: argparse.Namespace, config: dict[str, Any], case_ids: list[str]) -> None:
    payload = {
        "args": vars(args),
        "resolved_cases": case_ids,
        "config": config,
        "thresholds": DEFAULT_THRESHOLDS,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_reason(reason: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in reason.lower()).strip("_")
    return cleaned[:60] or "failure"


if __name__ == "__main__":
    raise SystemExit(main())

