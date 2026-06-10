"""CSV and Markdown reporting for manifold debug runs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from debug_manifold.metrics import DEFAULT_THRESHOLDS


SUMMARY_FIELDS = [
    "case_id",
    "num_steps",
    "feasible_rate",
    "infeasible_rate",
    "invalid_output_rate",
    "closure_error_mean",
    "closure_error_max",
    "self_intersection_rate",
    "target_inside_rate",
    "boundary_violation_rate",
    "obstacle_penetration_rate",
    "min_clearance_p05",
    "min_boundary_margin_p05",
    "pointwise_shift_p95",
    "hausdorff_shift_p95",
    "curvature_p95",
    "num_failure_frames",
]


DIAGNOSTIC_TABLE = [
    (
        "Curve penetrates obstacle",
        "obstacle repulsion too weak or clearance not enforced",
        "inspect min_obstacle_clearance and obstacle_weight",
    ),
    (
        "Curve self-intersects",
        "local deformation too strong or topology not preserved",
        "inspect self_intersection_count and curvature_p95",
    ),
    (
        "Curve jumps suddenly",
        "manifold parameterization discontinuity or hard obstacle influence threshold",
        "inspect pointwise_shift_p95 and hausdorff_shift",
    ),
    (
        "Boundary violation",
        "generate-then-clip logic or radius too large",
        "inspect min_boundary_margin and boundary_violation_rate",
    ),
    (
        "Target not inside curve",
        "enclosure constraint broken by obstacle or boundary deformation",
        "inspect winding_number and target_inside",
    ),
    (
        "INFEASIBLE scene outputs malformed curve",
        "missing feasibility detection",
        "inspect infeasible_rate and invalid_output_rate",
    ),
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_case(case_id: str, rows: list[dict]) -> dict:
    n = len(rows)
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    return {
        "case_id": case_id,
        "num_steps": n,
        "feasible_rate": _rate(rows, lambda r: r.get("status") == "OK"),
        "infeasible_rate": _rate(rows, lambda r: r.get("status") == "INFEASIBLE"),
        "invalid_output_rate": _rate(rows, lambda r: r.get("status") == "INVALID" or bool(r.get("invalid_geometry_reason"))),
        "closure_error_mean": _mean(ok_rows, "closure_error"),
        "closure_error_max": _max(ok_rows, "closure_error"),
        "self_intersection_rate": _rate(ok_rows, lambda r: _float(r.get("self_intersection_count")) > 0.0),
        "target_inside_rate": _rate(ok_rows, lambda r: _float(r.get("target_inside")) >= 0.5),
        "boundary_violation_rate": _rate(ok_rows, lambda r: _float(r.get("boundary_violation_rate")) > 0.0),
        "obstacle_penetration_rate": _rate(ok_rows, lambda r: _float(r.get("obstacle_penetration_count")) > 0.0),
        "min_clearance_p05": _percentile(ok_rows, "min_obstacle_clearance", 5),
        "min_boundary_margin_p05": _percentile(ok_rows, "min_boundary_margin", 5),
        "pointwise_shift_p95": _percentile(ok_rows, "pointwise_shift_p95", 95),
        "hausdorff_shift_p95": _percentile(ok_rows, "hausdorff_shift", 95),
        "curvature_p95": _percentile(ok_rows, "curvature_p95", 95),
        "num_failure_frames": int(sum(1 for r in rows if str(r.get("failure_reasons", "")).strip())),
    }


def write_report(path: Path, case_infos: dict[str, dict], all_rows: list[dict], summary_rows: list[dict], thresholds: dict | None = None) -> None:
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = Counter()
    for row in all_rows:
        for reason in str(row.get("failure_reasons", "")).split(";"):
            reason = reason.strip()
            if reason:
                failures[reason] += 1

    lines: list[str] = []
    lines.append("# Manifold Debug Report")
    lines.append("")
    lines.append("This suite checks manifold geometry, obstacle response, boundary safety, and temporal continuity only. It does not evaluate capture rate or validate the SCE method.")
    lines.append("")
    lines.append("## Case Overview")
    lines.append("")
    lines.append("| case_id | num_steps | expected |")
    lines.append("|---|---:|---|")
    for case_id, info in case_infos.items():
        lines.append(f"| {case_id} | {info.get('num_steps')} | {info.get('expected')} |")

    lines.append("")
    lines.append("## Pass/Fail Summary")
    lines.append("")
    lines.append("| case_id | pass | feasible_rate | invalid_output_rate | self_intersection_rate | boundary_violation_rate | obstacle_penetration_rate | target_inside_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        passed = _summary_passed(row, th)
        lines.append(
            f"| {row['case_id']} | {passed} | {_fmt(row['feasible_rate'])} | {_fmt(row['invalid_output_rate'])} | "
            f"{_fmt(row['self_intersection_rate'])} | {_fmt(row['boundary_violation_rate'])} | "
            f"{_fmt(row['obstacle_penetration_rate'])} | {_fmt(row['target_inside_rate'])} |"
        )

    lines.append("")
    lines.append("## Most Common Failure Reasons")
    lines.append("")
    if failures:
        for reason, count in failures.most_common(10):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None recorded.")

    _worst_table(lines, "Worst 10 Timesteps By Min Obstacle Clearance", all_rows, "min_obstacle_clearance", reverse=False)
    _worst_table(lines, "Worst 10 Timesteps By Pointwise Shift P95", all_rows, "pointwise_shift_p95", reverse=True)
    _worst_table(lines, "Worst 10 Timesteps By Self Intersection Count", all_rows, "self_intersection_count", reverse=True)

    lines.append("")
    lines.append("## Diagnostic Interpretation")
    lines.append("")
    lines.append("| Observed failure | likely cause | what to inspect |")
    lines.append("|---|---|---|")
    for observed, cause, inspect in DIAGNOSTIC_TABLE:
        lines.append(f"| {observed} | {cause} | {inspect} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostic_table_markdown() -> str:
    lines = ["| Observed failure | likely cause | what to inspect |", "|---|---|---|"]
    for observed, cause, inspect in DIAGNOSTIC_TABLE:
        lines.append(f"| {observed} | {cause} | {inspect} |")
    return "\n".join(lines)


def _summary_passed(row: dict, thresholds: dict) -> bool:
    checks = [
        _float(row.get("closure_error_mean")) < float(thresholds["closure_error_mean_max"]),
        _float(row.get("self_intersection_rate")) <= float(thresholds["self_intersection_rate_max"]),
        _float(row.get("boundary_violation_rate")) <= float(thresholds["boundary_violation_rate_max"]),
        _float(row.get("obstacle_penetration_rate")) <= float(thresholds["obstacle_penetration_rate_max"]),
        _float(row.get("target_inside_rate")) >= float(thresholds["target_inside_rate_min"]),
        _float(row.get("invalid_output_rate")) <= float(thresholds["invalid_output_rate_max"]),
    ]
    return bool(all(checks))


def _worst_table(lines: list[str], title: str, rows: list[dict], key: str, *, reverse: bool) -> None:
    finite_rows = [r for r in rows if np.isfinite(_float(r.get(key)))]
    finite_rows.sort(key=lambda r: _float(r.get(key)), reverse=reverse)
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"| case_id | timestep | {key} | status | failure_reasons |")
    lines.append("|---|---:|---:|---|---|")
    for row in finite_rows[:10]:
        lines.append(
            f"| {row.get('case_id')} | {row.get('timestep')} | {_fmt(row.get(key))} | "
            f"{row.get('status')} | {row.get('failure_reasons', '')} |"
        )


def _rate(rows: Iterable[dict], pred) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    return float(sum(1 for r in rows if pred(r)) / len(rows))


def _values(rows: Iterable[dict], key: str) -> np.ndarray:
    vals = np.asarray([_float(r.get(key)) for r in rows], dtype=np.float64)
    return vals[np.isfinite(vals)]


def _mean(rows: Iterable[dict], key: str) -> float:
    vals = _values(rows, key)
    return float(np.mean(vals)) if vals.size else float("nan")


def _max(rows: Iterable[dict], key: str) -> float:
    vals = _values(rows, key)
    return float(np.max(vals)) if vals.size else float("nan")


def _percentile(rows: Iterable[dict], key: str, q: float) -> float:
    vals = _values(rows, key)
    return float(np.percentile(vals, q)) if vals.size else float("nan")


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value) -> str:
    v = _float(value)
    return f"{v:.4g}" if np.isfinite(v) else "nan"

