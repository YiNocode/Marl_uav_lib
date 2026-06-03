"""BC baseline metrics and fine-tune regression flags."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def compute_regression_metrics(
    *,
    current: dict[str, float],
    baseline: dict[str, float],
    min_bc_retention: float = 0.95,
) -> dict[str, Any]:
    """Compare current eval to BC baseline; set regression_flag if capture drops too much."""
    bc_cap = float(baseline.get("capture_rate", baseline.get("bc_eval/capture_rate", 0.0)))
    cur_cap = float(current.get("capture_rate", current.get("bc_eval/capture_rate", 0.0)))
    bc_col = float(baseline.get("collision_rate", baseline.get("bc_eval/collision_rate", 0.0)))
    cur_col = float(current.get("collision_rate", current.get("bc_eval/collision_rate", 0.0)))
    bc_struct = float(
        baseline.get(
            f"D_ang_last30",
            baseline.get("mean_D_ang_last30", baseline.get("bc_eval/D_ang_last30", 0.0)),
        )
    )
    cur_struct = float(
        current.get(
            f"D_ang_last30",
            current.get("mean_D_ang_last30", current.get("bc_eval/D_ang_last30", 0.0)),
        )
    )

    capture_drop = bc_cap - cur_cap if bc_cap > 0 else 0.0
    collision_increase = cur_col - bc_col
    structure_drop = bc_struct - cur_struct if bc_struct > 0 else 0.0

    regression = False
    if bc_cap > 0 and cur_cap < bc_cap * min_bc_retention:
        regression = True

    return {
        "best_eval_capture_rate": max(bc_cap, cur_cap),
        "current_eval_capture_rate": cur_cap,
        "capture_drop_from_bc": float(capture_drop),
        "collision_increase_from_bc": float(collision_increase),
        "structure_drop_from_bc": float(structure_drop),
        "regression_flag": int(regression),
        "min_bc_retention": float(min_bc_retention),
    }


def append_train_log_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    fieldnames = list(row.keys())
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for k in reader.fieldnames:
                    if k not in fieldnames:
                        fieldnames.append(k)
                for k in row:
                    if k not in fieldnames:
                        fieldnames.append(k)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
