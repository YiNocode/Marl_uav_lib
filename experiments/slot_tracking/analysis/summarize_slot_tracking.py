"""Summarize slot-tracking benchmark outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.slot_tracking.run_slot_tracking_benchmark import (
    a_group_failure_counts,
    a_group_subscenario_summary,
    aggregate_summary,
    boundary_failure_subtype_counts,
    feasible_only_summary,
    infeasible_stress_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    metrics_path = input_dir / "summary" / "episode_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing episode metrics: {metrics_path}")
    df = pd.read_csv(metrics_path)
    summary = aggregate_summary(df)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    feasible = feasible_only_summary(df, {})
    if not feasible.empty:
        feasible.to_csv(out.parent / "feasible_only_summary.csv", index=False)
    a_summary = a_group_subscenario_summary(df)
    if not a_summary.empty:
        a_summary.to_csv(out.parent / "a_group_subscenario_summary.csv", index=False)
    a_failures = a_group_failure_counts(df)
    if not a_failures.empty:
        a_failures.to_csv(out.parent / "a_group_failure_counts.csv", index=False)
    infeasible = infeasible_stress_summary(df)
    if not infeasible.empty:
        infeasible.to_csv(out.parent / "infeasible_stress_summary.csv", index=False)
    boundary = boundary_failure_subtype_counts(df)
    if not boundary.empty:
        boundary.to_csv(out.parent / "boundary_failure_subtype_counts.csv", index=False)
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
