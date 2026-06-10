"""Plot slot-tracking benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.slot_tracking.run_slot_tracking_benchmark import generate_existing_debug_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_trajectories", type=int, default=12)
    return parser.parse_args()


def _save_bar(df: pd.DataFrame, column: str, output: Path, ylabel: str) -> None:
    pivot = df.pivot_table(index="scenario_group", columns="controller", values=column, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("scenario group")
    ax.legend(title="controller")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _save_box(df: pd.DataFrame, column: str, output: Path, ylabel: str) -> None:
    controllers = sorted(df["controller"].unique())
    data = [df[df["controller"] == c][column].dropna().to_numpy() for c in controllers]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=controllers, showfliers=False)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _plot_raw_file(path: Path, output_dir: Path) -> None:
    raw = pd.read_csv(path)
    plt.figure(figsize=(7, 7))
    for agent_id, part in raw.groupby("agent_id"):
        plt.plot(part["uav_x"], part["uav_y"], label=f"uav {agent_id}")
        plt.plot(part["slot_x"], part["slot_y"], "--", label=f"slot {agent_id}")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"trajectory_{path.stem}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    for agent_id, part in raw.groupby("agent_id"):
        plt.plot(part["t"], part["tracking_error"], label=f"agent {agent_id}")
    plt.xlabel("time [s]")
    plt.ylabel("tracking error [m]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"tracking_error_{path.stem}.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_dir / "summary" / "episode_metrics.csv")
    work = df.copy()
    for col in ["success", "obstacle_collision", "boundary_violation"]:
        work[col] = work[col].astype(float)

    _save_bar(work, "success", output_dir / "success_rate_by_group_controller.png", "success rate")
    _save_box(work, "p95_error", output_dir / "p95_tracking_error_boxplot.png", "P95 tracking error [m]")
    _save_bar(work, "obstacle_collision", output_dir / "obstacle_collision_rate.png", "collision rate")
    _save_bar(work, "boundary_violation", output_dir / "boundary_violation_rate.png", "boundary violation rate")
    _save_box(work, "decision_time_ms_p95", output_dir / "decision_time_p95_distribution.png", "decision time p95 [ms]")

    raw_dir = input_dir / "raw"
    for path in sorted(raw_dir.glob("*.csv"))[: max(int(args.max_trajectories), 0)]:
        _plot_raw_file(path, output_dir)
    generate_existing_debug_plots(work, input_dir, max_plots=max(int(args.max_trajectories), 0))
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
