"""Summarize ex1 reward-grid eval results and sort by capture rate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute ex1 reward-grid summaries from an existing manifest."
    )
    parser.add_argument(
        "--search-config",
        type=str,
        default=str((Path("configs") / "search" / "ex1_reward_grid.yaml")),
        help="Reward-grid search config used to generate the manifest.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke the summary backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        args.python,
        "scripts/grid_search_ex1_rewards.py",
        "--search-config",
        args.search_config,
        "--summarize-only",
    ]
    print(f"[summary] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
