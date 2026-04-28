"""Recompute summaries for the ex1 evader-speed sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute ex1 evader-speed summaries from an existing manifest."
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=str((Path("configs") / "search" / "ex1_evader_speed_sweep.yaml")),
        help="Evader-speed sweep config used to generate the manifest.",
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
        "scripts/sweep_ex1_evader_speed.py",
        "--sweep-config",
        args.sweep_config,
        "--summarize-only",
    ]
    print(f"[summary] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
