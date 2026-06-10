"""E4 SCE component ablation benchmark across E1/E2/E3 scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_e2_obstacles import load_suite, run_evaluation, selected_methods


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E4 SCE ablation PyFlyt benchmark.")
    p.add_argument(
        "--suite-config",
        type=str,
        default="configs/benchmark/e4_sce_ablation_suite.yaml",
    )
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-seeds", nargs="*", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_suite(ROOT / args.suite_config)
    methods = selected_methods(suite, args.methods)
    run_evaluation(
        suite,
        methods,
        episodes_override=args.episodes,
        eval_seeds_override=args.eval_seeds,
    )


if __name__ == "__main__":
    main()
