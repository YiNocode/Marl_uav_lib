"""Generate and optionally launch 3v1 pursuit-evasion training runs for the server.

This script covers three experiment groups:
1. speed_ratio: base task with multiple speed ratios and multiple seeds
2. role_assignment: ex1 task
3. obstacles: ex2 task

Default assumption for experiment 1:
- ratio strings like "1:2" mean evader_speed : pursuer_speed = 1 : 2
- evader speed stays fixed, pursuer speed is derived from the ratio
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "configs" / "generated"


def _base_train_cfg() -> dict[str, Any]:
    return {
        "env": "configs/env/pyflyt_3v1.yaml",
        "algo": "configs/algo/mappo.yaml",
        "model": "configs/model/centralized_critic.yaml",
        "task": {
            "name": "pursuit_evasion_3v1",
            "world_xy": 20.0,
            "z_min": 0.5,
            "z_max": 5.0,
            "capture_dist": 1.0,
            "episode_limit": 1000,
            "pursuer_speed": 0.20,
            "evader_speed": 0.06,
            "min_pursuer_sep": 0.6,
            "progress_reward_scale": 2.0,
            "min_progress_reward_scale": 1.0,
            "time_penalty": 0.001,
            "capture_bonus": 200.0,
            "collision_penalty": 2.0,
            "oob_penalty": 30.0,
            "evader_boundary_gain": 0.8,
            "mean_progress_reward_scale": 2.0,
            "capture_bonus_team": 30.0,
            "capture_bonus_individual": 10.0,
            "progress_dist_norm": 2.0,
        },
        "seed": 100,
        "num_epochs": 30000,
        "rollout_steps": 2048,
        "log_interval": 1,
        "eval_episodes": 20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch 3v1 pursuit-evasion training experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=("speed_ratio", "role_assignment", "obstacles"),
        help="Which experiment group to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 102, 103],
        help="Seeds to run for the selected experiment.",
    )
    parser.add_argument(
        "--speed-ratios",
        nargs="+",
        default=["1:1", "1:2", "1:3"],
        help="Only used by speed_ratio. Interpreted as evader:pursuer by default.",
    )
    parser.add_argument(
        "--base-evader-speed",
        type=float,
        default=0.06,
        help="Only used by speed_ratio when ratio mode is evader:pursuer.",
    )
    parser.add_argument(
        "--base-pursuer-speed",
        type=float,
        default=0.20,
        help="Only used by speed_ratio when ratio mode is pursuer:evader.",
    )
    parser.add_argument(
        "--ratio-mode",
        choices=("evader:pursuer", "pursuer:evader"),
        default="evader:pursuer",
        help="Interpretation rule for --speed-ratios.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print generated configs and commands without launching training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated config files.",
    )
    return parser.parse_args()


def parse_ratio(text: str) -> tuple[float, float]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid ratio {text!r}, expected format like 1:3.")
    left = float(parts[0].strip())
    right = float(parts[1].strip())
    if left <= 0.0 or right <= 0.0:
        raise ValueError(f"Invalid ratio {text!r}, both sides must be positive.")
    return left, right


def speed_pair_for_ratio(args: argparse.Namespace, ratio_text: str) -> tuple[float, float]:
    left, right = parse_ratio(ratio_text)
    if args.ratio_mode == "evader:pursuer":
        evader_speed = float(args.base_evader_speed)
        pursuer_speed = evader_speed * (right / left)
    else:
        pursuer_speed = float(args.base_pursuer_speed)
        evader_speed = pursuer_speed * (right / left)
    return pursuer_speed, evader_speed


def build_speed_ratio_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for ratio_text in args.speed_ratios:
        pursuer_speed, evader_speed = speed_pair_for_ratio(args, ratio_text)
        ratio_tag = ratio_text.replace(":", "to")
        for seed in args.seeds:
            cfg = _base_train_cfg()
            cfg["seed"] = int(seed)
            cfg["task"]["name"] = "pursuit_evasion_3v1"
            cfg["task"]["pursuer_speed"] = float(pursuer_speed)
            cfg["task"]["evader_speed"] = float(evader_speed)
            run_name = f"pursuit_speed_ratio_{ratio_tag}_seed{seed}"
            runs.append((run_name, cfg))
    return runs


def build_role_assignment_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for seed in args.seeds:
        cfg = _base_train_cfg()
        cfg["seed"] = int(seed)
        cfg["task"]["name"] = "pursuit_evasion_3v1_ex1"
        run_name = f"pursuit_role_assignment_seed{seed}"
        runs.append((run_name, cfg))
    return runs


def build_obstacle_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for seed in args.seeds:
        cfg = _base_train_cfg()
        cfg["seed"] = int(seed)
        cfg["task"]["name"] = "pursuit_evasion_3v1_ex2"
        cfg["task"]["num_obstacles_min"] = 5
        cfg["task"]["num_obstacles_max"] = 10
        cfg["task"]["obstacle_collision_penalty"] = 15.0
        run_name = f"pursuit_obstacles_seed{seed}"
        runs.append((run_name, cfg))
    return runs


def build_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    if args.experiment == "speed_ratio":
        return build_speed_ratio_runs(args)
    if args.experiment == "role_assignment":
        return build_role_assignment_runs(args)
    if args.experiment == "obstacles":
        return build_obstacle_runs(args)
    raise ValueError(f"Unsupported experiment {args.experiment!r}")


def write_config(run_name: str, cfg: dict[str, Any], overwrite: bool) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = GENERATED_DIR / f"{run_name}.yaml"
    if cfg_path.exists() and not overwrite:
        raise FileExistsError(
            f"Config already exists: {cfg_path}. Use --overwrite if you want to replace it."
        )
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg_path


def launch_run(python_bin: str, cfg_path: Path) -> None:
    cmd = [python_bin, "scripts/train.py", "--train-config", str(cfg_path.relative_to(ROOT))]
    print(f"\n[launch] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    runs = build_runs(args)

    summary = {
        "experiment": args.experiment,
        "num_runs": len(runs),
        "seeds": [int(x) for x in args.seeds],
    }
    if args.experiment == "speed_ratio":
        summary["speed_ratios"] = list(args.speed_ratios)
        summary["ratio_mode"] = args.ratio_mode
        if args.ratio_mode == "evader:pursuer":
            summary["base_evader_speed"] = float(args.base_evader_speed)
        else:
            summary["base_pursuer_speed"] = float(args.base_pursuer_speed)

    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())

    generated: list[tuple[Path, dict[str, Any]]] = []
    for run_name, cfg in runs:
        cfg_copy = deepcopy(cfg)
        cfg_path = write_config(run_name, cfg_copy, overwrite=args.overwrite)
        generated.append((cfg_path, cfg_copy))
        task_cfg = cfg_copy["task"]
        print(
            f"[config] {cfg_path.relative_to(ROOT)} | seed={cfg_copy['seed']} "
            f"| task={task_cfg['name']} | pursuer_speed={task_cfg['pursuer_speed']:.6f} "
            f"| evader_speed={task_cfg['evader_speed']:.6f}"
        )

    if args.dry_run:
        print("\nDry run only. No training process was launched.")
        return

    for cfg_path, _ in generated:
        launch_run(args.python, cfg_path)


if __name__ == "__main__":
    main()
