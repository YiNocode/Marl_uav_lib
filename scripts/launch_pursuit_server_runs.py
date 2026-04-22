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
GENERATED_ENV_DIR = GENERATED_DIR / "env"


def _load_yaml(rel_path: str) -> dict[str, Any]:
    with open(ROOT / rel_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dream_ex1_train_cfg() -> dict[str, Any]:
    return _load_yaml("configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml")


def _mappo_ex2_train_cfg() -> dict[str, Any]:
    return _load_yaml("configs/experiment/pursuit_evasion_mappo_3v1_ex2.yaml")


def _base_env_cfg() -> dict[str, Any]:
    with open(ROOT / "configs" / "env" / "pyflyt_3v1.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        default=[101],
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
        default=None,
        help="Only used by speed_ratio when ratio mode is evader:pursuer. Defaults to the base train config.",
    )
    parser.add_argument(
        "--base-pursuer-speed",
        type=float,
        default=None,
        help="Only used by speed_ratio when ratio mode is pursuer:evader. Defaults to the base train config.",
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


def speed_pair_for_ratio(
    args: argparse.Namespace,
    ratio_text: str,
    *,
    default_pursuer_speed: float,
    default_evader_speed: float,
) -> tuple[float, float]:
    left, right = parse_ratio(ratio_text)
    if args.ratio_mode == "evader:pursuer":
        evader_speed = float(
            default_evader_speed if args.base_evader_speed is None else args.base_evader_speed
        )
        pursuer_speed = evader_speed * (right / left)
    else:
        pursuer_speed = float(
            default_pursuer_speed if args.base_pursuer_speed is None else args.base_pursuer_speed
        )
        evader_speed = pursuer_speed * (right / left)
    return pursuer_speed, evader_speed


def build_speed_ratio_runs(
    args: argparse.Namespace,
) -> list[tuple[str, dict[str, Any], str | None, dict[str, Any] | None]]:
    runs: list[tuple[str, dict[str, Any], str | None, dict[str, Any] | None]] = []
    base_cfg = _dream_ex1_train_cfg()
    base_task = base_cfg.get("task", {}) or {}
    default_pursuer_speed = float(base_task.get("pursuer_speed", 0.20))
    default_evader_speed = float(base_task.get("evader_speed", 0.06))

    for ratio_text in args.speed_ratios:
        pursuer_speed, evader_speed = speed_pair_for_ratio(
            args,
            ratio_text,
            default_pursuer_speed=default_pursuer_speed,
            default_evader_speed=default_evader_speed,
        )
        ratio_tag = ratio_text.replace(":", "to")
        for seed in args.seeds:
            cfg = _dream_ex1_train_cfg()
            env_cfg = _base_env_cfg()
            cfg["seed"] = int(seed)
            cfg["task"]["name"] = "pursuit_evasion_3v1_ex1"
            # Continuous control under PyFlyt 3v1 uses env.action_low/high as the true
            # pursuer velocity bounds. Keep task speed in sync for normalization only.
            cfg["task"]["pursuer_speed"] = float(pursuer_speed)
            cfg["task"]["evader_speed"] = float(evader_speed)
            action_low = list(env_cfg.get("action_low", [-1.0, -1.0, -1.0, -1.0]))
            action_high = list(env_cfg.get("action_high", [1.0, 1.0, 1.0, 1.0]))
            if len(action_low) != 4 or len(action_high) != 4:
                raise ValueError("Expected 4D action_low/action_high for PyFlyt 3v1 continuous control.")
            # User requirement: only scale the x/y velocity bounds; keep yaw_rate and vz unchanged.
            action_low[0] = -float(pursuer_speed)
            action_low[1] = -float(pursuer_speed)
            action_high[0] = float(pursuer_speed)
            action_high[1] = float(pursuer_speed)
            env_cfg["action_low"] = action_low
            env_cfg["action_high"] = action_high
            run_name = f"pursuit_speed_ratio_{ratio_tag}_seed{seed}"
            env_name = f"pyflyt_3v1_speed_ratio_{ratio_tag}_seed{seed}"
            runs.append((run_name, cfg, env_name, env_cfg))
    return runs


def build_role_assignment_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for seed in args.seeds:
        cfg = _dream_ex1_train_cfg()
        cfg["seed"] = int(seed)
        cfg["task"]["name"] = "pursuit_evasion_3v1_ex1"
        run_name = f"pursuit_role_assignment_seed{seed}"
        runs.append((run_name, cfg, None, None))
    return runs


def build_obstacle_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for seed in args.seeds:
        cfg = _mappo_ex2_train_cfg()
        cfg["seed"] = int(seed)
        cfg["task"]["name"] = "pursuit_evasion_3v1_ex2"
        cfg["task"]["num_obstacles_min"] = 5
        cfg["task"]["num_obstacles_max"] = 10
        cfg["task"]["obstacle_collision_penalty"] = 15.0
        run_name = f"pursuit_obstacles_seed{seed}"
        runs.append((run_name, cfg, None, None))
    return runs


def build_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any], str | None, dict[str, Any] | None]]:
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


def write_env_config(env_name: str, cfg: dict[str, Any], overwrite: bool) -> Path:
    GENERATED_ENV_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = GENERATED_ENV_DIR / f"{env_name}.yaml"
    if cfg_path.exists() and not overwrite:
        raise FileExistsError(
            f"Env config already exists: {cfg_path}. Use --overwrite if you want to replace it."
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
            summary["base_evader_speed"] = (
                None if args.base_evader_speed is None else float(args.base_evader_speed)
            )
        else:
            summary["base_pursuer_speed"] = (
                None if args.base_pursuer_speed is None else float(args.base_pursuer_speed)
            )

    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False).strip())

    generated: list[tuple[Path, dict[str, Any]]] = []
    for run_name, cfg, env_name, env_cfg in runs:
        cfg_copy = deepcopy(cfg)
        if env_name is not None and env_cfg is not None:
            env_cfg_copy = deepcopy(env_cfg)
            env_path = write_env_config(env_name, env_cfg_copy, overwrite=args.overwrite)
            cfg_copy["env"] = str(env_path.relative_to(ROOT)).replace("\\", "/")
        cfg_path = write_config(run_name, cfg_copy, overwrite=args.overwrite)
        generated.append((cfg_path, cfg_copy))
        task_cfg = cfg_copy["task"]
        env_rel = cfg_copy["env"]
        msg = (
            f"[config] {cfg_path.relative_to(ROOT)} | seed={cfg_copy['seed']} "
            f"| env={env_rel} | task={task_cfg['name']} | task.pursuer_speed={task_cfg['pursuer_speed']:.6f} "
            f"| task.evader_speed={task_cfg['evader_speed']:.6f}"
        )
        if env_name is not None and env_cfg is not None:
            env_hi = env_cfg["action_high"]
            msg += f" | env.vxy_max={float(env_hi[0]):.6f}"
        print(
            msg
        )

    if args.dry_run:
        print("\nDry run only. No training process was launched.")
        return

    for cfg_path, _ in generated:
        launch_run(args.python, cfg_path)


if __name__ == "__main__":
    main()
