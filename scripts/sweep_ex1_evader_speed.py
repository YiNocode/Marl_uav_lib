"""Sweep evader speeds for ex1 with fixed rewards, then summarize capture rates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_CFG = ROOT / "configs" / "search" / "ex1_evader_speed_sweep.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ex1 evader speeds while keeping reward settings fixed."
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=str(DEFAULT_SWEEP_CFG.relative_to(ROOT)),
        help="Path to the evader-speed sweep YAML config.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for training/eval subprocesses.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch train/eval runs after generating configs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="When used with --launch, skip the post-train eval pass.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override the number of eval episodes per seed.",
    )
    parser.add_argument(
        "--eval-num-seeds",
        type=int,
        default=1,
        help="Number of eval seeds passed to scripts/eval.py.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only recompute summaries from an existing manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of generated speed points.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def set_by_dotted_path(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    cur: dict[str, Any] = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def get_by_dotted_path(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in [part for part in str(dotted_key).split(".") if part]:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def normalize_scalar_for_name(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace("-", "m").replace(".", "p")
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_"):
            out.append(ch)
        elif ch == ".":
            out.append("p")
    return "".join(out) or "x"


def build_speed_list(cfg: dict[str, Any]) -> list[float]:
    if "evader_speeds" in cfg:
        speeds = [float(x) for x in cfg.get("evader_speeds", [])]
    else:
        speed_min = float(cfg.get("speed_min", 0.10))
        speed_max = float(cfg.get("speed_max", 0.25))
        speed_step = float(cfg.get("speed_step", 0.03))
        if speed_step <= 0.0:
            raise ValueError("speed_step must be positive.")
        speeds = []
        cur = speed_min
        while cur <= speed_max + 1e-9:
            speeds.append(round(cur, 6))
            cur += speed_step
    if not speeds:
        raise ValueError("No evader speeds configured for sweep.")
    return speeds


def launch_run(python_bin: str, cfg_path: Path) -> None:
    cmd = [python_bin, "scripts/train.py", "--train-config", str(cfg_path.relative_to(ROOT))]
    print(f"[launch] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def launch_eval(
    python_bin: str,
    cfg_path: Path,
    *,
    seed: int,
    episodes: int,
    num_seeds: int,
) -> None:
    cmd = [
        python_bin,
        "scripts/eval.py",
        "--config",
        str(cfg_path.relative_to(ROOT)),
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--num-seeds",
        str(num_seeds),
    ]
    print(f"[eval] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_eval_records(results_dir: Path, seed: int) -> list[dict[str, Any]]:
    records_path = results_dir / f"eval_records_seed{seed}.json"
    if not records_path.exists():
        return []
    with open(records_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def summarize_eval_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "capture_rate": 0.0,
            "timeout_rate": 0.0,
            "collision_rate": 0.0,
            "success_capture_rate": 0.0,
            "avg_episode_return": 0.0,
            "avg_episode_len": 0.0,
            "mean_C_cov": 0.0,
            "mean_C_col": 0.0,
        }
    captured = [1.0 if bool(r.get("captured", False)) else 0.0 for r in records]
    timeout = [1.0 if bool(r.get("timeout", False)) else 0.0 for r in records]
    collision = [1.0 if bool(r.get("collision", False)) else 0.0 for r in records]
    success_capture = [1.0 if bool(r.get("success_or_capture", False)) else 0.0 for r in records]
    ep_return = [float(r.get("episode_return", 0.0)) for r in records]
    ep_len = [float(r.get("episode_len", 0.0)) for r in records]
    mean_cov = [float(r.get("mean_C_cov", 0.0)) for r in records if r.get("mean_C_cov") is not None]
    mean_col = [float(r.get("mean_C_col", 0.0)) for r in records if r.get("mean_C_col") is not None]
    return {
        "capture_rate": float(mean(captured)),
        "timeout_rate": float(mean(timeout)),
        "collision_rate": float(mean(collision)),
        "success_capture_rate": float(mean(success_capture)),
        "avg_episode_return": float(mean(ep_return)),
        "avg_episode_len": float(mean(ep_len)),
        "mean_C_cov": float(mean(mean_cov)) if mean_cov else 0.0,
        "mean_C_col": float(mean(mean_col)) if mean_col else 0.0,
    }


def write_summary_files(
    *,
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    run_prefix: str,
) -> tuple[Path, Path]:
    per_run_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}

    for row in manifest_rows:
        results_dir = ROOT / str(row["results_dir_relpath"])
        seed = int(row["seed"])
        records = load_eval_records(results_dir, seed)
        metrics = summarize_eval_records(records)

        per_run = dict(row)
        per_run.update(metrics)
        per_run["eval_episodes_found"] = len(records)
        per_run_rows.append(per_run)

        speed_id = str(row["speed_id"])
        group = grouped.setdefault(
            speed_id,
            {
                "speed_id": speed_id,
                "num_seeds": 0,
                "run_prefix": run_prefix,
                "task.evader_speed": row["task.evader_speed"],
                "_metrics": {
                    "capture_rate": [],
                    "timeout_rate": [],
                    "collision_rate": [],
                    "success_capture_rate": [],
                    "avg_episode_return": [],
                    "avg_episode_len": [],
                    "mean_C_cov": [],
                    "mean_C_col": [],
                },
            },
        )
        group["num_seeds"] += 1
        for key in group["_metrics"]:
            group["_metrics"][key].append(float(metrics[key]))

    per_run_rows.sort(
        key=lambda x: (-float(x["capture_rate"]), float(x["task.evader_speed"]), str(x["run_name"]))
    )

    grouped_rows: list[dict[str, Any]] = []
    for speed_id, group in grouped.items():
        out = {k: v for k, v in group.items() if k != "_metrics"}
        for key, values in group["_metrics"].items():
            out[key] = float(mean(values)) if values else 0.0
        grouped_rows.append(out)
    grouped_rows.sort(
        key=lambda x: (-float(x["capture_rate"]), float(x["task.evader_speed"]))
    )

    per_run_path = output_dir / f"{run_prefix}_summary_per_run.csv"
    grouped_path = output_dir / f"{run_prefix}_summary_by_speed.csv"

    with open(per_run_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_name",
            "speed_id",
            "seed",
            "config_relpath",
            "results_dir_relpath",
            "task.evader_speed",
            "capture_rate",
            "timeout_rate",
            "collision_rate",
            "success_capture_rate",
            "avg_episode_return",
            "avg_episode_len",
            "mean_C_cov",
            "mean_C_col",
            "eval_episodes_found",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_run_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(grouped_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "speed_id",
            "num_seeds",
            "task.evader_speed",
            "capture_rate",
            "timeout_rate",
            "collision_rate",
            "success_capture_rate",
            "avg_episode_return",
            "avg_episode_len",
            "mean_C_cov",
            "mean_C_col",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grouped_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return per_run_path, grouped_path


def main() -> None:
    args = parse_args()
    sweep_cfg_path = ROOT / args.sweep_config
    sweep_cfg = load_yaml(sweep_cfg_path)

    base_config_rel = str(
        sweep_cfg.get("base_config", "configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml")
    )
    base_cfg = load_yaml(ROOT / base_config_rel)

    run_prefix = str(sweep_cfg.get("run_prefix", "ex1_evader_speed")).strip() or "ex1_evader_speed"
    output_dir_rel = str(sweep_cfg.get("output_dir", "configs/generated/evader_speed_sweep"))
    output_dir = ROOT / output_dir_rel
    manifest_name = str(sweep_cfg.get("manifest_name", f"{run_prefix}_manifest.csv"))
    manifest_path = output_dir / manifest_name

    seeds = [int(x) for x in sweep_cfg.get("seeds", [int(base_cfg.get("seed", 42))])]
    eval_episodes = int(
        args.eval_episodes
        if args.eval_episodes is not None
        else sweep_cfg.get("eval_episodes", base_cfg.get("eval_episodes", 20))
    )
    fixed_overrides = dict(sweep_cfg.get("fixed_overrides", {}) or {})
    evader_speeds = build_speed_list(sweep_cfg)

    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs must be positive when provided.")
        evader_speeds = evader_speeds[: args.max_runs]

    if args.summarize_only:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for summarize-only mode: {manifest_path}")
        with open(manifest_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)
        per_run_path, grouped_path = write_summary_files(
            output_dir=output_dir,
            manifest_rows=manifest_rows,
            run_prefix=run_prefix,
        )
        print(f"[summary] {per_run_path.relative_to(ROOT)}")
        print(f"[summary] {grouped_path.relative_to(ROOT)}")
        return

    planned_rows: list[dict[str, Any]] = []
    generated_paths: list[Path] = []

    print(
        yaml.safe_dump(
            {
                "sweep_config": str(sweep_cfg_path.relative_to(ROOT)).replace("\\", "/"),
                "base_config": base_config_rel,
                "num_speed_points": len(evader_speeds),
                "num_seeds": len(seeds),
                "total_runs": len(evader_speeds) * len(seeds),
                "output_dir": output_dir_rel,
                "eval_episodes": eval_episodes,
            },
            allow_unicode=True,
            sort_keys=False,
        ).strip()
    )

    for speed_idx, evader_speed in enumerate(evader_speeds, start=1):
        speed_tag = normalize_scalar_for_name(evader_speed)
        for seed in seeds:
            cfg = deepcopy(base_cfg)
            for key, value in fixed_overrides.items():
                set_by_dotted_path(cfg, key, value)
            set_by_dotted_path(cfg, "task.evader_speed", float(evader_speed))
            set_by_dotted_path(cfg, "seed", int(seed))

            task_name = get_by_dotted_path(cfg, "task.name")
            if task_name != "pursuit_evasion_3v1_ex1":
                raise ValueError(
                    f"Evader-speed sweep base config must target ex1, got task.name={task_name!r}."
                )

            run_name = f"{run_prefix}_s{speed_idx:03d}_evs{speed_tag}_seed{seed}"
            cfg_path = output_dir / f"{run_name}.yaml"
            results_dir = ROOT / "results" / cfg_path.stem

            row = {
                "run_name": run_name,
                "speed_id": f"s{speed_idx:03d}",
                "seed": int(seed),
                "config_relpath": str(cfg_path.relative_to(ROOT)).replace("\\", "/"),
                "results_dir_relpath": str(results_dir.relative_to(ROOT)).replace("\\", "/"),
                "task.evader_speed": float(evader_speed),
            }
            planned_rows.append(row)

            print(
                f"[plan] {row['config_relpath']} | seed={seed} | task.evader_speed={evader_speed:.3f}"
            )

            if args.dry_run:
                continue

            save_yaml(cfg_path, cfg, overwrite=args.overwrite)
            generated_paths.append(cfg_path)

    if args.dry_run:
        print("\nDry run only. No files were written.")
        return

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. Use --overwrite to replace it."
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "speed_id",
        "seed",
        "config_relpath",
        "results_dir_relpath",
        "task.evader_speed",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in planned_rows:
            writer.writerow(row)

    print(f"\n[manifest] {manifest_path.relative_to(ROOT)}")

    if not args.launch:
        print("Configs generated. Add --launch to start training.")
    else:
        for row, cfg_path in zip(planned_rows, generated_paths, strict=False):
            launch_run(args.python, cfg_path)
            if not args.skip_eval:
                launch_eval(
                    args.python,
                    cfg_path,
                    seed=int(row["seed"]),
                    episodes=eval_episodes,
                    num_seeds=int(args.eval_num_seeds),
                )

    per_run_path, grouped_path = write_summary_files(
        output_dir=output_dir,
        manifest_rows=planned_rows,
        run_prefix=run_prefix,
    )
    print(f"[summary] {per_run_path.relative_to(ROOT)}")
    print(f"[summary] {grouped_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
