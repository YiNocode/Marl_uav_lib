"""Benchmark ex1 training with single-env and AsyncVectorEnv rollout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment" / "pursuit_evasion_mappo_3v1_ex1.yaml"

ROLL_LINE_RE = re.compile(
    r"^rollout=(?P<rollout_s>[-+eE0-9.]+)s\s+\((?P<rollout_ms_per_step>[-+eE0-9.]+)ms/step\)\s+"
    r"update=(?P<update_s>[-+eE0-9.]+)s\s+\((?P<update_ms_per_step>[-+eE0-9.]+)ms/step\)\s+"
    r"(?:log=(?P<log_s>[-+eE0-9.]+)s\s+)?"
    r"episodes=(?P<episodes>\d+)\s+"
    r"(?:(?:steps|env_steps)=(?P<steps>\d+))"
    r"(?:\s+env_step_ms:\s+total=(?P<env_total_ms>[-+eE0-9.]+)"
    r"\s+backend=(?P<env_backend_ms>[-+eE0-9.]+)"
    r"\s+reward=(?P<env_reward_ms>[-+eE0-9.]+)"
    r"\s+done=(?P<env_done_ms>[-+eE0-9.]+)"
    r"\s+obs_state=(?P<env_obs_state_ms>[-+eE0-9.]+)"
    r"\s+info=(?P<env_info_ms>[-+eE0-9.]+)"
    r"\s+action=(?P<env_action_ms>[-+eE0-9.]+))?"
)

TRAIN_LINE_RE = re.compile(
    r"^\[(?P<tag>train|vec-train)\]\s+epoch=(?P<epoch>\d+)/(?P<num_epochs>\d+)\s+"
    r"(?:num_envs=(?P<num_envs>\d+)\s+)?"
    r"steps=(?P<steps>\d+)\s+avg_return=(?P<avg_return>[-+eE0-9.]+)\s+"
    r"avg_len=(?P<avg_len>[-+eE0-9.]+)"
)


@dataclass
class RunResult:
    num_envs: int
    return_code: int
    wall_time_s: float
    rollout_s: float | None
    rollout_ms_per_step: float | None
    rollout_fps: float | None
    update_s: float | None
    update_ms_per_step: float | None
    episodes: int | None
    steps: int | None
    avg_return: float | None
    avg_len: float | None
    env_total_ms: float | None
    env_backend_ms: float | None
    env_reward_ms: float | None
    env_done_ms: float | None
    env_obs_state_ms: float | None
    env_info_ms: float | None
    env_action_ms: float | None
    stdout_log: str
    profile_train_config: str
    profile_env_config: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base ex1 training config.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Env counts to benchmark. Include 1 as baseline.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Override num_epochs for each benchmark run.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=256,
        help="Override rollout_steps for each benchmark run.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Override final eval episodes for each benchmark run.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=256,
        help="Override ex1 task episode_limit to keep the benchmark short and comparable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override.",
    )
    parser.add_argument(
        "--disable-render",
        action="store_true",
        default=True,
        help="Disable backend rendering in benchmark copies.",
    )
    parser.add_argument(
        "--vec-env-context",
        type=str,
        default="spawn",
        help="Multiprocessing context passed to train.py for VecEnv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "vecenv_benchmarks",
        help="Directory for benchmark summaries and logs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def build_profile_env_config(
    env_cfg: dict[str, Any],
    run_dir: Path,
    *,
    disable_render: bool,
) -> Path:
    out = dict(env_cfg)
    backend_cfg = dict(out.get("backend", {}) or {})
    if disable_render and "render" in backend_cfg:
        backend_cfg["render"] = False
    out["backend"] = backend_cfg
    path = run_dir / "env_profile.yaml"
    dump_yaml(path, out)
    return path


def build_profile_train_config(
    train_cfg: dict[str, Any],
    run_dir: Path,
    *,
    env_cfg_path: Path,
    num_envs: int,
    num_epochs: int,
    rollout_steps: int,
    eval_episodes: int,
    episode_limit: int,
    seed: int | None,
    vec_env_context: str,
) -> Path:
    out = dict(train_cfg)
    out["env"] = str(env_cfg_path.relative_to(ROOT)).replace("\\", "/")
    out["num_envs"] = int(num_envs)
    out["num_epochs"] = int(num_epochs)
    out["rollout_steps"] = int(rollout_steps)
    out["eval_episodes"] = int(eval_episodes)
    out["log_interval"] = 1
    out["vec_env_context"] = str(vec_env_context)
    out["vec_env_shared_memory"] = True
    out["vec_env_copy"] = False
    if seed is not None:
        out["seed"] = int(seed)

    task_cfg = dict(out.get("task", {}) or {})
    task_cfg["episode_limit"] = int(episode_limit)
    out["task"] = task_cfg

    path = run_dir / f"train_ex1_num_envs_{num_envs}.yaml"
    dump_yaml(path, out)
    return path


def parse_stdout(stdout_text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    last_train: dict[str, Any] | None = None
    last_roll: dict[str, Any] | None = None
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        train_match = TRAIN_LINE_RE.search(line)
        if train_match:
            gd = train_match.groupdict()
            last_train = {
                "tag": gd["tag"],
                "epoch": int(gd["epoch"]),
                "num_epochs": int(gd["num_epochs"]),
                "num_envs": None if gd["num_envs"] is None else int(gd["num_envs"]),
                "steps": int(gd["steps"]),
                "avg_return": float(gd["avg_return"]),
                "avg_len": float(gd["avg_len"]),
            }
            continue

        roll_match = ROLL_LINE_RE.search(line)
        if roll_match:
            gd = roll_match.groupdict()
            last_roll = {
                key: (
                    None
                    if value is None
                    else int(value)
                    if key in {"episodes", "steps"}
                    else float(value)
                )
                for key, value in gd.items()
            }
            if last_roll["rollout_ms_per_step"] is not None:
                last_roll["rollout_fps"] = 1000.0 / float(last_roll["rollout_ms_per_step"])
    return last_train, last_roll


def run_train(train_config_path: Path, *, cwd: Path) -> tuple[int, str, float]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--train-config",
        str(train_config_path.relative_to(ROOT)),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall_time_s = time.perf_counter() - t0
    return proc.returncode, proc.stdout, wall_time_s


def benchmark_one(
    *,
    train_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    run_dir: Path,
    num_envs: int,
    args: argparse.Namespace,
) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg_path = build_profile_env_config(env_cfg, run_dir, disable_render=args.disable_render)
    train_cfg_path = build_profile_train_config(
        train_cfg,
        run_dir,
        env_cfg_path=env_cfg_path,
        num_envs=num_envs,
        num_epochs=args.num_epochs,
        rollout_steps=args.rollout_steps,
        eval_episodes=args.eval_episodes,
        episode_limit=args.episode_limit,
        seed=args.seed,
        vec_env_context=args.vec_env_context,
    )

    return_code, stdout_text, wall_time_s = run_train(train_cfg_path, cwd=ROOT)
    log_path = run_dir / f"stdout_num_envs_{num_envs}.log"
    log_path.write_text(stdout_text, encoding="utf-8")
    train_row, roll_row = parse_stdout(stdout_text)

    return RunResult(
        num_envs=num_envs,
        return_code=return_code,
        wall_time_s=wall_time_s,
        rollout_s=None if roll_row is None else roll_row.get("rollout_s"),
        rollout_ms_per_step=None if roll_row is None else roll_row.get("rollout_ms_per_step"),
        rollout_fps=None if roll_row is None else roll_row.get("rollout_fps"),
        update_s=None if roll_row is None else roll_row.get("update_s"),
        update_ms_per_step=None if roll_row is None else roll_row.get("update_ms_per_step"),
        episodes=None if roll_row is None else roll_row.get("episodes"),
        steps=None if roll_row is None else roll_row.get("steps"),
        avg_return=None if train_row is None else train_row.get("avg_return"),
        avg_len=None if train_row is None else train_row.get("avg_len"),
        env_total_ms=None if roll_row is None else roll_row.get("env_total_ms"),
        env_backend_ms=None if roll_row is None else roll_row.get("env_backend_ms"),
        env_reward_ms=None if roll_row is None else roll_row.get("env_reward_ms"),
        env_done_ms=None if roll_row is None else roll_row.get("env_done_ms"),
        env_obs_state_ms=None if roll_row is None else roll_row.get("env_obs_state_ms"),
        env_info_ms=None if roll_row is None else roll_row.get("env_info_ms"),
        env_action_ms=None if roll_row is None else roll_row.get("env_action_ms"),
        stdout_log=str(log_path),
        profile_train_config=str(train_cfg_path),
        profile_env_config=str(env_cfg_path),
    )


def build_summary(results: list[RunResult]) -> dict[str, Any]:
    baseline = next((r for r in results if r.num_envs == 1 and r.return_code == 0), None)
    comparisons: list[dict[str, Any]] = []
    for result in results:
        row = {
            "num_envs": result.num_envs,
            "return_code": result.return_code,
            "wall_time_s": result.wall_time_s,
            "rollout_s": result.rollout_s,
            "rollout_ms_per_step": result.rollout_ms_per_step,
            "rollout_fps": result.rollout_fps,
            "update_s": result.update_s,
            "update_ms_per_step": result.update_ms_per_step,
            "episodes": result.episodes,
            "steps": result.steps,
            "avg_return": result.avg_return,
            "avg_len": result.avg_len,
            "env_step_ms": {
                "total": result.env_total_ms,
                "backend": result.env_backend_ms,
                "reward": result.env_reward_ms,
                "done": result.env_done_ms,
                "obs_state": result.env_obs_state_ms,
                "info": result.env_info_ms,
                "action": result.env_action_ms,
            },
            "artifacts": {
                "stdout_log": result.stdout_log,
                "profile_train_config": result.profile_train_config,
                "profile_env_config": result.profile_env_config,
            },
        }
        if baseline is not None and baseline.rollout_fps and result.rollout_fps:
            row["throughput_speedup_vs_1env"] = result.rollout_fps / baseline.rollout_fps
        else:
            row["throughput_speedup_vs_1env"] = None
        if baseline is not None and baseline.wall_time_s > 0:
            row["wall_time_ratio_vs_1env"] = result.wall_time_s / baseline.wall_time_s
        else:
            row["wall_time_ratio_vs_1env"] = None
        comparisons.append(row)

    best = None
    valid = [r for r in results if r.return_code == 0 and r.rollout_fps is not None]
    if valid:
        best = max(valid, key=lambda x: float(x.rollout_fps))

    return {
        "baseline_num_envs": None if baseline is None else baseline.num_envs,
        "best_num_envs_by_rollout_fps": None if best is None else best.num_envs,
        "runs": comparisons,
    }


def print_summary(results: list[RunResult]) -> None:
    baseline = next((r for r in results if r.num_envs == 1 and r.return_code == 0), None)
    print("\n=== Ex1 VecEnv Benchmark Summary ===")
    for result in results:
        fps = "N/A" if result.rollout_fps is None else f"{result.rollout_fps:.2f}"
        ms = "N/A" if result.rollout_ms_per_step is None else f"{result.rollout_ms_per_step:.2f}"
        speedup = None
        if baseline is not None and baseline.rollout_fps and result.rollout_fps:
            speedup = result.rollout_fps / baseline.rollout_fps
        speedup_str = "N/A" if speedup is None else f"{speedup:.2f}x"
        print(
            f"num_envs={result.num_envs:<4d} rc={result.return_code} "
            f"rollout_fps={fps:<10} ms_per_step={ms:<10} "
            f"speedup_vs_1env={speedup_str:<8} wall_time={result.wall_time_s:.2f}s"
        )


def main() -> None:
    args = parse_args()
    train_config_path = args.train_config.resolve()
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_config_path}")

    requested_num_envs = sorted({int(x) for x in args.num_envs})
    if 1 not in requested_num_envs:
        requested_num_envs = [1, *requested_num_envs]

    train_cfg = load_yaml(train_config_path)
    env_cfg_rel = train_cfg.get("env")
    if not env_cfg_rel:
        raise ValueError(f"Missing env path in {train_config_path}.")
    env_cfg = load_yaml(ROOT / env_cfg_rel)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / f"ex1_vecenv_benchmark_{run_stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for num_envs in requested_num_envs:
        print(f"\n=== Running ex1 benchmark with num_envs={num_envs} ===")
        run_dir = output_root / f"num_envs_{num_envs}"
        result = benchmark_one(
            train_cfg=train_cfg,
            env_cfg=env_cfg,
            run_dir=run_dir,
            num_envs=num_envs,
            args=args,
        )
        results.append(result)
        if result.return_code != 0:
            print(
                f"[warn] num_envs={num_envs} failed with return_code={result.return_code}. "
                f"See {result.stdout_log}"
            )

    summary = {
        "command_args": to_jsonable(vars(args)),
        "base_train_config": str(train_config_path),
        "created_at": run_stamp,
        "comparison": build_summary(results),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(results)
    print(f"\nSummary written to: {summary_path}")

    if any(r.return_code != 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
