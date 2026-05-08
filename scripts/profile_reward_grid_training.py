"""Profile a short reward-grid training run and record env/perf metrics.

This script runs one generated reward-grid experiment for a small number of
epochs, samples CPU/GPU utilization while training, parses rollout timing from
the trainer logs, and writes a JSON summary plus CSV samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = (
    ROOT
    / "configs"
    / "generated"
    / "reward_grid"
    / "ex1_reward_grid_c001_baseline_srs0p25_shs0p05_sis0p2_rcs1_rop0p5_cpss0p3_cpcs2_mcr0p01_msgs0p75_seed101.yaml"
)

TRAIN_LINE_RE = re.compile(
    r"^\[train\]\s+epoch=(?P<epoch>\d+)/(?P<num_epochs>\d+)\s+"
    r"steps=(?P<steps>\d+)\s+avg_return=(?P<avg_return>[-+eE0-9.]+)\s+"
    r"avg_len=(?P<avg_len>[-+eE0-9.]+)"
)

ROLLOUT_LINE_RE = re.compile(
    r"^rollout=(?P<rollout_s>[-+eE0-9.]+)s\s+\((?P<rollout_ms_per_step>[-+eE0-9.]+)ms/step\)\s+"
    r"update=(?P<update_s>[-+eE0-9.]+)s\s+\((?P<update_ms_per_step>[-+eE0-9.]+)ms/step\)\s+"
    r"log=(?P<log_s>[-+eE0-9.]+)s\s+episodes=(?P<episodes>\d+)\s+steps=(?P<steps>\d+)"
    r"(?:\s+env_step_ms:\s+total=(?P<env_total_ms>[-+eE0-9.]+)"
    r"\s+backend=(?P<env_backend_ms>[-+eE0-9.]+)"
    r"\s+reward=(?P<env_reward_ms>[-+eE0-9.]+)"
    r"\s+done=(?P<env_done_ms>[-+eE0-9.]+)"
    r"\s+obs_state=(?P<env_obs_state_ms>[-+eE0-9.]+)"
    r"\s+info=(?P<env_info_ms>[-+eE0-9.]+)"
    r"\s+action=(?P<env_action_ms>[-+eE0-9.]+))?"
)


try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None


@dataclass
class Sample:
    elapsed_s: float
    cpu_total_percent: float | None
    cpu_process_percent: float | None
    gpu_percent: float | None
    gpu_mem_percent: float | None


def safe_cpu_total_percent() -> float | None:
    if psutil is None:
        return None
    try:
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def safe_process_cpu_percent(proc: Any | None) -> float | None:
    if proc is None:
        return None
    try:
        return float(proc.cpu_percent(interval=None))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Path to a generated reward-grid train config.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Override num_epochs in the copied config so profiling finishes quickly.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Override eval_episodes in the copied config.",
    )
    parser.add_argument(
        "--disable-render",
        action="store_true",
        default=True,
        help="Disable backend rendering in the copied env config.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="Optionally override task.episode_limit. Defaults to min(existing, rollout_steps).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Resource sampling interval in seconds.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Run an unprofiled warmup training pass first to exclude first simulator startup cost. Use 0 to disable.",
    )
    parser.add_argument(
        "--warmup-eval-episodes",
        type=int,
        default=0,
        help="Override eval_episodes for the warmup pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "perf_profiles",
        help="Directory for summary/log/sample outputs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def infer_policy_agent_count(train_cfg: dict[str, Any], env_cfg: dict[str, Any]) -> int:
    task_cfg = dict(train_cfg.get("task", {}) or {})
    task_name = str(task_cfg.get("name", "")).strip().lower()
    if "3v1" in task_name:
        return 3
    backend_cfg = dict(env_cfg.get("backend", {}) or {})
    return int(backend_cfg.get("num_agents", 1))


def infer_num_envs() -> tuple[int, str]:
    return 1, "Current trainer uses one RolloutWorker with a single env instance."


def configured_ppo_batch_size(algo_cfg: dict[str, Any]) -> int | None:
    for key in ("minibatch_size", "batch_size"):
        value = algo_cfg.get(key)
        if value is not None:
            return int(value)
    return None


def try_init_nvml() -> bool:
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def query_gpu_percent_nvml() -> tuple[float | None, float | None]:
    if pynvml is None:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_pct = 100.0 * float(mem.used) / float(mem.total) if mem.total else None
        return float(util.gpu), mem_pct
    except Exception:
        return None, None


def query_gpu_percent_nvidia_smi() -> tuple[float | None, float | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None, None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None, None
    try:
        gpu_pct = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
        mem_pct = 100.0 * mem_used / mem_total if mem_total else None
        return gpu_pct, mem_pct
    except ValueError:
        return None, None


def query_gpu_percent(use_nvml: bool) -> tuple[float | None, float | None]:
    if use_nvml:
        gpu_pct, mem_pct = query_gpu_percent_nvml()
        if gpu_pct is not None:
            return gpu_pct, mem_pct
    return query_gpu_percent_nvidia_smi()


def enqueue_stdout(pipe: Any, q: queue.Queue[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            q.put(line)
    finally:
        pipe.close()


def build_profile_env_config(
    env_cfg: dict[str, Any],
    run_dir: Path,
    *,
    disable_render: bool,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg_profile = dict(env_cfg)
    backend_cfg_profile = dict(env_cfg_profile.get("backend", {}) or {})
    if disable_render and "render" in backend_cfg_profile:
        backend_cfg_profile["render"] = False
    env_cfg_profile["backend"] = backend_cfg_profile
    env_cfg_profile_path = run_dir / "profile_env_config.yaml"
    dump_yaml(env_cfg_profile_path, env_cfg_profile)
    return env_cfg_profile_path


def build_profile_train_config(
    train_cfg: dict[str, Any],
    run_dir: Path,
    *,
    env_cfg_profile_path: Path,
    num_epochs: int,
    eval_episodes: int,
    episode_limit: int | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_cfg = dict(train_cfg)
    profile_cfg["num_epochs"] = int(num_epochs)
    profile_cfg["eval_episodes"] = int(eval_episodes)
    profile_cfg["log_interval"] = 1
    profile_cfg["env"] = str(env_cfg_profile_path.relative_to(ROOT)).replace("\\", "/")
    profile_task_cfg = dict(profile_cfg.get("task", {}) or {})
    current_episode_limit = int(profile_task_cfg.get("episode_limit", profile_cfg.get("rollout_steps", 256)))
    target_episode_limit = (
        int(episode_limit)
        if episode_limit is not None
        else min(current_episode_limit, int(profile_cfg.get("rollout_steps", current_episode_limit)))
    )
    profile_task_cfg["episode_limit"] = target_episode_limit
    profile_cfg["task"] = profile_task_cfg
    profile_cfg_path = run_dir / "profile_train_config.yaml"
    dump_yaml(profile_cfg_path, profile_cfg)
    return profile_cfg_path


def run_warmup(command: list[str], *, cwd: Path, log_path: Path) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[warmup] {line}", end="")
    return_code = process.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    return return_code


def summarize_samples(samples: list[Sample]) -> dict[str, Any]:
    def stats(values: list[float | None]) -> dict[str, float | None]:
        clean = [float(v) for v in values if v is not None]
        if not clean:
            return {"mean": None, "max": None, "min": None}
        return {
            "mean": sum(clean) / len(clean),
            "max": max(clean),
            "min": min(clean),
        }

    return {
        "cpu_total_percent": stats([s.cpu_total_percent for s in samples]),
        "cpu_process_percent": stats([s.cpu_process_percent for s in samples]),
        "gpu_percent": stats([s.gpu_percent for s in samples]),
        "gpu_mem_percent": stats([s.gpu_mem_percent for s in samples]),
        "num_samples": len(samples),
    }


def write_samples_csv(path: Path, samples: list[Sample]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "elapsed_s",
                "cpu_total_percent",
                "cpu_process_percent",
                "gpu_percent",
                "gpu_mem_percent",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    f"{s.elapsed_s:.3f}",
                    "" if s.cpu_total_percent is None else f"{s.cpu_total_percent:.3f}",
                    "" if s.cpu_process_percent is None else f"{s.cpu_process_percent:.3f}",
                    "" if s.gpu_percent is None else f"{s.gpu_percent:.3f}",
                    "" if s.gpu_mem_percent is None else f"{s.gpu_mem_percent:.3f}",
                ]
            )


def main() -> None:
    args = parse_args()

    train_config_path = args.train_config.resolve()
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_config_path}")

    train_cfg = load_yaml(train_config_path)
    env_cfg = load_yaml(ROOT / train_cfg["env"])
    algo_cfg = load_yaml(ROOT / train_cfg["algo"])

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"profile_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg_profile_path = build_profile_env_config(
        env_cfg,
        run_dir,
        disable_render=args.disable_render,
    )
    profile_cfg_path = build_profile_train_config(
        train_cfg,
        run_dir,
        env_cfg_profile_path=env_cfg_profile_path,
        num_epochs=int(args.num_epochs),
        eval_episodes=int(args.eval_episodes),
        episode_limit=args.episode_limit,
    )
    warmup_cfg_path = build_profile_train_config(
        train_cfg,
        run_dir / "warmup",
        env_cfg_profile_path=env_cfg_profile_path,
        num_epochs=max(int(args.warmup_epochs), 1),
        eval_episodes=int(args.warmup_eval_episodes),
        episode_limit=args.episode_limit,
    ) if args.warmup_epochs > 0 else None

    num_envs, num_envs_reason = infer_num_envs()
    profile_cfg = load_yaml(profile_cfg_path)
    policy_agent_count = infer_policy_agent_count(profile_cfg, env_cfg)
    rollout_steps = int(profile_cfg.get("rollout_steps", 0))
    effective_update_batch_size = rollout_steps * policy_agent_count
    configured_batch_size = configured_ppo_batch_size(algo_cfg)

    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--train-config",
        str(profile_cfg_path.relative_to(ROOT)),
    ]

    warmup_command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--train-config",
        str(warmup_cfg_path.relative_to(ROOT)),
    ] if warmup_cfg_path is not None else None

    warmup_log_path = run_dir / "warmup_stdout.log"
    warmup_return_code: int | None = None
    if warmup_command is not None:
        warmup_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        print("Running warmup pass to exclude first simulator startup cost...")
        warmup_return_code = run_warmup(
            warmup_command,
            cwd=ROOT,
            log_path=warmup_log_path,
        )
        if warmup_return_code != 0:
            summary = {
                "command": command,
                "warmup_command": warmup_command,
                "train_config_original": str(train_config_path),
                "train_config_profile_copy": str(profile_cfg_path),
                "env_config_profile_copy": str(env_cfg_profile_path),
                "warmup_train_config_copy": str(warmup_cfg_path),
                "return_code": None,
                "warmup_return_code": warmup_return_code,
                "warnings": ["Warmup run failed; profiling run was not started."],
                "artifacts": {
                    "warmup_stdout_log": str(warmup_log_path),
                },
            }
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nSummary written to: {summary_path}")
            raise SystemExit(warmup_return_code)

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stdout_queue: queue.Queue[str] = queue.Queue()
    reader_thread = threading.Thread(
        target=enqueue_stdout, args=(process.stdout, stdout_queue), daemon=True
    )
    reader_thread.start()

    proc_psutil = psutil.Process(process.pid) if psutil is not None else None
    if psutil is not None:
        safe_cpu_total_percent()
        safe_process_cpu_percent(proc_psutil)

    use_nvml = try_init_nvml()
    started = time.perf_counter()
    last_sample_at = 0.0
    stdout_lines: list[str] = []
    parsed_train_lines: list[dict[str, Any]] = []
    parsed_rollout_lines: list[dict[str, Any]] = []
    warnings: list[str] = []
    samples: list[Sample] = []

    if psutil is None:
        warnings.append("psutil not installed; CPU utilization samples will be unavailable.")
    if not use_nvml:
        gpu_probe, _ = query_gpu_percent_nvidia_smi()
        if gpu_probe is None:
            warnings.append(
                "Neither pynvml nor nvidia-smi was available; GPU utilization samples will be unavailable."
            )

    try:
        while process.poll() is None or not stdout_queue.empty():
            while True:
                try:
                    line = stdout_queue.get_nowait()
                except queue.Empty:
                    break
                stdout_lines.append(line)
                print(line, end="")

                match = TRAIN_LINE_RE.search(line.strip())
                if match:
                    row = {k: float(v) if "." in v or "e" in v.lower() else int(v) for k, v in match.groupdict().items()}
                    parsed_train_lines.append(row)
                    continue

                match = ROLLOUT_LINE_RE.search(line.strip())
                if match:
                    row: dict[str, Any] = {}
                    for key, value in match.groupdict().items():
                        if value is None:
                            row[key] = None
                        elif key in {"episodes", "steps"}:
                            row[key] = int(value)
                        else:
                            row[key] = float(value)
                    if row.get("rollout_ms_per_step") is not None:
                        row["rollout_fps"] = 1000.0 / float(row["rollout_ms_per_step"])
                    parsed_rollout_lines.append(row)

            now = time.perf_counter()
            if now - last_sample_at >= args.sample_interval and process.poll() is None:
                cpu_total = safe_cpu_total_percent()
                cpu_proc = safe_process_cpu_percent(proc_psutil)
                gpu_pct, gpu_mem_pct = query_gpu_percent(use_nvml)
                samples.append(
                    Sample(
                        elapsed_s=now - started,
                        cpu_total_percent=cpu_total,
                        cpu_process_percent=cpu_proc,
                        gpu_percent=gpu_pct,
                        gpu_mem_percent=gpu_mem_pct,
                    )
                )
                last_sample_at = now

            time.sleep(0.1)
    finally:
        return_code = process.wait()
        reader_thread.join(timeout=2.0)
        if use_nvml and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    raw_log_path = run_dir / "train_stdout.log"
    raw_log_path.write_text("".join(stdout_lines), encoding="utf-8")

    samples_csv_path = run_dir / "resource_samples.csv"
    write_samples_csv(samples_csv_path, samples)

    latest_rollout = parsed_rollout_lines[-1] if parsed_rollout_lines else {}
    env_step_structure = {
        "pipeline": [
            "task.action_to_setpoint(actions, prev_backend_state, task_state)",
            "backend.step(setpoints)",
            "task.compute_rewards(prev_backend_state, backend_state, task_state)",
            "task.compute_terminated_truncated(backend_state, task_state, step_count)",
            "task.build_obs(backend_state, task_state)",
            "task.build_state(backend_state, task_state)",
            "build info['timing'] diagnostics",
        ],
        "timing_ms_per_step": {
            "total": latest_rollout.get("env_total_ms"),
            "action_to_setpoint": latest_rollout.get("env_action_ms"),
            "backend_step": latest_rollout.get("env_backend_ms"),
            "compute_rewards": latest_rollout.get("env_reward_ms"),
            "compute_done": latest_rollout.get("env_done_ms"),
            "build_obs_state": latest_rollout.get("env_obs_state_ms"),
            "build_info": latest_rollout.get("env_info_ms"),
        },
    }

    summary = {
        "command": command,
        "warmup_command": warmup_command,
        "train_config_original": str(train_config_path),
        "train_config_profile_copy": str(profile_cfg_path),
        "env_config_profile_copy": str(env_cfg_profile_path),
        "warmup_train_config_copy": str(warmup_cfg_path) if warmup_cfg_path is not None else None,
        "return_code": return_code,
        "warmup_return_code": warmup_return_code,
        "requested_metrics": {
            "num_envs": num_envs,
            "rollout_fps": latest_rollout.get("rollout_fps"),
            "ppo_batch_size": {
                "configured_minibatch_size": configured_batch_size,
                "effective_update_batch_size": effective_update_batch_size,
                "rollout_steps": rollout_steps,
                "policy_agent_count": policy_agent_count,
                "note": "Current learner updates on the full flattened rollout batch; minibatch_size is config-only unless learner code is extended.",
            },
            "cpu_utilization": summarize_samples(samples)["cpu_total_percent"],
            "gpu_utilization": summarize_samples(samples)["gpu_percent"],
            "env_step_structure": env_step_structure,
        },
        "derived_context": {
            "num_envs_reason": num_envs_reason,
            "resource_samples": summarize_samples(samples),
            "parsed_train_lines": parsed_train_lines,
            "parsed_rollout_lines": parsed_rollout_lines,
        },
        "artifacts": {
            "warmup_stdout_log": str(warmup_log_path) if warmup_command is not None else None,
            "stdout_log": str(raw_log_path),
            "resource_samples_csv": str(samples_csv_path),
        },
        "warnings": warnings,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSummary written to: {summary_path}")

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
