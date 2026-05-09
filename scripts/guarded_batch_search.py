"""Guarded batch launcher for reward-grid and evader-speed sweep experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from monitor_training_run import TrainingMonitor

DEFAULT_GRID_CFG = ROOT / "configs" / "search" / "ex1_reward_grid.yaml"
DEFAULT_SPEED_CFG = ROOT / "configs" / "search" / "ex1_evader_speed_sweep.yaml"
DEFAULT_ROLLOUT_STEPS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configs, run monitored training, then optional eval and summaries."
    )
    parser.add_argument(
        "--mode",
        choices=("grid", "speed"),
        default="speed",
        required=False,
        help="Batch mode: reward grid search or evader speed sweep.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Search/sweep YAML path. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable for subprocesses.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse an existing manifest instead of regenerating configs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval after each successful training run.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the batch immediately when one training/eval run fails.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass through to config generation scripts.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Pass through to generation scripts.",
    )
    parser.add_argument(
        "--max-train-seeds",
        type=int,
        default=1,
        help="Reward-grid only: pass through to generation script.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Reward-grid only: pass through to generation script.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Reward-grid only: pass through to generation script.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override eval episodes passed to eval.py and generation scripts.",
    )
    parser.add_argument(
        "--eval-num-seeds",
        type=int,
        default=1,
        help="Eval seeds passed to eval.py and generation scripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "guarded_batch_runs",
        help="Directory for guardian logs and summaries.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help=(
            "Override train config rollout_steps for each guarded run "
            f"(default {DEFAULT_ROLLOUT_STEPS}). "
            "Written into run_dir/<run folder name>.yaml before invoking train.py."
        ),
    )
    parser.add_argument("--min-rollout-fps", type=float, default=20.0)
    parser.add_argument("--max-rollout-ms-per-step", type=float, default=50.0)
    parser.add_argument("--max-update-ms-per-step", type=float, default=10.0)
    parser.add_argument("--max-approx-kl", type=float, default=0.20)
    parser.add_argument("--max-grad-norm", type=float, default=100.0)
    parser.add_argument("--max-clip-fraction", type=float, default=0.80)
    parser.add_argument("--min-episodes-per-rollout", type=int, default=1)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_batch_config(args: argparse.Namespace) -> Path:
    if args.config:
        path = ROOT / args.config
    elif args.mode == "grid":
        path = DEFAULT_GRID_CFG
    else:
        path = DEFAULT_SPEED_CFG
    return path.resolve()


def manifest_path_for_config(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    output_dir_rel = str(cfg.get("output_dir", "configs/generated"))
    manifest_name = str(cfg.get("manifest_name", "manifest.csv"))
    manifest_path = ROOT / output_dir_rel / manifest_name
    if args.mode == "grid" and int(args.num_shards) > 1:
        stem = manifest_path.stem
        suffix = manifest_path.suffix
        manifest_path = manifest_path.with_name(
            f"{stem}_shard{int(args.shard_index) + 1}of{int(args.num_shards)}{suffix}"
        )
    return manifest_path


def build_generation_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    if args.mode == "grid":
        script = ROOT / "scripts" / "grid_search_ex1_rewards.py"
        cmd = [args.python, str(script), "--search-config", str(config_path.relative_to(ROOT))]
        if args.max_train_seeds is not None:
            cmd += ["--max-train-seeds", str(args.max_train_seeds)]
        if args.num_shards is not None:
            cmd += ["--num-shards", str(args.num_shards), "--shard-index", str(args.shard_index)]
    else:
        script = ROOT / "scripts" / "sweep_ex1_evader_speed.py"
        cmd = [args.python, str(script), "--sweep-config", str(config_path.relative_to(ROOT))]

    if args.eval_episodes is not None:
        cmd += ["--eval-episodes", str(args.eval_episodes)]
    if args.eval_num_seeds is not None:
        cmd += ["--eval-num-seeds", str(args.eval_num_seeds)]
    if args.max_runs is not None:
        cmd += ["--max-runs", str(args.max_runs)]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_logged_command(command: list[str], *, log_path: Path) -> int:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    return proc.returncode


def materialize_train_config_with_rollout(
    *,
    source_relpath: str,
    run_dir: Path,
    rollout_steps: int,
) -> str:
    """Copy manifest train YAML into run_dir with rollout_steps and train_results_dir set.

    TensorBoard / checkpoints go under ``run_dir`` (see ``train_results_dir`` in scripts/train.py).
    """
    src = ROOT / source_relpath
    cfg = load_yaml(src)
    cfg["rollout_steps"] = int(rollout_steps)
    try:
        cfg["train_results_dir"] = run_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        cfg["train_results_dir"] = str(run_dir.resolve())
    stem = run_dir.name
    out_path = run_dir / f"{stem}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path.relative_to(ROOT).as_posix()


def make_monitor_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        min_rollout_fps=float(args.min_rollout_fps),
        max_rollout_ms_per_step=float(args.max_rollout_ms_per_step),
        max_update_ms_per_step=float(args.max_update_ms_per_step),
        max_approx_kl=float(args.max_approx_kl),
        max_grad_norm=float(args.max_grad_norm),
        max_clip_fraction=float(args.max_clip_fraction),
        min_episodes_per_rollout=int(args.min_episodes_per_rollout),
    )


def run_monitored_training(
    *,
    args: argparse.Namespace,
    config_relpath: str,
    run_dir: Path,
) -> tuple[int, dict[str, Any], str]:
    train_cfg_relpath = materialize_train_config_with_rollout(
        source_relpath=config_relpath,
        run_dir=run_dir,
        rollout_steps=int(args.rollout_steps),
    )
    command = [
        args.python,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--train-config",
        train_cfg_relpath,
    ]
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None

    stdout_lines: list[str] = []
    monitor = TrainingMonitor(make_monitor_args(args))
    line_no = 0
    for raw_line in proc.stdout:
        line_no += 1
        print(raw_line, end="")
        stdout_lines.append(raw_line)
        monitor.inspect_line(line_no, raw_line)

    return_code = proc.wait()
    stdout_log = run_dir / "train_stdout.log"
    stdout_log.write_text("".join(stdout_lines), encoding="utf-8")
    summary = monitor.build_summary(
        return_code=return_code,
        source="guarded_batch_train",
        stdout_log=str(stdout_log),
    )
    (run_dir / "train_monitor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return return_code, summary, train_cfg_relpath


def run_eval(
    *,
    args: argparse.Namespace,
    train_config_relpath: str,
    seed: int,
    episodes: int | None,
    run_dir: Path,
    train_seed: int,
) -> int:
    """Invoke eval.py using the same materialized train YAML as training (includes ``train_results_dir``)."""
    eval_episodes = 20 if episodes is None else int(episodes)
    ckpt_path = run_dir / "checkpoints" / str(train_seed) / "best.pt"
    try:
        ckpt_rel = ckpt_path.relative_to(ROOT)
    except ValueError:
        ckpt_rel = ckpt_path
    command = [
        args.python,
        str(ROOT / "scripts" / "eval.py"),
        "--config",
        train_config_relpath,
        "--seed",
        str(seed),
        "--episodes",
        str(eval_episodes),
        "--num-seeds",
        str(args.eval_num_seeds),
        "--ckpt",
        ckpt_rel.as_posix(),
    ]
    return run_logged_command(command, log_path=run_dir / "eval_stdout.log")


def call_summary_script(args: argparse.Namespace, config_path: Path, run_dir: Path) -> int:
    if args.mode == "grid":
        script = ROOT / "scripts" / "grid_search_ex1_rewards.py"
        cmd = [args.python, str(script), "--search-config", str(config_path.relative_to(ROOT)), "--summarize-only"]
        cmd += ["--max-train-seeds", str(args.max_train_seeds)]
        cmd += ["--num-shards", str(args.num_shards), "--shard-index", str(args.shard_index)]
    else:
        script = ROOT / "scripts" / "sweep_ex1_evader_speed.py"
        cmd = [args.python, str(script), "--sweep-config", str(config_path.relative_to(ROOT)), "--summarize-only"]
    return run_logged_command(cmd, log_path=run_dir / "summary_stdout.log")


def main() -> None:
    args = parse_args()
    config_path = resolve_batch_config(args)
    cfg = load_yaml(config_path)
    manifest_path = manifest_path_for_config(args, cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    guardian_dir = args.output_dir / f"{args.mode}_guardian_{stamp}"
    guardian_dir.mkdir(parents=True, exist_ok=True)

    skip_generate = bool(args.skip_generate)
    if manifest_path.exists() and not args.overwrite and not skip_generate:
        print(
            f"[guardian] existing manifest detected at {manifest_path.relative_to(ROOT)}; "
            "reusing generated configs. Pass --overwrite to regenerate them."
        )
        skip_generate = True

    if not skip_generate:
        gen_cmd = build_generation_command(args, config_path)
        print(f"[generate] {' '.join(gen_cmd)}")
        gen_rc = run_logged_command(gen_cmd, log_path=guardian_dir / "generate_stdout.log")
        if gen_rc != 0:
            raise SystemExit(gen_rc)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_rows = read_manifest(manifest_path)
    batch_records: list[dict[str, Any]] = []

    print(f"[guardian] manifest={manifest_path.relative_to(ROOT)} runs={len(manifest_rows)}")
    for idx, row in enumerate(manifest_rows, start=1):
        run_name = str(row.get("run_name", f"run_{idx:04d}"))
        config_relpath = str(row["config_relpath"])
        run_dir = guardian_dir / f"{idx:04d}_{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Guarded Run {idx}/{len(manifest_rows)}: {run_name} ===")
        train_rc, monitor_summary, train_cfg_effective = run_monitored_training(
            args=args,
            config_relpath=config_relpath,
            run_dir=run_dir,
        )

        run_record: dict[str, Any] = {
            "run_name": run_name,
            "index": idx,
            "config_relpath": config_relpath,
            "rollout_steps": int(args.rollout_steps),
            "train_results_dir": str(run_dir.resolve()),
            "train_config_effective": train_cfg_effective,
            "seed": int(row.get("seed", 0)),
            "train_return_code": int(train_rc),
            "monitor_alerts": int(monitor_summary.get("num_alerts", 0)),
            "last_train": monitor_summary.get("last_train"),
            "last_rollout": monitor_summary.get("last_rollout"),
            "artifacts": {
                "train_stdout_log": str(run_dir / "train_stdout.log"),
                "train_monitor_summary": str(run_dir / "train_monitor_summary.json"),
            },
        }

        eval_rc: int | None = None
        if train_rc == 0 and not args.skip_eval:
            eval_rc = run_eval(
                args=args,
                train_config_relpath=train_cfg_effective,
                seed=int(row.get("seed", 0)),
                train_seed=int(row.get("seed", 0)),
                episodes=args.eval_episodes,
                run_dir=run_dir,
            )
            run_record["eval_return_code"] = int(eval_rc)
            run_record["artifacts"]["eval_stdout_log"] = str(run_dir / "eval_stdout.log")
        else:
            run_record["eval_return_code"] = None

        batch_records.append(run_record)
        batch_summary_path = guardian_dir / "batch_summary.json"
        batch_summary_path.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "batch_config": str(config_path),
                    "manifest": str(manifest_path),
                    "runs": batch_records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        failed = train_rc != 0 or (eval_rc is not None and eval_rc != 0)
        if failed and args.stop_on_failure:
            print("[guardian] stop-on-failure enabled, terminating batch early.")
            break

    summary_rc = call_summary_script(args, config_path, guardian_dir)
    if summary_rc != 0:
        print("[guardian] warning: summary recomputation failed.")

    print(f"\n[guardian] batch summary: {guardian_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
