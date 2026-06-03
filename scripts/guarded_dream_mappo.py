"""Guarded launcher for Dream-MAPPO pursuit-evasion experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from monitor_training_run import TrainingMonitor
from marl_uav.utils.stdio import configure_utf8_stdio, utf8_subprocess_env


configure_utf8_stdio()


DEFAULT_DREAM_PYFLYT_EX1_CFG = ROOT / "configs" / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"
DEFAULT_DREAM_GENESIS_EX1_CFG = ROOT / "configs" / "experiment" / "pursuit_evasion_dream_mappo_3v1_genesis.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one guarded Dream-MAPPO training job with isolated artifacts.")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Top-level train config path. Defaults to the Dream ex1 config.",
    )
    p.add_argument(
        "--backend",
        choices=("genesis", "pyflyt"),
        default="genesis",
        help="Pick the simulator backend for the default Dream-MAPPO config.",
    )
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocesses.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "guarded_dream_mappo_runs",
        help="Parent directory for isolated run folders.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional custom run folder suffix. Defaults to config stem + timestamp.",
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Override rollout_steps in the materialized run config.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval.py after successful training.",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes passed to eval.py.",
    )
    p.add_argument("--eval-num-seeds", type=int, default=1)
    p.add_argument("--min-rollout-fps", type=float, default=20.0)
    p.add_argument("--max-rollout-ms-per-step", type=float, default=50.0)
    p.add_argument("--max-update-ms-per-step", type=float, default=10.0)
    p.add_argument("--max-approx-kl", type=float, default=0.20)
    p.add_argument("--max-grad-norm", type=float, default=100.0)
    p.add_argument("--max-clip-fraction", type=float, default=0.80)
    p.add_argument("--min-episodes-per-rollout", type=int, default=1)
    return p.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config(args: argparse.Namespace) -> Path:
    if args.config:
        return (ROOT / args.config).resolve()
    if args.backend == "genesis":
        return DEFAULT_DREAM_GENESIS_EX1_CFG
    return DEFAULT_DREAM_PYFLYT_EX1_CFG


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


def materialize_train_config(
    *,
    source_cfg: Path,
    run_dir: Path,
    rollout_steps: int | None,
) -> str:
    cfg = load_yaml(source_cfg)
    if rollout_steps is not None:
        cfg["rollout_steps"] = int(rollout_steps)
    try:
        cfg["train_results_dir"] = run_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        cfg["train_results_dir"] = str(run_dir.resolve())
    out_path = run_dir / f"{run_dir.name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path.relative_to(ROOT).as_posix()


def run_monitored_training(
    *,
    args: argparse.Namespace,
    train_cfg_relpath: str,
    run_dir: Path,
) -> tuple[int, dict[str, Any]]:
    cmd = [
        args.python,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--train-config",
        train_cfg_relpath,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=utf8_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None

    monitor = TrainingMonitor(make_monitor_args(args))
    stdout_lines: list[str] = []
    stdout_log = run_dir / "train_stdout.log"
    line_no = 0
    with open(stdout_log, "w", encoding="utf-8") as log_f:
        for raw_line in proc.stdout:
            line_no += 1
            print(raw_line, end="")
            stdout_lines.append(raw_line)
            log_f.write(raw_line)
            log_f.flush()
            monitor.inspect_line(line_no, raw_line)

    rc = proc.wait()
    summary = monitor.build_summary(
        return_code=rc,
        source="guarded_dream_mappo_train",
        stdout_log=str(stdout_log),
    )
    (run_dir / "train_monitor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rc, summary


def run_logged_command(command: list[str], *, log_path: Path) -> int:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=utf8_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    return proc.returncode


def run_eval(
    *,
    args: argparse.Namespace,
    train_cfg_relpath: str,
    seed: int,
    run_dir: Path,
) -> int:
    ckpt_path = run_dir / "checkpoints" / str(seed) / "best.pt"
    try:
        ckpt_rel = ckpt_path.relative_to(ROOT)
    except ValueError:
        ckpt_rel = ckpt_path
    cmd = [
        args.python,
        str(ROOT / "scripts" / "eval.py"),
        "--config",
        train_cfg_relpath,
        "--seed",
        str(seed),
        "--episodes",
        str(int(args.eval_episodes)),
        "--num-seeds",
        str(int(args.eval_num_seeds)),
        "--ckpt",
        ckpt_rel.as_posix(),
    ]
    return run_logged_command(cmd, log_path=run_dir / "eval_stdout.log")


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args)
    cfg = load_yaml(config_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = config_path.stem if not args.run_name else args.run_name
    run_dir = args.output_dir / f"{stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_relpath = materialize_train_config(
        source_cfg=config_path,
        run_dir=run_dir,
        rollout_steps=args.rollout_steps,
    )
    train_rc, monitor_summary = run_monitored_training(
        args=args,
        train_cfg_relpath=train_cfg_relpath,
        run_dir=run_dir,
    )

    eval_rc: int | None = None
    if train_rc == 0 and not args.skip_eval:
        eval_rc = run_eval(
            args=args,
            train_cfg_relpath=train_cfg_relpath,
            seed=int(cfg.get("seed", 0)),
            run_dir=run_dir,
        )

    summary = {
        "config": str(config_path.relative_to(ROOT)),
        "train_config_effective": train_cfg_relpath,
        "train_results_dir": str(run_dir.resolve()),
        "train_return_code": int(train_rc),
        "eval_return_code": None if eval_rc is None else int(eval_rc),
        "monitor_summary": monitor_summary,
        "artifacts": {
            "train_stdout_log": str(run_dir / "train_stdout.log"),
            "train_monitor_summary": str(run_dir / "train_monitor_summary.json"),
            "tb_dir": str(run_dir / "tb_" / str(int(cfg.get("seed", 0)))),
            "checkpoints_dir": str(run_dir / "checkpoints" / str(int(cfg.get("seed", 0)))),
        },
    }
    if eval_rc is not None:
        summary["artifacts"]["eval_stdout_log"] = str(run_dir / "eval_stdout.log")
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[guarded-dream-mappo] run dir: {run_dir}")
    print(f"[guarded-dream-mappo] summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
