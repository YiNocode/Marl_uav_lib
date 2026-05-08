"""Monitor single-env or VecEnv training logs and flag common rollout/update failures."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TRAIN_LINE_RE = re.compile(
    r"^\[(?P<tag>train|vec-train)\]\s+epoch=(?P<epoch>\d+)/(?P<num_epochs>\d+)\s+"
    r"(?:num_envs=(?P<num_envs>\d+)\s+)?"
    r"steps=(?P<steps>\d+)\s+avg_return=(?P<avg_return>[-+eE0-9.]+)\s+"
    r"avg_len=(?P<avg_len>[-+eE0-9.]+)"
    r"(?P<metrics>.*)$"
)

ROLLOUT_LINE_RE = re.compile(
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

EVAL_LINE_RE = re.compile(
    r"^\[eval\]\s+episodes=(?P<episodes>\d+)\s+avg_return=(?P<avg_return>[-+eE0-9.]+)\s+avg_len=(?P<avg_len>[-+eE0-9.]+)"
)
BAD_NUMERIC_TOKEN_RE = re.compile(r"(?<![a-zA-Z_])(nan|inf|-inf|\+inf)(?![a-zA-Z_])", re.IGNORECASE)


@dataclass
class Alert:
    level: str
    kind: str
    message: str
    line_no: int
    raw_line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-config",
        type=str,
        default="",
        help="Optional train config. If set with --launch, the monitor runs train.py itself.",
    )
    parser.add_argument(
        "--stdout-log",
        type=Path,
        default=None,
        help="Analyze an existing stdout log instead of launching training.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch scripts/train.py and monitor stdout in real time.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used when --launch is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "training_monitors",
        help="Where monitor logs and JSON summaries are written.",
    )
    parser.add_argument(
        "--min-rollout-fps",
        type=float,
        default=20.0,
        help="Alert when rollout FPS falls below this threshold.",
    )
    parser.add_argument(
        "--max-rollout-ms-per-step",
        type=float,
        default=50.0,
        help="Alert when rollout ms/step exceeds this threshold.",
    )
    parser.add_argument(
        "--max-update-ms-per-step",
        type=float,
        default=10.0,
        help="Alert when PPO update ms/step exceeds this threshold.",
    )
    parser.add_argument(
        "--max-approx-kl",
        type=float,
        default=0.20,
        help="Alert when approx_kl exceeds this threshold, if present in logs.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=100.0,
        help="Alert when grad_norm exceeds this threshold, if present in logs.",
    )
    parser.add_argument(
        "--max-clip-fraction",
        type=float,
        default=0.80,
        help="Alert when clip_fraction exceeds this threshold, if present in logs.",
    )
    parser.add_argument(
        "--min-episodes-per-rollout",
        type=int,
        default=1,
        help="Alert when a rollout reports fewer finished episodes than expected.",
    )
    return parser.parse_args()


def _safe_float(text: str) -> float | None:
    try:
        value = float(text)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _parse_train_metrics(metrics_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for token in metrics_text.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed = _safe_float(value)
        if parsed is not None:
            metrics[key] = parsed
    return metrics


class TrainingMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.alerts: list[Alert] = []
        self.rollout_rows: list[dict[str, Any]] = []
        self.train_rows: list[dict[str, Any]] = []
        self.eval_rows: list[dict[str, Any]] = []

    def _push_alert(self, *, level: str, kind: str, message: str, line_no: int, raw_line: str) -> None:
        alert = Alert(level=level, kind=kind, message=message, line_no=line_no, raw_line=raw_line.rstrip("\n"))
        self.alerts.append(alert)
        print(f"[{level.upper()}][{kind}] line={line_no}: {message}")

    def inspect_line(self, line_no: int, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        lowered = line.lower()
        if "traceback" in lowered:
            self._push_alert(
                level="error",
                kind="traceback",
                message="Training emitted a Python traceback.",
                line_no=line_no,
                raw_line=raw_line,
            )
        if BAD_NUMERIC_TOKEN_RE.search(line) is not None:
            self._push_alert(
                level="error",
                kind="numeric",
                message="Detected NaN/Inf token in training output.",
                line_no=line_no,
                raw_line=raw_line,
            )

        train_match = TRAIN_LINE_RE.search(line)
        if train_match:
            gd = train_match.groupdict()
            metrics = _parse_train_metrics(gd.get("metrics", ""))
            row = {
                "tag": gd["tag"],
                "epoch": int(gd["epoch"]),
                "num_epochs": int(gd["num_epochs"]),
                "num_envs": None if gd["num_envs"] is None else int(gd["num_envs"]),
                "steps": int(gd["steps"]),
                "avg_return": float(gd["avg_return"]),
                "avg_len": float(gd["avg_len"]),
                "metrics": metrics,
            }
            self.train_rows.append(row)

            if not math.isfinite(row["avg_return"]) or not math.isfinite(row["avg_len"]):
                self._push_alert(
                    level="error",
                    kind="train-metric",
                    message="avg_return or avg_len is not finite.",
                    line_no=line_no,
                    raw_line=raw_line,
                )
            if row["avg_len"] <= 0:
                self._push_alert(
                    level="warning",
                    kind="episode-length",
                    message="avg_len <= 0, rollout may be broken.",
                    line_no=line_no,
                    raw_line=raw_line,
                )

            approx_kl = metrics.get("train/approx_kl")
            if approx_kl is not None and approx_kl > self.args.max_approx_kl:
                self._push_alert(
                    level="warning",
                    kind="approx-kl",
                    message=f"approx_kl={approx_kl:.4f} exceeded threshold {self.args.max_approx_kl:.4f}.",
                    line_no=line_no,
                    raw_line=raw_line,
                )

            grad_norm = metrics.get("train/grad_norm")
            if grad_norm is not None and grad_norm > self.args.max_grad_norm:
                self._push_alert(
                    level="warning",
                    kind="grad-norm",
                    message=f"grad_norm={grad_norm:.4f} exceeded threshold {self.args.max_grad_norm:.4f}.",
                    line_no=line_no,
                    raw_line=raw_line,
                )

            clip_fraction = metrics.get("train/clip_fraction")
            if clip_fraction is not None and clip_fraction > self.args.max_clip_fraction:
                self._push_alert(
                    level="warning",
                    kind="clip-fraction",
                    message=f"clip_fraction={clip_fraction:.4f} exceeded threshold {self.args.max_clip_fraction:.4f}.",
                    line_no=line_no,
                    raw_line=raw_line,
                )
            return

        rollout_match = ROLLOUT_LINE_RE.search(line)
        if rollout_match:
            gd = rollout_match.groupdict()
            row: dict[str, Any] = {}
            for key, value in gd.items():
                if value is None:
                    row[key] = None
                elif key in {"episodes", "steps"}:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            if row["rollout_ms_per_step"] is not None:
                row["rollout_fps"] = 1000.0 / float(row["rollout_ms_per_step"])
            self.rollout_rows.append(row)

            if row["rollout_fps"] is not None and row["rollout_fps"] < self.args.min_rollout_fps:
                self._push_alert(
                    level="warning",
                    kind="throughput",
                    message=f"rollout_fps={row['rollout_fps']:.2f} dropped below threshold {self.args.min_rollout_fps:.2f}.",
                    line_no=line_no,
                    raw_line=raw_line,
                )
            if row["rollout_ms_per_step"] is not None and row["rollout_ms_per_step"] > self.args.max_rollout_ms_per_step:
                self._push_alert(
                    level="warning",
                    kind="rollout-latency",
                    message=(
                        f"rollout_ms_per_step={row['rollout_ms_per_step']:.2f} exceeded "
                        f"threshold {self.args.max_rollout_ms_per_step:.2f}."
                    ),
                    line_no=line_no,
                    raw_line=raw_line,
                )
            if row["update_ms_per_step"] is not None and row["update_ms_per_step"] > self.args.max_update_ms_per_step:
                self._push_alert(
                    level="warning",
                    kind="update-latency",
                    message=(
                        f"update_ms_per_step={row['update_ms_per_step']:.2f} exceeded "
                        f"threshold {self.args.max_update_ms_per_step:.2f}."
                    ),
                    line_no=line_no,
                    raw_line=raw_line,
                )
            if row["episodes"] is not None and row["episodes"] < self.args.min_episodes_per_rollout:
                self._push_alert(
                    level="warning",
                    kind="episode-count",
                    message=(
                        f"Only {row['episodes']} finished episodes were reported in one rollout; "
                        "this may indicate stuck envs or extremely long horizons."
                    ),
                    line_no=line_no,
                    raw_line=raw_line,
                )
            return

        eval_match = EVAL_LINE_RE.search(line)
        if eval_match:
            gd = eval_match.groupdict()
            row = {
                "episodes": int(gd["episodes"]),
                "avg_return": float(gd["avg_return"]),
                "avg_len": float(gd["avg_len"]),
            }
            self.eval_rows.append(row)
            if row["episodes"] <= 0:
                self._push_alert(
                    level="warning",
                    kind="eval-empty",
                    message="Eval reported zero episodes.",
                    line_no=line_no,
                    raw_line=raw_line,
                )

    def build_summary(self, *, return_code: int | None, source: str, stdout_log: str) -> dict[str, Any]:
        last_train = self.train_rows[-1] if self.train_rows else None
        last_rollout = self.rollout_rows[-1] if self.rollout_rows else None
        last_eval = self.eval_rows[-1] if self.eval_rows else None
        return {
            "source": source,
            "stdout_log": stdout_log,
            "return_code": return_code,
            "num_train_lines": len(self.train_rows),
            "num_rollout_lines": len(self.rollout_rows),
            "num_eval_lines": len(self.eval_rows),
            "num_alerts": len(self.alerts),
            "alerts": [asdict(a) for a in self.alerts],
            "last_train": last_train,
            "last_rollout": last_rollout,
            "last_eval": last_eval,
        }


def analyze_existing_log(log_path: Path, monitor: TrainingMonitor) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for idx, line in enumerate(text.splitlines(keepends=True), start=1):
        monitor.inspect_line(idx, line)
    return monitor.build_summary(return_code=None, source="existing_log", stdout_log=str(log_path))


def launch_and_monitor(args: argparse.Namespace, monitor: TrainingMonitor, out_dir: Path) -> dict[str, Any]:
    if not args.train_config:
        raise ValueError("--train-config is required when using --launch.")

    command = [
        args.python,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--train-config",
        args.train_config,
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

    stdout_log = out_dir / "train_stdout.log"
    line_no = 0
    lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line_no += 1
        print(raw_line, end="")
        lines.append(raw_line)
        monitor.inspect_line(line_no, raw_line)

    return_code = proc.wait()
    stdout_log.write_text("".join(lines), encoding="utf-8")
    return monitor.build_summary(return_code=return_code, source="live_launch", stdout_log=str(stdout_log))


def main() -> None:
    args = parse_args()
    if not args.launch and args.stdout_log is None:
        raise ValueError("Use either --launch or --stdout-log.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"monitor_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = TrainingMonitor(args)
    if args.launch:
        summary = launch_and_monitor(args, monitor, out_dir)
    else:
        summary = analyze_existing_log(args.stdout_log.resolve(), monitor)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Monitor Summary ===")
    print(f"alerts={summary['num_alerts']} train_lines={summary['num_train_lines']} rollout_lines={summary['num_rollout_lines']}")
    print(f"summary: {summary_path}")

    if summary["return_code"] not in (None, 0):
        raise SystemExit(int(summary["return_code"]))


if __name__ == "__main__":
    main()
