"""E1.1 open-space main benchmark runner.

This script keeps the main benchmark separate from older ex1/ex2 scripts:

- generates per-method/per-seed training configs under configs/generated/e1_1_open_space/
- trains RL methods through the existing scripts/train.py entry point
- evaluates RL checkpoints and the fixed-ring heuristic on PyFlyt
- writes per-episode records plus by-seed/by-method summaries under
  results/e1_1_open_space_pyflyt/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_uav.agents.mac import MAC
from marl_uav.control.fixed_ring_pursuit import make_fixed_ring_get_actions_fn
from marl_uav.envs.factories import build_env_from_config
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.checkpoint import load_checkpoint
from marl_uav.utils.config import load_config


def _load_eval_helpers():
    path = ROOT / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("e11_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import eval helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e11_eval_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod.build_policy, mod.build_learner


build_policy, build_learner = _load_eval_helpers()


def load_suite(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    if "methods" not in cfg:
        raise ValueError(f"Suite config missing 'methods': {path}")
    return cfg


def selected_methods(suite: dict[str, Any], names: list[str] | None) -> list[str]:
    available = list((suite.get("methods") or {}).keys())
    if not names:
        return available
    missing = [m for m in names if m not in available]
    if missing:
        raise ValueError(f"Unknown method(s): {missing}. Available: {available}")
    return names


def generated_config_path(suite: dict[str, Any], method: str, seed: int) -> Path:
    gen_dir = ROOT / str(suite.get("generated_config_dir", "configs/generated/e1_1_open_space"))
    scenario = str(suite.get("scenario", "e1_1_open_space"))
    return gen_dir / f"{scenario}_pyflyt_{method}_seed{seed}.yaml"


def generate_train_configs(
    suite: dict[str, Any],
    methods: list[str],
    *,
    overwrite: bool,
) -> list[Path]:
    seeds = [int(s) for s in suite.get("seeds", [101, 102, 103])]
    method_cfgs = suite["methods"]
    written: list[Path] = []
    for method in methods:
        meta = dict(method_cfgs[method] or {})
        if not bool(meta.get("train", False)):
            continue
        base_path = ROOT / str(meta["config"])
        cfg = load_config(base_path)
        for seed in seeds:
            out = generated_config_path(suite, method, seed)
            if out.exists() and not overwrite:
                written.append(out)
                continue
            cfg_seed = dict(cfg)
            cfg_seed["seed"] = int(seed)
            train_root = Path(str(cfg_seed.get("train_results_dir", "")))
            if train_root:
                cfg_seed["train_results_dir"] = str(train_root).replace("\\", "/")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_seed, f, sort_keys=False, allow_unicode=True)
            written.append(out)
    return written


def checkpoint_path_for_config(cfg_path: Path) -> Path:
    cfg = load_config(cfg_path)
    seed = int(cfg.get("seed", 0))
    result_dir = Path(str(cfg.get("train_results_dir") or f"results/{cfg_path.stem}"))
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    return result_dir / "checkpoints" / str(seed) / "best.pt"


def run_training(
    suite: dict[str, Any],
    methods: list[str],
    *,
    skip_existing: bool,
    overwrite_configs: bool,
) -> None:
    cfg_paths = generate_train_configs(suite, methods, overwrite=overwrite_configs)
    for cfg_path in cfg_paths:
        ckpt = checkpoint_path_for_config(cfg_path)
        if skip_existing and ckpt.is_file():
            print(f"[train] skip existing checkpoint: {ckpt}")
            continue
        rel = cfg_path.relative_to(ROOT)
        print(f"[train] {rel}")
        subprocess.run(
            [sys.executable, "scripts/train.py", "--train-config", str(rel)],
            cwd=ROOT,
            check=True,
        )


def _window_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return math.nan
    kk = min(int(k), int(values.size))
    return float(np.mean(values[-kk:]))


def _structure_series_from_trajectory(traj_xyz: np.ndarray) -> list[dict[str, float]]:
    from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
        compute_pursuit_structure_metrics_3v1,
    )

    arr = np.asarray(traj_xyz, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return []
    return [
        compute_pursuit_structure_metrics_3v1(arr[t, :3, :], arr[t, 3, :])
        for t in range(arr.shape[0])
    ]


def _value_array(series: list[dict[str, Any]], key: str) -> np.ndarray:
    if not series:
        return np.zeros(0, dtype=np.float64)
    vals: list[float] = []
    for row in series:
        if key in row:
            vals.append(float(row[key]))
    return np.asarray(vals, dtype=np.float64)


def _xy_velocity_series(traj_xyz: np.ndarray) -> np.ndarray:
    xy = np.asarray(traj_xyz[:, :, :2], dtype=np.float64)
    if xy.shape[0] <= 1:
        return np.zeros_like(xy, dtype=np.float64)
    dxy = np.diff(xy, axis=0)
    vel = np.zeros_like(xy, dtype=np.float64)
    vel[1:] = dxy
    vel[0] = dxy[0]
    return vel


def _fesc_series(traj_xyz: np.ndarray, n_theta: int = 72) -> np.ndarray:
    arr = np.asarray(traj_xyz, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return np.zeros(0, dtype=np.float64)
    vel_xy = _xy_velocity_series(arr)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    ev_proj = vel_xy[:, 3, :] @ dirs.T
    pu_proj = np.einsum("tad,kd->tak", vel_xy[:, :3, :], dirs)
    return np.max(ev_proj - np.max(pu_proj, axis=1), axis=1).astype(np.float64)


def _role_stability_series(traj_xyz: np.ndarray) -> np.ndarray:
    arr = np.asarray(traj_xyz, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 4:
        return np.zeros(0, dtype=np.float64)
    rel_xy = arr[:, :3, :2] - arr[:, 3:4, :2]
    theta = np.arctan2(rel_xy[:, :, 1], rel_xy[:, :, 0])
    ranks = np.zeros_like(theta, dtype=np.int64)
    for t in range(theta.shape[0]):
        order = np.argsort(theta[t])
        ranks[t, order] = np.arange(3, dtype=np.int64)
    stability = np.ones(theta.shape[0], dtype=np.float64)
    if theta.shape[0] >= 2:
        stability[1:] = 1.0 - np.mean((ranks[1:] != ranks[:-1]).astype(np.float64), axis=1)
    return stability


def _episode_metrics(
    *,
    info: dict[str, Any],
    trajectory: np.ndarray,
    terminal_window: int,
    control_hz: float,
) -> dict[str, Any]:
    series = info.get("pursuit_structure_series")
    if not isinstance(series, list) or not series:
        series = _structure_series_from_trajectory(trajectory)
    cov = _value_array(series, "C_cov")
    col = _value_array(series, "C_col")
    dang = _value_array(series, "D_ang")
    phi_max = _value_array(series, "phi_max")
    fesc = _fesc_series(trajectory)
    role_stability = _role_stability_series(trajectory)

    captured = bool(info.get("capture", False))
    capture_step = int(info.get("capture_step", -1))
    if captured and capture_step < 0:
        capture_step = int(info.get("episode_len", 0))
    capture_time_s = float(capture_step / control_hz) if captured and control_hz > 0 else math.nan

    max_gap = _window_mean(phi_max, terminal_window)
    return {
        "episode_return": float(info.get("episode_return", math.nan)),
        "episode_len": int(info.get("episode_len", 0)),
        "captured": int(captured),
        "capture_step": capture_step if captured else "",
        "capture_time_s": capture_time_s,
        "collision": int(bool(info.get("collision", False))),
        "timeout": int(bool(info.get("timeout", False))),
        "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
        "pursuer_oob": int(bool(info.get("pursuer_oob", False))),
        f"C_cov_last{terminal_window}": _window_mean(cov, terminal_window),
        f"C_col_last{terminal_window}": _window_mean(col, terminal_window),
        f"D_ang_last{terminal_window}": _window_mean(dang, terminal_window),
        f"max_escape_gap_last{terminal_window}": max_gap,
        f"max_escape_gap_deg_last{terminal_window}": float(np.degrees(max_gap)) if np.isfinite(max_gap) else math.nan,
        f"F_esc_last{terminal_window}": _window_mean(fesc, terminal_window),
        f"role_stability_last{terminal_window}": _window_mean(role_stability, terminal_window),
        f"role_instability_last{terminal_window}": 1.0 - _window_mean(role_stability, terminal_window),
    }


def _build_rl_worker(cfg_path: Path, seed: int) -> RolloutWorker:
    cfg = load_config(cfg_path)
    env_cfg_path = ROOT / str(cfg["env"])
    algo_cfg_path = ROOT / str(cfg["algo"])
    model_cfg_path = ROOT / str(cfg["model"])
    env = build_env_from_config(env_cfg_path, seed=seed, task_cfg=cfg.get("task", {}))
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        env.reset(seed=seed)
    policy_core = build_policy(model_cfg_path, env, algo_cfg_path)
    n_actions = (
        env.n_actions
        if getattr(policy_core, "action_space_type", "discrete") == "discrete"
        else (getattr(policy_core, "action_dim", None) or 0)
    )
    mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions, n_agents=env.num_agents)
    mac.policy = policy_core
    learner = build_learner(algo_cfg_path, policy_core)
    ckpt = checkpoint_path_for_config(cfg_path)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint for {cfg_path.relative_to(ROOT)}: {ckpt}. "
            "Run benchmark_e1_1_open_space.py --mode train first."
        )
    load_checkpoint(ckpt, learner)
    mac.set_test_mode(True)
    return RolloutWorker(env=env, policy=mac)


def _build_fixed_ring_worker(cfg_path: Path, seed: int) -> RolloutWorker:
    cfg = load_config(cfg_path)
    env = build_env_from_config(ROOT / str(cfg["env"]), seed=seed, task_cfg=cfg.get("task", {}))
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        env.reset(seed=seed)
    ring_cfg = dict(cfg.get("fixed_ring", {}) or {})
    get_actions = make_fixed_ring_get_actions_fn(env, **ring_cfg)
    return RolloutWorker(env=env, policy=object(), get_actions_fn=get_actions)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        fv = float(value)
        if np.isfinite(fv):
            vals.append(fv)
    return float(np.mean(vals)) if vals else math.nan


def _finite_std(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        fv = float(value)
        if np.isfinite(fv):
            vals.append(fv)
    return float(np.std(vals)) if vals else math.nan


def summarize(records: list[dict[str, Any]], *, terminal_window: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_keys = [
        "episode_return",
        "episode_len",
        "capture_time_s",
        f"C_cov_last{terminal_window}",
        f"C_col_last{terminal_window}",
        f"D_ang_last{terminal_window}",
        f"max_escape_gap_last{terminal_window}",
        f"max_escape_gap_deg_last{terminal_window}",
        f"F_esc_last{terminal_window}",
        f"role_stability_last{terminal_window}",
        f"role_instability_last{terminal_window}",
    ]

    by_seed_rows: list[dict[str, Any]] = []
    seed_keys = sorted({(r["method"], r["train_seed"]) for r in records})
    for method, seed in seed_keys:
        sub = [r for r in records if r["method"] == method and r["train_seed"] == seed]
        row: dict[str, Any] = {
            "method": method,
            "train_seed": seed,
            "num_episodes": len(sub),
            "capture_rate": _finite_mean(sub, "captured"),
            "collision_rate": _finite_mean(sub, "collision"),
            "timeout_rate": _finite_mean(sub, "timeout"),
            "out_of_bounds_rate": _finite_mean(sub, "out_of_bounds"),
        }
        for key in metric_keys:
            row[f"mean_{key}"] = _finite_mean(sub, key)
            row[f"std_{key}"] = _finite_std(sub, key)
        by_seed_rows.append(row)

    by_method_rows: list[dict[str, Any]] = []
    for method in sorted({r["method"] for r in records}):
        sub = [r for r in records if r["method"] == method]
        row = {
            "method": method,
            "num_train_seeds": len({r["train_seed"] for r in sub}),
            "num_episodes": len(sub),
            "capture_rate": _finite_mean(sub, "captured"),
            "collision_rate": _finite_mean(sub, "collision"),
            "timeout_rate": _finite_mean(sub, "timeout"),
            "out_of_bounds_rate": _finite_mean(sub, "out_of_bounds"),
        }
        for key in metric_keys:
            row[f"mean_{key}"] = _finite_mean(sub, key)
            row[f"std_{key}"] = _finite_std(sub, key)
        by_method_rows.append(row)
    return by_seed_rows, by_method_rows


def run_evaluation(
    suite: dict[str, Any],
    methods: list[str],
    *,
    episodes_override: int | None,
    allow_missing_checkpoints: bool,
) -> None:
    seeds = [int(s) for s in suite.get("seeds", [101, 102, 103])]
    eval_cfg = dict(suite.get("eval", {}) or {})
    episodes = int(episodes_override or eval_cfg.get("episodes_per_seed", 200))
    base_seed = int(eval_cfg.get("base_seed", 7010))
    terminal_window = int(eval_cfg.get("terminal_window", 30))
    result_root = ROOT / str(suite.get("result_root", "results/e1_1_open_space_pyflyt"))
    record_dir = result_root / "eval_records"
    all_records: list[dict[str, Any]] = []

    for method in methods:
        meta = dict(suite["methods"][method] or {})
        base_cfg_path = ROOT / str(meta["config"])
        method_kind = str(meta.get("kind", "rl"))
        for seed in seeds:
            if method_kind == "rl":
                cfg_path = generated_config_path(suite, method, seed)
                if not cfg_path.is_file():
                    generate_train_configs(suite, [method], overwrite=False)
                try:
                    worker = _build_rl_worker(cfg_path, seed=base_seed + seed)
                except FileNotFoundError as exc:
                    if allow_missing_checkpoints:
                        print(f"[eval] skip missing checkpoint: {exc}")
                        continue
                    raise
            elif method_kind == "heuristic":
                worker = _build_fixed_ring_worker(base_cfg_path, seed=base_seed + seed)
            else:
                raise ValueError(f"Unsupported method kind={method_kind!r} for {method}")

            env_cfg = load_config(ROOT / str(load_config(base_cfg_path)["env"]))
            control_hz = float((env_cfg.get("backend", {}) or {}).get("control_hz", 60))
            method_seed_rows: list[dict[str, Any]] = []
            print(f"[eval] method={method} seed={seed} episodes={episodes}")
            try:
                for ep in range(episodes):
                    ep_seed = base_seed * 1_000_000 + seed * 10_000 + ep
                    _, info = worker.collect_episode(seed=ep_seed, record_trajectory=True)
                    traj = np.asarray(info.get("trajectory"), dtype=np.float32)
                    rec = {
                        "suite": str(suite.get("suite", "E1")),
                        "scenario": str(suite.get("scenario", "e1_1_open_space")),
                        "backend": str(suite.get("backend", "pyflyt")),
                        "method": method,
                        "train_seed": seed,
                        "eval_episode": ep,
                        "eval_seed": ep_seed,
                    }
                    rec.update(
                        _episode_metrics(
                            info=info,
                            trajectory=traj,
                            terminal_window=terminal_window,
                            control_hz=control_hz,
                        )
                    )
                    method_seed_rows.append(rec)
                    all_records.append(rec)
            finally:
                try:
                    worker.env.close()
                except Exception:
                    pass
            _write_csv(record_dir / f"{method}_seed{seed}_records.csv", method_seed_rows)

    _write_csv(record_dir / "e1_1_open_space_all_records.csv", all_records)
    by_seed, by_method = summarize(all_records, terminal_window=terminal_window)
    _write_csv(result_root / "e1_1_open_space_summary_by_seed.csv", by_seed)
    _write_csv(result_root / "e1_1_open_space_summary_by_method.csv", by_method)
    result_root.mkdir(parents=True, exist_ok=True)
    with open(result_root / "e1_1_open_space_eval_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": suite,
                "num_records": len(all_records),
                "terminal_window": terminal_window,
                "episodes_per_seed": episodes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E1.1 open-space PyFlyt benchmark.")
    p.add_argument(
        "--suite-config",
        type=str,
        default="configs/benchmark/e1_1_open_space_suite.yaml",
    )
    p.add_argument(
        "--mode",
        choices=("generate-configs", "train", "eval", "all"),
        default="generate-configs",
    )
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--overwrite-configs", action="store_true")
    p.add_argument("--skip-existing-checkpoints", action="store_true")
    p.add_argument("--allow-missing-checkpoints", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_suite(ROOT / args.suite_config)
    methods = selected_methods(suite, args.methods)

    if args.mode in ("generate-configs", "train", "all"):
        paths = generate_train_configs(suite, methods, overwrite=bool(args.overwrite_configs))
        print(f"[generate] {len(paths)} generated/available train configs")

    if args.mode in ("train", "all"):
        run_training(
            suite,
            methods,
            skip_existing=bool(args.skip_existing_checkpoints),
            overwrite_configs=bool(args.overwrite_configs),
        )

    if args.mode in ("eval", "all"):
        run_evaluation(
            suite,
            methods,
            episodes_override=args.episodes,
            allow_missing_checkpoints=bool(args.allow_missing_checkpoints),
        )


if __name__ == "__main__":
    main()

