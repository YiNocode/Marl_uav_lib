"""Pure manifold tracking test: drive each pursuer with controller(g_i - p_i) only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_uav.envs.adapters.pyflyt_aviary_env import PURSUIT_EVASION_3V1_TASK_TYPES
from marl_uav.utils.config import load_config


def _load_eval_module():
    path = ROOT / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("marl_eval_pure_manifold", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval = _load_eval_module()
build_env = _eval.build_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a pure manifold-tracking dynamics test.")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"),
    )
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--rho", type=float, default=3.0, help="Desired manifold radius in world coordinates.")
    p.add_argument("--psi", type=float, default=0.0, help="Global phase offset of the manifold.")
    p.add_argument("--kp-xy", type=float, default=0.9)
    p.add_argument("--kp-z", type=float, default=0.8)
    p.add_argument("--evader-mode", type=str, default="fixed", choices=["fixed", "constant"])
    p.add_argument("--evader-vx", type=float, default=0.03)
    p.add_argument("--evader-vy", type=float, default=0.0)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def _world_manifold_targets(pursuer_xyz: np.ndarray, evader_xyz: np.ndarray, rho: float, psi: float) -> np.ndarray:
    pursuer_xyz = np.asarray(pursuer_xyz, dtype=np.float64).reshape(-1, 3)
    evader_xyz = np.asarray(evader_xyz, dtype=np.float64).reshape(3)
    n = int(pursuer_xyz.shape[0])
    rel_xy = pursuer_xyz[:, :2] - evader_xyz[None, :2]
    theta = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])
    order = np.argsort(theta)
    inv_rank = np.zeros((n,), dtype=np.int64)
    inv_rank[order] = np.arange(n, dtype=np.int64)
    phi = (2.0 * np.pi / float(n)) * inv_rank.astype(np.float64)
    ang = phi + float(psi)
    targets = np.zeros((n, 3), dtype=np.float64)
    targets[:, 0] = evader_xyz[0] + float(rho) * np.cos(ang)
    targets[:, 1] = evader_xyz[1] + float(rho) * np.sin(ang)
    targets[:, 2] = evader_xyz[2]
    return targets


def _clip_vec(v: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(v, low), high).astype(np.float32)


def _make_evader_setpoint(
    evader_xyz: np.ndarray,
    hold_xyz: np.ndarray,
    mode: str,
    low: np.ndarray,
    high: np.ndarray,
    vx: float,
    vy: float,
    kp_xy: float,
    kp_z: float,
) -> np.ndarray:
    out = np.zeros((4,), dtype=np.float32)
    if mode == "fixed":
        err = hold_xyz - evader_xyz
        out[0] = float(np.clip(kp_xy * err[0], low[0], high[0]))
        out[1] = float(np.clip(kp_xy * err[1], low[1], high[1]))
        out[2] = float(np.clip(0.0, low[2], high[2]))
        out[3] = float(np.clip(kp_z * err[2], low[3], high[3]))
        return out

    out[0] = float(np.clip(vx, low[0], high[0]))
    out[1] = float(np.clip(vy, low[1], high[1]))
    out[2] = float(np.clip(0.0, low[2], high[2]))
    out[3] = float(np.clip(kp_z * (hold_xyz[2] - evader_xyz[2]), low[3], high[3]))
    return out


def _plot_results(
    save_path: Path,
    traj_xyz: np.ndarray,
    target_xyz: np.ndarray,
    tracking_error: np.ndarray,
    rho: float,
    psi: float,
    mode: str,
) -> None:
    fig = plt.figure(figsize=(12.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, :])

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = ["pursuer_0", "pursuer_1", "pursuer_2", "evader"]
    n_agents = traj_xyz.shape[1]

    for i in range(n_agents):
        ax3d.plot(traj_xyz[:, i, 0], traj_xyz[:, i, 1], traj_xyz[:, i, 2], color=colors[i], lw=1.8, label=labels[i])
        ax3d.scatter(traj_xyz[0, i, 0], traj_xyz[0, i, 1], traj_xyz[0, i, 2], color=colors[i], marker="o", s=35)
        ax3d.scatter(traj_xyz[-1, i, 0], traj_xyz[-1, i, 1], traj_xyz[-1, i, 2], color=colors[i], marker="*", s=60)

    evader_final = traj_xyz[-1, 3]
    th = np.linspace(0.0, 2.0 * np.pi, 181, dtype=np.float64)
    circle = np.zeros((th.shape[0], 3), dtype=np.float64)
    circle[:, 0] = evader_final[0] + float(rho) * np.cos(th)
    circle[:, 1] = evader_final[1] + float(rho) * np.sin(th)
    circle[:, 2] = evader_final[2]
    ax3d.plot(circle[:, 0], circle[:, 1], circle[:, 2], "--", color="purple", lw=1.8, label="target manifold")
    ax3d.scatter(target_xyz[-1, :, 0], target_xyz[-1, :, 1], target_xyz[-1, :, 2], c="purple", marker="x", s=80, linewidths=1.6, label="slot targets")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title(f"Pure manifold tracking | mode={mode} | rho={rho:.2f} psi={psi:.2f}")
    ax3d.grid(True, alpha=0.3)
    ax3d.legend(loc="best")

    for i in range(3):
        ax_xy.plot(traj_xyz[:, i, 0], traj_xyz[:, i, 1], color=colors[i], lw=1.8, label=f"p{i}")
        ax_xy.scatter(target_xyz[-1, i, 0], target_xyz[-1, i, 1], color="purple", marker="x", s=70)
    ax_xy.plot(traj_xyz[:, 3, 0], traj_xyz[:, 3, 1], color=colors[3], lw=1.8, label="evader")
    ax_xy.plot(circle[:, 0], circle[:, 1], "--", color="purple", lw=1.5)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("XY projection")
    ax_xy.legend(loc="best")

    steps = np.arange(tracking_error.shape[0], dtype=np.int32)
    for i in range(3):
        ax_err.plot(steps, tracking_error[:, i], lw=1.6, label=f"pursuer_{i}")
    ax_err.plot(steps, tracking_error.mean(axis=1), color="black", lw=2.2, label="mean")
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("||g_i - p_i||")
    ax_err.grid(True, alpha=0.35)
    ax_err.set_title("Manifold tracking error")
    ax_err.legend(loc="best")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    exp_cfg_path = ROOT / args.config
    exp_cfg = load_config(exp_cfg_path)
    env_cfg_path = ROOT / exp_cfg.get("env", "configs/env/pyflyt_3v1.yaml")
    task_cfg: dict[str, Any] = dict(exp_cfg.get("task", {}) or {})
    env = build_env(env_cfg_path, seed=args.seed, task_cfg=task_cfg)

    if not isinstance(env.task, PURSUIT_EVASION_3V1_TASK_TYPES):
        raise TypeError("This script only supports pursuit-evasion 3v1 tasks.")
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("Pure manifold tracking requires continuous action space.")

    obs_state, _ = env.reset(seed=args.seed)
    del obs_state
    backend = env.backend
    low = np.asarray(env.action_low_np, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_high_np, dtype=np.float32).reshape(-1)

    if env.prev_backend_state is None or env.task_state is None:
        raise RuntimeError("Environment is not initialized after reset.")

    traj: list[np.ndarray] = [np.asarray(env.prev_backend_state.states[:, 3, :], dtype=np.float32)]
    target_hist: list[np.ndarray] = []
    err_hist: list[np.ndarray] = []

    hold_evader_xyz = traj[0][3].astype(np.float64).copy()

    for _ in range(int(args.steps)):
        current = np.asarray(backend.all_states[:, 3, :], dtype=np.float64)
        pursuer_xyz = current[:3]
        evader_xyz = current[3]
        targets = _world_manifold_targets(pursuer_xyz, evader_xyz, float(args.rho), float(args.psi))
        err = targets - pursuer_xyz

        pursuer_setpoints = np.zeros((3, 4), dtype=np.float32)
        pursuer_setpoints[:, 0] = np.clip(args.kp_xy * err[:, 0], low[0], high[0])
        pursuer_setpoints[:, 1] = np.clip(args.kp_xy * err[:, 1], low[1], high[1])
        pursuer_setpoints[:, 2] = np.clip(0.0, low[2], high[2])
        pursuer_setpoints[:, 3] = np.clip(args.kp_z * err[:, 2], low[3], high[3])

        evader_setpoint = _make_evader_setpoint(
            evader_xyz=evader_xyz,
            hold_xyz=hold_evader_xyz,
            mode=str(args.evader_mode),
            low=low,
            high=high,
            vx=float(args.evader_vx),
            vy=float(args.evader_vy),
            kp_xy=float(args.kp_xy),
            kp_z=float(args.kp_z),
        )
        joint_setpoints = np.concatenate([pursuer_setpoints, evader_setpoint[None, :]], axis=0).astype(np.float32)

        backend_state = backend.step(joint_setpoints)
        env.prev_backend_state = backend_state
        traj.append(np.asarray(backend_state.states[:, 3, :], dtype=np.float32))
        target_hist.append(targets.astype(np.float32))
        err_hist.append(np.linalg.norm(err, axis=1).astype(np.float32))

    traj_xyz = np.stack(traj, axis=0)
    target_xyz = np.stack(target_hist, axis=0)
    tracking_error = np.stack(err_hist, axis=0)

    mean_err_start = float(np.mean(tracking_error[: min(20, tracking_error.shape[0])]))
    mean_err_end = float(np.mean(tracking_error[max(0, tracking_error.shape[0] - 20) :]))
    summary = {
        "seed": int(args.seed),
        "steps": int(args.steps),
        "rho": float(args.rho),
        "psi": float(args.psi),
        "evader_mode": str(args.evader_mode),
        "kp_xy": float(args.kp_xy),
        "kp_z": float(args.kp_z),
        "mean_tracking_error_start20": mean_err_start,
        "mean_tracking_error_end20": mean_err_end,
        "final_tracking_error_per_agent": tracking_error[-1].tolist(),
    }

    exp_name = exp_cfg_path.stem
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = ROOT / "results" / exp_name / "pure_manifold_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"summary_seed{args.seed}{tag}.json"
    fig_path = out_dir / f"tracking_seed{args.seed}{tag}.png"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _plot_results(fig_path, traj_xyz, target_xyz, tracking_error, float(args.rho), float(args.psi), str(args.evader_mode))

    print("\n=== Pure Manifold Tracking Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved summary to {json_path}")
    print(f"Saved plot to {fig_path}")

    backend.close()


if __name__ == "__main__":
    main()
