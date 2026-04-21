"""Evaluation entry script for config-driven IPPO / MAPPO experiments."""

from __future__ import annotations

import json
from typing import Any, Dict

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.envs.adapters.toy_uav_env import ToyUavEnv
from marl_uav.envs.adapters.pyflyt_aviary_env import (
    PURSUIT_EVASION_3V1_TASK_TYPES,
    PyFlytAviaryEnv,
)
from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend
from marl_uav.envs.tasks.navigation_task import NavigationTask
from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
    PursuitEvasion3v1Task,
    compute_pursuit_structure_metrics_3v1,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx1,
)
from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex2 import (
    PursuitEvasion3v1Task as PursuitEvasion3v1TaskEx2,
)
from marl_uav.learners.on_policy import IPPOLearner, MAPPOLearner, SCMAPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy
from marl_uav.policies.dream_mappo_policy import DreamMappoCentralizedCriticPolicy
from marl_uav.runners.evaluator import Evaluator
from marl_uav.runners.rollout_worker import PURSUIT_STRUCTURE_MEAN_LAST_STEPS, RolloutWorker
from marl_uav.utils.checkpoint import load_checkpoint
from marl_uav.utils.config import load_config
from marl_uav.utils.env_action_bounds import (
    boxed_action_bounds,
    parse_continuous_action_bounds_from_env_cfg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_mappo_3v1.yaml"),
        help="顶层实验配置 (包含 env/algo/model/task 子配置路径)",
    )
    p.add_argument("--seed", type=int, default=6, help="评估用随机种子（首个种子）")
    p.add_argument("--num-seeds", type=int, default=1, help="评估种子数量，每个种子独立跑若干 episode")
    p.add_argument("--episodes", type=int, default=20, help="每个种子评估 episode 数量")
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help=(
            "可选：显式指定 checkpoint 路径 "
            "(默认使用 results/<exp_name>/checkpoints/<seed>/best.pt，与 train.py 一致)"
        ),
    )
    return p.parse_args()


def build_env(env_cfg_path: Path, seed: int, task_cfg: Dict[str, Any] | None = None):
    """根据 env 配置构建评估环境，与训练时保持一致。"""
    cfg = load_config(env_cfg_path)
    env_id = cfg.get("env_id", "toy_uav")

    if env_id == "toy_uav":
        return ToyUavEnv.from_config(cfg, seed=seed)

    if env_id == "pyflyt_navigation":
        backend_cfg = cfg.get("backend", {})
        num_agents = int(backend_cfg.get("num_agents", 1))
        backend = PyFlytAviaryBackend(
            num_agents=num_agents,
            drone_type=backend_cfg.get("drone_type", "quadx"),
            render=bool(backend_cfg.get("render", False)),
            physics_hz=int(backend_cfg.get("physics_hz", 240)),
            control_hz=int(backend_cfg.get("control_hz", 60)),
            world_scale=float(backend_cfg.get("world_scale", 5.0)),
            drone_options=backend_cfg.get("drone_options", {}) or {},
            seed=seed + int(backend_cfg.get("seed_offset", 0)),
            flight_mode=int(backend_cfg.get("flight_mode", 6)),
        )

        task_params = dict(task_cfg or {})
        task_name = str(task_params.pop("name", "navigation"))

        if task_name == "navigation":
            task = NavigationTask(**task_params) if task_params else NavigationTask()
        elif task_name == "pursuit_evasion_3v1":
            task = PursuitEvasion3v1Task(**task_params) if task_params else PursuitEvasion3v1Task()
        elif task_name == "pursuit_evasion_3v1_ex1":
            task = PursuitEvasion3v1TaskEx1(**task_params) if task_params else PursuitEvasion3v1TaskEx1()
        elif task_name == "pursuit_evasion_3v1_ex2":
            task = PursuitEvasion3v1TaskEx2(**task_params) if task_params else PursuitEvasion3v1TaskEx2()
        else:
            raise ValueError(f"Unsupported task name={task_name!r} for env_id={env_id!r}")
        _aspace = str(cfg.get("action_space", "discrete")).lower()
        _adim = int(cfg.get("action_dim", 4))
        _alow, _ahigh = parse_continuous_action_bounds_from_env_cfg(
            cfg, action_space=_aspace, action_dim=_adim
        )
        return PyFlytAviaryEnv(
            backend=backend,
            task=task,
            seed=seed,
            action_space=cfg.get("action_space", "discrete"),
            action_dim=_adim,
            action_low=_alow,
            action_high=_ahigh,
        )

    raise ValueError(f"Unsupported env_id={env_id!r} in {env_cfg_path}")


def build_policy(
    model_cfg_path: Path,
    env: Any,
    algo_cfg_path: Path,
) -> Any:
    """与 train.py 一致：根据 algo 的 action_space 构建离散/连续策略。"""
    cfg = load_config(model_cfg_path)
    algo_cfg = load_config(algo_cfg_path)
    model_type = cfg.get("type", "mlp")
    action_space = str(algo_cfg.get("action_space", "discrete")).lower()

    if action_space not in ("discrete", "continuous"):
        raise ValueError(
            f"action_space must be 'discrete' or 'continuous', got {algo_cfg.get('action_space')!r}"
        )

    # 若 env 提供自身的动作空间类型，则与 algo 配置进行一致性校验
    env_action_space = str(
        getattr(env, "_action_space_type", getattr(env, "action_space_type", "")) or ""
    ).lower()
    if env_action_space in ("discrete", "continuous") and env_action_space != action_space:
        raise ValueError(
            "Mismatch between algo.action_space and env.action_space: "
            f"algo={action_space!r}, env={env_action_space!r}. "
            "请在 env config 和 algo config 中使用一致的 action_space 设置，"
            "否则可能导致 env.n_actions 为 0 或 logits 维度错误。"
        )

    if model_type == "centralized_critic":
        if action_space == "discrete":
            return CentralizedCriticPolicy(
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                n_actions=env.n_actions,
                action_space_type="discrete",
            )
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
            action_dim = int(env.action_space.shape[0])
        if action_dim is None:
            raise ValueError(
                "For centralized_critic + continuous, env must provide action_dim or action_space.shape."
            )
        low, high = boxed_action_bounds(env, action_dim)
        log_std_init = float(algo_cfg.get("log_std_init", -0.5))
        return CentralizedCriticPolicy(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_space_type="continuous",
            action_dim=action_dim,
            action_low=low,
            action_high=high,
            log_std_init=log_std_init,
        )

    if model_type == "dream_mappo_centralized_critic":
        if action_space != "continuous":
            raise ValueError("dream_mappo_centralized_critic 仅支持 continuous action_space。")
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
            action_dim = int(env.action_space.shape[0])
        if action_dim is None:
            raise ValueError(
                "For dream_mappo_centralized_critic, env must provide action_dim or action_space.shape."
            )
        low, high = boxed_action_bounds(env, action_dim)
        log_std_init = float(algo_cfg.get("log_std_init", -0.5))
        dream_cfg = cfg.get("dream", {}) or {}
        return DreamMappoCentralizedCriticPolicy(
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=action_dim,
            action_low=low,
            action_high=high,
            log_std_init=log_std_init,
            num_pursuers=int(dream_cfg.get("num_pursuers", 3)),
            a_max_geom=float(dream_cfg.get("a_max_geom", 0.15)),
            sigma_p=float(dream_cfg.get("sigma_p", 0.5)),
            rho_scale=float(dream_cfg.get("rho_scale", 0.5)),
            rho_min=float(dream_cfg.get("rho_min", 0.05)),
            psi_scale=float(dream_cfg.get("psi_scale", 3.14159265)),
            a_max_residual=float(dream_cfg.get("a_max_residual", 0.08)),
        )

    # IPPO 的 critic 只用 obs，不传 state_dim，与训练时 checkpoint 一致
    algo_name = algo_cfg.get("algo", "ippo").lower()
    state_dim = None if algo_name == "ippo" else getattr(env, "state_dim", None)
    if action_space == "discrete":
        return ActorCriticPolicy(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            state_dim=state_dim,
            action_space_type="discrete",
        )
    action_dim = getattr(env, "action_dim", None)
    if action_dim is None and hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        action_dim = int(env.action_space.shape[0])
    if action_dim is None:
        raise ValueError(
            "For action_space=continuous, env must provide action_dim or action_space.shape."
        )
    log_std_init = float(algo_cfg.get("log_std_init", -0.5))
    low, high = boxed_action_bounds(env, action_dim)
    return ActorCriticPolicy(
        obs_dim=env.obs_dim,
        action_space_type="continuous",
        action_dim=action_dim,
        state_dim=state_dim,
        log_std_init=log_std_init,
        action_low=low,
        action_high=high,
    )


def build_learner(algo_cfg_path: Path, policy: Any) -> IPPOLearner | MAPPOLearner | SCMAPPOLearner:
    cfg = load_config(algo_cfg_path)
    algo_name = cfg.get("algo", "ippo").lower()

    lr = float(cfg.get("lr", 3e-4))
    clip_ratio = float(cfg.get("clip_ratio", 0.2))
    value_coef = float(cfg.get("value_coef", cfg.get("vf_coef", 0.5)))
    entropy_coef = float(cfg.get("entropy_coef", cfg.get("ent_coef", 0.01)))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    num_epochs = int(cfg.get("epochs", 4))

    learner_kwargs = dict(
        lr=lr,
        clip_range=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
    )

    if algo_name == "sc_mappo":
        dispersion_coef = float(cfg.get("dispersion_coef", 0.05))
        num_pursuers = int(cfg.get("num_pursuers", 3))
        spatial_dim = int(cfg.get("spatial_dim", 3))
        rels_from_end = bool(cfg.get("rels_from_end", True))
        rels_start = cfg.get("rels_start_idx")
        rels_start_idx = None if rels_start is None else int(rels_start)
        return SCMAPPOLearner(
            policy=policy,
            dispersion_coef=dispersion_coef,
            num_pursuers=num_pursuers,
            spatial_dim=spatial_dim,
            rels_from_end=rels_from_end,
            rels_start_idx=rels_start_idx,
            **learner_kwargs,
        )
    if algo_name == "mappo":
        return MAPPOLearner(policy=policy, **learner_kwargs)
    if algo_name == "dream_mappo":
        return MAPPOLearner(policy=policy, **learner_kwargs)
    if algo_name == "ippo":
        return IPPOLearner(policy=policy, **learner_kwargs)

    raise ValueError(f"Unsupported algo={algo_name!r} in {algo_cfg_path}")


def _maybe_plot_navigation_trajectories(
    root: Path,
    exp_cfg_path: Path,
    env_cfg_path: Path,
    env: Any,
    mac: MAC,
    episodes: int,
    seed: int,
) -> None:
    """若为 NavigationTask + PyFlytAviaryEnv，则绘制带动作着色的 3D 轨迹."""
    from dataclasses import dataclass
    from typing import Callable

    if not isinstance(env, PyFlytAviaryEnv) or not isinstance(env.task, NavigationTask):
        return

    @dataclass
    class EpisodeStats:
        length: int
        total_return: float
        success: bool
        out_of_bounds: bool
        collision: bool
        crash: bool

    def collect_episode_eval(
        env_nav: PyFlytAviaryEnv,
        act_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        max_steps: int = 512,
    ) -> tuple[EpisodeStats, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_state, info = env_nav.reset()
        obs = obs_state["obs"]
        state = obs_state["state"]

        start_pos = env_nav.backend.all_states[:, 3, :]  # [N, 3]
        start_xyz = np.asarray(start_pos[0, :3], dtype=np.float32)
        goal_xyz = np.asarray(env_nav.task_state.goals[0, :3], dtype=np.float32)  # type: ignore[union-attr]

        traj_xyz: list[np.ndarray] = [start_xyz.copy()]
        actions_seq: list[int] = []

        total_return = 0.0
        step_count = 0
        done = False
        last_info = info

        while not done and step_count < max_steps:
            actions = act_fn(obs, state)
            next_obs_state, rewards, terminated, truncated, info = env_nav.step(actions)
            next_obs = next_obs_state["obs"]
            next_state = next_obs_state["state"]

            curr_pos = env_nav.prev_backend_state.states[:, 3, :]  # type: ignore[union-attr]
            curr_xyz = np.asarray(curr_pos[0, :3], dtype=np.float32)
            traj_xyz.append(curr_xyz)

            act_arr = np.asarray(actions, dtype=np.int64)
            act_idx = int(act_arr[0]) if act_arr.ndim > 0 else int(act_arr)
            actions_seq.append(act_idx)

            r = np.asarray(rewards, dtype=np.float32)
            total_return += float(r.sum())
            step_count += 1
            obs = next_obs
            state = next_state
            last_info = info
            done = bool(terminated or truncated)

        stats = EpisodeStats(
            length=step_count,
            total_return=total_return,
            success=bool(last_info.get("all_reached", False)),
            out_of_bounds=bool(last_info.get("out_of_bounds", False)),
            collision=bool(last_info.get("has_collision", False)),
            crash=bool(last_info.get("crash", False)),
        )
        traj_xyz_arr = np.stack(traj_xyz, axis=0)
        actions_arr = np.asarray(actions_seq, dtype=np.int64)
        return stats, traj_xyz_arr, start_xyz, goal_xyz, actions_arr

    # 结果目录：使用 experiment 配置名
    exp_name = exp_cfg_path.stem
    results_root = root / "results" / exp_name
    traj_dir = results_root / "eval_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    mac.set_test_mode(True)

    def ippo_act_fn(obs: np.ndarray, state_arr: np.ndarray) -> np.ndarray:
        avail = env.get_avail_actions()
        actions, _, _ = mac.select_actions(
            obs, state=state_arr, avail_actions=avail, deterministic=True
        )
        actions_np = np.asarray(actions)
        if actions_np.ndim > 1 and actions_np.shape[0] == 1:
            actions_np = actions_np[0]
        return actions_np.astype(np.int64)

    stats_list: list[EpisodeStats] = []
    for ep in range(episodes):
        stats, traj_xyz, start_xyz, goal_xyz, actions_arr = collect_episode_eval(
            env, act_fn=ippo_act_fn, max_steps=512
        )
        stats_list.append(stats)
        print(
            f"[Eval-Nav] Ep {ep}: len={stats.length}, return={stats.total_return:.3f}, "
            f"success={stats.success}, out_of_bounds={stats.out_of_bounds}, "
            f"collision={stats.collision}, crash={stats.crash}"
        )

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            traj_xyz[:, 0],
            traj_xyz[:, 1],
            traj_xyz[:, 2],
            "-k",
            alpha=0.3,
            label="trajectory",
        )

        if len(actions_arr) > 0:
            base_cmap = plt.cm.get_cmap("tab10")
            from matplotlib.colors import ListedColormap, BoundaryNorm

            cmap9 = ListedColormap(base_cmap.colors[:9])
            bounds = np.arange(-0.5, 9.5, 1.0)
            norm = BoundaryNorm(bounds, cmap9.N)
            sc = ax.scatter(
                traj_xyz[1:, 0],
                traj_xyz[1:, 1],
                traj_xyz[1:, 2],
                c=actions_arr,
                cmap=cmap9,
                norm=norm,
                s=25,
                edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("action index")
            cbar.set_ticks(np.arange(0, 9, 1))

        ax.scatter(
            start_xyz[0],
            start_xyz[1],
            start_xyz[2],
            c="g",
            marker="o",
            s=60,
            label="start",
        )
        ax.scatter(
            goal_xyz[0],
            goal_xyz[1],
            goal_xyz[2],
            c="r",
            marker="*",
            s=80,
            label="goal",
        )

        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_zlim(0.0, 2.0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(
            f"Episode {ep} | return={stats.total_return:.2f} | success={stats.success}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(traj_dir / f"episode_{ep:04d}_3d.png", dpi=120)
        plt.close(fig)


def _pursuit_series_from_trajectory(traj_xyz: np.ndarray) -> list[dict[str, Any]]:
    """由轨迹逐时刻重算围捕结构指标。traj_xyz: [T+1, N, 3]，agent 0..2 为 pursuer，3 为 evader。"""
    Tp1, n, _ = traj_xyz.shape
    if n < 4:
        return []
    out: list[dict[str, Any]] = []
    for t in range(Tp1):
        p = traj_xyz[t, :3, :]
        e = traj_xyz[t, 3, :]
        out.append(compute_pursuit_structure_metrics_3v1(p, e))
    return out


def _dream_manifold_targets_from_snapshot(
    pursuer_xyz: np.ndarray,
    evader_xyz: np.ndarray,
    rho: float,
    psi: float,
) -> np.ndarray:
    """Rebuild Dream-MAPPO manifold target points in the xy plane."""
    pursuer_xyz = np.asarray(pursuer_xyz, dtype=np.float64)
    evader_xyz = np.asarray(evader_xyz, dtype=np.float64).reshape(3)
    n = int(pursuer_xyz.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    rel_xy = pursuer_xyz[:, :2] - evader_xyz[None, :2]
    alpha = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])
    order = np.argsort(alpha)
    inv_rank = np.zeros((n,), dtype=np.int64)
    inv_rank[order] = np.arange(n, dtype=np.int64)

    phi = (2.0 * np.pi / float(n)) * inv_rank.astype(np.float64)
    ang = phi + float(psi)
    targets = np.zeros((n, 3), dtype=np.float64)
    targets[:, 0] = evader_xyz[0] + float(rho) * np.cos(ang)
    targets[:, 1] = evader_xyz[1] + float(rho) * np.sin(ang)
    targets[:, 2] = evader_xyz[2]
    return targets


def _plot_pursuit_polar_schematic(
    ax: Any,
    metrics: dict[str, Any],
    *,
    timestep: int | None = None,
) -> None:
    """以 evader 为圆心的 xy 方位示意图：三 pursuer bearing、φ1/φ2/φ3 扇区与指标文字。"""
    two_pi = 2.0 * np.pi
    ts = metrics["theta_sorted"]
    t1, t2, t3 = float(ts[0]), float(ts[1]), float(ts[2])
    phi1 = float(metrics["phi_1"])
    phi2 = float(metrics["phi_2"])
    phi3 = float(metrics["phi_3"])
    th_p = [float(x) for x in metrics["theta_pursuer"]]

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.35)

    th_ring = np.linspace(0.0, two_pi, 180)
    ax.plot(th_ring, np.ones_like(th_ring), color="0.45", lw=0.8, alpha=0.7)

    sector_colors = ("#cfe8ff", "#d6f5d6", "#fff4c2")
    n_arc = 48

    def _fill_arc(t_a: float, t_b: float, color: str, alpha: float = 0.55) -> None:
        if t_b < t_a:
            t_b += two_pi
        tt = np.linspace(t_a, t_b, n_arc)
        ax.fill_between(tt, 0.0, 1.0, color=color, alpha=alpha)

    _fill_arc(t1, t2, sector_colors[0])
    _fill_arc(t2, t3, sector_colors[1])
    _fill_arc(t3, two_pi, sector_colors[2])
    _fill_arc(0.0, t1, sector_colors[2])

    base_cmap = plt.cm.get_cmap("tab10")
    for i in range(3):
        c = base_cmap(i)
        ax.plot([th_p[i], th_p[i]], [0.0, 1.0], color=c, lw=2.2, solid_capstyle="round")
        ax.scatter([th_p[i]], [1.0], color=c, s=36, zorder=5)
        ax.text(
            th_p[i],
            1.06,
            f"P{i}",
            color=c,
            fontsize=9,
            ha="center",
            va="bottom",
        )

    mid1 = 0.5 * (t1 + t2)
    mid2 = 0.5 * (t2 + t3)
    mid3 = t3 + 0.5 * phi3
    if mid3 >= two_pi:
        mid3 -= two_pi
    for mid, lab in ((mid1, r"$\phi_1$"), (mid2, r"$\phi_2$"), (mid3, r"$\phi_3$")):
        ax.text(mid, 0.52, lab, ha="center", va="center", fontsize=11, color="0.15")

    cc = float(metrics["C_cov"])
    cl = float(metrics["C_col"])
    da = float(metrics["D_ang"])
    ax.text(
        0.02,
        0.98,
        f"$C_{{cov}}={cc:.3f}$\n$C_{{col}}={cl:.3f}$\n$D_{{ang}}={da:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
    )
    title = "Pursuit structure (xy, evader @ center)"
    if timestep is not None:
        title += f" · step {timestep}"
    ax.set_title(title, fontsize=10, pad=12)


def _plot_pursuit_evasion_trajectories_from_data(
    trajectories: list[Dict[str, Any]],
    results_root: Path,
) -> None:
    """根据 evaluator 返回的轨迹数据绘制 3v1 追逃：3D 轨迹 + 极坐标结构示意图 + 指标曲线。

    trajectories: list of dict，每个含 "trajectory" ([T+1, N, 3]) 及可选 pursuit_structure_series。
    """
    if not trajectories:
        return
    traj_dir = results_root / "eval_trajectories_pe3v1"
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(traj_dir)
    base_cmap = plt.cm.get_cmap("tab10")
    for ep, data in enumerate(trajectories):
        traj_xyz = np.asarray(data["trajectory"], dtype=np.float32)  # [T+1, N, 3]
        Tp1, n_real, _ = traj_xyz.shape
        colors = [base_cmap(i) for i in range(min(n_real, 4))]
        labels = [f"pursuer_{i}" for i in range(max(0, n_real - 1))] + ["evader"]
        if n_real != 4:
            labels = [f"agent_{i}" for i in range(n_real)]

        series_raw = data.get("pursuit_structure_series")
        if isinstance(series_raw, list) and len(series_raw) == Tp1:
            series = [dict(x) for x in series_raw]
        else:
            series = _pursuit_series_from_trajectory(traj_xyz)

        manifold_raw = data.get("dream_manifold_series")
        manifold_series = manifold_raw if isinstance(manifold_raw, list) else []

        if not series:
            metrics_sel: dict[str, Any] = {}
        else:
            t_show = Tp1 - 1
            metrics_sel = series[min(t_show, len(series) - 1)]

        fig = plt.figure(figsize=(13.5, 9.5))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.35, 1.0],
            width_ratios=[1.15, 1.0],
            hspace=0.28,
            wspace=0.14,
        )
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_polar = fig.add_subplot(gs[0, 1], projection="polar")
        ax_curves = fig.add_subplot(gs[1, :])

        for agent_idx in range(n_real):
            c = colors[agent_idx % len(colors)]
            ax3d.plot(
                traj_xyz[:, agent_idx, 0],
                traj_xyz[:, agent_idx, 1],
                traj_xyz[:, agent_idx, 2],
                "-",
                color=c,
                alpha=0.9,
                label=labels[agent_idx],
            )
            ax3d.scatter(
                traj_xyz[0, agent_idx, 0],
                traj_xyz[0, agent_idx, 1],
                traj_xyz[0, agent_idx, 2],
                c=[c],
                marker="o",
                s=40,
            )
            ax3d.scatter(
                traj_xyz[-1, agent_idx, 0],
                traj_xyz[-1, agent_idx, 1],
                traj_xyz[-1, agent_idx, 2],
                c=[c],
                marker="*",
                s=60,
            )

        oxy_raw = data.get("obstacle_xy")
        orad_raw = data.get("obstacle_r")
        if oxy_raw is not None and orad_raw is not None:
            oxy = np.asarray(oxy_raw, dtype=np.float64)
            orad = np.asarray(orad_raw, dtype=np.float64).reshape(-1)
            if oxy.ndim == 2 and oxy.shape[0] > 0 and orad.shape[0] == oxy.shape[0]:
                z_plot = float(0.5 * (float(np.min(traj_xyz[..., 2])) + float(np.max(traj_xyz[..., 2]))))
                rmax = float(np.max(orad)) if orad.size else 1.0
                rmax = max(rmax, 1e-6)
                sizes = np.clip(36.0 + 220.0 * (orad / rmax), 32.0, 220.0)
                zs = np.full((oxy.shape[0],), z_plot, dtype=np.float64)
                ax3d.scatter(
                    oxy[:, 0],
                    oxy[:, 1],
                    zs,
                    c="dimgray",
                    s=sizes,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.95,
                    label="obstacles (xy center)",
                )

        if len(manifold_series) == Tp1 and n_real >= 4:
            manifold_last = manifold_series[-1]
            rho = manifold_last.get("rho")
            psi = manifold_last.get("psi")
            if rho is not None and psi is not None:
                evader_xyz = np.asarray(traj_xyz[-1, 3, :], dtype=np.float64)
                pursuer_xyz = np.asarray(traj_xyz[-1, :3, :], dtype=np.float64)
                th = np.linspace(0.0, 2.0 * np.pi, 181, dtype=np.float64)
                circle_xyz = np.zeros((th.shape[0], 3), dtype=np.float64)
                circle_xyz[:, 0] = evader_xyz[0] + float(rho) * np.cos(th)
                circle_xyz[:, 1] = evader_xyz[1] + float(rho) * np.sin(th)
                circle_xyz[:, 2] = evader_xyz[2]
                ax3d.plot(
                    circle_xyz[:, 0],
                    circle_xyz[:, 1],
                    circle_xyz[:, 2],
                    "--",
                    color="purple",
                    lw=1.8,
                    alpha=0.95,
                    label="dream manifold",
                )
                manifold_targets = _dream_manifold_targets_from_snapshot(
                    pursuer_xyz,
                    evader_xyz,
                    float(rho),
                    float(psi),
                )
                if manifold_targets.shape[0] > 0:
                    ax3d.scatter(
                        manifold_targets[:, 0],
                        manifold_targets[:, 1],
                        manifold_targets[:, 2],
                        c="purple",
                        marker="x",
                        s=70,
                        linewidths=1.6,
                        label="manifold targets",
                    )

        ax3d.set_xlim(-20, 20)
        ax3d.set_ylim(-20, 20)
        ax3d.set_zlim(0.5, 5.0)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(
            f"PE3v1 Ep {ep} | return={data['episode_return']:.2f} | "
            f"captured={data.get('captured', False)} | timeout={data.get('timeout', False)} | "
            f"obs_term={data.get('obstacle_termination', False)}"
        )
        ax3d.grid(True, alpha=0.3)
        ax3d.legend(loc="best")

        if metrics_sel and all(k in metrics_sel for k in ("theta_sorted", "theta_pursuer", "phi_1")):
            _plot_pursuit_polar_schematic(ax_polar, metrics_sel, timestep=Tp1 - 1)
        else:
            ax_polar.set_visible(False)

        steps = np.arange(len(series), dtype=np.float64)
        if len(series) > 0:
            c_cov = np.array([float(s["C_cov"]) for s in series], dtype=np.float64)
            c_col = np.array([float(s["C_col"]) for s in series], dtype=np.float64)
            d_ang = np.array([float(s["D_ang"]) for s in series], dtype=np.float64)
            ax_curves.plot(steps, c_cov, label=r"$C_{cov}$", color="steelblue", lw=1.8)
            ax_curves.plot(steps, c_col, label=r"$C_{col}$", color="coral", lw=1.8)
            ax_curves.plot(steps, d_ang, label=r"$D_{ang}$", color="seagreen", lw=1.8)
            ax_curves.set_xlabel("step (trajectory index, reset=0)")
            ax_curves.set_ylabel("score")
            ax_curves.set_ylim(-0.05, 1.05)
            ax_curves.grid(True, alpha=0.35)
            ax_curves.legend(loc="best")
            ax_curves.set_title("Pursuit structure metrics vs step")
        else:
            ax_curves.text(0.5, 0.5, "No pursuit structure series", ha="center", va="center")

        fig.suptitle(
            f"Episode {ep} | $C_{{cov}}, C_{{col}}, D_{{ang}}$ curves + polar at last step",
            fontsize=11,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(traj_dir / f"episode_{ep:04d}_3d.png", dpi=120)
        plt.close(fig)


def run_multi_seed_eval(
    worker: RolloutWorker,
    num_seeds: int,
    episodes_per_seed: int,
    base_seed: int,
    record_trajectories: bool,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]] | None]:
    """
    多种子评估：遍历 num_seeds 个种子，每种子跑 episodes_per_seed 次，记录每次指标。
    Returns:
        all_records: 每条 episode 的指标列表
        trajectories: 仅在第一颗种子时记录轨迹（用于 3v1 轨迹图），其余为 None
    """
    all_records: list[Dict[str, Any]] = []
    trajectories: list[Dict[str, Any]] | None = [] if record_trajectories else None

    for s in range(num_seeds):
        print(s)
        seed = base_seed + s
        for ep in range(episodes_per_seed):
            ep_seed = seed * 1000 + ep
            _, info = worker.collect_episode(
                seed=ep_seed,
                record_trajectory=record_trajectories and (s == 0),
            )
            success_or_capture = bool(
                info.get("capture", False) or info.get("success", False)
            )
            ep_rec: Dict[str, Any] = {
                "seed": seed,
                "episode": ep,
                "episode_return": float(info["episode_return"]),
                "episode_len": int(info["episode_len"]),
                "success_or_capture": success_or_capture,
                "collision": bool(info.get("collision", False)),
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "captured": bool(info.get("capture", False)),
                "timeout": bool(info.get("timeout", False)),
                "obstacle_termination": bool(info.get("obstacle_termination", False)),
            }
            if "mean_C_cov" in info and "mean_C_col" in info:
                ep_rec["mean_C_cov"] = float(info["mean_C_cov"])
                ep_rec["mean_C_col"] = float(info["mean_C_col"])
            all_records.append(ep_rec)
            if record_trajectories and s == 0 and "trajectory" in info:
                ep_traj: Dict[str, Any] = {
                    "trajectory": info["trajectory"],
                    "episode_return": info["episode_return"],
                    "episode_len": info["episode_len"],
                    "captured": info.get("capture", False),
                    "timeout": info.get("timeout", False),
                    "pursuer_oob": info.get("pursuer_oob", False),
                    "collision": info.get("collision", False),
                    "obstacle_termination": bool(info.get("obstacle_termination", False)),
                }
                if "pursuit_structure_series" in info:
                    ep_traj["pursuit_structure_series"] = info["pursuit_structure_series"]
                if "dream_manifold_series" in info:
                    ep_traj["dream_manifold_series"] = info["dream_manifold_series"]
                if "obstacle_xy" in info and "obstacle_r" in info:
                    ep_traj["obstacle_xy"] = np.asarray(info["obstacle_xy"], dtype=np.float32).copy()
                    ep_traj["obstacle_r"] = np.asarray(info["obstacle_r"], dtype=np.float32).copy()
                trajectories.append(ep_traj)
    return all_records, trajectories


def compute_aggregate_metrics(
    all_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """根据每条 episode 记录计算各类指标平均值（及标准差）。"""
    if not all_records:
        return {}
    n = len(all_records)
    returns = [r["episode_return"] for r in all_records]
    lens = [r["episode_len"] for r in all_records]
    success = [float(r["success_or_capture"]) for r in all_records]
    collision = [float(r["collision"]) for r in all_records]
    oob = [float(r["out_of_bounds"]) for r in all_records]
    obs_term = [float(r.get("obstacle_termination", False)) for r in all_records]
    return {
        "eval/avg_return": float(np.mean(returns)),
        "eval/std_return": float(np.std(returns)),
        "eval/avg_episode_len": float(np.mean(lens)),
        "eval/std_episode_len": float(np.std(lens)),
        "eval/success_capture_rate": float(np.mean(success)),
        "eval/collision_rate": float(np.mean(collision)),
        "eval/out_of_bounds_rate": float(np.mean(oob)),
        "eval/obstacle_termination_rate": float(np.mean(obs_term)) if obs_term else 0.0,
        "eval/num_episodes": n,
        "eval/num_seeds": len({r["seed"] for r in all_records}),
    }


def plot_eval_statistics(
    metrics: Dict[str, Any],
    save_path: Path,
) -> None:
    """将评估指标绘制成一张图内 4 个子图，分别对应各指标。"""
    n = metrics.get("eval/num_episodes", 1)

    def _std_return():
        return metrics.get("eval/std_return", 0.0)

    def _std_len():
        return metrics.get("eval/std_episode_len", 0.0)

    def _std_rate(p: float):
        return np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 子图1: Episode Return
    ax = axes[0, 0]
    mean_r = metrics.get("eval/avg_return", 0.0)
    std_r = _std_return()
    ax.bar([0], [mean_r], 0.5, yerr=[std_r], capsize=6, color="steelblue", edgecolor="navy")
    ax.set_xticks([0])
    ax.set_xticklabels(["Episode Return"])
    ax.set_ylabel("Value")
    ax.set_title("Episode Return")

    # 子图2: Avg Episode Length
    ax = axes[0, 1]
    mean_len = metrics.get("eval/avg_episode_len", 0.0)
    std_len = _std_len()
    ax.bar([0], [mean_len], 0.5, yerr=[std_len], capsize=6, color="seagreen", edgecolor="darkgreen")
    ax.set_xticks([0])
    ax.set_xticklabels(["Avg Episode Length"])
    ax.set_ylabel("Steps")
    ax.set_title("Average Episode Length")

    # 子图3: Success/Capture Rate
    ax = axes[1, 0]
    mean_ok = metrics.get("eval/success_capture_rate", 0.0)
    std_ok = _std_rate(mean_ok)
    ax.bar([0], [mean_ok], 0.5, yerr=[std_ok], capsize=6, color="mediumseagreen", edgecolor="darkgreen")
    ax.set_xticks([0])
    ax.set_xticklabels(["Success/Capture Rate"])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success/Capture Rate")

    # 子图4: Collision / OOB / 撞柱终止（ex2 等）
    ax = axes[1, 1]
    mean_col = metrics.get("eval/collision_rate", 0.0)
    mean_oob = metrics.get("eval/out_of_bounds_rate", 0.0)
    mean_obs = metrics.get("eval/obstacle_termination_rate", 0.0)
    std_col = _std_rate(mean_col)
    std_oob = _std_rate(mean_oob)
    std_obs = _std_rate(mean_obs)
    x = np.array([0, 1, 2])
    means = [mean_col, mean_oob, mean_obs]
    stds = [std_col, std_oob, std_obs]
    ax.bar(
        x,
        means,
        0.55,
        yerr=stds,
        capsize=5,
        color=["coral", "gold", "slategray"],
        edgecolor=["darkred", "darkorange", "0.2"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Collision", "Out-of-Bounds", "Obstacle term."])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Collision, OOB & Obstacle Termination")

    fig.suptitle(
        f"Eval Statistics (n={metrics.get('eval/num_episodes', 0)} episodes, "
        f"{metrics.get('eval/num_seeds', 0)} seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pursuit_mean_cov_col_scatter(
    all_records: list[Dict[str, Any]],
    save_path: Path,
) -> bool:
    """横轴/纵轴为每局最后若干步上 C_cov、C_col 的均值（见 RolloutWorker 中 PURSUIT_STRUCTURE_MEAN_LAST_STEPS）；每点一条 episode；颜色 captured/timeout；点大小 ∝ episode_len。"""
    rows = [r for r in all_records if "mean_C_cov" in r and "mean_C_col" in r]
    if not rows:
        return False

    xs = np.array([float(r["mean_C_cov"]) for r in rows], dtype=np.float64)
    ys = np.array([float(r["mean_C_col"]) for r in rows], dtype=np.float64)
    lens = np.array([int(r["episode_len"]) for r in rows], dtype=np.float64)

    # 点面积：与 episode 长度成比例，并限制在合理范围
    s = np.clip(22.0 + lens * 5.5, 35.0, 650.0)

    # 颜色：优先 captured；否则 timeout；其余为 other
    color_list: list[str] = []
    for r in rows:
        if bool(r.get("captured", False)):
            color_list.append("#2ca02c")
        elif bool(r.get("timeout", False)):
            color_list.append("#ff7f0e")
        else:
            color_list.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(
        xs,
        ys,
        s=s,
        c=color_list,
        alpha=0.82,
        edgecolors="0.25",
        linewidths=0.45,
        zorder=3,
    )
    ax.set_xlabel(
        rf"$\bar{{C}}_{{\mathrm{{cov}}}}^{{\mathrm{{term}}}}$ (mean over last {PURSUIT_STRUCTURE_MEAN_LAST_STEPS} steps)"
    )
    ax.set_ylabel(
        rf"$\bar{{C}}_{{\mathrm{{col}}}}^{{\mathrm{{term}}}}$ (mean over last {PURSUIT_STRUCTURE_MEAN_LAST_STEPS} steps)"
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="0.25", markersize=9, label="captured"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markeredgecolor="0.25", markersize=9, label="timeout"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markeredgecolor="0.25", markersize=9, label="other"),
    ]
    leg1 = ax.legend(handles=legend_elems, loc="upper right", title="outcome")
    ax.add_artist(leg1)

    # 示意：不同 episode 长度对应的点大小
    if lens.size > 0:
        lo, hi = int(lens.min()), int(lens.max())
        if lo == hi:
            size_note = f"marker size ∝ episode length (all {lo})"
        else:
            s_lo = float(np.clip(22.0 + lo * 5.5, 35.0, 650.0))
            s_hi = float(np.clip(22.0 + hi * 5.5, 35.0, 650.0))
            size_note = f"marker size ∝ episode length (min={lo}→{s_lo:.0f}, max={hi}→{s_hi:.0f})"
        ax.text(
            0.02,
            0.02,
            size_note,
            transform=ax.transAxes,
            fontsize=8,
            color="0.35",
            va="bottom",
        )

    n = len(rows)
    ax.set_title(f"Pursuit structure scatter (n={n} episodes)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


def _peek_checkpoint_actor_obs_dim(ckpt_path: Path) -> int | None:
    """Read actor input dim from checkpoint without loading the full learner."""
    try:
        data = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None
    learner_state = data.get("learner")
    if not isinstance(learner_state, dict):
        return None
    policy_state = learner_state.get("policy")
    if not isinstance(policy_state, dict):
        return None
    weight = policy_state.get("actor_encoder.net.0.weight")
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    try:
        return int(shape[1])
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    exp_cfg_path = root / args.config
    exp_cfg = load_config(exp_cfg_path)

    env_cfg_path = root / exp_cfg.get("env", "configs/env/toy_uav.yaml")
    algo_cfg_path = root / exp_cfg.get("algo", "configs/algo/ippo.yaml")
    model_cfg_path = root / exp_cfg.get("model", "configs/model/mlp.yaml")
    task_cfg: Dict[str, Any] = exp_cfg.get("task", {}) or {}

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        exp_name = Path(args.config).stem
        ckpt_dir = root / "results" / exp_name / "checkpoints" / str(args.seed)
        ckpt_path = ckpt_dir / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. 璇峰厛閫氳繃 train.py 璁粌骞朵繚瀛樻ā鍨嬨€?"
        )

    # 构建环境 / 策略 / learner，并从 checkpoint 恢复参数
    env = build_env(env_cfg_path, seed=args.seed, task_cfg=task_cfg)
    # 若环境还未初始化 obs_dim/state_dim，则在构建 policy 前先 reset 一次
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        try:
            env.reset(seed=args.seed)
        except TypeError:
            # 兼容不接受 seed 参数的 reset 签名
            env.reset()
    task_name = str(task_cfg.get("name", "navigation"))
    ckpt_obs_dim = _peek_checkpoint_actor_obs_dim(ckpt_path)
    if (
        ckpt_obs_dim is not None
        and getattr(env, "obs_dim", None) is not None
        and int(env.obs_dim) != int(ckpt_obs_dim)
        and task_name in ("pursuit_evasion_3v1_ex1", "pursuit_evasion_3v1_ex2")
        and int(env.obs_dim) - int(ckpt_obs_dim) == 3
        and "structure_obs_include_deltas" not in task_cfg
    ):
        compat_task_cfg = dict(task_cfg)
        compat_task_cfg["structure_obs_include_deltas"] = False
        env = build_env(env_cfg_path, seed=args.seed, task_cfg=compat_task_cfg)
        try:
            env.reset(seed=args.seed)
        except TypeError:
            env.reset()
        task_cfg = compat_task_cfg
        print(
            f"[Eval] Detected legacy checkpoint obs_dim={ckpt_obs_dim}; "
            "rebuilding env with structure_obs_include_deltas=False for compatibility."
        )
    policy_core = build_policy(model_cfg_path, env, algo_cfg_path)
    n_actions_for_mac = (
        env.n_actions
        if getattr(policy_core, "action_space_type", "discrete") == "discrete"
        else (getattr(policy_core, "action_dim", None) or 0)
    )
    mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions_for_mac, n_agents=env.num_agents)
    mac.policy = policy_core

    learner = build_learner(algo_cfg_path, policy_core)

    # checkpoint 路径：优先使用命令行
    # 默认：与 train.py 一致 -> results/<exp_name>/checkpoints/<seed>/best.pt
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        exp_name = Path(args.config).stem
        ckpt_dir = root / "results" / exp_name / "checkpoints" / str(args.seed)
        ckpt_path = ckpt_dir / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. 请先通过 train.py 训练并保存模型。"
        )

    state = load_checkpoint(ckpt_path, learner)
    print(
        f"Loaded checkpoint from {ckpt_path} "
        f"(epoch={state.get('epoch')}, "
        f"metric={state.get('metrics', {}).get('train/avg_return', 'N/A')})"
    )

    worker = RolloutWorker(env=env, policy=mac)
    is_pe3v1 = isinstance(env.task, PURSUIT_EVASION_3V1_TASK_TYPES)
    num_seeds = args.num_seeds
    episodes_per_seed = args.episodes

    print(f"\n=== Multi-seed Eval: {num_seeds} seeds x {episodes_per_seed} episodes ===")
    all_records, trajectories = run_multi_seed_eval(
        worker=worker,
        num_seeds=num_seeds,
        episodes_per_seed=episodes_per_seed,
        base_seed=args.seed,
        record_trajectories=is_pe3v1,
    )
    metrics = compute_aggregate_metrics(all_records)

    print("\n=== Eval Metrics (mean over all episodes) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    exp_name = Path(args.config).stem
    results_dir = root / "results" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    # 与 checkpoints/<seed>/ 一致，文件名带种子便于区分不同 checkpoint 的评估输出
    seed_tag = f"seed{args.seed}"
    records_path = results_dir / f"eval_records_{seed_tag}.json"
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"Per-episode records saved to {records_path}")

    stats_path = results_dir / f"eval_statistics_{seed_tag}.png"
    plot_eval_statistics(metrics, stats_path)
    print(f"Eval statistics plot saved to {stats_path}")

    if is_pe3v1:
        scatter_path = results_dir / f"eval_pursuit_cov_col_scatter_{seed_tag}.png"
        if plot_pursuit_mean_cov_col_scatter(all_records, scatter_path):
            print(f"Pursuit episode-mean C_cov vs C_col scatter saved to {scatter_path}")

    # 若为 NavigationTask + PyFlyt 环境，则额外绘制 3D 轨迹（按动作着色）
    _maybe_plot_navigation_trajectories(
        root=root,
        exp_cfg_path=exp_cfg_path,
        env_cfg_path=env_cfg_path,
        env=env,
        mac=mac,
        episodes=episodes_per_seed,
        seed=args.seed,
    )

    # 3v1 追逃：用多种子评估中第一颗种子的轨迹画图
    if trajectories is not None:
        task_name = str(task_cfg.get("name", "navigation"))
        results_root = (
            root / "results" / "PursuitEvasion3v1Task"
            if task_name in ("pursuit_evasion_3v1", "pursuit_evasion_3v1_ex1", "pursuit_evasion_3v1_ex2")
            else results_dir
        )
        _plot_pursuit_evasion_trajectories_from_data(trajectories, results_root)


if __name__ == "__main__":
    main()
