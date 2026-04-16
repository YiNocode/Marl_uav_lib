from __future__ import annotations

"""使用已训练 IPPO Checkpoint 在 PyFlyt NavigationTask 上做评估."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_uav.agents.mac import MAC
from marl_uav.envs.backends.pyflyt_aviary_backend import PyFlytAviaryBackend
from marl_uav.envs.tasks.navigation_task import NavigationTask
from marl_uav.envs.adapters.pyflyt_aviary_env import PyFlytAviaryEnv
from marl_uav.learners.on_policy import IPPOLearner
from marl_uav.utils.checkpoint import load_checkpoint


@dataclass
class EpisodeStats:
    length: int
    total_return: float
    success: bool
    out_of_bounds: bool
    collision: bool
    crash: bool


def collect_episode_eval(
    env: PyFlytAviaryEnv,
    act_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_steps: int = 512,
) -> tuple[EpisodeStats, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单条评估 episode rollout，并记录 3D 轨迹和动作序列."""
    obs_state, info = env.reset()
    obs = obs_state["obs"]
    state = obs_state["state"]

    # 起点与目标位置（单智能体 num_agents=1，记录完整 3D 位置）
    start_pos = env.backend.all_states[:, 3, :]  # [N, 3]
    start_xyz = np.asarray(start_pos[0, :3], dtype=np.float32)
    goal_xyz = np.asarray(env.task_state.goals[0, :3], dtype=np.float32)  # type: ignore[union-attr]

    traj_xyz: list[np.ndarray] = [start_xyz.copy()]
    actions_seq: list[int] = []

    total_return = 0.0
    step_count = 0
    done = False
    last_info = info

    while not done and step_count < max_steps:
        actions = act_fn(obs, state)
        # actions = np.array([2])
        next_obs_state, rewards, terminated, truncated, info = env.step(actions)
        next_obs = next_obs_state["obs"]
        next_state = next_obs_state["state"]

        # 当前 step 后的位姿：PyFlytAviaryEnv.step 中已将 prev_backend_state 更新为新的 backend_state
        curr_pos = env.prev_backend_state.states[:, 3, :]  # type: ignore[union-attr]
        curr_xyz = np.asarray(curr_pos[0, :3], dtype=np.float32)
        traj_xyz.append(curr_xyz)

        # 记录单智能体的离散动作编号
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
    traj_xyz_arr = np.stack(traj_xyz, axis=0)  # [T+1, 3]
    actions_arr = np.asarray(actions_seq, dtype=np.int64)  # [T]
    return stats, traj_xyz_arr, start_xyz, goal_xyz, actions_arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval IPPO NavigationTask with checkpoint.")
    root = Path(__file__).resolve().parents[1]
    default_ckpt = root / "results" / "navigation_ippo_vs_random" / "checkpoints" / "best.pt"
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(default_ckpt),
        help="IPPO learner checkpoint 路径（best.pt 或 latest.pt）",
    )
    p.add_argument("--episodes", type=int, default=20, help="评估 episode 数量")
    p.add_argument("--seed", type=int, default=123, help="环境随机种子（评估）")
    p.add_argument(
        "--render",
        action="store_true",
        help="是否在评估时打开渲染（默认为关闭，更快）",
    )
    return p.parse_args()


def build_eval_env(seed: int, render: bool) -> PyFlytAviaryEnv:
    """构建与训练脚本一致的 NavigationTask 环境."""
    num_agents = 1
    backend = PyFlytAviaryBackend(
        num_agents=num_agents,
        drone_type="quadx",
        render=render,
        physics_hz=240,
        control_hz=60,
        world_scale=5.0,
        drone_options={},
        seed=seed,
        flight_mode=6,
    )
    task = NavigationTask()
    return PyFlytAviaryEnv(backend=backend, task=task, seed=seed)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "请先运行 scripts/run_ippo_vs_random_navigation.py 产生 checkpoint。"
        )

    # 构建评估环境
    env = build_eval_env(seed=args.seed, render=args.render)

    # 通过环境尺寸构建 MAC 与 IPPOLearner，随后从 checkpoint 恢复参数
    obs_state, _ = env.reset(seed=args.seed)
    obs = obs_state["obs"]
    obs_dim = int(obs.shape[1])
    n_actions = env.n_actions
    num_agents = env.num_agents

    mac = MAC(obs_dim=obs_dim, n_actions=n_actions, n_agents=num_agents)
    learner = IPPOLearner(
        policy=mac.policy,
        lr=3e-4,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
    )

    # 从 checkpoint 恢复：这一步将把 policy 参数加载到 mac.policy 中
    state = load_checkpoint(ckpt_path, learner)
    print(
        f"Loaded checkpoint from {ckpt_path} "
        f"(epoch={state.get('epoch')}, "
        f"metric={state.get('metrics', {}).get('train/episode_return', 'N/A')})"
    )

    # 评估时使用确定性策略
    mac.set_test_mode(False)

    def ippo_act_fn(obs: np.ndarray, state_arr: np.ndarray) -> np.ndarray:
        avail = env.get_avail_actions()
        actions, _, _ = mac.select_actions(
            obs, state=state_arr, avail_actions=avail, deterministic=False
        )
        actions_np = np.asarray(actions)
        if actions_np.ndim > 1 and actions_np.shape[0] == 1:
            actions_np = actions_np[0]
        return actions_np.astype(np.int64)

    # 多 episode 统计与轨迹可视化
    results_root = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    traj_dir = results_root / "eval_trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    stats_list: list[EpisodeStats] = []
    for ep in range(args.episodes):
        stats, traj_xyz, start_xyz, goal_xyz, actions_arr = collect_episode_eval(
            env, act_fn=ippo_act_fn, max_steps=512
        )
        stats_list.append(stats)
        print(
            f"[Eval] Ep {ep}: len={stats.length}, return={stats.total_return:.3f}, "
            f"success={stats.success}, out_of_bounds={stats.out_of_bounds}, "
            f"collision={stats.collision}, crash={stats.crash}，last_position={traj_xyz[-1]},goal={goal_xyz},dist={np.linalg.norm(traj_xyz[-1] - goal_xyz)}"
        )

        # 绘制并保存 3D 轨迹图（x-y-z），用颜色标记每一步的动作编号
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        # 轨迹线
        ax.plot(
            traj_xyz[:, 0],
            traj_xyz[:, 1],
            traj_xyz[:, 2],
            "-k",
            alpha=0.3,
            label="trajectory",
        )

        # 每个 step 的位置用动作编号着色（长度为 T，对应从 step 1 开始的位置）
        if len(actions_arr) > 0:
            # 只需要 9 个离散颜色，对应动作编号 0-8，显式使用离散 normalization
            base_cmap = plt.cm.get_cmap("tab10")
            from matplotlib.colors import ListedColormap, BoundaryNorm  # 局部导入

            cmap9 = ListedColormap(base_cmap.colors[:9])  # index 0-8 -> 颜色 0-8
            bounds = np.arange(-0.5, 9.5, 1.0)  # [-0.5, 0.5, ..., 8.5, 9.5]
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

        # 起点/终点
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
        ax.scatter(
            traj_xyz[-1][0],
            traj_xyz[-1][1],
            traj_xyz[-1][2],
            c="r",
            marker="+",
            s=80,
            label="goal",
        )

        # 固定坐标轴范围，便于不同 episode 对比
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
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

    lengths = np.array([s.length for s in stats_list], dtype=np.float32)
    total_returns = np.array([s.total_return for s in stats_list], dtype=np.float32)
    successes = np.array([s.success for s in stats_list], dtype=np.float32)
    out_of_bounds = np.array([s.out_of_bounds for s in stats_list], dtype=np.float32)
    collisions = np.array([s.collision for s in stats_list], dtype=np.float32)
    crashes = np.array([s.crash for s in stats_list], dtype=np.float32)

    print("\n=== IPPO Checkpoint Eval Summary ===")
    print(f"episodes: {len(stats_list)}")
    print(f"avg_len: {lengths.mean():.2f}")
    print(f"avg_return: {total_returns.mean():.3f}")
    print(f"success_rate: {successes.mean():.3f}")
    print(f"out_of_bounds_rate: {out_of_bounds.mean():.3f}")
    print(f"collision_rate: {collisions.mean():.3f}")
    print(f"crash_rate: {crashes.mean():.3f}")

    env.close()


if __name__ == "__main__":
    main()

