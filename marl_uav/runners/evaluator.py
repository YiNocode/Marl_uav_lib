"""Evaluator: run evaluation episodes."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from marl_uav.runners.base_runner import BaseRunner
from marl_uav.runners.rollout_worker import RolloutWorker


class Evaluator(BaseRunner):
    """Runs evaluation without exploration."""

    def __init__(self, rollout_worker: RolloutWorker) -> None:
        self.rollout_worker = rollout_worker

    def run(
        self,
        *,
        num_episodes: int = 5,
        seed: int = 123,
        record_trajectories: bool = False,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]] | None]:
        """
        Run evaluation episodes.

        Args:
            num_episodes: 评估 episode 数
            seed: 随机种子
            record_trajectories: 若为 True，则在 collect_episode 中记录轨迹，并返回轨迹列表

        Returns:
            metrics: 聚合的评估指标
            trajectories: 当 record_trajectories=True 时，为 list of dict，每个 dict 含
                "trajectory" (np.ndarray [T+1, N, 3]) 以及 episode_return, captured, timeout 等；
                否则为 None
        """
        returns = []
        lens = []
        terminated_cnt = 0
        truncated_cnt = 0
        capture_cnt = 0
        timeout_cnt = 0
        pursuer_oob_cnt = 0
        collision_cnt = 0
        obstacle_term_cnt = 0
        capture_steps: list[float] = []
        trajectories: List[Dict[str, Any]] | None = [] if record_trajectories else None

        for i in range(num_episodes):
            _, info = self.rollout_worker.collect_episode(
                seed=seed + i,
                record_trajectory=record_trajectories,
            )
            returns.append(info["episode_return"])
            lens.append(info["episode_len"])
            terminated_cnt += int(info["terminated"])
            truncated_cnt += int(info["truncated"])
            if info.get("capture", False):
                capture_cnt += 1
                if int(info.get("capture_step", -1)) >= 0:
                    capture_steps.append(float(info["capture_step"]))
            if info.get("timeout", False):
                timeout_cnt += 1
            if info.get("pursuer_oob", False):
                pursuer_oob_cnt += 1
            if info.get("collision", False):
                collision_cnt += 1
            if info.get("obstacle_termination", False):
                obstacle_term_cnt += 1

            if record_trajectories and "trajectory" in info:
                row: Dict[str, Any] = {
                    "trajectory": info["trajectory"],
                    "episode_return": info["episode_return"],
                    "episode_len": info["episode_len"],
                    "captured": info.get("capture", False),
                    "timeout": info.get("timeout", False),
                    "pursuer_oob": info.get("pursuer_oob", False),
                    "collision": info.get("collision", False),
                    "obstacle_termination": bool(info.get("obstacle_termination", False)),
                }
                if "obstacle_xy" in info and "obstacle_r" in info:
                    row["obstacle_xy"] = np.asarray(info["obstacle_xy"], dtype=np.float32).copy()
                    row["obstacle_r"] = np.asarray(info["obstacle_r"], dtype=np.float32).copy()
                trajectories.append(row)

        metrics: Dict[str, Any] = {
            "eval/num_episodes": int(num_episodes),
            "eval/avg_return": float(np.mean(returns)) if returns else 0.0,
            "eval/avg_len": float(np.mean(lens)) if lens else 0.0,
            "eval/terminated_rate": float(terminated_cnt / num_episodes) if num_episodes else 0.0,
            "eval/truncated_rate": float(truncated_cnt / num_episodes) if num_episodes else 0.0,
            "eval/capture_rate": float(capture_cnt / num_episodes) if num_episodes else 0.0,
            "eval/avg_capture_step": float(np.mean(capture_steps)) if capture_steps else -1.0,
            "eval/timeout_rate": float(timeout_cnt / num_episodes) if num_episodes else 0.0,
            "eval/pursuer_oob_rate": float(pursuer_oob_cnt / num_episodes) if num_episodes else 0.0,
            "eval/collision_rate": float(collision_cnt / num_episodes) if num_episodes else 0.0,
            "eval/obstacle_termination_rate": float(obstacle_term_cnt / num_episodes)
            if num_episodes
            else 0.0,
        }
        print(
            f"[eval] episodes={num_episodes} avg_return={metrics['eval/avg_return']:.3f} avg_len={metrics['eval/avg_len']:.1f}"
        )
        return metrics, trajectories
