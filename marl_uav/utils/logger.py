from __future__ import annotations

"""Logging utilities (stdout logger + TensorBoard)."""

from pathlib import Path
from typing import Mapping
import logging

from torch.utils.tensorboard import SummaryWriter


ScalarMapping = Mapping[str, float]


class Logger:
    """TensorBoard logger封装.

    约定的 tag 结构：
    - 环境训练指标：
      - train/episode_return
      - train/episode_length
      - train/success_rate
      - train/out_of_bounds_rate
      - train/collision_rate
      - train/capture_rate
      - train/timeout_rate
      - train/pursuer_oob_rate
      - train/obstacle_termination_rate  # ex2：因碰柱 terminated 的 episode 记 1
    - PPO 训练指标：
      - ppo/policy_loss
      - ppo/value_loss
      - ppo/entropy
      - ppo/approx_kl
      - ppo/clip_fraction
      - ppo/grad_norm
    - 环境诊断指标：
      - env/mean_goal_distance
      - env/final_goal_distance
      - env/reward_progress
      - env/reward_time_penalty
      - env/reward_reach_bonus
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.log_dir))

    # --------------------------- 基础接口 ---------------------------
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录单个标量到 TensorBoard."""
        self.tb.add_scalar(tag, float(value), step)

    def log_dict(self, metrics: ScalarMapping, step: int, prefix: str | None = None) -> None:
        """记录一组标量到 TensorBoard.

        如果提供 prefix，则最终 tag 形式为 ``f"{prefix}/{key}"``。
        """
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.tb.add_scalar(tag, float(v), step)

    # ------------------------ 语义化封装接口 ------------------------
    def log_train_env_metrics(self, metrics: ScalarMapping, step: int) -> None:
        """记录环境训练相关指标到 TensorBoard（前缀为 ``train/``）。

        期望的 key：
        - episode_return
        - episode_length
        - success_rate
        - out_of_bounds_rate
        - collision_rate
        - capture_rate
        - timeout_rate
        - pursuer_oob_rate
        - obstacle_termination_rate
        """
        self.log_dict(metrics, step, prefix="train")

    def log_ppo_metrics(self, metrics: ScalarMapping, step: int) -> None:
        """记录 PPO 相关指标到 TensorBoard（前缀为 ``ppo/``）。

        期望的 key：
        - policy_loss
        - value_loss
        - entropy
        - approx_kl
        - clip_fraction
        - grad_norm
        """
        self.log_dict(metrics, step, prefix="ppo")

    def log_env_diagnostics(self, metrics: ScalarMapping, step: int) -> None:
        """记录环境诊断相关指标到 TensorBoard（前缀为 ``env/``）。

        期望的 key：
        - mean_goal_distance
        - final_goal_distance
        - reward_progress
        - reward_time_penalty
        - reward_reach_bonus
        """
        self.log_dict(metrics, step, prefix="env")

    def flush(self) -> None:
        """立即将缓存写入磁盘."""
        self.tb.flush()

    def close(self) -> None:
        self.tb.close()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """返回配置好的标准输出 logger（与 TensorBoard Logger 相互独立）。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(h)
    return logger
