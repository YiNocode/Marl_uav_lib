from __future__ import annotations

"""模型 checkpoint 工具：保存/加载最新与最优 learner 状态."""

from pathlib import Path
from typing import Any, Mapping

import torch


class CheckpointManager:
    """管理训练过程中的 checkpoint（最新 / 当前最优）。"""

    def __init__(
        self,
        ckpt_dir: str | Path,
        *,
        best_metric: str = "train/avg_return",
        mode: str = "max",
    ) -> None:
        """
        Args:
            ckpt_dir: 保存 checkpoint 的目录
            best_metric: 用于挑选“最优模型”的指标名
            mode: "max" 表示该指标越大越好, "min" 表示越小越好
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.best_metric = best_metric
        self.mode = mode
        self.best_value: float | None = None

    # ---------------------------- public API ---------------------------- #
    def save(
        self,
        *,
        learner: Any,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, float],
    ) -> None:
        """保存当前 learner 状态为 latest.pt，并在更优时更新 best.pt."""

        if not hasattr(learner, "state_dict") or not hasattr(learner, "load_state_dict"):
            raise TypeError(
                "learner must implement state_dict() and load_state_dict() "
                "to be used with CheckpointManager."
            )

        state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": dict(metrics),
            "learner": learner.state_dict(),  # type: ignore[no-untyped-call]
        }

        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(state, latest_path)

        metric_val = metrics.get(self.best_metric)
        if metric_val is None:
            return

        metric_val = float(metric_val)
        if self._is_better(metric_val):
            self.best_value = metric_val
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)

    # --------------------------- helper methods ------------------------- #
    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value


def load_checkpoint(path: str | Path, learner: Any) -> dict[str, Any]:
    """从 checkpoint 文件加载 learner 参数并返回完整 state 字典."""

    data = torch.load(Path(path), map_location="cpu")
    learner_state = data.get("learner")
    if learner_state is not None and hasattr(learner, "load_state_dict"):
        # #region agent log
        try:
            import json
            policy_state = learner_state.get("policy") or {}
            ckpt_critic = policy_state.get("critic_encoder.net.0.weight")
            ckpt_actor = policy_state.get("actor_encoder.net.0.weight")
            cur = getattr(learner, "policy", None)
            cur_critic_sh = list(cur.state_dict()["critic_encoder.net.0.weight"].shape) if cur and "critic_encoder.net.0.weight" in cur.state_dict() else None
            cur_actor_sh = list(cur.state_dict()["actor_encoder.net.0.weight"].shape) if cur and "actor_encoder.net.0.weight" in cur.state_dict() else None

        except Exception:
            pass
        # #endregion
        learner.load_state_dict(learner_state)  # type: ignore[no-untyped-call]
    return data

