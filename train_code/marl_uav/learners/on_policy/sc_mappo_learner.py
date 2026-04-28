"""SC-MAPPO：在 MAPPO 基础上加入空间分散度指标，鼓励 pursuer 从 evader 周围不同方向接近。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from marl_uav.learners.on_policy.mappo_learner import MAPPOLearner
from marl_uav.learners.utils.spatial_dispersion import spatial_dispersion_penalty_per_timestep


class SCMAPPOLearner(MAPPOLearner):
    """MAPPO + 空间分散度 shaping。

    分散度：各 pursuer 指向 evader 的单位方向向量两两余弦相似度的均值；越一致值越大。
    在 PPO 中通过对 advantage 减去（中心化后的）惩罚，使策略在训练中抑制「并排同向追击」。

    与 MAPPO 共用同一套 CentralizedCriticPolicy / heads，仅 learner 不同。

    全局 state 须包含 pursuer 相对 evader 的位移（如 PursuitEvasion3v1Task.build_state 末尾 P*3 维）。
    """

    def __init__(
        self,
        policy: Any,
        *,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        dispersion_coef: float = 0.05,
        num_pursuers: int = 3,
        spatial_dim: int = 3,
        rels_from_end: bool = True,
        rels_start_idx: int | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            lr=lr,
            clip_range=clip_range,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
        )
        self.dispersion_coef = float(dispersion_coef)
        self.num_pursuers = int(num_pursuers)
        self.spatial_dim = int(spatial_dim)
        self.rels_from_end = bool(rels_from_end)
        self.rels_start_idx = rels_start_idx
        self._last_dispersion_mean: float | None = None

    def update(self, batch: Any) -> Dict[str, Any]:
        self._last_dispersion_mean = None
        metrics = super().update(batch)
        if self._last_dispersion_mean is not None:
            metrics["loss/dispersion_penalty"] = float(self._last_dispersion_mean)
        return metrics

    def _maybe_adjust_advantages(
        self,
        advantages_bt: np.ndarray,
        state_bt: np.ndarray,
        obs_bt: np.ndarray,
    ) -> np.ndarray:
        del obs_bt
        if self.dispersion_coef == 0.0:
            return advantages_bt

        penalty_t = spatial_dispersion_penalty_per_timestep(
            state_bt,
            num_pursuers=self.num_pursuers,
            spatial_dim=self.spatial_dim,
            rels_from_end=self.rels_from_end,
            rels_start_idx=self.rels_start_idx,
        )
        self._last_dispersion_mean = float(np.mean(penalty_t))
        # 中心化，避免整体平移 advantage
        penalty_t = penalty_t - float(np.mean(penalty_t))
        # 广播到所有 agent（与 MAPPO 展平后的 (T_tot, N) 一致）
        advantages_bt = advantages_bt - self.dispersion_coef * penalty_t[:, np.newaxis]
        return advantages_bt.astype(np.float32, copy=False)

    def state_dict(self) -> Dict[str, Any]:
        out = super().state_dict()
        hp = out.setdefault("hyperparams", {})
        hp.update(
            {
                "dispersion_coef": self.dispersion_coef,
                "num_pursuers": self.num_pursuers,
                "spatial_dim": self.spatial_dim,
                "rels_from_end": self.rels_from_end,
                "rels_start_idx": self.rels_start_idx,
            }
        )
        return out

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        hp = state_dict.get("hyperparams") or {}
        if "dispersion_coef" in hp:
            self.dispersion_coef = float(hp["dispersion_coef"])
        if "num_pursuers" in hp:
            self.num_pursuers = int(hp["num_pursuers"])
        if "spatial_dim" in hp:
            self.spatial_dim = int(hp["spatial_dim"])
        if "rels_from_end" in hp:
            self.rels_from_end = bool(hp["rels_from_end"])
        if "rels_start_idx" in hp:
            v = hp["rels_start_idx"]
            self.rels_start_idx = None if v is None else int(v)
