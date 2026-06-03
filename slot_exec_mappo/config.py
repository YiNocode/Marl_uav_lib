"""Configuration dataclass for slot execution MAPPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecObsConfig:
    obstacle_slots: int = 16
    include_prev_action: bool = True


@dataclass
class ExecRewardConfig:
    w_progress: float = 2.0
    w_alive: float = 0.01
    w_time: float = 0.02
    w_clearance: float = 0.15
    w_arrive: float = 5.0
    w_collision: float = 15.0
    w_oob: float = 10.0
    w_smooth: float = 0.05
    w_team_collision: float = 5.0
    progress_dist_norm: float = 2.0
    arrive_dist: float = 0.25
    clearance_margin: float = 0.30
    success_all_agents: bool = False


@dataclass
class MAPPOTrainConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.001
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    rollout_steps: int = 128
    hidden_sizes: tuple[int, ...] = (256, 256)
    target_kl: float = 0.03


@dataclass
class SlotExecConfig:
    obs: ExecObsConfig = field(default_factory=ExecObsConfig)
    reward: ExecRewardConfig = field(default_factory=ExecRewardConfig)
    train: MAPPOTrainConfig = field(default_factory=MAPPOTrainConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SlotExecConfig":
        d = dict(raw or {})
        obs_d = dict(d.get("obs") or {})
        rew_d = dict(d.get("reward") or {})
        train_d = dict(d.get("train") or {})
        hidden = train_d.get("hidden_sizes", [256, 256])
        return cls(
            obs=ExecObsConfig(
                obstacle_slots=int(obs_d.get("obstacle_slots", 16)),
                include_prev_action=bool(obs_d.get("include_prev_action", True)),
            ),
            reward=ExecRewardConfig(
                w_progress=float(rew_d.get("w_progress", 2.0)),
                w_alive=float(rew_d.get("w_alive", 0.01)),
                w_time=float(rew_d.get("w_time", 0.02)),
                w_clearance=float(rew_d.get("w_clearance", 0.15)),
                w_arrive=float(rew_d.get("w_arrive", 5.0)),
                w_collision=float(rew_d.get("w_collision", 15.0)),
                w_oob=float(rew_d.get("w_oob", 10.0)),
                w_smooth=float(rew_d.get("w_smooth", 0.05)),
                w_team_collision=float(rew_d.get("w_team_collision", 5.0)),
                progress_dist_norm=float(rew_d.get("progress_dist_norm", 2.0)),
                arrive_dist=float(rew_d.get("arrive_dist", 0.25)),
                clearance_margin=float(rew_d.get("clearance_margin", 0.30)),
                success_all_agents=bool(rew_d.get("success_all_agents", False)),
            ),
            train=MAPPOTrainConfig(
                gamma=float(train_d.get("gamma", 0.99)),
                gae_lambda=float(train_d.get("gae_lambda", 0.95)),
                clip_ratio=float(train_d.get("clip_ratio", 0.2)),
                lr=float(train_d.get("lr", 3e-4)),
                value_coef=float(train_d.get("value_coef", 0.5)),
                entropy_coef=float(train_d.get("entropy_coef", 0.01)),
                entropy_coef_final=float(train_d.get("entropy_coef_final", 0.001)),
                max_grad_norm=float(train_d.get("max_grad_norm", 0.5)),
                ppo_epochs=int(train_d.get("ppo_epochs", 4)),
                rollout_steps=int(train_d.get("rollout_steps", 128)),
                hidden_sizes=tuple(int(x) for x in hidden),
                target_kl=float(train_d.get("target_kl", 0.03)),
            ),
        )
