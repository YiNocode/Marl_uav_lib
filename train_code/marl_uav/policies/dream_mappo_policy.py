"""Dream-MAPPO：centralized critic + 全局 state 决定流形 (u_ρ,u_ψ)，局部 obs 决定残差。"""

from __future__ import annotations

from typing import Any, Dict, Literal, Sequence, Tuple

import torch
from torch import nn

from marl_uav.modules.encoders.mlp_encoder import MLPEncoder
from marl_uav.modules.heads.dream_mappo_actor_heads import (
    DreamMappoActorHead,
    geom_actions_from_pursuit_state,
    manifold_targets_from_pursuit_state,
    structure_uv_to_rho_psi,
)
from marl_uav.policies.base_policy import BasePolicy


class DreamMappoCentralizedCriticPolicy(nn.Module, BasePolicy):
    """全局结构变量 (u_ρ,u_ψ) 仅由全局 state 经 structure 网络得到；个体残差仅由局部 obs 经 actor_encoder 得到。"""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        *,
        action_space_type: Literal["discrete", "continuous"] = "continuous",
        action_dim: int | None = None,
        action_low: Sequence[float] | None = None,
        action_high: Sequence[float] | None = None,
        encoder_hidden_dims: tuple[int, ...] = (64, 64),
        encoder_output_dim: int | None = None,
        actor_encoder: nn.Module | None = None,
        critic_encoder: nn.Module | None = None,
        structure_encoder: nn.Module | None = None,
        dream_actor_head: nn.Module | None = None,
        value_head: nn.Module | None = None,
        log_std_init: float = -0.5,
        num_pursuers: int = 3,
        a_max_geom: float = 0.15,
        sigma_p: float = 0.5,
        rho_scale: float = 0.5,
        rho_min: float = 0.05,
        psi_scale: float = 3.14159265,
        a_max_residual: float = 0.08,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_space_type = str(action_space_type).lower()
        if self.action_space_type != "continuous":
            raise ValueError("DreamMappoCentralizedCriticPolicy 仅支持 continuous。")
        if action_dim is None or action_low is None or action_high is None:
            raise ValueError("continuous 模式需要 action_dim / action_low / action_high。")
        self.n_actions = 0
        self.action_dim = int(action_dim)
        self._action_low = list(float(x) for x in action_low)
        self._action_high = list(float(x) for x in action_high)
        if len(self._action_low) != self.action_dim or len(self._action_high) != self.action_dim:
            raise ValueError("action_low/high 长度须等于 action_dim。")

        self.num_pursuers = int(num_pursuers)
        self.a_max_geom = float(a_max_geom)
        self.sigma_p = float(sigma_p)
        self.rho_scale = float(rho_scale)
        self.rho_min = float(rho_min)
        self.psi_scale = float(psi_scale)
        self.actor_condition_dim = 7

        self.actor_encoder: nn.Module = (
            actor_encoder
            if actor_encoder is not None
            else MLPEncoder(
                input_dim=self.obs_dim + self.actor_condition_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )
        self.critic_encoder: nn.Module = (
            critic_encoder
            if critic_encoder is not None
            else MLPEncoder(
                input_dim=self.state_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )
        self.structure_encoder: nn.Module = (
            structure_encoder
            if structure_encoder is not None
            else MLPEncoder(
                input_dim=self.state_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
            )
        )

        actor_feat_dim = int(
            getattr(self.actor_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1])
        )
        critic_feat_dim = int(
            getattr(self.critic_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1])
        )
        struct_feat_dim = int(
            getattr(self.structure_encoder, "output_dim", encoder_output_dim or encoder_hidden_dims[-1])
        )

        self.structure_uv_head = nn.Linear(struct_feat_dim, 2)

        self.dream_actor_head: nn.Module = (
            dream_actor_head
            if dream_actor_head is not None
            else DreamMappoActorHead(
                feat_dim=actor_feat_dim,
                num_pursuers=self.num_pursuers,
                action_dim=self.action_dim,
                a_max_residual=float(a_max_residual),
                log_std_init=float(log_std_init),
            )
        )

        self.value_head: nn.Module = (
            value_head if value_head is not None else nn.Linear(critic_feat_dim, 1)
        )

    def _structure_uv(self, state_b: torch.Tensor) -> torch.Tensor:
        """全局 state -> (B, 2) 的 (u_ρ, u_ψ) logits（仅依赖 state）。"""
        h = self.structure_encoder(state_b)
        return self.structure_uv_head(h)

    def _geom_from_state(self, state_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """state -> uv -> ρ,ψ -> a_geom。"""
        uv = self._structure_uv(state_b)
        rho, psi = structure_uv_to_rho_psi(
            uv,
            rho_scale=self.rho_scale,
            rho_min=self.rho_min,
            psi_scale=self.psi_scale,
        )
        a_geom = geom_actions_from_pursuit_state(
            state_b,
            rho,
            psi,
            num_pursuers=self.num_pursuers,
            a_max_geom=self.a_max_geom,
            sigma_p=self.sigma_p,
            action_dim=self.action_dim,
        )
        return a_geom, rho, psi

    def _actor_condition_from_state(
        self,
        state_b: torch.Tensor,
        rho: torch.Tensor,
        psi: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build per-agent manifold conditioning features for the actor."""
        targets, pursuer_pos, weights = manifold_targets_from_pursuit_state(
            state_b,
            rho,
            psi,
            num_pursuers=self.num_pursuers,
        )
        rel_slot = targets - pursuer_pos
        rel_norm = torch.linalg.norm(rel_slot, dim=-1, keepdim=True).clamp_min(1e-6)
        slot_dir = rel_slot / rel_norm
        actor_cond = torch.cat([rel_slot, slot_dir, weights], dim=-1)
        return {
            "actor_cond": actor_cond,
            "slot_rel": rel_slot,
            "slot_dir": slot_dir,
            "slot_weight": weights,
            "slot_target": targets,
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_obs(self, obs: Any) -> tuple[torch.Tensor, int, int]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs shape 须为 (batch, n_agents, {self.obs_dim}) 或 (n_agents, {self.obs_dim})，"
                f"当前 {tuple(x.shape)}"
            )
        B, N, _ = x.shape
        return x, B, N

    def _prepare_state(self, state: Any, *, B: int, N: int) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0).unsqueeze(0)
        if s.ndim == 2:
            s = s.unsqueeze(1)
        if s.ndim != 3 or s.shape[-1] != self.state_dim:
            raise ValueError(
                f"state shape 须为 (B,N,{self.state_dim}) / (B,{self.state_dim}) / ({self.state_dim},)，"
                f"当前 {tuple(s.shape)}"
            )
        if s.shape[0] != B:
            if s.shape[0] == 1 and B > 1:
                s = s.expand(B, s.shape[1], self.state_dim)
            else:
                raise ValueError(f"obs batch B={B} 与 state batch {s.shape[0]} 不一致。")
        if s.shape[1] != N:
            if s.shape[1] == 1:
                s = s.expand(B, N, self.state_dim)
            else:
                raise ValueError(f"obs agents N={N} 与 state agents {s.shape[1]} 不一致。")
        return s

    def _state_b(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """取 batch 内一份全局 state 向量 (B, S)。"""
        return state_tensor[:, 0, :].contiguous()

    def forward(
        self,
        obs: Any,
        state: Any,
        *,
        avail_actions: Any | None = None,
        deterministic: bool = False,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        obs_tensor, B, N = self._prepare_obs(obs)
        state_tensor = self._prepare_state(state, B=B, N=N)
        state_b = self._state_b(state_tensor)

        a_geom, rho, psi = self._geom_from_state(state_b)
        cond = self._actor_condition_from_state(state_b, rho, psi)

        actor_input = torch.cat([obs_tensor, cond["actor_cond"]], dim=-1)
        flat_actor_input = actor_input.reshape(B * N, self.obs_dim + self.actor_condition_dim)
        actor_feat_flat = self.actor_encoder(flat_actor_input)

        actor_out_flat = self.dream_actor_head(
            actor_feat_flat,
            a_geom,
            B=B,
            N=N,
            deterministic=deterministic,
        )

        adim = self.action_dim
        actions = actor_out_flat["actions"].reshape(B, N, adim)
        log_probs = actor_out_flat["log_probs"].reshape(B, N)
        entropy = actor_out_flat["entropy"].reshape(B, N)
        logits = actor_out_flat["logits"].reshape(B, N, adim)

        actor_out = {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": logits,
            "rho": rho,
            "psi": psi,
            "a_geom": a_geom,
            "actor_cond": cond["actor_cond"],
            "slot_rel": cond["slot_rel"],
            "slot_dir": cond["slot_dir"],
            "slot_weight": cond["slot_weight"],
            "slot_target": cond["slot_target"],
        }

        flat_state = state_tensor.reshape(B * N, self.state_dim)
        critic_feat_flat = self.critic_encoder(flat_state)
        values_flat = self.value_head(critic_feat_flat)
        if values_flat.ndim > 1:
            values_flat = values_flat.squeeze(-1)
        values = values_flat.reshape(B, N)
        critic_out = {"values": values}
        return actor_out, critic_out

    def act(
        self,
        obs: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        deterministic: bool = False,
        return_entropy: bool = False,
        **kwargs: Any,
    ):
        if state is None:
            raise ValueError("DreamMappoCentralizedCriticPolicy.act 需要 state。")
        actor_out, critic_out = self.forward(
            obs, state, avail_actions=avail_actions, deterministic=deterministic
        )
        actions = actor_out["actions"]
        log_probs = actor_out["log_probs"]
        values = critic_out["values"]
        if return_entropy:
            return actions, log_probs, values, actor_out["entropy"]
        return actions, log_probs, values

    def evaluate_actions(
        self,
        obs: Any,
        actions: Any,
        *,
        state: Any | None = None,
        avail_actions: Any | None = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            raise ValueError("DreamMappoCentralizedCriticPolicy.evaluate_actions 需要 state。")
        obs_tensor, B, N = self._prepare_obs(obs)
        state_tensor = self._prepare_state(state, B=B, N=N)
        state_b = self._state_b(state_tensor)

        a_geom, rho, psi = self._geom_from_state(state_b)
        cond = self._actor_condition_from_state(state_b, rho, psi)

        actor_input = torch.cat([obs_tensor, cond["actor_cond"]], dim=-1)
        flat_actor_input = actor_input.reshape(B * N, self.obs_dim + self.actor_condition_dim)
        actor_feat_flat = self.actor_encoder(flat_actor_input)
        flat_actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        ).reshape(B * N, self.action_dim)

        actor_eval = self.dream_actor_head.evaluate_actions(
            actor_feat_flat,
            flat_actions,
            a_geom,
            B=B,
            N=N,
        )
        new_log_probs = actor_eval["log_probs"].reshape(B, N)
        entropy = actor_eval["entropy"].reshape(B, N)

        flat_state = state_tensor.reshape(B * N, self.state_dim)
        critic_feat_flat = self.critic_encoder(flat_state)
        values_flat = self.value_head(critic_feat_flat)
        if values_flat.ndim > 1:
            values_flat = values_flat.squeeze(-1)
        values = values_flat.reshape(B, N)
        return new_log_probs, entropy, values
