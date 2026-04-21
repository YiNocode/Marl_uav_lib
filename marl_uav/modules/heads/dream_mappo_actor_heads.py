"""Dream-MAPPO：流形几何由 policy 用全局 state 预计算；本模块仅负责 obs 特征上的残差 tanh-高斯。"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn
from torch.distributions import Normal


def pursuit_state_slices(num_pursuers: int) -> tuple[int, int, int]:
    """与 PursuitEvasion3v1Task.build_state 拼接顺序一致。"""
    n = int(num_pursuers)
    p = 3 * n
    evader_start = 3 * p  # pursuer pos + vel + ang
    rels_start = evader_start + 6  # evader pos(3) + vel(3)
    return p, evader_start, rels_start


def manifold_targets_from_pursuit_state(
    state_b: torch.Tensor,
    rho: torch.Tensor,
    psi: torch.Tensor,
    *,
    num_pursuers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-pursuer manifold targets in normalized state coordinates."""
    B = state_b.shape[0]
    n = int(num_pursuers)
    p3, evader_start, _ = pursuit_state_slices(n)
    device = state_b.device
    dtype = state_b.dtype

    P = state_b[:, 0:p3].reshape(B, n, 3)
    E = state_b[:, evader_start : evader_start + 3].reshape(B, 3)
    Pxy = P[:, :, :2]
    Exy = E[:, :2]

    alpha = torch.atan2(Pxy[:, :, 1] - Exy[:, 1:2], Pxy[:, :, 0] - Exy[:, 0:1])
    order = torch.argsort(alpha, dim=1)
    inv_rank = torch.zeros(B, n, dtype=torch.long, device=device)
    k_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    inv_rank.scatter_(1, order, k_idx)

    phi = (2.0 * math.pi / float(n)) * inv_rank.to(dtype=dtype)
    ang = phi + psi.unsqueeze(-1)
    targets = torch.zeros(B, n, 3, device=device, dtype=dtype)
    targets[:, :, 0] = Exy[:, 0:1] + rho.unsqueeze(-1) * torch.cos(ang)
    targets[:, :, 1] = Exy[:, 1:2] + rho.unsqueeze(-1) * torch.sin(ang)
    targets[:, :, 2] = E[:, 2:3]
    weights = torch.ones(B, n, 1, device=device, dtype=dtype)
    return targets, P, weights


def geom_actions_from_pursuit_state(
    state_b: torch.Tensor,
    rho: torch.Tensor,
    psi: torch.Tensor,
    *,
    num_pursuers: int,
    a_max_geom: float,
    sigma_p: float,
    action_dim: int,
) -> torch.Tensor:
    """在归一化 xy 平面上构造圆形流形目标点，并输出几何动作 (仅前两维非零)。

    state_b: (B, state_dim)，取 pursuer 与 evader 的归一化位置段。
    rho, psi: (B,) 当前步包围半径与相位偏置（已由全局头产生）。
    """
    targets, P, _ = manifold_targets_from_pursuit_state(
        state_b,
        rho,
        psi,
        num_pursuers=num_pursuers,
    )
    e_xy = targets[:, :, :2] - P[:, :, :2]
    a_xy = float(a_max_geom) * torch.tanh(e_xy / float(sigma_p))

    B, n = P.shape[0], P.shape[1]
    out = torch.zeros(B, n, int(action_dim), device=P.device, dtype=P.dtype)
    out[:, :, :2] = a_xy
    return out


def structure_uv_to_rho_psi(
    uv: torch.Tensor,
    *,
    rho_scale: float,
    rho_min: float,
    psi_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(u_ρ, u_ψ) logits -> (ρ, ψ)，供几何模块使用。"""
    u_rho, u_psi = uv[:, 0], uv[:, 1]
    rho = torch.nn.functional.softplus(u_rho) * float(rho_scale) + float(rho_min)
    psi = torch.tanh(u_psi) * float(psi_scale)
    return rho, psi


def _squashed_scaled_tanh_log_prob(
    dist: Normal,
    z: torch.Tensor,
    a_max: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """a_res = a_max * tanh(z)，log p(a_res)=log p(z) - sum log|a_max * (1-tanh(z)^2)|。"""
    u = torch.tanh(z)
    base = dist.log_prob(z)
    scale = a_max.to(device=z.device, dtype=z.dtype)
    if scale.ndim == 1 and z.ndim > 1:
        scale = scale.expand_as(z)
    correction = torch.log(torch.clamp(scale * (1.0 - u.pow(2)), min=eps))
    return (base - correction).sum(dim=-1)


def _z_from_a_res(a_res: torch.Tensor, a_max: torch.Tensor, eps: float) -> torch.Tensor:
    """由 a_res = a_max*tanh(z) 反解 z = atanh(a_res/a_max)。"""
    scale = a_max.to(device=a_res.device, dtype=a_res.dtype)
    if scale.ndim == 1 and a_res.ndim > 1:
        scale = scale.expand_as(a_res)
    u = a_res / torch.clamp(scale, min=eps)
    u = torch.clamp(u, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(u) - torch.log1p(-u))


class DreamMappoActorHead(nn.Module):
    """仅局部 obs 特征上的残差：z~N(μ(obs),σ)，a_res=a_max*tanh(z)，a=a_geom+a_res。

    几何动作 a_geom 由 policy 用全局 state 单独计算后传入。
    """

    def __init__(
        self,
        feat_dim: int,
        num_pursuers: int,
        action_dim: int,
        *,
        a_max_residual: float,
        log_std_init: float = -0.5,
        log_std_min: float = -2.5,
        log_std_max: float = 1.0,
        squash_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_pursuers = int(num_pursuers)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.squash_eps = float(squash_eps)

        am = torch.full((self.action_dim,), float(a_max_residual), dtype=torch.float32)
        self.register_buffer("a_max_res", am)

        self.residual_mean_xy = nn.Linear(self.feat_dim, 2)
        self.residual_mean_rest = (
            nn.Linear(self.feat_dim, self.action_dim - 2) if self.action_dim > 2 else None
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    def _clamp_log_std(self) -> torch.Tensor:
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

    def _residual_mean_z(self, feat_flat: torch.Tensor) -> torch.Tensor:
        mxy = self.residual_mean_xy(feat_flat)
        if self.residual_mean_rest is None:
            return mxy
        mr = self.residual_mean_rest(feat_flat)
        return torch.cat([mxy, mr], dim=-1)

    def forward(
        self,
        feat_flat: torch.Tensor,
        a_geom: torch.Tensor,
        *,
        B: int,
        N: int,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """feat_flat: (B*N, F)；a_geom: (B, N, A)，由 policy 用 state 预先计算。"""
        if N != self.num_pursuers:
            raise ValueError(
                f"DreamMappoActorHead expects N={self.num_pursuers}, got N={N}."
            )
        mean_z = self._residual_mean_z(feat_flat)
        log_std = self._clamp_log_std()
        if mean_z.ndim > 1:
            log_std = log_std.expand_as(mean_z)
        std = log_std.exp()
        dist = Normal(mean_z, std)

        if deterministic:
            z = mean_z
        else:
            z = dist.rsample()

        a_res = self.a_max_res * torch.tanh(z)
        actions = a_geom.reshape(B * N, self.action_dim) + a_res
        log_probs = _squashed_scaled_tanh_log_prob(
            dist, z, self.a_max_res, self.squash_eps
        )
        entropy = dist.entropy().sum(dim=-1)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": mean_z,
            "a_geom": a_geom,
            "mean_z": mean_z,
            "log_std": self._clamp_log_std(),
        }

    def evaluate_actions(
        self,
        feat_flat: torch.Tensor,
        actions: torch.Tensor,
        a_geom: torch.Tensor,
        *,
        B: int,
        N: int,
    ) -> Dict[str, Any]:
        a_geom_f = a_geom.reshape(B * N, self.action_dim)

        mean_z = self._residual_mean_z(feat_flat)
        log_std = self._clamp_log_std()
        if mean_z.ndim > 1:
            log_std = log_std.expand_as(mean_z)
        std = log_std.exp()
        dist = Normal(mean_z, std)

        actions = actions.to(device=mean_z.device, dtype=mean_z.dtype)
        a_res = actions - a_geom_f
        z = _z_from_a_res(a_res, self.a_max_res, self.squash_eps)
        log_probs = _squashed_scaled_tanh_log_prob(
            dist, z, self.a_max_res, self.squash_eps
        )
        entropy = dist.entropy().sum(dim=-1)

        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "logits": mean_z,
            "mean": mean_z,
            "log_std": self._clamp_log_std(),
        }
