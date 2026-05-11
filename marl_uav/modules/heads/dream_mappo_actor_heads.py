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


def pursuit_state_base_dim(num_pursuers: int) -> int:
    n = int(num_pursuers)
    return 16 * n + 6


def _wrap_angle_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return torch.remainder(angle + math.pi, two_pi) - math.pi


def _extract_obstacle_block_from_state(
    state_b: torch.Tensor,
    *,
    num_pursuers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_dim = pursuit_state_base_dim(num_pursuers)
    if state_b.shape[-1] <= base_dim:
        zf = torch.zeros((*state_b.shape[:-1], 0), device=state_b.device, dtype=state_b.dtype)
        return zf.reshape(state_b.shape[0], 0, 2), zf.reshape(state_b.shape[0], 0)

    extra_dim = int(state_b.shape[-1] - base_dim)
    if extra_dim <= 0 or extra_dim % 4 != 0:
        zf = torch.zeros((*state_b.shape[:-1], 0), device=state_b.device, dtype=state_b.dtype)
        return zf.reshape(state_b.shape[0], 0, 2), zf.reshape(state_b.shape[0], 0)

    obstacle_block = state_b[:, base_dim:].reshape(state_b.shape[0], extra_dim // 4, 4)
    valid = obstacle_block[:, :, 3] > 0.5
    obstacle_xy = obstacle_block[:, :, 0:2]
    obstacle_r = obstacle_block[:, :, 2]
    obstacle_r = torch.where(valid, obstacle_r, torch.zeros_like(obstacle_r))
    return obstacle_xy, obstacle_r


def _obstacle_aware_radii_from_state(
    state_b: torch.Tensor,
    rho: torch.Tensor,
    psi: torch.Tensor,
    *,
    num_pursuers: int,
    rho_min: float,
    obstacle_manifold_top_k: int = 4,
    obstacle_manifold_influence_radius_scale: float = 2.5,
    obstacle_manifold_clearance_margin_scale: float = 0.35,
    obstacle_manifold_fourier_scale: float = 0.55,
    obstacle_manifold_fourier_order: int = 2,
    obstacle_manifold_bump_sigma_deg: float = 28.0,
    obstacle_manifold_bump_scale: float = 0.45,
    obstacle_manifold_max_extra_radius_scale: float = 1.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = state_b.shape[0]
    n = int(num_pursuers)
    _, evader_start, _ = pursuit_state_slices(n)
    device = state_b.device
    dtype = state_b.dtype

    E = state_b[:, evader_start : evader_start + 3].reshape(B, 3)
    Exy = E[:, :2]
    obstacle_xy, obstacle_r = _extract_obstacle_block_from_state(state_b, num_pursuers=n)

    inv_rank = torch.arange(n, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
    phi = (2.0 * math.pi / float(n)) * inv_rank
    theta = phi + psi.unsqueeze(-1)

    if obstacle_xy.shape[1] == 0:
        return theta, rho.unsqueeze(-1).expand(B, n)

    rel = obstacle_xy - Exy.unsqueeze(1)
    dist = torch.linalg.norm(rel, dim=-1)
    valid = obstacle_r > 0.0
    clear_r = obstacle_r + float(obstacle_manifold_clearance_margin_scale) * float(rho_min)
    surface_dist = dist - clear_r
    influence_radius = float(obstacle_manifold_influence_radius_scale) * torch.maximum(
        rho,
        torch.full_like(rho, float(rho_min)),
    )
    masked_surface = torch.where(valid, surface_dist, torch.full_like(surface_dist, float("inf")))
    sorted_idx = torch.argsort(masked_surface, dim=1)
    top_k = min(int(obstacle_manifold_top_k), int(obstacle_xy.shape[1]))
    gather_idx = sorted_idx[:, :top_k]

    rel_top = torch.gather(rel, 1, gather_idx.unsqueeze(-1).expand(B, top_k, 2))
    clear_top = torch.gather(clear_r, 1, gather_idx)
    dist_top = torch.linalg.norm(rel_top, dim=-1)
    valid_top = torch.gather(valid, 1, gather_idx)
    surface_top = torch.gather(masked_surface, 1, gather_idx)
    phi_obs = torch.atan2(rel_top[:, :, 1], rel_top[:, :, 0])

    influence = influence_radius.unsqueeze(-1).clamp_min(1e-6)
    closeness = torch.clamp(1.0 - surface_top / influence, 0.0, 1.0)
    radial_weight = torch.clamp(clear_top / torch.clamp(dist_top, min=clear_top + 1e-6), 0.0, 1.0)
    weights = torch.where(valid_top, closeness * radial_weight, torch.zeros_like(closeness))
    weight_sum = weights.sum(dim=1, keepdim=True)

    base_shift = torch.where(
        weight_sum > 1e-6,
        float(obstacle_manifold_fourier_scale)
        * torch.minimum(rho, influence_radius)
        * (
            (
                weights
                * torch.clamp(clear_top / torch.clamp(dist_top, min=1e-6), 0.0, 1.0)
            ).sum(dim=1)
            / weight_sum.squeeze(1)
        ),
        torch.zeros_like(rho),
    )
    radius = rho.unsqueeze(-1).expand(B, n) + base_shift.unsqueeze(-1)

    amp_base = float(obstacle_manifold_fourier_scale) * torch.minimum(rho, influence_radius)
    for k in range(1, int(obstacle_manifold_fourier_order) + 1):
        cos_coeff = torch.where(
            weight_sum.squeeze(1) > 1e-6,
            amp_base * ((weights * torch.cos(float(k) * phi_obs)).sum(dim=1) / weight_sum.squeeze(1)),
            torch.zeros_like(rho),
        )
        sin_coeff = torch.where(
            weight_sum.squeeze(1) > 1e-6,
            amp_base * ((weights * torch.sin(float(k) * phi_obs)).sum(dim=1) / weight_sum.squeeze(1)),
            torch.zeros_like(rho),
        )
        radius = radius + cos_coeff.unsqueeze(-1) * torch.cos(float(k) * theta)
        radius = radius + sin_coeff.unsqueeze(-1) * torch.sin(float(k) * theta)

    sigma = math.radians(float(obstacle_manifold_bump_sigma_deg))
    sigma = max(sigma, 1e-3)
    required = torch.zeros_like(radius)
    for j in range(top_k):
        dj = dist_top[:, j].unsqueeze(-1)
        rj = clear_top[:, j].unsqueeze(-1)
        vj = valid_top[:, j].unsqueeze(-1)
        phi_j = phi_obs[:, j].unsqueeze(-1)
        delta = _wrap_angle_pi_torch(theta - phi_j)
        sin_abs = torch.abs(torch.sin(delta))
        cos_val = torch.cos(delta)
        valid_ray = vj & (sin_abs < (rj / torch.clamp(dj, min=1e-6))) & (cos_val > 0.0)
        root = torch.sqrt(torch.clamp(rj * rj - (dj * sin_abs) ** 2, min=0.0))
        radial = dj * cos_val + root
        required = torch.maximum(required, torch.where(valid_ray, radial, torch.zeros_like(radial)))
        bump = torch.exp(-0.5 * (delta / sigma) ** 2)
        required = torch.maximum(
            required,
            torch.where(vj, float(obstacle_manifold_bump_scale) * rj * bump, torch.zeros_like(bump)),
        )

    rho_floor = torch.full_like(radius, float(rho_min))
    extra_cap = rho.unsqueeze(-1) + float(obstacle_manifold_max_extra_radius_scale) * float(rho_min)
    radius = torch.maximum(radius, torch.maximum(rho_floor, required))
    radius = torch.minimum(radius, extra_cap)
    return theta, radius


def manifold_targets_from_pursuit_state(
    state_b: torch.Tensor,
    rho: torch.Tensor,
    psi: torch.Tensor,
    *,
    num_pursuers: int,
    rho_min: float = 0.05,
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
    ang_ref, rho_ref = _obstacle_aware_radii_from_state(
        state_b,
        rho,
        psi,
        num_pursuers=n,
        rho_min=float(rho_min),
    )
    targets = torch.zeros(B, n, 3, device=device, dtype=dtype)
    targets[:, :, 0] = Exy[:, 0:1] + rho_ref * torch.cos(ang_ref)
    targets[:, :, 1] = Exy[:, 1:2] + rho_ref * torch.sin(ang_ref)
    targets[:, :, 2] = E[:, 2:3]
    weights = torch.ones(B, n, 1, device=device, dtype=dtype)
    return targets, P, weights


def geom_actions_from_pursuit_state(
    state_b: torch.Tensor,
    rho: torch.Tensor,
    psi: torch.Tensor,
    *,
    num_pursuers: int,
    rho_min: float,
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
        rho_min=rho_min,
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
