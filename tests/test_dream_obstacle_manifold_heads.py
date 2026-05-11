from __future__ import annotations

import math

import torch

from marl_uav.modules.heads.dream_mappo_actor_heads import (
    manifold_targets_from_pursuit_state,
    pursuit_state_base_dim,
)


def _base_state() -> torch.Tensor:
    n = 3
    state = torch.zeros((1, pursuit_state_base_dim(n)), dtype=torch.float32)
    pursuers = torch.tensor(
        [[-0.4, -0.1, 0.0], [-0.4, 0.0, 0.0], [-0.4, 0.1, 0.0]],
        dtype=torch.float32,
    ).reshape(-1)
    evader = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float32)
    rels = torch.tensor(
        [[-0.6, -0.1, 0.0], [-0.6, 0.0, 0.0], [-0.6, 0.1, 0.0]],
        dtype=torch.float32,
    ).reshape(-1)
    assigned_targets = torch.zeros((9,), dtype=torch.float32)
    assignment = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    state[0, 0:9] = pursuers
    state[0, 27:30] = evader
    state[0, 33:42] = rels
    state[0, 42:51] = assigned_targets
    state[0, 51:54] = assignment
    return state


def test_dream_manifold_targets_match_circle_without_obstacles():
    state = _base_state()
    rho = torch.tensor([0.3], dtype=torch.float32)
    psi = torch.tensor([0.0], dtype=torch.float32)
    targets, _, _ = manifold_targets_from_pursuit_state(
        state,
        rho,
        psi,
        num_pursuers=3,
        rho_min=0.05,
    )

    exy = torch.tensor([0.2, 0.0], dtype=torch.float32)
    expected = torch.tensor(
        [
            [exy[0] + 0.3 * math.cos(0.0), exy[1] + 0.3 * math.sin(0.0), 0.0],
            [exy[0] + 0.3 * math.cos(2.0 * math.pi / 3.0), exy[1] + 0.3 * math.sin(2.0 * math.pi / 3.0), 0.0],
            [exy[0] + 0.3 * math.cos(4.0 * math.pi / 3.0), exy[1] + 0.3 * math.sin(4.0 * math.pi / 3.0), 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    torch.testing.assert_close(targets, expected, atol=1e-5, rtol=1e-5)


def test_dream_manifold_targets_expand_blocked_direction_for_ex2_state():
    state = _base_state()
    obstacle_block = torch.tensor([[0.36, 0.0, 0.09, 1.0]], dtype=torch.float32).reshape(1, 4)
    state = torch.cat([state, obstacle_block], dim=1)
    rho = torch.tensor([0.3], dtype=torch.float32)
    psi = torch.tensor([0.0], dtype=torch.float32)

    targets, _, _ = manifold_targets_from_pursuit_state(
        state,
        rho,
        psi,
        num_pursuers=3,
        rho_min=0.05,
    )
    exy = torch.tensor([0.2, 0.0], dtype=torch.float32)
    radii = torch.linalg.norm(targets[0, :, :2] - exy.unsqueeze(0), dim=1)

    assert radii[0] > rho[0]
    assert radii[0] > radii[1]
    assert radii[0] > radii[2]
