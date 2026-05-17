from __future__ import annotations

import pytest

from scripts.train import resolve_vec_rollout_steps


def test_vec_rollout_steps_per_env_is_legacy_default() -> None:
    assert resolve_vec_rollout_steps(
        rollout_steps=1024,
        num_envs=8,
        train_cfg={},
    ) == 1024


def test_vec_rollout_steps_total_preserves_update_cadence() -> None:
    assert resolve_vec_rollout_steps(
        rollout_steps=1024,
        num_envs=8,
        train_cfg={"vec_rollout_steps_mode": "total"},
    ) == 128


def test_vec_rollout_steps_total_rounds_up() -> None:
    assert resolve_vec_rollout_steps(
        rollout_steps=1001,
        num_envs=8,
        train_cfg={"vec_rollout_steps_mode": "total"},
    ) == 126


def test_vec_rollout_steps_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="vec_rollout_steps_mode"):
        resolve_vec_rollout_steps(
            rollout_steps=1024,
            num_envs=8,
            train_cfg={"vec_rollout_steps_mode": "episode"},
        )
