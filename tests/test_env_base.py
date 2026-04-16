"""Tests for base env."""
import pytest

from marl_uav.envs.base_env import BaseEnv


def test_base_env_is_abc():
    """BaseEnv should be abstract."""
    with pytest.raises(TypeError):
        BaseEnv()
