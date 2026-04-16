"""Tests for base policy."""
import pytest

from marl_uav.policies.base_policy import BasePolicy


def test_base_policy_is_abc():
    """BasePolicy should be abstract."""
    with pytest.raises(TypeError):
        BasePolicy()
