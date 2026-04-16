"""Tests for base learner."""
import pytest

from marl_uav.learners.base_learner import BaseLearner


def test_base_learner_is_abc():
    """BaseLearner should be abstract."""
    with pytest.raises(TypeError):
        BaseLearner()
