"""Tests for base buffer."""
import pytest

from marl_uav.buffers.base_buffer import BaseBuffer


def test_base_buffer_is_abc():
    """BaseBuffer should be abstract."""
    with pytest.raises(TypeError):
        BaseBuffer()
