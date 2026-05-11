from __future__ import annotations

import pytest

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task


def test_residual_control_gain_decays_linearly_across_epochs():
    task = PursuitEvasion3v1Task(
        residual_control_gain=0.5,
        residual_control_gain_final=0.1,
        residual_control_gain_decay_epochs=5,
    )

    gains = [task.set_training_progress(epoch=epoch, num_epochs=5) for epoch in range(5)]

    assert gains == pytest.approx([0.5, 0.4, 0.3, 0.2, 0.1])


def test_residual_control_gain_uses_num_epochs_when_decay_epochs_omitted():
    task = PursuitEvasion3v1Task(
        residual_control_gain=0.5,
        residual_control_gain_final=0.1,
    )

    first = task.set_training_progress(epoch=0, num_epochs=3)
    middle = task.set_training_progress(epoch=1, num_epochs=3)
    last = task.set_training_progress(epoch=2, num_epochs=3)

    assert first == pytest.approx(0.5)
    assert middle == pytest.approx(0.3)
    assert last == pytest.approx(0.1)
