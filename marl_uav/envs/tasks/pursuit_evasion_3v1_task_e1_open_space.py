from __future__ import annotations

"""E1.1 open-space pursuit task alias.

This file intentionally adds a new task module for the main benchmark series
without changing the existing task factory. The class subclasses the current
ex1 pursuit task so it remains compatible with the PyFlyt adapter's existing
3v1 diagnostics, reference-manifold logging, and structure metrics.
"""

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import (
    PursuitEvasion3v1Task as _Ex1PursuitTask,
)


class E11OpenSpacePursuitTask(_Ex1PursuitTask):
    """Open-space E1.1 3v1 pursuit task.

    The benchmark configs use the existing factory-compatible task name
    ``pursuit_evasion_3v1_ex1`` so old training/eval entry points continue to
    work without modifying repository code. This class is a named extension
    point for future E1 scenarios or scripts that want to instantiate the
    benchmark task directly.
    """

    scenario_id = "E1.1-open-space"
    scenario_name = "open_space_obstacle_free"


PursuitEvasion3v1Task = E11OpenSpacePursuitTask

