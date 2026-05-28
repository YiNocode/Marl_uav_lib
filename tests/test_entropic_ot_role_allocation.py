from __future__ import annotations

import numpy as np

from marl_uav.framework.role_allocation import (
    entropic_ot_assignment,
    sinkhorn_transport_plan,
)


def test_sinkhorn_plan_is_doubly_stochastic_3x3() -> None:
    cost = np.array(
        [[0.0, 2.0, 4.0], [2.0, 0.0, 3.0], [5.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    p = sinkhorn_transport_plan(cost, epsilon=0.2, num_iters=50)
    assert p.shape == (3, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0 / 3.0, atol=2e-3)
    np.testing.assert_allclose(p.sum(axis=0), 1.0 / 3.0, atol=2e-3)
    assert np.all(p >= -1e-8)


def test_entropic_ot_prefers_low_cost_pairing() -> None:
    cost = np.array(
        [[0.1, 5.0, 5.0], [5.0, 0.1, 5.0], [5.0, 5.0, 0.1]],
        dtype=np.float64,
    )
    assign = entropic_ot_assignment(cost, epsilon=0.05, num_iters=40)
    assert tuple(int(assign[i]) for i in range(3)) == (0, 1, 2)


def test_entropic_ot_inertia_holds_assignment() -> None:
    cost = np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    prev = np.array([0, 1, 2], dtype=np.int64)
    assign = entropic_ot_assignment(
        cost,
        epsilon=0.1,
        num_iters=30,
        prev_assignment=prev,
        inertia_margin=1.0,
    )
    np.testing.assert_array_equal(assign, prev)


def test_sce_config_uses_entropic_ot_and_proportional_block() -> None:
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "experiment" / "e1_1_open_space_pyflyt_sce.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["benchmark"]["method"] == "sce"
    assert cfg["task"]["role_assignment_mode"] == "entropic_ot"
    assert "sce" in cfg
    assert "algo" not in cfg
