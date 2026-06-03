"""Tests for phi_max recovery from cached pursuit structure metrics."""

from __future__ import annotations

import math

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task import (
    compute_pursuit_structure_metrics_3v1,
    phi_max_from_c_cov,
    pursuit_structure_from_cached_metrics,
)


def test_phi_max_from_c_cov_inverts_full_metrics() -> None:
    pursuers = np.array([[2.0, 0.0, 1.0], [-1.0, 1.7, 1.0], [-1.0, -1.7, 1.0]], dtype=np.float64)
    evader = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    metrics = compute_pursuit_structure_metrics_3v1(pursuers, evader)
    recovered = phi_max_from_c_cov(metrics["C_cov"])
    assert abs(recovered - metrics["phi_max"]) < 1e-9


def test_pursuit_structure_from_cached_metrics_includes_phi_max() -> None:
    out = pursuit_structure_from_cached_metrics(0.5, 0.8, 0.9)
    assert "phi_max" in out
    assert math.isfinite(out["phi_max"])


def test_phi_max_recovered_from_c_cov_only_series() -> None:
    series = [{"C_cov": 0.2, "C_col": 0.9, "D_ang": 0.8} for _ in range(40)]
    phi = np.asarray([phi_max_from_c_cov(float(s["C_cov"])) for s in series], dtype=np.float64)
    assert phi.shape == (40,)
    assert np.all(np.isfinite(phi))
    assert math.isfinite(float(np.mean(phi[-30:])))
