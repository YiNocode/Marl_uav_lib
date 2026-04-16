"""APF 追逃控制单元测试（无仿真）。"""

from __future__ import annotations

import numpy as np

from marl_uav.control.apf_pursuit import apf_acceleration_3d, apf_action_from_force


def test_attraction_points_toward_evader():
    p = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    e = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    f = apf_acceleration_3d(p, e, [], k_att=1.0, k_rep=0.0, rho0=5.0)
    assert f[0] > 0.9 and abs(f[1]) < 0.1


def test_repulsion_when_close():
    p0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    e = np.array([10.0, 0.0, 1.0], dtype=np.float32)
    p1 = np.array([0.5, 0.0, 1.0], dtype=np.float32)
    f = apf_acceleration_3d(p0, e, [p1], k_att=0.0, k_rep=10.0, rho0=2.0)
    assert f[0] < -0.1


def test_action_clip():
    low = np.array([-1.0, -1.0, -0.1, -0.2], dtype=np.float32)
    high = np.array([1.0, 1.0, 0.1, 0.2], dtype=np.float32)
    a = apf_action_from_force(np.array([100.0, 0.0, 300.0], dtype=np.float32), low, high)
    assert np.isclose(a[0], 1.0) and np.isclose(a[3], 0.2)
