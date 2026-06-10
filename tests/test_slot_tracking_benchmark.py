from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from experiments.slot_tracking.controllers.baseline_pure_pursuit import ControllerObservation
from experiments.slot_tracking.controllers.wrapped_existing_controller import ExistingControllerWrapper
from experiments.slot_tracking.metrics.safety_metrics import (
    obstacle_collision,
    outside_boundary,
)
from experiments.slot_tracking.run_slot_tracking_benchmark import _initial_boundary_state, run_episode
from experiments.slot_tracking.scenarios.slot_scenarios import ScenarioSpec, instantiate_scenario, scenario_specs
from marl_uav.framework.geometry.obstacle_geometry import Obstacle


def _cfg() -> dict:
    with Path("experiments/slot_tracking/configs/slot_tracking_default.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["dynamics"]["episode_steps"] = 260
    cfg["benchmark"]["num_seeds"] = 1
    cfg["output"]["save_raw_trajectories"] = False
    return cfg


def test_static_slot_solved_by_pure_pursuit_and_pd() -> None:
    cfg = _cfg()
    spec = ScenarioSpec(name="static_slot", group="A", speed_scale=0.2)
    instance = instantiate_scenario(spec, cfg, seed=123)
    instance.init_positions[0, :2] = np.array([1.0, 0.0])

    for controller in ["pure_pursuit", "pd"]:
        row = run_episode(
            controller_name=controller,
            controller_cfg=cfg["controllers"][controller],
            instance=instance,
            cfg=cfg,
            seed=123,
            raw_path=None,
        )
        assert row["success"] is True
        assert row["final_error"] <= cfg["success"]["final_error_threshold"]
        assert np.isfinite(row["rmse_error"])
        assert np.isfinite(row["p95_error"])


def test_collision_checker_detects_obstacle_overlap() -> None:
    obs = [Obstacle(kind="circle", center=np.array([0.0, 0.0]), radius=0.5)]
    assert obstacle_collision(np.array([0.4, 0.0]), obs, uav_radius=0.2)
    assert not obstacle_collision(np.array([2.0, 0.0]), obs, uav_radius=0.2)


def test_boundary_checker_detects_out_of_bounds() -> None:
    assert outside_boundary(np.array([20.1, 0.0]), world_xy=20.0, uav_radius=0.15)
    assert not outside_boundary(np.array([19.0, 0.0]), world_xy=20.0, uav_radius=0.15)


def test_metrics_return_finite_values_for_short_episode() -> None:
    cfg = _cfg()
    cfg["dynamics"]["episode_steps"] = 80
    spec = ScenarioSpec(name="linear_slot", group="A", speed_scale=0.2)
    instance = instantiate_scenario(spec, cfg, seed=321)
    row = run_episode(
        controller_name="pure_pursuit",
        controller_cfg=cfg["controllers"]["pure_pursuit"],
        instance=instance,
        cfg=cfg,
        seed=321,
        raw_path=None,
    )
    for key in ["rmse_error", "mean_error", "p95_error", "path_length", "control_effort", "decision_time_ms_mean"]:
        assert np.isfinite(row[key])


def test_boundary_clean_starts_near_valid_boundary_slots() -> None:
    with Path("experiments/slot_tracking/configs/slot_tracking_B_boundary_clean.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    specs = scenario_specs(
        cfg["benchmark"]["scenario_groups"],
        [float(x) for x in cfg["benchmark"]["speed_levels"]],
        cfg.get("robustness", {}),
        scenario_names=cfg["benchmark"].get("scenario_names"),
    )
    assert specs
    for spec in specs:
        instance = instantiate_scenario(spec, cfg, seed=int(cfg["benchmark"]["seeds_start"]))
        initial_error = float(np.linalg.norm(instance.init_positions[0, :2] - instance.slot_positions[0, 0, :2]))
        invalid, margin, outward = _initial_boundary_state(
            instance.init_positions,
            instance.init_velocities,
            world_xy=float(cfg["world"]["world_xy"]),
            uav_radius=float(cfg["dynamics"]["uav_radius"]),
            safety_margin=float(cfg["dynamics"]["safety_margin"]),
            amax=float(cfg["dynamics"]["uav_amax"]),
        )
        assert initial_error <= float(cfg["dynamics"]["lost_threshold"])
        assert invalid is False
        assert margin >= float(cfg["benchmark"]["initial_state"]["min_boundary_margin"])
        if spec.name == "outward_initial_velocity":
            assert outward > 0.0


def test_existing_boundary_filter_preserves_tangent_and_handles_corner() -> None:
    wrapper = ExistingControllerWrapper(
        {
            "boundary_activation_distance": 2.5,
            "boundary_hard_margin": 0.25,
            "boundary_braking_margin": 0.25,
            "boundary_braking_gain": 1.0,
            "max_inward_correction": 1.0,
            "vmax": 1.0,
        }
    )
    obs = ControllerObservation(
        position=np.array([19.75, 19.75, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        slot_position=np.array([19.0, 18.0, 1.0]),
        slot_velocity=np.zeros(3),
        obstacles=[],
        world_xy=20.0,
        dt=0.05,
        uav_vmax=1.0,
        uav_amax=2.0,
        uav_radius=0.15,
        safety_margin=0.25,
    )

    action, diag = wrapper._apply_boundary_filter(obs, np.array([0.4, -0.7]))
    assert action[0] <= 1e-9
    assert np.isclose(action[1], -0.7)
    assert diag["boundary_filter_active"] is True

    corner_action, corner_diag = wrapper._apply_boundary_filter(obs, np.array([0.4, 0.5]))
    assert corner_action[0] <= 1e-9
    assert corner_action[1] <= 1e-9
    assert set(corner_diag["boundary_active_names"].split(",")) == {"x_max", "y_max"}
