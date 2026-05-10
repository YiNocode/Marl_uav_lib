from __future__ import annotations

import numpy as np

from marl_uav.runners.vecenv_trainer import (
    VecEnvTrainer,
    _vec_info_bool_any,
    _vec_info_bool,
    _vec_info_float,
    _vec_info_pick,
    _vec_info_pick_terminal_aware,
)


def test_vec_info_pick_terminal_aware_falls_back_to_final_info_when_mask_is_false():
    infos = {
        "captured": np.array([False, False], dtype=np.bool_),
        "_captured": np.array([False, False], dtype=np.bool_),
        "final_info": np.array(
            [
                {"captured": True, "capture_step": 17, "mean_goal_distance": 1.25},
                None,
            ],
            dtype=object,
        ),
        "_final_info": np.array([True, False], dtype=np.bool_),
    }

    assert bool(_vec_info_pick(infos, "captured", 0)) is False
    assert bool(_vec_info_pick_terminal_aware(infos, "captured", 0)) is True
    assert _vec_info_bool(infos, "captured", 0) is True
    assert _vec_info_pick_terminal_aware(infos, "capture_step", 0) == 17
    assert _vec_info_float(infos, "mean_goal_distance", 0) == 1.25


def test_vec_info_pick_terminal_aware_prefers_regular_info_when_mask_is_true():
    infos = {
        "captured": np.array([True, False], dtype=np.bool_),
        "_captured": np.array([True, True], dtype=np.bool_),
        "capture_step": np.array([9, -1], dtype=np.int32),
        "_capture_step": np.array([True, True], dtype=np.bool_),
        "final_info": np.array([{"captured": False, "capture_step": 99}, None], dtype=object),
        "_final_info": np.array([True, False], dtype=np.bool_),
    }

    assert bool(_vec_info_pick_terminal_aware(infos, "captured", 0)) is True
    assert _vec_info_pick_terminal_aware(infos, "capture_step", 0) == 9


def test_vec_info_pick_terminal_aware_handles_dict_of_batched_final_info():
    infos = {
        "captured": np.array([False, False], dtype=np.bool_),
        "_captured": np.array([False, False], dtype=np.bool_),
        "final_info": {
            "captured": np.array([True, False], dtype=np.bool_),
            "capture_step": np.array([23, -1], dtype=np.int32),
        },
        "_final_info": np.array([True, False], dtype=np.bool_),
    }

    assert bool(_vec_info_pick_terminal_aware(infos, "captured", 0)) is True
    assert _vec_info_bool(infos, "captured", 0) is True
    assert _vec_info_pick_terminal_aware(infos, "capture_step", 0) == 23


def test_mean_metric_dicts_averages_completed_episode_metrics():
    metrics_list = [
        {"capture_rate": 1.0, "episode_return": 12.0, "mean_goal_distance": 3.0},
        {"capture_rate": 0.0, "episode_return": 6.0, "mean_goal_distance": 5.0},
        {"capture_rate": 1.0, "episode_return": 9.0, "mean_goal_distance": 4.0},
    ]

    mean_metrics = VecEnvTrainer._mean_metric_dicts(metrics_list)

    assert mean_metrics["capture_rate"] == 2.0 / 3.0
    assert mean_metrics["episode_return"] == 9.0
    assert mean_metrics["mean_goal_distance"] == 4.0


def test_vec_info_bool_any_supports_capture_alias():
    infos = {
        "final_info": np.array(
            [
                {"capture": True, "is_success": True, "terminated": True},
                None,
            ],
            dtype=object,
        ),
        "_final_info": np.array([True, False], dtype=np.bool_),
    }

    assert _vec_info_bool_any(infos, ("capture", "captured"), 0) is True
    assert _vec_info_bool(infos, "is_success", 0) is True
    assert _vec_info_bool(infos, "terminated", 0) is True


def test_episode_debug_counts_uses_completed_episode_denominator():
    metrics_list = [
        {"capture_rate": 1.0, "timeout_rate": 0.0, "collision_rate": 0.0},
        {"capture_rate": 0.0, "timeout_rate": 1.0, "collision_rate": 0.0},
        {"capture_rate": 0.0, "timeout_rate": 0.0, "collision_rate": 1.0},
        {"capture_rate": 1.0, "timeout_rate": 0.0, "collision_rate": 0.0},
    ]

    debug_counts = VecEnvTrainer._episode_debug_counts(metrics_list)

    assert debug_counts["completed_episodes"] == 4.0
    assert debug_counts["captured_episodes"] == 2.0
    assert debug_counts["timeout_episodes"] == 1.0
    assert debug_counts["collision_episodes"] == 1.0
    assert debug_counts["train_capture_rate"] == 0.5


def test_build_tb_episode_metrics_requires_terminated_success_for_capture():
    trainer = object.__new__(VecEnvTrainer)
    trainer._ep_any_capture = np.array([True], dtype=np.bool_)
    trainer._ep_any_collision = np.array([False], dtype=np.bool_)
    trainer._ep_any_pursuer_oob = np.array([False], dtype=np.bool_)
    trainer._ep_any_timeout = np.array([False], dtype=np.bool_)
    trainer._ep_any_obstacle_term = np.array([False], dtype=np.bool_)
    trainer._ep_first_capture_step = np.array([-1], dtype=np.int32)
    trainer._ep_mgd_sum = np.array([5.0], dtype=np.float64)
    trainer._ep_prog_sum = np.array([0.0], dtype=np.float64)
    trainer._ep_time_penalty_sum = np.array([0.0], dtype=np.float64)
    trainer._ep_reach_bonus_sum = np.array([0.0], dtype=np.float64)
    trainer._ep_collision_penalty_sum = np.array([0.0], dtype=np.float64)
    trainer._ep_ps_pairs = [[]]

    timeout_infos = {
        "final_info": np.array(
            [
                {
                    "capture": True,
                    "is_success": False,
                    "terminated": False,
                    "truncated": True,
                    "timeout": True,
                    "termination_reason": "timeout",
                    "mean_goal_distance": 1.0,
                }
            ],
            dtype=object,
        ),
        "_final_info": np.array([True], dtype=np.bool_),
    }
    train_metrics, _ = trainer._build_tb_episode_metrics(0, 7.0, 5, timeout_infos)

    assert train_metrics["capture_rate"] == 0.0
    assert train_metrics["timeout_rate"] == 1.0
    assert train_metrics["truncated_rate"] == 1.0
    assert train_metrics["terminated_rate"] == 0.0
