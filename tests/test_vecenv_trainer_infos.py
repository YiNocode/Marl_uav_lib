from __future__ import annotations

import numpy as np

from marl_uav.runners.vecenv_trainer import (
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
