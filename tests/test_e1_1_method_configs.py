from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
E11_GEN = ROOT / "configs" / "generated" / "e1_1_open_space"

RESIDUAL_CONTROL_KEYS = {
    "residual_control_gain",
    "residual_control_start_epoch",
    "residual_control_end_epoch",
    "residual_control_start_frac",
    "residual_control_end_frac",
    "residual_control_gain_final",
    "residual_control_gain_decay_epochs",
}

STRUCTURE_METHOD_KEYS = {
    "structure_reward_scale",
    "structure_improve_scale",
    "structure_hold_reward_scale",
    "role_progress_reward_scale",
    "radial_compress_reward_scale",
    "radial_overshoot_penalty_scale",
    "contraction_phase_structure_scale",
    "contraction_phase_compress_scale",
    "encirclement_capture_enabled",
    "structure_cov_weight",
    "structure_col_weight",
    "structure_ang_weight",
    "structure_cov_threshold",
    "structure_col_threshold",
    "structure_ang_threshold",
    "structure_hold_steps_cap",
    "manifold_contraction_rate",
    "manifold_structure_gate_scale",
    "role_features_enabled",
    "role_assignment_mode",
    "manifold_target_phase",
    "manifold_target_radius_scale",
    "assignment_inertia_margin",
}


def _load(rel_path: str | Path) -> dict:
    path = ROOT / rel_path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), path
    return cfg


def _method_config_paths(method: str) -> list[Path]:
    paths = [
        ROOT / "configs" / "experiment" / f"e1_1_open_space_pyflyt_{method}.yaml",
    ]
    paths.extend(sorted(E11_GEN.glob(f"e1_1_open_space_pyflyt_{method}_seed*.yaml")))
    smoke = E11_GEN / "smoke_training" / f"e1_1_open_space_pyflyt_{method}_smoke.yaml"
    if smoke.exists():
        paths.append(smoke)
    return paths


def _task(cfg: dict) -> dict:
    task = cfg.get("task")
    assert isinstance(task, dict)
    return task


def _model(cfg: dict) -> dict:
    return _load(cfg["model"])


def _assert_no_residual_controller(task: dict) -> None:
    assert not (set(task) & RESIDUAL_CONTROL_KEYS)


def test_e1_1_mappo_is_pure_mappo() -> None:
    for path in _method_config_paths("mappo"):
        cfg = _load(path.relative_to(ROOT))
        task = _task(cfg)
        assert cfg["benchmark"]["method"] == "mappo"
        assert cfg["algo"] == "configs/algo/mappo.yaml"
        assert cfg["model"] == "configs/model/centralized_critic.yaml"
        assert _model(cfg).get("type") == "centralized_critic"
        assert task["name"] == "pursuit_evasion_3v1"
        assert not (set(task) & STRUCTURE_METHOD_KEYS)
        _assert_no_residual_controller(task)


def test_e1_1_reward_shaped_mappo_has_structure_inputs_and_rewards_only() -> None:
    for path in _method_config_paths("reward_shaped_mappo"):
        cfg = _load(path.relative_to(ROOT))
        task = _task(cfg)
        assert cfg["benchmark"]["method"] == "reward_shaped_mappo"
        assert cfg["algo"] == "configs/algo/mappo.yaml"
        assert cfg["model"] == "configs/model/centralized_critic.yaml"
        assert _model(cfg).get("type") == "centralized_critic"
        assert task["name"] == "pursuit_evasion_3v1_ex1"
        assert bool(task["role_features_enabled"]) is False
        assert float(task["structure_reward_scale"]) > 0.0
        assert float(task["role_progress_reward_scale"]) == 0.0
        assert float(task["radial_compress_reward_scale"]) > 0.0
        assert bool(task["encirclement_capture_enabled"]) is False
        _assert_no_residual_controller(task)


def test_e1_1_dream_mappo_full_has_manifold_policy_and_role_assignment() -> None:
    for path in _method_config_paths("dream_mappo_full"):
        cfg = _load(path.relative_to(ROOT))
        task = _task(cfg)
        assert cfg["benchmark"]["method"] == "dream_mappo_full"
        assert cfg["algo"] == "configs/algo/dream_mappo.yaml"
        assert cfg["model"] == "configs/model/dream_mappo_centralized.yaml"
        model = _model(cfg)
        dream_cfg = model.get("dream", {})
        assert model.get("type") == "dream_mappo_centralized_critic"
        assert float(dream_cfg["a_max_geom"]) > 0.0
        assert float(dream_cfg["a_max_residual"]) > 0.0
        assert task["name"] == "pursuit_evasion_3v1_ex1"
        assert bool(task["role_features_enabled"]) is True
        assert task["role_assignment_mode"] == "nearest"
        assert float(task["manifold_target_radius_scale"]) > 0.0
        assert float(task["structure_reward_scale"]) > 0.0
        assert float(task["role_progress_reward_scale"]) > 0.0
        assert bool(task["encirclement_capture_enabled"]) is True
        _assert_no_residual_controller(task)
