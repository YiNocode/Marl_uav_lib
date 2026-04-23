"""Sanity check for ex1 role assignment + residual control before training."""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_uav.envs.tasks.pursuit_evasion_3v1_task_ex1 import PursuitEvasion3v1Task
from marl_uav.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check ex1 role assignment and residual control.")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"),
    )
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--output-tag", type=str, default="")
    return p.parse_args()


def _variant_task_cfg(base_task_cfg: dict[str, Any], name: str) -> dict[str, Any]:
    cfg = deepcopy(base_task_cfg)
    cfg["name"] = "pursuit_evasion_3v1_ex1"
    if name == "residual_off":
        cfg["residual_control_gain"] = 0.0
        cfg["assignment_inertia_margin"] = 0.0
    elif name == "residual_on_no_inertia":
        cfg["residual_control_gain"] = 0.5
        cfg["assignment_inertia_margin"] = 0.0
    elif name == "residual_on_with_inertia":
        cfg["residual_control_gain"] = 0.5
        cfg["assignment_inertia_margin"] = 0.05
    else:
        raise ValueError(f"Unknown variant {name!r}")
    cfg["role_assignment_mode"] = "nearest"
    cfg.setdefault("manifold_target_rho_min", None)
    return cfg


def _assignment_switches(assign_hist: list[np.ndarray]) -> int:
    if len(assign_hist) <= 1:
        return 0
    switches = 0
    for prev, cur in zip(assign_hist[:-1], assign_hist[1:]):
        if not np.array_equal(prev, cur):
            switches += 1
    return switches


def _run_one_episode(config_path: Path, task_cfg: dict[str, Any], seed: int, steps: int) -> dict[str, Any]:
    exp_cfg = load_config(config_path)
    del exp_cfg
    task_cfg = deepcopy(task_cfg)
    task_cfg.pop("name", None)
    task = PursuitEvasion3v1Task(**task_cfg)
    rng = np.random.default_rng(seed)
    start_pos, _, task_state = task.sample_initial_conditions(4, rng)

    states = np.zeros((4, 4, 3), dtype=np.float32)
    states[:, 3, :] = start_pos
    backend_state = SimpleNamespace(
        states=states.copy(),
        contact_array=np.zeros((4, 4), dtype=np.int32),
    )
    prev_backend_state = SimpleNamespace(
        states=states.copy(),
        contact_array=np.zeros((4, 4), dtype=np.int32),
    )
    zero_actions = np.zeros((3, 4), dtype=np.float32)

    role_dist_hist: list[np.ndarray] = []
    min_evader_dist_hist: list[float] = []
    struct_hist: list[dict[str, float]] = []
    reward_hist: list[np.ndarray] = []
    assignment_hist: list[np.ndarray] = []
    termination_reason = "running"

    for _ in range(int(steps)):
        lin_pos = np.asarray(backend_state.states[:, 3, :], dtype=np.float32)
        pursuer_pos = lin_pos[task_state.pursuer_ids]
        evader_pos = lin_pos[task_state.evader_id]
        _, assignment, assigned_targets = task._assigned_targets_from_state(
            pursuer_pos,
            evader_pos,
            task_state=task_state,
        )
        role_dists = np.linalg.norm(assigned_targets - pursuer_pos, axis=1).astype(np.float32)
        role_dist_hist.append(role_dists)
        assignment_hist.append(assignment.astype(np.int64).copy())
        min_evader_dist_hist.append(float(np.min(np.linalg.norm(pursuer_pos - evader_pos[None, :], axis=1))))
        struct_hist.append(task._structure_aware_features_19d(
            lin_pos,
            np.asarray(backend_state.states[:, 2, :], dtype=np.float32),
            task_state.pursuer_ids,
            task_state.evader_id,
            task_state,
        )[:, -3:].mean(axis=0).astype(np.float32))

        setpoints = task.action_to_setpoint(
            zero_actions,
            backend_state,
            task_state,
            action_space_type="continuous",
            action_dim=4,
        )
        next_states = np.asarray(backend_state.states, dtype=np.float32).copy()
        next_states[:, 2, :] = 0.0
        next_states[:, 2, 0] = setpoints[:, 0]
        next_states[:, 2, 1] = setpoints[:, 1]
        next_states[:, 2, 2] = setpoints[:, 3]
        next_states[:, 3, 0] += setpoints[:, 0]
        next_states[:, 3, 1] += setpoints[:, 1]
        next_states[:, 3, 2] += setpoints[:, 3]
        next_backend_state = SimpleNamespace(
            states=next_states,
            contact_array=np.zeros((4, 4), dtype=np.int32),
        )

        rewards = task.compute_rewards(prev_backend_state, next_backend_state, task_state)
        terminated, truncated = task.compute_terminated_truncated(
            next_backend_state,
            task_state,
            len(role_dist_hist),
        )
        reward_hist.append(np.asarray(rewards, dtype=np.float32))

        backend_state = next_backend_state
        prev_backend_state = next_backend_state

        if terminated or truncated:
            if bool(getattr(task_state, "captured", False)):
                termination_reason = "captured"
            elif bool(task._get_oob_mask(backend_state.states[:, 3, :][task_state.evader_id][None, :])[0]):
                termination_reason = "evader_oob"
            elif int(np.sum(task._get_oob_mask(backend_state.states[:, 3, :][task_state.pursuer_ids]))) >= task.max_pursuers_oob_before_terminate:
                termination_reason = "too_many_pursuers_oob"
            elif truncated:
                termination_reason = "timeout"
            else:
                termination_reason = "other_terminated"
            break

    role_dist_arr = np.stack(role_dist_hist, axis=0)
    reward_arr = np.stack(reward_hist, axis=0) if reward_hist else np.zeros((0, 3), dtype=np.float32)
    struct_delta_arr = np.stack(struct_hist, axis=0) if struct_hist else np.zeros((0, 3), dtype=np.float32)
    first_k = min(20, role_dist_arr.shape[0])
    last_k = min(20, role_dist_arr.shape[0])
    start_mean = float(np.mean(role_dist_arr[:first_k]))
    end_mean = float(np.mean(role_dist_arr[-last_k:]))

    return {
        "seed": int(seed),
        "steps_executed": int(role_dist_arr.shape[0]),
        "termination_reason": termination_reason,
        "assignment_switches": int(_assignment_switches(assignment_hist)),
        "switch_rate": float(_assignment_switches(assignment_hist) / max(role_dist_arr.shape[0] - 1, 1)),
        "role_distance_start20": start_mean,
        "role_distance_end20": end_mean,
        "role_distance_improvement": float(start_mean - end_mean),
        "final_role_distance_per_agent": role_dist_arr[-1].tolist(),
        "final_assignment": assignment_hist[-1].tolist(),
        "mean_episode_reward": float(np.mean(reward_arr)) if reward_arr.size else 0.0,
        "min_evader_distance_end20": float(np.mean(min_evader_dist_hist[-last_k:])),
        "mean_structure_delta_end20": struct_delta_arr[-last_k:].mean(axis=0).tolist() if struct_delta_arr.size else [0.0, 0.0, 0.0],
    }


def _aggregate(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str) -> float:
        return float(np.mean([float(r[key]) for r in records]))

    capture_rate = float(np.mean([1.0 if r["termination_reason"] == "captured" else 0.0 for r in records]))
    return {
        "variant": name,
        "num_runs": len(records),
        "capture_rate": capture_rate,
        "mean_role_distance_start20": mean_of("role_distance_start20"),
        "mean_role_distance_end20": mean_of("role_distance_end20"),
        "mean_role_distance_improvement": mean_of("role_distance_improvement"),
        "mean_assignment_switches": mean_of("assignment_switches"),
        "mean_switch_rate": mean_of("switch_rate"),
        "mean_episode_reward": mean_of("mean_episode_reward"),
        "mean_min_evader_distance_end20": mean_of("min_evader_distance_end20"),
        "records": records,
    }


def _decision(aggregates: dict[str, dict[str, Any]]) -> str:
    off = aggregates["residual_off"]
    on = aggregates["residual_on_with_inertia"]
    no_inertia = aggregates["residual_on_no_inertia"]

    better_tracking = on["mean_role_distance_end20"] < 0.85 * off["mean_role_distance_end20"]
    lower_switch = on["mean_switch_rate"] <= no_inertia["mean_switch_rate"] + 1e-6
    non_collapse = on["mean_episode_reward"] >= off["mean_episode_reward"] - 1.0

    if better_tracking and lower_switch and non_collapse:
        return "recommended_to_train"
    return "hold_before_training"


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    variants = [
        "residual_off",
        "residual_on_no_inertia",
        "residual_on_with_inertia",
    ]

    results: dict[str, Any] = {
        "config": str(config_path.relative_to(ROOT)),
        "steps": int(args.steps),
        "seeds": [int(s) for s in args.seeds],
        "variants": {},
    }

    exp_cfg = load_config(config_path)
    base_task_cfg = dict(exp_cfg.get("task", {}) or {})

    for variant in variants:
        records: list[dict[str, Any]] = []
        task_cfg = _variant_task_cfg(base_task_cfg, variant)
        for seed in args.seeds:
            record = _run_one_episode(config_path, task_cfg, int(seed), int(args.steps))
            records.append(record)
        results["variants"][variant] = _aggregate(variant, records)

    results["decision"] = _decision(results["variants"])

    exp_name = config_path.stem
    tag = f"_{args.output_tag}" if args.output_tag else ""
    out_dir = ROOT / "results" / exp_name / "role_assignment_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sanity_steps{int(args.steps)}{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== EX1 Role Assignment Sanity Check ===")
    for variant in variants:
        agg = results["variants"][variant]
        print(
            f"{variant:>24} | role_end20={agg['mean_role_distance_end20']:.4f} | "
            f"improve={agg['mean_role_distance_improvement']:.4f} | "
            f"switch_rate={agg['mean_switch_rate']:.4f} | "
            f"reward={agg['mean_episode_reward']:.4f} | capture_rate={agg['capture_rate']:.2f}"
        )
    print(f"decision: {results['decision']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
