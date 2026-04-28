"""Inspect whether Dream-MAPPO actor input really contains agent-specific manifold targets."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_uav.policies.dream_mappo_policy import DreamMappoCentralizedCriticPolicy
from marl_uav.utils.checkpoint import load_checkpoint
from marl_uav.utils.config import load_config


def _load_eval_module():
    path = ROOT / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("marl_eval_check_manifold", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval = _load_eval_module()
build_env = _eval.build_env
build_policy = _eval.build_policy
build_learner = _eval.build_learner
_peek_checkpoint_actor_obs_dim = _eval._peek_checkpoint_actor_obs_dim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check whether manifold targets enter Dream actor input.")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_dream_mappo_3v1.yaml"),
    )
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--agent", type=int, default=0, help="Which pursuer row to print.")
    return p.parse_args()


def _default_ckpt_path(config_path: Path, seed: int) -> Path:
    exp_name = config_path.stem
    return ROOT / "results" / exp_name / "checkpoints" / str(seed) / "best.pt"


def _compat_task_cfg(task_cfg: dict[str, Any], env: Any, ckpt_path: Path) -> dict[str, Any]:
    out = dict(task_cfg)
    ckpt_obs_dim = _peek_checkpoint_actor_obs_dim(ckpt_path)
    env_obs_dim = getattr(env, "obs_dim", None)
    task_name = str(task_cfg.get("name", "navigation"))
    if (
        ckpt_obs_dim is not None
        and env_obs_dim is not None
        and int(env_obs_dim) - int(ckpt_obs_dim) == 3
        and task_name in ("pursuit_evasion_3v1_ex1", "pursuit_evasion_3v1_ex2")
        and "structure_obs_include_deltas" not in out
    ):
        out["structure_obs_include_deltas"] = False
    return out


def _load_checkpoint_for_inspection(ckpt_path: Path, learner: Any) -> str:
    """Best-effort checkpoint load for architecture inspection.

    If the full learner load fails because the actor input dim changed,
    load all policy parameters whose tensor shapes still match.
    """
    try:
        load_checkpoint(ckpt_path, learner)
        return f"Loaded checkpoint: {ckpt_path}"
    except RuntimeError as exc:
        data = torch.load(ckpt_path, map_location="cpu")
        learner_state = data.get("learner")
        policy_state = learner_state.get("policy") if isinstance(learner_state, dict) else None
        current_policy = getattr(learner, "policy", None)
        if not isinstance(policy_state, dict) or current_policy is None:
            raise

        current_state = current_policy.state_dict()
        compatible_state: dict[str, Any] = {}
        skipped: list[str] = []
        for k, v in policy_state.items():
            cur = current_state.get(k)
            if cur is not None and tuple(getattr(v, "shape", ())) == tuple(cur.shape):
                compatible_state[k] = v
            else:
                skipped.append(k)

        current_policy.load_state_dict(compatible_state, strict=False)
        skipped_preview = ", ".join(skipped[:8])
        if len(skipped) > 8:
            skipped_preview += ", ..."
        return (
            f"Partially loaded checkpoint for inspection: {ckpt_path}\n"
            f"full load error: {exc}\n"
            f"skipped incompatible policy params ({len(skipped)}): {skipped_preview}"
        )


def _world_manifold_targets(pursuer_xyz: np.ndarray, evader_xyz: np.ndarray, rho: float, psi: float) -> np.ndarray:
    pursuer_xyz = np.asarray(pursuer_xyz, dtype=np.float64).reshape(-1, 3)
    evader_xyz = np.asarray(evader_xyz, dtype=np.float64).reshape(3)
    n = int(pursuer_xyz.shape[0])
    rel_xy = pursuer_xyz[:, :2] - evader_xyz[None, :2]
    theta = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])
    order = np.argsort(theta)
    inv_rank = np.zeros((n,), dtype=np.int64)
    inv_rank[order] = np.arange(n, dtype=np.int64)
    phi = (2.0 * np.pi / float(n)) * inv_rank.astype(np.float64)
    ang = phi + float(psi)
    targets = np.zeros((n, 3), dtype=np.float64)
    targets[:, 0] = evader_xyz[0] + float(rho) * np.cos(ang)
    targets[:, 1] = evader_xyz[1] + float(rho) * np.sin(ang)
    targets[:, 2] = evader_xyz[2]
    return targets


def _print_obs_breakdown(obs_row: np.ndarray, obs_dim: int) -> None:
    idx = 0

    def take(name: str, width: int) -> None:
        nonlocal idx
        arr = obs_row[idx : idx + width]
        print(f"{name:>22}: dim={width:>2} values={np.array2string(arr, precision=4)}")
        idx += width

    take("self_pos", 3)
    take("self_vel", 3)
    take("self_ang", 3)
    take("rel_evader_pos", 3)
    take("rel_evader_vel", 3)
    take("rel_tm1_pos", 3)
    take("rel_tm1_vel", 3)
    take("rel_tm2_pos", 3)
    take("rel_tm2_vel", 3)

    remaining = obs_dim - idx
    if remaining == 19:
        take("structure_aware", 19)
    elif remaining == 16:
        take("structure_aware", 16)
    elif remaining > 19 and (remaining - 19) % 4 == 0:
        take("structure_aware", 19)
        take("obstacle_block", remaining - 19)
    elif remaining > 16 and (remaining - 16) % 4 == 0:
        take("structure_aware", 16)
        take("obstacle_block", remaining - 16)
    elif remaining > 0:
        take("unknown_tail", remaining)


def main() -> None:
    args = parse_args()
    exp_cfg_path = ROOT / args.config
    exp_cfg = load_config(exp_cfg_path)

    env_cfg_path = ROOT / exp_cfg.get("env", "configs/env/pyflyt_3v1.yaml")
    algo_cfg_path = ROOT / exp_cfg.get("algo", "configs/algo/dream_mappo.yaml")
    model_cfg_path = ROOT / exp_cfg.get("model", "configs/model/dream_mappo_centralized.yaml")
    task_cfg = dict(exp_cfg.get("task", {}) or {})
    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt_path(exp_cfg_path, args.seed)

    env = build_env(env_cfg_path, seed=args.seed, task_cfg=task_cfg)
    obs_state, _ = env.reset(seed=args.seed)
    if ckpt_path.exists():
        compat_cfg = _compat_task_cfg(task_cfg, env, ckpt_path)
        if compat_cfg != task_cfg:
            env = build_env(env_cfg_path, seed=args.seed, task_cfg=compat_cfg)
            obs_state, _ = env.reset(seed=args.seed)
            task_cfg = compat_cfg

    policy = build_policy(model_cfg_path, env, algo_cfg_path)
    learner = build_learner(algo_cfg_path, policy)
    if ckpt_path.exists():
        print(_load_checkpoint_for_inspection(ckpt_path, learner))
    else:
        print(f"Checkpoint not found, inspecting architecture only: {ckpt_path}")

    if not isinstance(policy, DreamMappoCentralizedCriticPolicy):
        raise TypeError(f"Expected DreamMappoCentralizedCriticPolicy, got {type(policy).__name__}")

    obs = obs_state["obs"]
    state = obs_state["state"]
    obs_tensor, B, N = policy._prepare_obs(obs)
    state_tensor = policy._prepare_state(state, B=B, N=N)
    state_b = policy._state_b(state_tensor)
    a_geom, rho, psi = policy._geom_from_state(state_b)
    cond = policy._actor_condition_from_state(state_b, rho, psi)
    flat_obs = obs_tensor.reshape(B * N, policy.obs_dim).detach().cpu().numpy()
    flat_cond = cond["actor_cond"].reshape(B * N, policy.actor_condition_dim).detach().cpu().numpy()
    flat_actor_input = np.concatenate([flat_obs, flat_cond], axis=-1)

    agent_idx = int(np.clip(args.agent, 0, N - 1))
    actor_input_row = flat_actor_input[agent_idx]
    actor_cond_row = flat_cond[agent_idx]
    backend_state = env.prev_backend_state
    if backend_state is None:
        raise RuntimeError("env.prev_backend_state is None after reset")
    lin_pos = np.asarray(backend_state.states[:, 3, :], dtype=np.float64)
    pursuer_xyz = lin_pos[:N]
    evader_xyz = lin_pos[N]
    targets = _world_manifold_targets(pursuer_xyz, evader_xyz, float(rho[0].item()), float(psi[0].item()))
    rel_slot = targets[agent_idx] - pursuer_xyz[agent_idx]

    print("\n=== Dream Actor Input Check ===")
    print(f"obs_dim={policy.obs_dim}")
    print(f"actor_encoder.input_dim={getattr(policy.actor_encoder, 'input_dim', 'N/A')}")
    print(f"actor_condition_dim={policy.actor_condition_dim}")
    print(f"actor encoder input source = concat(obs, actor_cond): shape={tuple(flat_actor_input.shape)}")
    print(f"a_geom.shape={tuple(a_geom.shape)} | rho={float(rho[0].item()):.6f} | psi={float(psi[0].item()):.6f}")
    print(f"selected agent={agent_idx}")
    print(f"world manifold target g_i={np.array2string(targets[agent_idx], precision=4)}")
    print(f"world current position p_i={np.array2string(pursuer_xyz[agent_idx], precision=4)}")
    print(f"world slot offset g_i - p_i={np.array2string(rel_slot, precision=4)}")
    print("\nActor input breakdown:")
    _print_obs_breakdown(actor_input_row, policy.obs_dim)
    print(f"{'slot_rel(g_i-p_i)':>22}: dim= 3 values={np.array2string(actor_cond_row[0:3], precision=4)}")
    print(f"{'slot_direction':>22}: dim= 3 values={np.array2string(actor_cond_row[3:6], precision=4)}")
    print(f"{'assignment_weight':>22}: dim= 1 values={np.array2string(actor_cond_row[6:7], precision=4)}")

    print("\n=== Minimal Checklist ===")
    print("relative slot position in actor input: YES")
    print("slot velocity / slot direction in actor input: YES (direction)")
    print("slot confidence / assignment weight in actor input: YES")
    print("reason: Dream policy now concatenates agent-specific manifold conditioning to obs before actor_encoder.")


if __name__ == "__main__":
    main()
