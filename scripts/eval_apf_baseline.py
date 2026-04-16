"""追逃任务人工势场（APF）基线评估：与 MAPPO 使用相同场景配置与指标，不加载策略网络。

顶层 YAML（如 pursuit_evasion_mappo_3v1.yaml）中仅使用：
  - env：环境路径与动作空间
  - task：PursuitEvasion3v1Task 参数

忽略：algo、model、num_epochs、rollout_steps、log_interval、eval_episodes、seed（训练种子）
等训练相关项；评估用随机种子由本脚本 `--seed` / `--num-seeds` / `--episodes` 指定。

APF：各 pursuer 受 evader 吸引（沿 evader 方向），受其他 pursuer 排斥；evader 仍由任务内
启发式控制（与 RL 评估一致）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marl_uav.control.apf_pursuit import apf_acceleration_3d, apf_action_from_force
from marl_uav.envs.adapters.pyflyt_aviary_env import PURSUIT_EVASION_3V1_TASK_TYPES
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.config import load_config


def _load_eval_module():
    """复用 scripts/eval.py 中的 build_env 与评估指标/绘图函数。"""
    path = ROOT / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("marl_scripts_eval_apf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval = _load_eval_module()
build_env = _eval.build_env
run_multi_seed_eval = _eval.run_multi_seed_eval
compute_aggregate_metrics = _eval.compute_aggregate_metrics
plot_eval_statistics = _eval.plot_eval_statistics
_plot_pursuit_evasion_trajectories_from_data = _eval._plot_pursuit_evasion_trajectories_from_data


def make_apf_get_actions_fn(
    env: Any,
    *,
    k_att: float,
    k_rep: float,
    rho0: float,
):
    """构造 RolloutWorker.get_actions_fn：连续 setpoint，形状 [num_pursuers, action_dim]。"""
    if not isinstance(env.task, PURSUIT_EVASION_3V1_TASK_TYPES):
        raise TypeError(
            "APF 基线仅支持 PursuitEvasion3v1Task / pursuit_evasion_3v1_ex1 / pursuit_evasion_3v1_ex2"
        )
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError(
            "APF 基线需要连续动作空间（与 MAPPO 追逃 eval 一致）；请检查 env 配置 action_space。"
        )

    low = env.action_low_np
    high = env.action_high_np

    def fn(obs_list: Any, state: Any, avail_actions: Any) -> np.ndarray:
        del obs_list, state, avail_actions
        bs = env.prev_backend_state
        if bs is None:
            raise RuntimeError("环境未 reset，缺少 prev_backend_state。")
        lin_pos = np.asarray(bs.states[:, 3, :], dtype=np.float32)
        ts = env.task_state
        pids = ts.pursuer_ids
        eid = ts.evader_id
        ev = lin_pos[int(eid)]
        rows: list[np.ndarray] = []
        for i in pids:
            p_i = lin_pos[int(i)]
            others = [lin_pos[int(j)] for j in pids if int(j) != int(i)]
            f = apf_acceleration_3d(
                p_i, ev, others, k_att=k_att, k_rep=k_rep, rho0=rho0
            )
            rows.append(apf_action_from_force(f, low, high))
        return np.stack(rows, axis=0).astype(np.float32)

    return fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="追逃 APF 基线评估（与 eval.py 相同指标）")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_mappo_3v1.yaml"),
        help="顶层实验配置：仅读取 env 与 task，忽略 algo/model 等",
    )
    p.add_argument("--seed", type=int, default=1111, help="评估基准种子（与 eval.py 一致）")
    p.add_argument("--num-seeds", type=int, default=1, help="评估种子数量")
    p.add_argument("--episodes", type=int, default=20, help="每个种子 episode 数")
    p.add_argument("--k-att", type=float, default=2.0, help="APF 对 evader 吸引增益")
    p.add_argument("--k-rep", type=float, default=0.8, help="APF 机间排斥增益")
    p.add_argument(
        "--rho0",
        type=float,
        default=3.0,
        help="APF 排斥作用距离上界（米量级，需小于典型间距时可调小）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg_path = ROOT / args.config
    exp_cfg: Dict[str, Any] = load_config(exp_cfg_path)

    env_cfg_path = ROOT / exp_cfg.get("env", "configs/env/pyflyt_3v1.yaml")
    task_cfg: Dict[str, Any] = dict(exp_cfg.get("task", {}) or {})

    env = build_env(env_cfg_path, seed=args.seed, task_cfg=task_cfg)
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        try:
            env.reset(seed=args.seed)
        except TypeError:
            env.reset()

    get_actions = make_apf_get_actions_fn(
        env,
        k_att=float(args.k_att),
        k_rep=float(args.k_rep),
        rho0=float(args.rho0),
    )
    worker = RolloutWorker(env=env, policy=object(), get_actions_fn=get_actions)

    is_pe3v1 = isinstance(env.task, PURSUIT_EVASION_3V1_TASK_TYPES)
    num_seeds = max(4, int(args.num_seeds))
    episodes_per_seed = int(args.episodes)

    print(
        f"\n=== APF Baseline Eval: {num_seeds} seeds x {episodes_per_seed} episodes "
        f"(k_att={args.k_att}, k_rep={args.k_rep}, rho0={args.rho0}) ==="
    )
    all_records, trajectories = run_multi_seed_eval(
        worker=worker,
        num_seeds=num_seeds,
        episodes_per_seed=episodes_per_seed,
        base_seed=args.seed,
        record_trajectories=is_pe3v1,
    )
    metrics = compute_aggregate_metrics(all_records)

    print("\n=== APF Baseline Metrics (same keys as eval.py) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    exp_name = Path(args.config).stem
    results_dir = ROOT / "results" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"seed{args.seed}"
    records_path = results_dir / f"apf_baseline_eval_records_{seed_tag}.json"
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\nPer-episode records saved to {records_path}")

    stats_path = results_dir / f"apf_baseline_eval_statistics_{seed_tag}.png"
    plot_eval_statistics(metrics, stats_path)
    print(f"Eval statistics plot saved to {stats_path}")

    if trajectories is not None and is_pe3v1:
        traj_root = results_dir / "apf_baseline_trajectories_pe3v1"
        _plot_pursuit_evasion_trajectories_from_data(trajectories, traj_root)
        print(f"Pursuit-evasion trajectory plots saved under {traj_root}")


if __name__ == "__main__":
    main()
