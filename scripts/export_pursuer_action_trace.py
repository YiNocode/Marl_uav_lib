"""Export one pursuer's full action trace from an ex1 heuristic rollout (e.g. oracle_slot).

Example:

    python scripts/export_pursuer_action_trace.py \\
        --config configs/experiment/e1_1_open_space_pyflyt_oracle_slot.yaml \\
        --seed 101 \\
        --pursuer 0 \\
        --output results/action_traces/oracle_slot_seed101_pursuer0.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_uav.control.geometric_pursuit_baselines import make_oracle_slot_get_actions_fn
from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.action_trace import collect_pursuer_action_trace
from marl_uav.utils.config import load_config
from marl_uav.utils.e1_1_suite import merge_rl_task_speed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export single-pursuer action trace for PX4 replay.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiment/e1_1_open_space_pyflyt_oracle_slot.yaml",
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--pursuer", type=int, default=0, help="Pursuer row index 0..2.")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path. Default: results/action_traces/<config_stem>_seed<seed>_pursuer<i>.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = merge_rl_task_speed(load_config(cfg_path))
    env_cfg = load_config(ROOT / str(cfg["env"]))

    if "oracle_slot" not in cfg:
        raise ValueError(
            f"Config {cfg_path} has no oracle_slot section. "
            "Pass e1_1_open_space_pyflyt_oracle_slot.yaml or extend this script."
        )

    env = build_env_from_config(ROOT / str(cfg["env"]), seed=args.seed, task_cfg=dict(cfg.get("task", {}) or {}))
    if getattr(env, "_action_space_type", "") != "continuous":
        raise ValueError("Action trace export requires a continuous-action env.")

    get_actions = make_oracle_slot_get_actions_fn(env, **dict(cfg.get("oracle_slot", {}) or {}))

    trace = collect_pursuer_action_trace(
        env,
        get_actions_fn=get_actions,
        seed=int(args.seed),
        pursuer_index=int(args.pursuer),
        env_cfg=env_cfg,
        meta={
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(cfg_path.relative_to(ROOT)).replace("\\", "/"),
            "method": "oracle_slot",
            "suite": str((cfg.get("benchmark") or {}).get("suite", "E1")),
            "scenario": str((cfg.get("benchmark") or {}).get("scenario", "e1_1_open_space")),
            "controller": dict(cfg.get("oracle_slot", {}) or {}),
        },
    )

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        out_path = (
            ROOT
            / "results"
            / "action_traces"
            / f"{cfg_path.stem}_seed{args.seed}_pursuer{args.pursuer}.json"
        )

    saved = trace.save_json(out_path)
    ep = trace.episode
    print(f"[export] saved -> {saved.relative_to(ROOT)}")
    print(
        f"[export] pursuer={ep['pursuer_index']} len={ep['episode_len']} "
        f"capture={ep['capture']} return={ep['episode_return']:.3f}"
    )
    print(f"[export] control_hz={trace.control['control_hz']:.1f} dt={trace.control['dt_s']:.6f}s")
    print("[export] Use setpoint fields (vx, vy, yaw_rate, vz) for PX4 velocity offboard replay.")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
