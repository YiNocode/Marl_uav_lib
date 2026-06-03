"""Standalone training entry for slot execution MAPPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_uav.envs.factories import build_env_from_config
from marl_uav.utils.config import load_config
from slot_exec_mappo.adapter import SlotExecEnvWrapper
from slot_exec_mappo.config import SlotExecConfig
from slot_exec_mappo.trainer import SlotExecMAPPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train standalone slot execution MAPPO.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiment/e2_obstacles_pyflyt_slot_exec_mappo.yaml",
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    task_cfg = dict(cfg.get("task") or {})
    env = build_env_from_config(ROOT / str(cfg["env"]), seed=args.seed, task_cfg=task_cfg)
    slot_cfg = SlotExecConfig.from_dict(cfg.get("slot_exec_mappo"))
    wrapped = SlotExecEnvWrapper(env, cfg=slot_cfg)
    wrapped.reset(seed=args.seed)

    save_path = args.save
    if save_path is None:
        out_dir = Path(str(cfg.get("train_results_dir", "results/slot_exec_mappo")))
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / "policy.pt")

    trainer = SlotExecMAPPOTrainer(wrapped, cfg=slot_cfg, device=args.device)
    trainer.train(total_updates=int(args.updates), save_path=save_path)
    print(f"[slot_exec_mappo] saved policy -> {save_path}")


if __name__ == "__main__":
    main()
