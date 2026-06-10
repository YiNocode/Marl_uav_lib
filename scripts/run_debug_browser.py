"""Run one pursuit episode with browser-based real-time debug visualization.

Example (SCE, no training required):

    python scripts/run_debug_browser.py \\
        --config configs/experiment/e2/sce.yaml \\
        --seed 101

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import os

# Must be set before PyFlyt/Numba are imported (see pyflyt_aviary_backend.py).
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import argparse
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_uav.control.fixed_ring_pursuit import make_fixed_ring_get_actions_fn
from marl_uav.control.geometric_pursuit_baselines import (
    make_hungarian_slot_get_actions_fn,
    make_oracle_slot_get_actions_fn,
    make_ot_slot_get_actions_fn,
    make_pure_pursuit_get_actions_fn,
)
from marl_uav.control.obstacle_aware_sce_baselines import (
    make_sce_cached_path_cbf_slot_get_actions_fn,
    make_sce_cached_path_slot_get_actions_fn,
    make_sce_los_slot_get_actions_fn,
    make_sce_reachability_cbf_slot_get_actions_fn,
    make_sce_reachability_slot_get_actions_fn,
    make_sce_turn_radius_slot_get_actions_fn,
)
from marl_uav.control.obstacle_apf_baselines import (
    make_fixed_ring_apf_get_actions_fn,
    make_pure_pursuit_apf_get_actions_fn,
)
from marl_uav.control.sce_controller import make_sce_get_actions_fn
from marl_uav.control.trajectory_planner import make_trajectory_planner_get_actions_fn
from slot_exec_mappo.adapter import make_slot_exec_get_actions_fn
from slot_exec_mappo.config import SlotExecConfig
from marl_uav.envs.factories import build_env_from_config
from marl_uav.runners.rollout_worker import RolloutWorker
from marl_uav.utils.config import load_config
from marl_uav.utils.debug_browser import configure_debug_browser, get_debug_browser_hub
from marl_uav.utils.debug_viz import resolve_viz_profile
from marl_uav.utils.e1_1_suite import merge_rl_task_speed, resolve_speed_bounds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pursuit episode with browser debug UI.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiment/e2/sce.yaml",
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to run sequentially.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open-browser", action="store_true")
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Playback speed multiplier (1.0≈real-time at control_hz; 0.25=4× slower).",
    )
    p.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep HTTP server running after episodes finish.",
    )
    p.add_argument("--sleep", type=float, default=0.0, help="Pause seconds between episodes.")
    return p.parse_args()


def _build_get_actions_fn(env: Any, cfg: dict[str, Any]):
    if "fixed_ring" in cfg:
        return make_fixed_ring_get_actions_fn(env, **dict(cfg.get("fixed_ring", {}) or {}))
    if "fixed_ring_apf" in cfg:
        return make_fixed_ring_apf_get_actions_fn(
            env, **dict(cfg.get("fixed_ring_apf", {}) or {})
        )
    if "pure_pursuit" in cfg:
        return make_pure_pursuit_get_actions_fn(env, **dict(cfg.get("pure_pursuit", {}) or {}))
    if "pure_pursuit_apf" in cfg:
        return make_pure_pursuit_apf_get_actions_fn(
            env, **dict(cfg.get("pure_pursuit_apf", {}) or {})
        )
    if "oracle_slot" in cfg:
        return make_oracle_slot_get_actions_fn(env, **dict(cfg.get("oracle_slot", {}) or {}))
    if "hungarian_slot" in cfg:
        return make_hungarian_slot_get_actions_fn(env, **dict(cfg.get("hungarian_slot", {}) or {}))
    if "ot_slot" in cfg:
        return make_ot_slot_get_actions_fn(env, **dict(cfg.get("ot_slot", {}) or {}))
    if "trajectory_planner" in cfg:
        return make_trajectory_planner_get_actions_fn(
            env, **dict(cfg.get("trajectory_planner", {}) or {})
        )
    if "slot_exec_mappo" in cfg:
        block = dict(cfg.get("slot_exec_mappo", {}) or {})
        checkpoint = block.pop("checkpoint", None)
        slot_cfg = SlotExecConfig.from_dict(block)
        ckpt_path = None if checkpoint is None else str(ROOT / checkpoint)
        return make_slot_exec_get_actions_fn(
            env,
            checkpoint=ckpt_path,
            cfg=slot_cfg,
            deterministic=bool(block.get("deterministic", True)),
            device=str(block.get("device", "cpu")),
        )
    if "sce" in cfg:
        return make_sce_get_actions_fn(env, **dict(cfg.get("sce", {}) or {}))
    if "sce_los_slot" in cfg:
        block = dict(cfg.get("sce_los_slot", {}) or {})
        reach = dict(block.pop("reachability", {}) or {})
        rates = dict(block.pop("runtime_rates", {}) or {})
        return make_sce_los_slot_get_actions_fn(env, reachability=reach, runtime_rates=rates, **block)
    if "sce_turn_radius_slot" in cfg:
        block = dict(cfg.get("sce_turn_radius_slot", {}) or {})
        reach = dict(block.pop("reachability", {}) or {})
        rates = dict(block.pop("runtime_rates", {}) or {})
        planner = dict(block.pop("turn_radius_planner", {}) or {})
        return make_sce_turn_radius_slot_get_actions_fn(
            env,
            reachability=reach,
            runtime_rates=rates,
            turn_radius_planner=planner,
            **block,
        )
    if "sce_cached_path_slot" in cfg or "sce_path_slot" in cfg:
        key = "sce_cached_path_slot" if "sce_cached_path_slot" in cfg else "sce_path_slot"
        block = dict(cfg.get(key, {}) or {})
        reach = dict(block.pop("reachability", {}) or {})
        rates = dict(block.pop("runtime_rates", {}) or {})
        return make_sce_cached_path_slot_get_actions_fn(env, reachability=reach, runtime_rates=rates, **block)
    if "sce_cached_path_cbf_slot" in cfg or "sce_path_cbf_slot" in cfg:
        key = "sce_cached_path_cbf_slot" if "sce_cached_path_cbf_slot" in cfg else "sce_path_cbf_slot"
        block = dict(cfg.get(key, {}) or {})
        reach = dict(block.pop("reachability", {}) or {})
        cbf = dict(block.pop("cbf", {}) or {})
        rates = dict(block.pop("runtime_rates", {}) or {})
        return make_sce_cached_path_cbf_slot_get_actions_fn(
            env, reachability=reach, cbf=cbf, runtime_rates=rates, **block
        )
    if "sce_reachability_slot" in cfg:
        block = dict(cfg.get("sce_reachability_slot", {}) or {})
        reach = dict(block.pop("reachability", {}) or {})
        rates = dict(block.pop("runtime_rates", {}) or {})
        return make_sce_reachability_slot_get_actions_fn(
            env, reachability=reach, runtime_rates=rates, **block
        )
    if "sce_reachability_cbf_slot" in cfg:
        block = dict(cfg.get("sce_reachability_cbf_slot", {}) or {})
        reach = dict(block.pop("reachability", {}) or {})
        cbf = dict(block.pop("cbf", {}) or {})
        rates = dict(block.pop("runtime_rates", {}) or {})
        return make_sce_reachability_cbf_slot_get_actions_fn(
            env, reachability=reach, cbf=cbf, runtime_rates=rates, **block
        )
    return None


def _build_rl_worker(env: Any, cfg: dict[str, Any], cfg_path: Path) -> RolloutWorker:
    import importlib.util

    from marl_uav.agents.mac import MAC

    eval_path = ROOT / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("debug_eval_helpers", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import eval helpers from {eval_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    algo_cfg_path = ROOT / str(cfg["algo"])
    model_cfg_path = ROOT / str(cfg["model"])
    policy_core = mod.build_policy(model_cfg_path, env, algo_cfg_path)
    n_actions = (
        env.n_actions
        if getattr(policy_core, "action_space_type", "discrete") == "discrete"
        else (getattr(policy_core, "action_dim", None) or 0)
    )
    mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions, n_agents=env.num_agents)
    mac.policy = policy_core
    learner = mod.build_learner(algo_cfg_path, policy_core)

    seed = int(cfg.get("seed", 0))
    result_dir = Path(str(cfg.get("train_results_dir") or f"results/{cfg_path.stem}"))
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    ckpt = result_dir / "checkpoints" / str(seed) / "best.pt"
    if not ckpt.is_file():
        ckpt = result_dir / "checkpoints" / str(seed) / "latest.pt"
    if not ckpt.is_file():
        ckpt = result_dir / "checkpoints" / str(seed) / "bc_pretrained.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint found under {result_dir / 'checkpoints' / str(seed)}")

    from marl_uav.utils.checkpoint import load_checkpoint

    load_checkpoint(ckpt, learner)
    mac.set_test_mode(True)
    return RolloutWorker(env=env, policy=mac)


def _resolve_step_dt(env_cfg: dict[str, Any]) -> float:
    backend = dict(env_cfg.get("backend", {}) or {})
    control_hz = float(backend.get("control_hz", 60))
    return 1.0 / max(control_hz, 1e-6)


def _start_step_publish_watchdog(env: Any, *, interval_s: float = 0.25) -> threading.Event:
    """Publish a backup live frame when env.step_count advances.

    Normal step frames are emitted by the env/worker path.  This watchdog keeps
    the browser live view moving even if a debug frame publication is swallowed
    by a backend-specific path.
    """
    stop = threading.Event()

    def _loop() -> None:
        last_step = -1
        while not stop.is_set():
            try:
                hub = get_debug_browser_hub()
                step = int(getattr(env, "step_count", 0))
                if hub is not None and step > last_step:
                    last_step = step
                    if step > 0:
                        from marl_uav.utils.debug_browser import publish_env_frame

                        publish_env_frame(
                            env,
                            {"termination_reason": "running"},
                            event="step",
                            extra={"debug_watchdog": True},
                        )
            except Exception:
                pass
            stop.wait(max(float(interval_s), 0.05))

    threading.Thread(target=_loop, name="debug-browser-step-watchdog", daemon=True).start()
    return stop


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = merge_rl_task_speed(load_config(cfg_path))
    task_cfg = dict(cfg.get("task", {}) or {})
    task_cfg["debug"] = True
    browser_cfg = dict(task_cfg.get("debug_browser", {}) or {})
    host = str(browser_cfg.get("host", args.host))
    port = int(browser_cfg.get("port", args.port))
    open_browser = bool(browser_cfg.get("open_browser", not args.no_open_browser))
    playback_speed = float(
        args.speed if args.speed is not None else browser_cfg.get("playback_speed", 0.25)
    )
    env_cfg = load_config(ROOT / str(cfg["env"]))
    step_dt = float(browser_cfg.get("step_dt", _resolve_step_dt(env_cfg)))

    record_root = browser_cfg.get("record_dir")
    if record_root:
        record_dir = Path(record_root)
        if not record_dir.is_absolute():
            record_dir = ROOT / record_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = ROOT / "results" / "debug_browser" / cfg_path.stem / stamp
    record_dir.mkdir(parents=True, exist_ok=True)

    viz_profile = resolve_viz_profile(cfg)
    controller_cfg: dict[str, Any] = {}
    for key in (
        "pure_pursuit",
        "pure_pursuit_apf",
        "fixed_ring",
        "fixed_ring_apf",
        "oracle_slot",
        "hungarian_slot",
        "ot_slot",
        "trajectory_planner",
        "slot_exec_mappo",
        "sce",
        "sce_los_slot",
        "sce_turn_radius_slot",
        "sce_cached_path_slot",
        "sce_cached_path_cbf_slot",
        "sce_reachability_slot",
        "sce_reachability_cbf_slot",
        "sce_path_slot",
        "sce_path_cbf_slot",
    ):
        if key in cfg:
            controller_cfg = dict(cfg.get(key) or {})

    hub = configure_debug_browser(
        enabled=True,
        host=host,
        port=port,
        playback_speed=playback_speed,
        step_dt=step_dt,
        start_paused=True,
        record_dir=record_dir,
        meta={
            "config": str(cfg_path.relative_to(ROOT)),
            "seed": int(args.seed),
            "playback_speed": playback_speed,
            "step_dt": step_dt,
            "total_episodes": int(args.episodes),
            "record_dir": str(record_dir.relative_to(ROOT)).replace("\\", "/"),
            "viz": viz_profile,
            "controller": controller_cfg,
        },
    )
    if hub is None:
        raise RuntimeError("Failed to start debug browser hub.")
    hub.set_run_plan(total_episodes=int(args.episodes))

    env = build_env_from_config(ROOT / str(cfg["env"]), seed=args.seed, task_cfg=task_cfg)
    if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
        env.reset(seed=args.seed)

    speed_bounds = resolve_speed_bounds(cfg, env=env, env_cfg=env_cfg)
    hub._meta["speed_bounds"] = speed_bounds
    hub._meta["execution_kind"] = "rl" if "algo" in cfg else "heuristic"

    get_actions = _build_get_actions_fn(env, cfg)
    if get_actions is not None:
        worker = RolloutWorker(env=env, policy=object(), get_actions_fn=get_actions)
    elif "algo" in cfg and "model" in cfg:
        worker = _build_rl_worker(env, cfg, cfg_path)
    else:
        raise ValueError(
            f"Config {cfg_path} must define sce/oracle_slot/hungarian_slot/ot_slot/"
            "pure_pursuit/pure_pursuit_apf/fixed_ring/fixed_ring_apf/"
            "trajectory_planner/slot_exec_mappo/"
            "sce_turn_radius_slot/sce_reachability_slot/sce_reachability_cbf_slot "
            "or algo+model for RL debug."
        )

    watchdog_stop = _start_step_publish_watchdog(env, interval_s=max(step_dt * 2.0, 0.10))

    if open_browser:
        webbrowser.open(hub.url)

    print(
        f"[debug-browser] {args.episodes} episode(s); UI at {hub.url} "
        f"speed={playback_speed}x step_dt={step_dt:.4f}s"
    )
    print(f"[debug-browser] recording -> {record_dir.relative_to(ROOT)}")
    print("[debug-browser] paused at start frame — click 开始 in browser to run")
    try:
        for ep in range(int(args.episodes)):
            ep_seed = int(args.seed) + ep
            print(f"[debug-browser] episode {ep + 1}/{args.episodes} seed={ep_seed}")
            try:
                _, info = worker.collect_episode(seed=ep_seed, record_trajectory=False)
            except Exception as exc:
                print(f"[debug-browser] episode failed: {exc}")
                import traceback

                traceback.print_exc()
                live_hub = get_debug_browser_hub()
                if live_hub is not None:
                    live_hub.publish(
                        {
                            "event": "sim_error",
                            "scene_id": "pursuit_3v1",
                            "error": str(exc),
                            "step": 0,
                        }
                    )
                raise
            print(
                f"[debug-browser] done: return={info.get('episode_return', 0):.3f} "
                f"len={info.get('episode_len', 0)} capture={info.get('capture', False)}"
            )
            if args.sleep > 0 and ep + 1 < int(args.episodes):
                time.sleep(args.sleep)
    finally:
        try:
            env.close()
        except Exception:
            pass
        watchdog_stop.set()
        if args.keep_alive:
            print("[debug-browser] keep-alive enabled. Press Ctrl+C to stop server.")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
        else:
            print("[debug-browser] finished.")


if __name__ == "__main__":
    main()
