"""Failure replay taxonomy for ``sce_cached_path_slot`` on E2 obstacle scenarios.

Runs (or replays) episodes with per-step deploy diagnostics and classifies failures
into obstacle-proximity / planner / slot reachability / tracking conflict / stale-cache
hypotheses.

Example::

    python scripts/analyze_sce_cached_path_failures.py \\
        --config configs/experiment/e2_obstacles_pyflyt_sce_cached_path_slot.yaml \\
        --episodes 50 --seed 101

    python scripts/analyze_sce_cached_path_failures.py \\
        --eval-records results/e2_obstacles_pyflyt/eval_records/e2_obstacles_all_records.csv \\
        --method sce_cached_path_slot --failures-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl_uav.control.obstacle_aware_sce_baselines import make_sce_cached_path_slot_get_actions_fn
from marl_uav.envs.factories import build_env_from_config
from marl_uav.framework.geometry.obstacle_adapter import obstacles_from_task_state
from marl_uav.framework.geometry.obstacle_geometry import (
    Obstacle,
    has_line_of_sight,
)
from marl_uav.framework.planning.path_tracking import closest_point_on_polyline
from marl_uav.framework.planning.path_validation import validate_planned_path
from marl_uav.utils.config import load_config
from marl_uav.utils.e1_1_suite import merge_rl_task_speed


TAG_NEAR_OBSTACLE = "near_obstacle"
TAG_PATH_NOT_AVOIDING = "path_not_avoiding"
TAG_SLOT_UNREACHABLE = "slot_unreachable"
TAG_TRACKING_CONFLICT = "tracking_encirclement_conflict"
TAG_STALE_UNSAFE_CACHE = "stale_unsafe_cache"

TAG_LABELS_ZH = {
    TAG_NEAR_OBSTACLE: "失败发生在障碍物附近",
    TAG_PATH_NOT_AVOIDING: "路径规划未绕开障碍",
    TAG_SLOT_UNREACHABLE: "slot 在障碍后方/不可达",
    TAG_TRACKING_CONFLICT: "管径跟踪与围捕目标冲突",
    TAG_STALE_UNSAFE_CACHE: "复用了不安全的旧路径缓存",
}


@dataclass
class StepSnapshot:
    step: int
    pursuer_xy: list[list[float]]
    evader_xy: list[float]
    hit_mask: list[bool]
    min_clearance: list[float]
    num_replans: float
    path_collision_free: float | None
    los_blocked_assigned: float | None
    limit_reasons: list[str]
    cross_track: list[float]
    slot_bearing_deg: list[float]
    path_tangent_deg: list[float]
    slot_los_clear: list[bool]
    path_deviation: list[float]
    slot_goal_drift: list[float]
    deploy_control: dict[str, Any] | None = None


@dataclass
class EpisodeTrace:
    seed: int
    episode_len: int
    captured: bool
    obstacle_termination: bool
    timeout: bool
    collision: bool
    outcome: str
    failure_step: int
    failure_pursuer_ids: list[int]
    steps: list[StepSnapshot] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tag_evidence: dict[str, str] = field(default_factory=dict)
    primary_tag: str = ""


def _build_worker(cfg_path: Path, seed: int):
    cfg = merge_rl_task_speed(load_config(cfg_path))
    env = build_env_from_config(ROOT / str(cfg["env"]), seed=seed, task_cfg=cfg.get("task", {}))
    if getattr(env, "obs_dim", None) is None:
        env.reset(seed=seed)
    if hasattr(env, "task"):
        env.task.debug = False

    block = dict(cfg.get("sce_cached_path_slot", cfg.get("sce_path_slot", {})) or {})
    reach = dict(block.pop("reachability", {}) or {})
    rates = dict(block.pop("runtime_rates", {}) or {})
    get_actions = make_sce_cached_path_slot_get_actions_fn(
        env, reachability=reach, runtime_rates=rates, **block,
    )
    return env, get_actions, reach


def _clearances(pursuer_xy: np.ndarray, obstacles: list[Obstacle]) -> tuple[list[float], list[bool]]:
    out_clear: list[float] = []
    out_hit: list[bool] = []
    for i in range(int(pursuer_xy.shape[0])):
        p = pursuer_xy[i]
        best = float("inf")
        hit = False
        for obs in obstacles:
            c = np.asarray(obs.center, dtype=np.float64).reshape(2)
            surface = float(np.linalg.norm(p - c)) - float(obs.radius)
            best = min(best, surface)
            if surface <= 0.0:
                hit = True
        out_clear.append(best if math.isfinite(best) else float("nan"))
        out_hit.append(hit)
    return out_clear, out_hit


def _bearing_deg(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    d = np.asarray(to_xy, dtype=np.float64).reshape(2) - np.asarray(from_xy, dtype=np.float64).reshape(2)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _angle_diff_deg(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def _path_tangent_deg(path_xy: list[list[float]] | None, pos_xy: np.ndarray) -> float:
    if not path_xy or len(path_xy) < 2:
        return float("nan")
    pts = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    _, tangent, _, _ = closest_point_on_polyline(pts, pos_xy)
    t = np.asarray(tangent, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(t)) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan2(t[1], t[0])))


def _path_deviation(path_xy: list[list[float]] | None, pos_xy: np.ndarray) -> float:
    if not path_xy or len(path_xy) < 2:
        return float("nan")
    pts = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    _, _, cross, _ = closest_point_on_polyline(pts, pos_xy)
    return float(abs(cross))


def _path_cache_snapshot(env: Any, obstacles: list[Obstacle], *, safety_margin: float, uav_r: float) -> tuple[list[float], list[float], list[bool]]:
    deviations: list[float] = []
    drifts: list[float] = []
    los_flags: list[bool] = []
    st = getattr(env, "_deploy_sce_state", None)
    if st is None:
        return [float("nan")] * 3, [float("nan")] * 3, [True] * 3
    pc = st.path_cache
    task_state = env.task_state
    targets = getattr(st, "cached_targets", None)
    assignment = getattr(st, "cached_assignment", None)
    if targets is None or assignment is None:
        return [float("nan")] * 3, [float("nan")] * 3, [True] * 3

    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    lin_pos = np.asarray(env.prev_backend_state.states[:, 3, :], dtype=np.float64)
    pursuer_pos = lin_pos[pursuer_ids]

    for i in range(3):
        slot_j = int(assignment[i])
        goal = np.asarray(targets[slot_j, :2], dtype=np.float64)
        p = pursuer_pos[i, :2]
        path = pc.get_agent_path(i)
        path_xy = None if path is None else [[float(x[0]), float(x[1])] for x in path]
        deviations.append(_path_deviation(path_xy, p))
        ap = pc.agent_paths.get(i)
        if ap is not None and ap.goal_xy is not None:
            drifts.append(float(np.linalg.norm(goal - ap.goal_xy.reshape(2))))
        else:
            drifts.append(float("nan"))
        los_flags.append(
            has_line_of_sight(p, goal, obstacles, safety_margin=safety_margin, uav_radius=uav_r)
        )
    return deviations, drifts, los_flags


def _collect_step_snapshot(
    env: Any,
    step_info: dict[str, Any],
    *,
    step: int,
    safety_margin: float,
    uav_r: float,
    store_deploy: bool = False,
) -> StepSnapshot:
    task_state = env.task_state
    task = env.task
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).reshape(3)
    lin_pos = np.asarray(env.prev_backend_state.states[:, 3, :], dtype=np.float64)
    pursuer_pos = lin_pos[pursuer_ids]
    evader_pos = lin_pos[int(task_state.evader_id)]
    obstacles = obstacles_from_task_state(task_state, task=task)

    clearances, hit_mask = _clearances(pursuer_pos[:, :2], obstacles)
    diag = getattr(env, "_obstacle_aware_diagnostics", {}) or {}
    deploy = diag.get("deploy_control") if isinstance(diag.get("deploy_control"), dict) else {}
    pursuers_dbg = deploy.get("pursuers") if isinstance(deploy.get("pursuers"), list) else []

    path_cf = diag.get("assigned_path_collision_free")
    if path_cf is not None:
        path_cf = float(path_cf)
    elif pursuers_dbg:
        cf_vals: list[float] = []
        for entry in pursuers_dbg:
            path_xy = entry.get("assigned_path_xy")
            if not path_xy or len(path_xy) < 2:
                continue
            path = [np.asarray(p, dtype=np.float64) for p in path_xy]
            cf_vals.append(
                1.0 if validate_planned_path(
                    path, obstacles, safety_margin=safety_margin, uav_radius=uav_r,
                ) else 0.0
            )
        if cf_vals:
            path_cf = float(np.mean(cf_vals))

    los_blocked = diag.get("assigned_pair_blocked_los")
    if los_blocked is not None:
        los_blocked = float(los_blocked)

    limit_reasons: list[str] = []
    cross_tracks: list[float] = []
    slot_bearings: list[float] = []
    tangents: list[float] = []

    for i in range(3):
        pxy = pursuer_pos[i, :2]
        entry = pursuers_dbg[i] if i < len(pursuers_dbg) else {}
        slot_xy = np.asarray(entry.get("slot_target_xy", evader_pos[:2]), dtype=np.float64)
        path_xy = entry.get("assigned_path_xy")
        limit_reasons.append(str(entry.get("limit_reason", "")))
        cross_tracks.append(float(entry.get("cross_track_xy", 0.0)))
        slot_bearings.append(_bearing_deg(pxy, slot_xy))
        tangents.append(_path_tangent_deg(path_xy, pxy))

    deviations, drifts, slot_los = _path_cache_snapshot(
        env, obstacles, safety_margin=safety_margin, uav_r=uav_r,
    )

    return StepSnapshot(
        step=step,
        pursuer_xy=[[float(x), float(y)] for x, y in pursuer_pos[:, :2]],
        evader_xy=[float(evader_pos[0]), float(evader_pos[1])],
        hit_mask=[bool(x) for x in hit_mask],
        min_clearance=[float(x) for x in clearances],
        num_replans=float(diag.get("num_replans_this_step", 0.0)),
        path_collision_free=path_cf,
        los_blocked_assigned=los_blocked,
        limit_reasons=limit_reasons,
        cross_track=cross_tracks,
        slot_bearing_deg=slot_bearings,
        path_tangent_deg=tangents,
        slot_los_clear=[bool(x) for x in slot_los],
        path_deviation=deviations,
        slot_goal_drift=drifts,
        deploy_control=deploy if store_deploy else None,
    )


def _run_episode(
    env: Any,
    get_actions: Any,
    seed: int,
    *,
    safety_margin: float,
    uav_r: float,
    lookback: int,
) -> EpisodeTrace:
    obs_dict, _ = env.reset(seed=seed)
    obs_list = obs_dict["obs"]
    state = obs_dict["state"]
    avail = env.get_avail_actions()

    steps: list[StepSnapshot] = []
    terminated = truncated = False
    last_info: dict[str, Any] = {}
    step_idx = 0

    while True:
        actions = get_actions(obs_list, state, avail)
        next_obs_dict, _, terminated, truncated, last_info = env.step(actions)
        step_idx += 1
        store_deploy = step_idx % 10 == 0 or bool(last_info.get("obstacle_terminated", False))
        snap = _collect_step_snapshot(
            env, last_info, step=step_idx,
            safety_margin=safety_margin, uav_r=uav_r, store_deploy=store_deploy,
        )
        steps.append(snap)
        if terminated or truncated:
            break
        obs_list = next_obs_dict["obs"]
        state = next_obs_dict["state"]
        avail = env.get_avail_actions()

    captured = bool(last_info.get("captured", False))
    obs_term = bool(last_info.get("obstacle_terminated", False))
    timeout = bool(last_info.get("timeout", False))
    collision = bool(last_info.get("has_collision", False))

    if captured:
        outcome = "captured"
    elif obs_term:
        outcome = "obstacle_termination"
    elif timeout:
        outcome = "timeout"
    elif collision:
        outcome = "collision"
    else:
        outcome = "other"

    fail_step = step_idx
    fail_pursuers = [i for i, h in enumerate(steps[-1].hit_mask) if h] if steps else []
    if not fail_pursuers and outcome == "obstacle_termination" and steps:
        last = steps[-1]
        fail_pursuers = [
            i for i, c in enumerate(last.min_clearance)
            if math.isfinite(c) and c < 0.05
        ]

    trace = EpisodeTrace(
        seed=int(seed),
        episode_len=step_idx,
        captured=captured,
        obstacle_termination=obs_term,
        timeout=timeout,
        collision=collision,
        outcome=outcome,
        failure_step=fail_step,
        failure_pursuer_ids=fail_pursuers,
        steps=steps[-lookback:] if lookback > 0 else steps,
    )
    _classify_episode(trace, env, safety_margin=safety_margin, uav_r=uav_r)
    return trace


def _classify_episode(trace: EpisodeTrace, env: Any, *, safety_margin: float, uav_r: float) -> None:
    if trace.outcome == "captured":
        trace.primary_tag = "success"
        return

    if not trace.steps:
        trace.primary_tag = "unknown"
        return

    last = trace.steps[-1]
    task_state = env.task_state
    task = env.task
    obstacles = obstacles_from_task_state(task_state, task=task)
    st = getattr(env, "_deploy_sce_state", None)
    slot_thresh = float(st.path_cache.cfg.slot_replan_threshold) if st is not None else 0.3

    near_m = 1.0
    min_clear = min((c for c in last.min_clearance if math.isfinite(c)), default=float("inf"))
    window_clears = [
        c for s in trace.steps for c in s.min_clearance if math.isfinite(c)
    ]
    mean_clear = float(np.mean(window_clears)) if window_clears else float("inf")

    if min_clear <= near_m or mean_clear <= near_m + 0.3:
        trace.tags.append(TAG_NEAR_OBSTACLE)
        trace.tag_evidence[TAG_NEAR_OBSTACLE] = (
            f"终止步最小净空={min_clear:.2f}m, 窗口均值={mean_clear:.2f}m"
        )

    path_invalid = last.path_collision_free is not None and last.path_collision_free < 0.5
    deploy = last.deploy_control or {}
    pursuers_dbg = deploy.get("pursuers") if isinstance(deploy.get("pursuers"), list) else []
    recomputed_invalid = False
    for entry in pursuers_dbg:
        path_xy = entry.get("assigned_path_xy")
        if not path_xy or len(path_xy) < 2:
            continue
        path = [np.asarray(p, dtype=np.float64) for p in path_xy]
        if not validate_planned_path(path, obstacles, safety_margin=safety_margin, uav_radius=uav_r):
            recomputed_invalid = True
            break

    if path_invalid or recomputed_invalid:
        trace.tags.append(TAG_PATH_NOT_AVOIDING)
        trace.tag_evidence[TAG_PATH_NOT_AVOIDING] = (
            f"assigned_path_collision_free={last.path_collision_free}, "
            f"recomputed_invalid={recomputed_invalid}"
        )

    slot_blocked = any(not x for x in last.slot_los_clear)
    los_assign_blocked = bool(last.los_blocked_assigned and last.los_blocked_assigned > 0.5)
    if slot_blocked or los_assign_blocked:
        trace.tags.append(TAG_SLOT_UNREACHABLE)
        blocked_ids = [i for i, ok in enumerate(last.slot_los_clear) if not ok]
        trace.tag_evidence[TAG_SLOT_UNREACHABLE] = (
            f"slot LOS blocked pursuers={blocked_ids}, assigned_pair_blocked_los={last.los_blocked_assigned}"
        )

    conflict_hits = 0
    for i in range(3):
        reason = last.limit_reasons[i] if i < len(last.limit_reasons) else ""
        cross = last.cross_track[i] if i < len(last.cross_track) else 0.0
        clr = last.min_clearance[i] if i < len(last.min_clearance) else float("inf")
        sb = last.slot_bearing_deg[i] if i < len(last.slot_bearing_deg) else float("nan")
        tg = last.path_tangent_deg[i] if i < len(last.path_tangent_deg) else float("nan")
        bearing_gap = _angle_diff_deg(sb, tg) if math.isfinite(sb) and math.isfinite(tg) else 0.0
        if (
            reason in ("tube_off_path", "tube_slowdown", "approach")
            and math.isfinite(clr)
            and clr < near_m + 0.5
            and (cross > 0.25 or bearing_gap > 45.0)
        ):
            conflict_hits += 1
    if conflict_hits > 0:
        trace.tags.append(TAG_TRACKING_CONFLICT)
        trace.tag_evidence[TAG_TRACKING_CONFLICT] = (
            f"{conflict_hits}/3 pursuers: tube/cross-track/bearing conflict near obstacles"
        )

    no_replan_steps = sum(1 for s in trace.steps if s.num_replans < 0.5)
    drift_high = any(
        math.isfinite(d) and d > slot_thresh for d in last.slot_goal_drift
    )
    stale_path = (path_invalid or recomputed_invalid) and no_replan_steps >= max(1, len(trace.steps) // 2)
    stale_drift = drift_high and no_replan_steps >= max(1, len(trace.steps) // 2)
    if stale_path or stale_drift:
        trace.tags.append(TAG_STALE_UNSAFE_CACHE)
        trace.tag_evidence[TAG_STALE_UNSAFE_CACHE] = (
            f"no_replan_steps={no_replan_steps}/{len(trace.steps)}, "
            f"drift_high={drift_high}, path_invalid={path_invalid or recomputed_invalid}"
        )

    if not trace.tags:
        trace.primary_tag = trace.outcome
    else:
        priority = [
            TAG_STALE_UNSAFE_CACHE,
            TAG_PATH_NOT_AVOIDING,
            TAG_SLOT_UNREACHABLE,
            TAG_TRACKING_CONFLICT,
            TAG_NEAR_OBSTACLE,
        ]
        trace.primary_tag = next((t for t in priority if t in trace.tags), trace.tags[0])


def _load_failure_seeds_from_csv(path: Path, method: str) -> list[int]:
    seeds: list[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") != method:
                continue
            captured = int(float(row.get("captured", 0)))
            obs_term = int(float(row.get("obstacle_termination", 0)))
            timeout = int(float(row.get("timeout", 0)))
            if captured:
                continue
            if obs_term or timeout:
                seeds.append(int(row["eval_seed"]))
    return seeds


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _summarize(traces: list[EpisodeTrace]) -> dict[str, Any]:
    failures = [t for t in traces if t.outcome != "captured"]
    n_fail = len(failures)
    tag_counts = Counter(tag for t in failures for tag in t.tags)
    primary_counts = Counter(t.primary_tag for t in failures)

    def pct(n: int) -> float:
        return 100.0 * n / n_fail if n_fail else 0.0

    questions = {
        "failures_near_obstacles": {
            "question": "失败是否集中发生在障碍物附近？",
            "yes": tag_counts.get(TAG_NEAR_OBSTACLE, 0),
            "pct": pct(tag_counts.get(TAG_NEAR_OBSTACLE, 0)),
        },
        "planner_did_not_avoid": {
            "question": "是否因路径规划没有绕开障碍物？",
            "yes": tag_counts.get(TAG_PATH_NOT_AVOIDING, 0),
            "pct": pct(tag_counts.get(TAG_PATH_NOT_AVOIDING, 0)),
        },
        "slot_unreachable": {
            "question": "是否因 slot 在障碍后方且无可达路径？",
            "yes": tag_counts.get(TAG_SLOT_UNREACHABLE, 0),
            "pct": pct(tag_counts.get(TAG_SLOT_UNREACHABLE, 0)),
        },
        "tracking_conflict": {
            "question": "是否因局部管径跟踪与围捕目标冲突？",
            "yes": tag_counts.get(TAG_TRACKING_CONFLICT, 0),
            "pct": pct(tag_counts.get(TAG_TRACKING_CONFLICT, 0)),
        },
        "stale_unsafe_cache": {
            "question": "是否因复用了已经不安全的旧路径缓存？",
            "yes": tag_counts.get(TAG_STALE_UNSAFE_CACHE, 0),
            "pct": pct(tag_counts.get(TAG_STALE_UNSAFE_CACHE, 0)),
        },
    }

    return {
        "num_episodes": len(traces),
        "num_failures": n_fail,
        "outcome_counts": dict(Counter(t.outcome for t in traces)),
        "tag_counts": dict(tag_counts),
        "primary_tag_counts": dict(primary_counts),
        "questions": questions,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# sce_cached_path_slot 失败回放分类报告",
        "",
        f"- 分析 episode 数: {summary['num_episodes']}",
        f"- 失败 episode 数: {summary['num_failures']}",
        f"- 结局分布: {summary['outcome_counts']}",
        "",
        "## 五个核心问题（占失败 episode 比例，可多标签）",
        "",
    ]
    for key, block in summary["questions"].items():
        lines.append(f"### {block['question']}")
        lines.append(f"- **是**: {block['yes']} / {summary['num_failures']} ({block['pct']:.1f}%)")
        lines.append("")
    lines.append("## 标签计数")
    for tag, count in sorted(summary.get("tag_counts", {}).items(), key=lambda x: -x[1]):
        label = TAG_LABELS_ZH.get(tag, tag)
        lines.append(f"- `{tag}` ({label}): {count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify sce_cached_path_slot failure replays.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiment/e2_obstacles_pyflyt_sce_cached_path_slot.yaml",
    )
    p.add_argument("--seed", type=int, default=101, help="Base train seed for env layout.")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--lookback", type=int, default=60, help="Steps kept per episode for analysis.")
    p.add_argument(
        "--eval-records",
        type=str,
        default=None,
        help="Optional CSV; when set, replay eval_seed rows for --method.",
    )
    p.add_argument("--method", type=str, default="sce_cached_path_slot")
    p.add_argument("--failures-only", action="store_true", help="With --eval-records, only failed rows.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: results/e2_obstacles_pyflyt/failure_analysis/sce_cached_path_slot/<timestamp>",
    )
    p.add_argument("--save-examples", type=int, default=2, help="JSON traces per primary tag (0=off).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    reach = dict((cfg.get("sce_cached_path_slot", {}) or {}).get("reachability", {}) or {})
    safety_margin = float(reach.get("safety_margin", 0.3))
    uav_r = float(reach.get("uav_radius", 0.15))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results" / "e2_obstacles_pyflyt" / "failure_analysis" / args.method / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_records:
        seeds = _load_failure_seeds_from_csv(ROOT / args.eval_records, args.method)
        if not args.failures_only:
            raise ValueError("--eval-records requires --failures-only (only failed rows are replayed).")
        if args.episodes > 0:
            seeds = seeds[: args.episodes]
    else:
        base = int(args.seed) * 1_000_000 + 8010 * 10_000
        seeds = [base + ep for ep in range(int(args.episodes))]

    env, get_actions, _ = _build_worker(cfg_path, seed=int(args.seed))
    traces: list[EpisodeTrace] = []
    rows: list[dict[str, Any]] = []

    print(f"[analyze] episodes={len(seeds)} output={out_dir.relative_to(ROOT)}")
    for i, ep_seed in enumerate(seeds):
        trace = _run_episode(
            env, get_actions, ep_seed,
            safety_margin=safety_margin, uav_r=uav_r, lookback=int(args.lookback),
        )
        traces.append(trace)
        rows.append({
            "eval_seed": trace.seed,
            "outcome": trace.outcome,
            "episode_len": trace.episode_len,
            "failure_step": trace.failure_step,
            "failure_pursuer_ids": ",".join(str(x) for x in trace.failure_pursuer_ids),
            "primary_tag": trace.primary_tag,
            "tags": "|".join(trace.tags),
            **{f"evidence_{k}": v for k, v in trace.tag_evidence.items()},
        })
        if (i + 1) % 10 == 0 or i + 1 == len(seeds):
            print(f"  [{i + 1}/{len(seeds)}] last outcome={trace.outcome} tag={trace.primary_tag}")

    summary = _summarize(traces)
    _write_csv(out_dir / "episode_classifications.csv", rows)
    _write_csv(
        out_dir / "tag_summary.csv",
        [
            {"tag": tag, "label_zh": TAG_LABELS_ZH.get(tag, tag), "count": count,
             "pct_of_failures": 100.0 * count / max(summary["num_failures"], 1)}
            for tag, count in sorted(summary["tag_counts"].items(), key=lambda x: -x[1])
        ],
    )
    _write_report(out_dir / "failure_analysis_report.md", summary)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if int(args.save_examples) > 0:
        examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trace in traces:
            if trace.outcome == "captured":
                continue
            if len(examples[trace.primary_tag]) >= int(args.save_examples):
                continue
            examples[trace.primary_tag].append({
                "seed": trace.seed,
                "outcome": trace.outcome,
                "tags": trace.tags,
                "evidence": trace.tag_evidence,
                "steps": [s.__dict__ for s in trace.steps],
            })
        with open(out_dir / "example_traces.json", "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    print("\n=== 失败分类摘要 ===")
    for block in summary["questions"].values():
        print(f"{block['question']}: {block['yes']}/{summary['num_failures']} ({block['pct']:.1f}%)")
    print(f"\nWrote: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
