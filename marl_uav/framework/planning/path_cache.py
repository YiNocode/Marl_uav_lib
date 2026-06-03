"""Event-triggered path cache for deployable SCE baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_adapter import obstacle_version_key
from marl_uav.framework.geometry.obstacle_geometry import (
    Obstacle,
    collision_check_path,
    has_line_of_sight,
    path_length,
)
from marl_uav.framework.planning.path_validation import validate_planned_path
from marl_uav.framework.planning.static_visibility_graph import StaticVisibilityGraph
from marl_uav.framework.planning.path_tracking import closest_point_on_polyline


def path_clearance_stats(
    path: list[np.ndarray] | np.ndarray,
    obstacles: list[Obstacle],
    *,
    uav_radius: float = 0.0,
    samples_per_segment: int = 5,
) -> tuple[float, float]:
    """Return min/mean surface clearance for a path against the same obstacle set.

    The cache uses this for invalidation; assignment and logging use the same
    helper so planning, scoring, and diagnostics agree on what "safe path" means.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0 or not obstacles:
        return float("inf"), float("inf")

    samples: list[np.ndarray] = []
    if pts.shape[0] == 1:
        samples.append(pts[0])
    else:
        n_seg = max(int(samples_per_segment), 2)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0.0, 1.0, n_seg, endpoint=False):
                samples.append(p0 + float(t) * (p1 - p0))
        samples.append(pts[-1])

    clearances: list[float] = []
    for p in samples:
        best = float("inf")
        for obs in obstacles:
            if getattr(obs, "kind", None) != "circle":
                continue
            center = np.asarray(obs.center, dtype=np.float64).reshape(2)
            clear = float(np.linalg.norm(p - center)) - float(obs.radius) - float(uav_radius)
            best = min(best, clear)
        if np.isfinite(best):
            clearances.append(best)
    if not clearances:
        return float("inf"), float("inf")
    arr = np.asarray(clearances, dtype=np.float64)
    return float(np.min(arr)), float(np.mean(arr))


@dataclass
class PathCacheConfig:
    enabled: bool = True
    pair_cost_mode: str = "los_penalty_cached"
    replan_interval: int = 10
    path_deviation_threshold: float = 0.25
    slot_replan_threshold: float = 0.3
    slot_replan_always: bool = False
    slot_projection_force_replan: bool = True
    goal_rounding_resolution: float = 0.5
    start_rounding_resolution: float = 0.5
    max_replans_per_step: int = 3
    fallback_blocked_penalty: float = 50.0
    validate_cached_path: bool = True
    cache_validation_interval: int = 25
    invalidate_on_plan_failure: bool = True
    replan_slot_move_thresh: float = 1.0
    replan_target_move_thresh: float = 1.0
    replan_endpoint_error_thresh: float = 1.0
    replan_min_clearance: float = 0.6
    replan_tracking_error_thresh: float = 1.5
    replan_cbf_active_steps: int = 20
    replan_time_budget: float = 80.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PathCacheConfig:
        d = dict(raw or {})
        slot_thresh = float(d.get("replan_slot_move_thresh", d.get("slot_replan_threshold", 1.0)))
        track_thresh = float(d.get("replan_tracking_error_thresh", d.get("path_deviation_threshold", 1.5)))
        return cls(
            enabled=bool(d.get("enabled", True)),
            pair_cost_mode=str(d.get("pair_cost_mode", "los_penalty_cached")),
            replan_interval=int(d.get("replan_interval", 10)),
            path_deviation_threshold=float(d.get("path_deviation_threshold", 0.25)),
            slot_replan_threshold=float(d.get("slot_replan_threshold", 0.3)),
            slot_replan_always=bool(d.get("slot_replan_always", False)),
            slot_projection_force_replan=bool(d.get("slot_projection_force_replan", True)),
            goal_rounding_resolution=float(d.get("goal_rounding_resolution", 0.5)),
            start_rounding_resolution=float(d.get("start_rounding_resolution", 0.5)),
            max_replans_per_step=int(d.get("max_replans_per_step", 3)),
            fallback_blocked_penalty=float(d.get("fallback_blocked_penalty", 50.0)),
            validate_cached_path=bool(d.get("validate_cached_path", True)),
            cache_validation_interval=max(int(d.get("cache_validation_interval", 25)), 1),
            invalidate_on_plan_failure=bool(d.get("invalidate_on_plan_failure", True)),
            replan_slot_move_thresh=slot_thresh,
            replan_target_move_thresh=float(d.get("replan_target_move_thresh", 1.0)),
            replan_endpoint_error_thresh=float(d.get("replan_endpoint_error_thresh", 1.0)),
            replan_min_clearance=float(d.get("replan_min_clearance", 0.6)),
            replan_tracking_error_thresh=track_thresh,
            replan_cbf_active_steps=max(int(d.get("replan_cbf_active_steps", 20)), 1),
            replan_time_budget=float(d.get("replan_time_budget", 80.0)),
        )


@dataclass
class AgentPathState:
    path: list[np.ndarray] | None = None
    slot_id: int = -1
    start_xy: np.ndarray | None = None
    goal_xy: np.ndarray | None = None
    target_xy: np.ndarray | None = None
    obstacle_version: tuple = field(default_factory=tuple)
    path_length: float = float("inf")
    min_clearance: float = float("inf")
    mean_clearance: float = float("inf")
    planned_step: int = -1
    last_replan_step: int = -1
    los_blocked: bool = False
    feasible: bool = False


class DeployPathCache:
    """
    Cached paths and pair costs for real-time deployable baselines.

    Assignment uses LOS + cached/heuristic costs only — never full replan for all pairs.
    """

    def __init__(self, cfg: PathCacheConfig | dict[str, Any] | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, PathCacheConfig) else PathCacheConfig.from_dict(cfg)
        self.agent_paths: dict[int, AgentPathState] = {}
        self.pair_cost_cache: dict[tuple[int, int, tuple], float] = {}
        self.static_graph = StaticVisibilityGraph()
        self.obstacle_version: tuple = ()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "replan_count": 0,
            "timeout_count": 0,
            "replan_time_ms": [],
        }

    def clear(self) -> None:
        self.agent_paths.clear()
        self.pair_cost_cache.clear()
        self.static_graph = StaticVisibilityGraph()
        self.obstacle_version = ()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "replan_count": 0,
            "timeout_count": 0,
            "replan_time_ms": [],
        }

    def _round(self, p: np.ndarray, res: float) -> tuple[float, float]:
        q = np.asarray(p, dtype=np.float64).reshape(2)
        r = max(float(res), 0.05)
        return (round(float(q[0]) / r) * r, round(float(q[1]) / r) * r)

    def ensure_static_graph(
        self,
        obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        planner_cfg: dict[str, Any],
        *,
        safety_margin: float,
        uav_radius: float,
    ) -> None:
        ver = obstacle_version_key(obstacles)
        if ver != self.obstacle_version or self.static_graph.nodes.shape[0] == 0:
            if bool(planner_cfg.get("build_static_graph_once", True)):
                self.static_graph.build_static_graph_once(
                    obstacles, bounds, planner_cfg,
                    safety_margin=safety_margin, uav_radius=uav_radius,
                )
            self.obstacle_version = ver
            self.pair_cost_cache.clear()

    def los_clear(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[Obstacle],
        *,
        safety_margin: float,
        uav_radius: float,
    ) -> bool:
        return has_line_of_sight(
            start, goal, obstacles,
            safety_margin=safety_margin, uav_radius=uav_radius,
        )

    def get_pair_cost(
        self,
        i: int,
        j: int,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[Obstacle],
        *,
        safety_margin: float,
        uav_radius: float,
    ) -> tuple[float, bool]:
        """Cached/heuristic pair cost for assignment (no planner by default)."""
        start = np.asarray(start, dtype=np.float64).reshape(2)
        goal = np.asarray(goal, dtype=np.float64).reshape(2)
        euclid = float(np.linalg.norm(goal - start))

        if self.los_clear(start, goal, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
            return euclid, False

        key = (
            int(i), int(j), self.obstacle_version,
            self._round(start, self.cfg.start_rounding_resolution),
            self._round(goal, self.cfg.goal_rounding_resolution),
        )
        cached = self.pair_cost_cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached, True

        agent_st = self.agent_paths.get(i)
        if agent_st is not None and agent_st.slot_id == j and agent_st.path is not None:
            if self.cfg.validate_cached_path and not validate_planned_path(
                agent_st.path,
                obstacles,
                safety_margin=safety_margin,
                uav_radius=uav_radius,
            ):
                self.invalidate_agent(i)
                self.stats["cache_misses"] += 1
                cost = euclid + self.cfg.fallback_blocked_penalty
                self.pair_cost_cache[key] = cost
                return cost, True
            plen = path_length(agent_st.path)
            self.pair_cost_cache[key] = plen
            self.stats["cache_hits"] += 1
            return plen, True

        self.stats["cache_misses"] += 1
        cost = euclid + self.cfg.fallback_blocked_penalty
        self.pair_cost_cache[key] = cost
        return cost, True

    def _path_deviation(self, path: list[np.ndarray], pos: np.ndarray) -> float:
        """Min distance from ``pos`` to the path polyline (not just waypoints)."""
        pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return 0.0
        _, _, cross_signed, _ = closest_point_on_polyline(pts, pos)
        return float(abs(cross_signed))

    def _slot_moved(self, st: AgentPathState, goal: np.ndarray) -> bool:
        if st.goal_xy is None:
            return True
        dg = float(np.linalg.norm(np.asarray(goal).reshape(2) - st.goal_xy.reshape(2)))
        return dg > self.cfg.replan_slot_move_thresh

    def path_diagnostics(
        self,
        agent_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        target_xy: np.ndarray | None = None,
        speed: float | None = None,
    ) -> dict[str, float | bool]:
        st = self.agent_paths.get(int(agent_id))
        if st is None or st.path is None:
            return {
                "path_endpoint_error": float("inf"),
                "path_tracking_error": float("inf"),
                "path_min_clearance": float("nan"),
                "expected_time_to_slot": float("inf"),
                "slot_moved_distance": float("inf"),
                "target_moved_distance": float("inf"),
                "path_is_invalid": True,
            }
        start2 = np.asarray(start, dtype=np.float64).reshape(2)
        goal2 = np.asarray(goal, dtype=np.float64).reshape(2)
        end = np.asarray(st.path[-1], dtype=np.float64).reshape(2)
        slot_move = (
            float(np.linalg.norm(goal2 - st.goal_xy.reshape(2)))
            if st.goal_xy is not None else float("inf")
        )
        target_move = 0.0
        if target_xy is not None:
            if st.target_xy is None:
                target_move = float("inf")
            else:
                target_move = float(np.linalg.norm(np.asarray(target_xy).reshape(2) - st.target_xy.reshape(2)))
        expected_time = (
            float(st.path_length) / max(float(speed), 1e-6)
            if speed is not None and np.isfinite(st.path_length) else float("inf")
        )
        return {
            "path_endpoint_error": float(np.linalg.norm(end - goal2)),
            "path_tracking_error": float(self._path_deviation(st.path, start2)),
            "path_min_clearance": float(st.min_clearance),
            "expected_time_to_slot": expected_time,
            "slot_moved_distance": slot_move,
            "target_moved_distance": target_move,
            "path_is_invalid": not bool(st.feasible),
        }

    def invalidate_agent(self, agent_id: int) -> None:
        """Drop one agent's cached path and pair costs."""
        aid = int(agent_id)
        self.agent_paths.pop(aid, None)
        drop = [k for k in self.pair_cost_cache if int(k[0]) == aid]
        for k in drop:
            del self.pair_cost_cache[k]

    def cached_path_collision_free(
        self,
        agent_id: int,
        validate_obstacles: list[Obstacle],
        *,
        safety_margin: float,
        uav_radius: float,
    ) -> bool:
        """True only when the cached path exists and validates against current obstacles."""
        st = self.agent_paths.get(int(agent_id))
        if st is None or st.path is None:
            return False
        return validate_planned_path(
            st.path,
            validate_obstacles,
            safety_margin=safety_margin,
            uav_radius=uav_radius,
        )

    def cached_path_stale_or_unsafe(
        self,
        agent_id: int,
        slot_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        validate_obstacles: list[Obstacle],
        *,
        safety_margin: float,
        uav_radius: float,
    ) -> bool:
        """Fast per-step guard before reusing a cached path."""
        st = self.agent_paths.get(int(agent_id))
        if st is None or st.path is None:
            return True
        if int(st.slot_id) != int(slot_id):
            return True
        if self.cfg.validate_cached_path and not self.cached_path_collision_free(
            agent_id,
            validate_obstacles,
            safety_margin=safety_margin,
            uav_radius=uav_radius,
        ):
            self.invalidate_agent(agent_id)
            return True
        return False

    def should_replan(
        self,
        agent_id: int,
        slot_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        validate_obstacles: list[Obstacle],
        step_count: int,
        *,
        safety_margin: float,
        uav_radius: float,
        force_slot: bool = False,
        force_projection: bool = False,
        target_xy: np.ndarray | None = None,
        cbf_active_steps: int = 0,
        speed: float | None = None,
    ) -> bool:
        st = self.agent_paths.get(agent_id)
        if st is None or st.path is None:
            return True
        if int(st.slot_id) != int(slot_id):
            return True
        if force_slot or force_projection or self._slot_moved(st, goal):
            return True
        if self.cfg.validate_cached_path and not self.cached_path_collision_free(
            agent_id,
            validate_obstacles,
            safety_margin=safety_margin,
            uav_radius=uav_radius,
        ):
            self.invalidate_agent(agent_id)
            return True
        diag = self.path_diagnostics(agent_id, start, goal, target_xy=target_xy, speed=speed)
        if float(diag["target_moved_distance"]) > self.cfg.replan_target_move_thresh:
            return True
        if float(diag["path_endpoint_error"]) > self.cfg.replan_endpoint_error_thresh:
            return True
        if float(diag["path_min_clearance"]) < self.cfg.replan_min_clearance:
            return True
        if float(diag["path_tracking_error"]) > self.cfg.replan_tracking_error_thresh:
            return True
        if int(cbf_active_steps) >= self.cfg.replan_cbf_active_steps:
            return True
        if float(diag["expected_time_to_slot"]) > self.cfg.replan_time_budget:
            return True
        if bool(diag["path_is_invalid"]):
            return True
        if step_count - st.last_replan_step >= self.cfg.replan_interval:
            return True
        return False

    def _sync_goal_slot(
        self,
        agent_id: int,
        slot_id: int,
        goal: np.ndarray,
        step_count: int,
    ) -> None:
        """Record slot/goal intent without replacing the cached path."""
        st = self.agent_paths.get(agent_id)
        if st is None:
            return
        st.slot_id = int(slot_id)
        st.goal_xy = np.asarray(goal, dtype=np.float64).reshape(2).copy()
        st.last_replan_step = int(step_count)

    def get_or_replan_assigned_path(
        self,
        agent_id: int,
        slot_id: int,
        start: np.ndarray,
        goal: np.ndarray,
        plan_obstacles: list[Obstacle],
        graph_obstacles: list[Obstacle],
        validate_obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        planner_cfg: dict[str, Any],
        step_count: int,
        *,
        safety_margin: float,
        uav_radius: float,
        force: bool = False,
        target_xy: np.ndarray | None = None,
        cbf_active_steps: int = 0,
        speed: float | None = None,
    ) -> tuple[list[np.ndarray] | None, bool, float]:
        """Event-triggered replan with post-plan validation on ``validate_obstacles``."""
        start = np.asarray(start, dtype=np.float64).reshape(2)
        goal = np.asarray(goal, dtype=np.float64).reshape(2)

        def _store(
            path: list[np.ndarray], *, replanned: bool, ms: float
        ) -> tuple[list[np.ndarray] | None, bool, float]:
            min_clear, mean_clear = path_clearance_stats(path, validate_obstacles, uav_radius=uav_radius)
            plen = path_length(path)
            if not validate_planned_path(
                path, validate_obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            ):
                if self.cfg.invalidate_on_plan_failure:
                    self.invalidate_agent(agent_id)
                return None, False, ms
            los_blocked = not self.los_clear(
                start, goal, validate_obstacles,
                safety_margin=safety_margin, uav_radius=uav_radius,
            )
            self.agent_paths[agent_id] = AgentPathState(
                path=path,
                slot_id=slot_id,
                start_xy=start.copy(),
                goal_xy=goal.copy(),
                target_xy=(
                    None if target_xy is None
                    else np.asarray(target_xy, dtype=np.float64).reshape(2).copy()
                ),
                obstacle_version=tuple(self.obstacle_version),
                path_length=plen,
                min_clearance=min_clear,
                mean_clearance=mean_clear,
                planned_step=step_count,
                last_replan_step=step_count,
                los_blocked=los_blocked,
                feasible=True,
            )
            key = (
                int(agent_id), int(slot_id), self.obstacle_version,
                self._round(start, self.cfg.start_rounding_resolution),
                self._round(goal, self.cfg.goal_rounding_resolution),
            )
            self.pair_cost_cache[key] = plen
            if replanned:
                self.stats["replan_count"] += 1
                self.stats["replan_time_ms"].append(ms)
            return path, replanned, ms

        def _existing_valid_path() -> list[np.ndarray] | None:
            st = self.agent_paths.get(agent_id)
            if st is None or st.path is None:
                return None
            if validate_planned_path(
                st.path, validate_obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            ):
                return st.path
            if self.cfg.invalidate_on_plan_failure:
                self.invalidate_agent(agent_id)
            return None

        if not force and not self.should_replan(
            agent_id, slot_id, start, goal, validate_obstacles, step_count,
            safety_margin=safety_margin, uav_radius=uav_radius,
            target_xy=target_xy, cbf_active_steps=cbf_active_steps, speed=speed,
        ):
            st = self.agent_paths.get(agent_id)
            self.stats["cache_hits"] += 1
            return st.path, False, 0.0

        # Event-triggered replans are cache misses even when the replacement is
        # a direct LOS segment; otherwise dynamic scenes misleadingly report a
        # perfect cache hit rate while paths are being refreshed.
        self.stats["cache_misses"] += 1

        if self.los_clear(start, goal, validate_obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
            candidate = [start.copy(), goal.copy()]
            if validate_planned_path(
                candidate, validate_obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            ):
                return _store(candidate, replanned=bool(force), ms=0.0)

        self.ensure_static_graph(
            graph_obstacles, bounds, planner_cfg,
            safety_margin=safety_margin, uav_radius=uav_radius,

        )
        timeout_ms = float(planner_cfg.get("planner_timeout_ms", 5.0))
        path, ms, timed_out = self.static_graph.query_path(
            start, goal, plan_obstacles, planner_cfg,
            safety_margin=safety_margin, uav_radius=uav_radius,
            timeout_ms=timeout_ms,
        )
        if timed_out:
            self.stats["timeout_count"] += 1
            existing = _existing_valid_path()
            if existing is not None:
                return existing, False, ms
            return None, False, ms

        if path is None:
            existing = _existing_valid_path()
            if existing is not None:
                return existing, False, ms
            return None, False, ms

        if not validate_planned_path(
            path, validate_obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
        ):
            path2, ms2, _ = self.static_graph.query_path(
                start, goal, graph_obstacles, planner_cfg,
                safety_margin=safety_margin, uav_radius=uav_radius,
                timeout_ms=timeout_ms,
            )
            ms = ms + ms2
            if path2 is not None and validate_planned_path(
                path2, validate_obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            ):
                path = path2
            elif self.cfg.invalidate_on_plan_failure:
                self.invalidate_agent(agent_id)
                return None, False, ms

        return _store(path, replanned=True, ms=ms)

    def invalidate_slot(self, slot_id: int) -> None:
        """Drop cached paths/costs tied to a slot that slid on the manifold."""
        sid = int(slot_id)
        for aid in list(self.agent_paths.keys()):
            st = self.agent_paths.get(aid)
            if st is not None and int(st.slot_id) == sid:
                del self.agent_paths[aid]
        drop = [k for k in self.pair_cost_cache if int(k[1]) == sid]
        for k in drop:
            del self.pair_cost_cache[k]

    def get_agent_path(self, agent_id: int) -> list[np.ndarray] | None:
        st = self.agent_paths.get(agent_id)
        return None if st is None else st.path

    @property
    def hit_rate(self) -> float:
        hits = self.stats["cache_hits"]
        misses = self.stats["cache_misses"]
        total = hits + misses
        return float(hits / total) if total > 0 else 1.0
