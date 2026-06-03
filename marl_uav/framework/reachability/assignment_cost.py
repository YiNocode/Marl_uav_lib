"""Reachability-aware assignment cost matrices (deployable + offline oracle)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_query import ObstacleQueryConfig, select_validation_obstacles
from marl_uav.framework.planning.path_cache import DeployPathCache
from marl_uav.framework.reachability.los_cost import build_los_cost_matrix
from marl_uav.framework.reachability.structure_assignment_cost import (
    StructureAssignmentConfig,
    build_structure_assignment_cost_matrix,
    select_structure_assignment,
)

INF_COST = 1e9


@dataclass
class ReachabilityConfig:
    """Configuration for obstacle-aware assignment costs."""

    enabled: bool = False
    mode: str = "euclidean"  # euclidean | los | cached_path | structure | structure_los | structure_cached_path | reachability_candidate_path
    safety_margin: float = 0.3
    uav_radius: float = 0.15
    obstacle_mode: str = "global"  # global | local
    local_obstacle_radius: float = 20.0
    local_obstacle_top_k: int | None = None
    los_block_penalty: float = 100.0
    use_infinite_los_block: bool = False
    switch_penalty: float = 0.2
    path_planner: dict[str, Any] = field(default_factory=dict)
    path_tracking: dict[str, Any] = field(default_factory=dict)
    path_cache: dict[str, Any] = field(default_factory=dict)
    obstacle_query: dict[str, Any] = field(default_factory=dict)
    structure_assignment: dict[str, Any] = field(default_factory=dict)
    structure_slot_selection: dict[str, Any] = field(default_factory=dict)
    slot_projection: dict[str, Any] = field(default_factory=dict)
    pair_reachability: dict[str, Any] = field(default_factory=dict)
    assignment: dict[str, Any] = field(default_factory=dict)
    candidate_slots: dict[str, Any] = field(default_factory=dict)
    assignment_cost: dict[str, Any] = field(default_factory=dict)
    structure_selection: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> ReachabilityConfig:
        raw = dict(cfg or {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "euclidean")).strip().lower(),
            safety_margin=float(raw.get("safety_margin", 0.3)),
            uav_radius=float(raw.get("uav_radius", 0.15)),
            obstacle_mode=str(raw.get("obstacle_mode", "global")).strip().lower(),
            local_obstacle_radius=float(raw.get("local_obstacle_radius", 20.0)),
            local_obstacle_top_k=(
                None if raw.get("local_obstacle_top_k", None) is None
                else int(raw.get("local_obstacle_top_k"))
            ),
            los_block_penalty=float(raw.get("los_block_penalty", 100.0)),
            use_infinite_los_block=bool(raw.get("use_infinite_los_block", False)),
            switch_penalty=float(raw.get("switch_penalty", 0.2)),
            path_planner=dict(raw.get("path_planner") or {}),
            path_tracking=dict(raw.get("path_tracking") or {}),
            path_cache=dict(raw.get("path_cache") or {}),
            obstacle_query=dict(raw.get("obstacle_query") or {}),
            structure_assignment=dict(raw.get("structure_assignment") or {}),
            structure_slot_selection=dict(raw.get("structure_slot_selection") or {}),
            slot_projection=dict(raw.get("slot_projection") or {}),
            pair_reachability=dict(raw.get("pair_reachability") or {}),
            assignment=dict(raw.get("assignment") or {}),
            candidate_slots=dict(raw.get("candidate_slots") or {}),
            assignment_cost=dict(raw.get("assignment_cost") or {}),
            structure_selection=dict(raw.get("structure_selection") or {}),
        )

    def structure_cfg(self) -> StructureAssignmentConfig:
        return StructureAssignmentConfig.from_dict(self.structure_assignment)

    def exclude_unreachable_pairs(self) -> bool:
        return bool(self.pair_reachability.get("exclude_unreachable", True))

    def assignment_method(self) -> str:
        return str(self.assignment.get("method", "permute")).strip().lower()


def build_cached_path_cost_matrix(
    pursuer_positions: np.ndarray,
    slots: np.ndarray,
    obstacles: list,
    previous_assignment: np.ndarray | None,
    path_cache: DeployPathCache,
    cfg: ReachabilityConfig,
    *,
    safety_margin: float,
    uav_radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assignment cost using LOS + cached/heuristic pair costs (corridor/full validation)."""
    import time

    t0 = time.perf_counter()
    p = np.asarray(pursuer_positions, dtype=np.float64).reshape(-1, 3)[:, :2]
    s = np.asarray(slots, dtype=np.float64).reshape(-1, 3)[:, :2]
    n_p, n_s = int(p.shape[0]), int(s.shape[0])
    cost = np.zeros((n_p, n_s), dtype=np.float64)
    los_blocked = np.zeros((n_p, n_s), dtype=bool)
    oq = ObstacleQueryConfig.from_dict(cfg.obstacle_query)

    for i in range(n_p):
        for j in range(n_s):
            obs_ij = select_validation_obstacles(p[i], s[j], obstacles, oq)
            c, blocked = path_cache.get_pair_cost(
                i, j, p[i], s[j], obs_ij,
                safety_margin=safety_margin, uav_radius=uav_radius,
            )
            cost[i, j] = c
            los_blocked[i, j] = blocked

    if previous_assignment is not None:
        prev = np.asarray(previous_assignment, dtype=np.int64).reshape(-1)
        if prev.shape[0] == n_p:
            for i in range(n_p):
                for j in range(n_s):
                    if int(prev[i]) != j:
                        cost[i, j] += cfg.switch_penalty

    ms = (time.perf_counter() - t0) * 1000.0
    return cost, {
        "assignment_cost_matrix": cost.copy(),
        "los_blocked_matrix": los_blocked.copy(),
        "num_los_blocked_pairs": int(np.sum(los_blocked)),
        "cost_matrix_time_ms": ms,
    }


def build_structure_assignment(
    pursuer_positions: np.ndarray,
    slots: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list,
    previous_assignment: np.ndarray | None,
    cfg: ReachabilityConfig,
    *,
    path_cache: DeployPathCache | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Structure-first assignment (D_ang, C_cov, C_col + reachability)."""
    scfg = cfg.structure_cfg()
    assign, diag = select_structure_assignment(
        pursuer_positions,
        slots,
        evader_pos,
        obstacles,
        previous_assignment,
        scfg,
        safety_margin=cfg.safety_margin,
        uav_radius=cfg.uav_radius,
        switch_penalty=cfg.switch_penalty,
        path_cache=path_cache,
        exclude_unreachable=cfg.exclude_unreachable_pairs(),
        method=cfg.assignment_method(),
    )
    return assign, diag


def build_reachability_cost_matrix(
    pursuer_positions: np.ndarray,
    slots: np.ndarray,
    obstacles: list,
    previous_assignment: np.ndarray | None,
    cfg: ReachabilityConfig | dict[str, Any],
    *,
    evader_pos: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    path_cache: DeployPathCache | None = None,
    planner_timing: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch to structure, LOS, or cached-path cost builders."""
    del bounds, planner_timing
    rcfg = cfg if isinstance(cfg, ReachabilityConfig) else ReachabilityConfig.from_dict(cfg)
    mode = rcfg.mode if rcfg.enabled else "euclidean"

    if mode.startswith("structure"):
        if evader_pos is None:
            raise ValueError("structure assignment requires evader_pos")
        assign, diag = build_structure_assignment(
            pursuer_positions, slots, evader_pos, obstacles, previous_assignment, rcfg,
            path_cache=path_cache,
        )
        cost = diag.get("assignment_cost_matrix")
        if cost is None:
            cost, _ = build_structure_assignment_cost_matrix(
                pursuer_positions, slots, evader_pos, obstacles, previous_assignment,
                rcfg.structure_cfg(),
                safety_margin=rcfg.safety_margin,
                uav_radius=rcfg.uav_radius,
                switch_penalty=rcfg.switch_penalty,
                path_cache=path_cache,
                exclude_unreachable=rcfg.exclude_unreachable_pairs(),
            )
        diag["_structure_assignment"] = assign
        return np.asarray(cost, dtype=np.float64), diag

    if mode in ("los", "euclidean"):
        return build_los_cost_matrix(
            pursuer_positions,
            slots,
            obstacles,
            previous_assignment,
            safety_margin=rcfg.safety_margin,
            uav_radius=rcfg.uav_radius,
            los_block_penalty=rcfg.los_block_penalty,
            use_infinite_los_block=rcfg.use_infinite_los_block,
            switch_penalty=rcfg.switch_penalty,
        )

    if mode in ("cached_path", "path") and path_cache is not None:
        return build_cached_path_cost_matrix(
            pursuer_positions,
            slots,
            obstacles,
            previous_assignment,
            path_cache,
            rcfg,
            safety_margin=rcfg.safety_margin,
            uav_radius=rcfg.uav_radius,
        )

    return build_los_cost_matrix(
        pursuer_positions, slots, obstacles, previous_assignment,
        safety_margin=rcfg.safety_margin,
        uav_radius=rcfg.uav_radius,
        los_block_penalty=rcfg.los_block_penalty,
        use_infinite_los_block=rcfg.use_infinite_los_block,
        switch_penalty=rcfg.switch_penalty,
    )
