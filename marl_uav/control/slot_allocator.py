"""Slot allocation for the E2 debug trajectory planner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight
from marl_uav.framework.planning.local_reachability import (
    LocalReachabilityScoringConfig,
    local_reachability_probe,
)
from marl_uav.framework.planning.turn_radius_obstacle_query import TurnRadiusObstacleQueryConfig
from marl_uav.framework.role_allocation import (
    default_ot_epsilon,
    entropic_ot_assignment,
    sinkhorn_transport_plan,
)


@dataclass(frozen=True)
class SlotAllocatorConfig:
    ot_epsilon: float = 0.05
    ot_epsilon_scale: float | None = 0.25
    ot_sinkhorn_iterations: int = 25
    assignment_inertia_margin: float = 0.05
    los_penalty: float = 0.0
    safety_margin: float = 0.30
    uav_radius: float = 0.15
    w_reach_block: float = 0.0
    w_clearance: float = 0.0
    w_path: float = 0.0
    world_xy: float | None = None
    boundary_margin: float = 0.30
    reachability: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SlotAllocatorConfig":
        d = dict(raw or {})
        eps_scale = d.get("ot_epsilon_scale", cls.ot_epsilon_scale)
        world_xy = d.get("world_xy", cls.world_xy)
        reach_raw = d.get("reachability")
        return cls(
            ot_epsilon=float(d.get("ot_epsilon", cls.ot_epsilon)),
            ot_epsilon_scale=None if eps_scale is None else float(eps_scale),
            ot_sinkhorn_iterations=int(d.get("ot_sinkhorn_iterations", cls.ot_sinkhorn_iterations)),
            assignment_inertia_margin=float(
                d.get("assignment_inertia_margin", cls.assignment_inertia_margin)
            ),
            los_penalty=float(d.get("los_penalty", cls.los_penalty)),
            safety_margin=float(d.get("safety_margin", cls.safety_margin)),
            uav_radius=float(d.get("uav_radius", cls.uav_radius)),
            w_reach_block=float(d.get("w_reach_block", cls.w_reach_block)),
            w_clearance=float(d.get("w_clearance", cls.w_clearance)),
            w_path=float(d.get("w_path", cls.w_path)),
            world_xy=None if world_xy is None else float(world_xy),
            boundary_margin=float(d.get("boundary_margin", cls.boundary_margin)),
            reachability=None if reach_raw is None else dict(reach_raw),
        )

    def reachability_query_cfg(self) -> TurnRadiusObstacleQueryConfig:
        raw = dict(self.reachability or {})
        raw.setdefault("safety_margin", self.safety_margin)
        raw.setdefault("uav_radius", self.uav_radius)
        return TurnRadiusObstacleQueryConfig.from_dict(raw)

    def reachability_scoring_cfg(self) -> LocalReachabilityScoringConfig:
        raw = dict(self.reachability or {})
        raw.setdefault("world_xy", self.world_xy)
        raw.setdefault("boundary_margin", self.boundary_margin)
        return LocalReachabilityScoringConfig.from_dict(raw)


class SlotAllocator:
    """Entropic-OT hard assignment from pursuers to manifold slots."""

    def __init__(self, cfg: dict[str, Any] | SlotAllocatorConfig | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, SlotAllocatorConfig) else SlotAllocatorConfig.from_dict(cfg)
        self.previous_assignment: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_assignment = None

    def allocate(
        self,
        pursuer_pos: np.ndarray,
        slot_targets: np.ndarray,
        obstacles: list[Any] | None = None,
        *,
        pursuer_yaws: np.ndarray | None = None,
        world_xy: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
        s = np.asarray(slot_targets, dtype=np.float64).reshape(3, 3)
        cost = np.linalg.norm(p[:, None, :2] - s[None, :, :2], axis=2)

        los_blocked = np.zeros_like(cost, dtype=bool)
        if self.cfg.los_penalty > 0.0 and obstacles:
            for i in range(3):
                for j in range(3):
                    clear = has_line_of_sight(
                        p[i, :2],
                        s[j, :2],
                        obstacles,
                        safety_margin=self.cfg.safety_margin,
                        uav_radius=self.cfg.uav_radius,
                    )
                    los_blocked[i, j] = not bool(clear)
            cost = cost + los_blocked.astype(np.float64) * float(self.cfg.los_penalty)

        reach_blocked = np.zeros_like(cost, dtype=bool)
        reach_costs = np.zeros_like(cost, dtype=np.float64)
        min_clearances = np.full_like(cost, np.inf, dtype=np.float64)
        use_reach = (
            self.cfg.w_reach_block > 0.0 or self.cfg.w_clearance > 0.0 or self.cfg.w_path > 0.0
        )
        if use_reach and obstacles is not None:
            yaws = (
                np.zeros(3, dtype=np.float64)
                if pursuer_yaws is None
                else np.asarray(pursuer_yaws, dtype=np.float64).reshape(3)
            )
            query_cfg = self.cfg.reachability_query_cfg()
            scoring_cfg = self.cfg.reachability_scoring_cfg()
            if world_xy is not None:
                scoring_cfg = replace(scoring_cfg, world_xy=float(world_xy))
            for i in range(3):
                for j in range(3):
                    reach = local_reachability_probe(
                        p[i, :2],
                        float(yaws[i]),
                        s[j, :2],
                        obstacles,
                        cfg=query_cfg,
                        scoring=scoring_cfg,
                    )
                    reach_blocked[i, j] = bool(reach.blocked)
                    reach_costs[i, j] = float(reach.best_cost)
                    min_clearances[i, j] = float(reach.min_clearance)
                    cost[i, j] += float(self.cfg.w_reach_block) * float(reach.blocked)
                    cost[i, j] += float(self.cfg.w_clearance) / max(float(reach.min_clearance), 1e-3)
                    cost[i, j] += float(self.cfg.w_path) * float(reach.best_cost)

        eps = default_ot_epsilon(cost, self.cfg.ot_epsilon, self.cfg.ot_epsilon_scale)
        plan = sinkhorn_transport_plan(
            cost,
            epsilon=eps,
            num_iters=self.cfg.ot_sinkhorn_iterations,
        )
        assignment = entropic_ot_assignment(
            cost,
            epsilon=eps,
            num_iters=self.cfg.ot_sinkhorn_iterations,
            prev_assignment=self.previous_assignment,
            inertia_margin=self.cfg.assignment_inertia_margin,
        )
        self.previous_assignment = np.asarray(assignment, dtype=np.int64).reshape(3).copy()

        assigned_targets = s[assignment].astype(np.float32)
        diag = {
            "cost_matrix": cost.astype(float).tolist(),
            "transport_plan": plan.astype(float).tolist(),
            "ot_epsilon": float(eps),
            "role_assignment": assignment.astype(int).tolist(),
            "los_blocked_matrix": los_blocked.astype(bool).tolist(),
            "reach_blocked_matrix": reach_blocked.astype(bool).tolist(),
            "reachability_cost_matrix": reach_costs.astype(float).tolist(),
            "reach_min_clearance_matrix": min_clearances.astype(float).tolist(),
        }
        return assignment.astype(np.int64), assigned_targets, diag
