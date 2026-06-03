"""Structure-aware assignment costs for deployable SCE baselines.

Assignment minimizes loss in encirclement structure (C_cov, C_col, D_ang)
subject to slot reachability and obstacle clearance — not Euclidean travel distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task import compute_pursuit_structure_metrics_3v1
from marl_uav.framework.geometry.obstacle_geometry import has_line_of_sight, path_length
from marl_uav.framework.planning.path_cache import DeployPathCache
from marl_uav.framework.planning.path_validation import validate_planned_path

INF_COST = 1e9


@dataclass
class StructureAssignmentConfig:
    """Weights for structure-first OT / Hungarian assignment."""

    w_cov: float = 1.0
    w_col: float = 1.0
    w_ang: float = 1.0
    structure_scale: float = 10.0
    w_travel: float = 0.05
    use_path_travel_tiebreak: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> StructureAssignmentConfig:
        d = dict(raw or {})
        return cls(
            w_cov=float(d.get("w_cov", 1.0)),
            w_col=float(d.get("w_col", 1.0)),
            w_ang=float(d.get("w_ang", 1.0)),
            structure_scale=float(d.get("structure_scale", 10.0)),
            w_travel=float(d.get("w_travel", 0.05)),
            use_path_travel_tiebreak=bool(d.get("use_path_travel_tiebreak", True)),
        )


def structure_score(metrics: dict[str, float], cfg: StructureAssignmentConfig) -> float:
    """Higher is better: coverage + angular uniformity + low collapse."""
    c_cov = float(metrics.get("C_cov", 0.0))
    c_col = float(metrics.get("C_col", 0.0))
    d_ang = float(metrics.get("D_ang", 0.0))
    return (
        cfg.w_cov * c_cov
        + cfg.w_ang * d_ang
        + cfg.w_col * (1.0 - c_col)
    )


def formation_structure_score(
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    cfg: StructureAssignmentConfig,
) -> float:
    metrics = compute_pursuit_structure_metrics_3v1(pursuer_pos, evader_pos)
    return structure_score(metrics, cfg)


def marginal_structure_gain(
    pursuer_idx: int,
    slot_idx: int,
    pursuer_pos: np.ndarray,
    slots: np.ndarray,
    evader_pos: np.ndarray,
    cfg: StructureAssignmentConfig,
) -> float:
    """
    Structure improvement if pursuer ``pursuer_idx`` moves to ``slots[slot_idx]``
    (other pursuers held at current positions).
    """
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    s = np.asarray(slots, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    base = formation_structure_score(p, e, cfg)
    hyp = p.copy()
    hyp[int(pursuer_idx), :2] = s[int(slot_idx), :2]
    new = formation_structure_score(hyp, e, cfg)
    return float(new - base)


def assignment_structure_score(
    pursuer_pos: np.ndarray,
    slots: np.ndarray,
    assignment: np.ndarray,
    evader_pos: np.ndarray,
    cfg: StructureAssignmentConfig,
) -> float:
    """Evaluate structure when each pursuer occupies its assigned slot."""
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3).copy()
    s = np.asarray(slots, dtype=np.float64).reshape(3, 3)
    assign = np.asarray(assignment, dtype=np.int64).reshape(3)
    for i in range(3):
        p[i, :2] = s[int(assign[i]), :2]
    return formation_structure_score(p, evader_pos, cfg)


def pair_reachable(
    pursuer_xy: np.ndarray,
    slot_xy: np.ndarray,
    obstacles: list,
    *,
    safety_margin: float,
    uav_radius: float,
    path_cache: DeployPathCache | None = None,
    agent_idx: int | None = None,
    slot_idx: int | None = None,
) -> bool:
    p = np.asarray(pursuer_xy, dtype=np.float64).reshape(2)
    g = np.asarray(slot_xy, dtype=np.float64).reshape(2)
    if has_line_of_sight(p, g, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
        return True
    if path_cache is None or agent_idx is None or slot_idx is None:
        return False
    st = path_cache.agent_paths.get(int(agent_idx))
    if (
        st is not None
        and int(st.slot_id) == int(slot_idx)
        and st.path is not None
        and validate_planned_path(
            st.path, obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
        )
    ):
        return True
    return False


def build_pair_reachability_matrix(
    pursuer_pos: np.ndarray,
    slots: np.ndarray,
    obstacles: list,
    *,
    safety_margin: float,
    uav_radius: float,
    path_cache: DeployPathCache | None = None,
) -> np.ndarray:
    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)[:, :2]
    s = np.asarray(slots, dtype=np.float64).reshape(3, 3)[:, :2]
    out = np.zeros((3, 3), dtype=bool)
    for i in range(3):
        for j in range(3):
            out[i, j] = pair_reachable(
                p[i], s[j], obstacles,
                safety_margin=safety_margin, uav_radius=uav_radius,
                path_cache=path_cache, agent_idx=i, slot_idx=j,
            )
    return out


def _travel_tiebreak(
    i: int,
    j: int,
    pursuer_xy: np.ndarray,
    slot_xy: np.ndarray,
    obstacles: list,
    path_cache: DeployPathCache | None,
    *,
    safety_margin: float,
    uav_radius: float,
    cfg: StructureAssignmentConfig,
) -> float:
    if not cfg.use_path_travel_tiebreak or cfg.w_travel <= 0.0:
        return 0.0
    p = np.asarray(pursuer_xy, dtype=np.float64).reshape(2)
    g = np.asarray(slot_xy, dtype=np.float64).reshape(2)
    if path_cache is not None:
        st = path_cache.agent_paths.get(i)
        if st is not None and int(st.slot_id) == j and st.path is not None:
            if validate_planned_path(
                st.path, obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            ):
                return cfg.w_travel * float(path_length(st.path))
            if getattr(path_cache.cfg, "invalidate_on_plan_failure", True):
                path_cache.invalidate_agent(i)
    if has_line_of_sight(p, g, obstacles, safety_margin=safety_margin, uav_radius=uav_radius):
        return cfg.w_travel * float(np.linalg.norm(g - p))
    return cfg.w_travel * float(np.linalg.norm(g - p))


def build_structure_assignment_cost_matrix(
    pursuer_positions: np.ndarray,
    slots: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list,
    previous_assignment: np.ndarray | None,
    cfg: StructureAssignmentConfig,
    *,
    safety_margin: float,
    uav_radius: float,
    switch_penalty: float,
    path_cache: DeployPathCache | None = None,
    exclude_unreachable: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build C_ij for OT/Hungarian:

    cost = -structure_scale * Δstructure(i,j) + w_travel * travel_tiebreak + switch
    unreachable pairs -> INF
    """
    import time

    t0 = time.perf_counter()
    p = np.asarray(pursuer_positions, dtype=np.float64).reshape(3, 3)
    s = np.asarray(slots, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    reachable = build_pair_reachability_matrix(
        p, s, obstacles,
        safety_margin=safety_margin, uav_radius=uav_radius,
        path_cache=path_cache,
    )
    cost = np.zeros((3, 3), dtype=np.float64)
    gains = np.zeros((3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            if exclude_unreachable and not reachable[i, j]:
                cost[i, j] = INF_COST
                continue
            gain = marginal_structure_gain(i, j, p, s, e, cfg)
            gains[i, j] = gain
            travel = _travel_tiebreak(
                i, j, p[i, :2], s[j, :2], obstacles, path_cache,
                safety_margin=safety_margin, uav_radius=uav_radius, cfg=cfg,
            )
            cost[i, j] = -cfg.structure_scale * gain + travel

    if previous_assignment is not None:
        prev = np.asarray(previous_assignment, dtype=np.int64).reshape(3)
        for i in range(3):
            for j in range(3):
                if int(prev[i]) != j and cost[i, j] < INF_COST * 0.5:
                    cost[i, j] += switch_penalty

    ms = (time.perf_counter() - t0) * 1000.0
    base_metrics = compute_pursuit_structure_metrics_3v1(p, e)
    return cost, {
        "assignment_cost_matrix": cost.copy(),
        "structure_gain_matrix": gains.copy(),
        "pair_reachable_matrix": reachable.copy(),
        "num_unreachable_pairs": int(np.sum(~reachable)),
        "base_C_cov": float(base_metrics["C_cov"]),
        "base_C_col": float(base_metrics["C_col"]),
        "base_D_ang": float(base_metrics["D_ang"]),
        "cost_matrix_time_ms": ms,
    }


def select_structure_assignment(
    pursuer_positions: np.ndarray,
    slots: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list,
    previous_assignment: np.ndarray | None,
    cfg: StructureAssignmentConfig,
    *,
    safety_margin: float,
    uav_radius: float,
    switch_penalty: float,
    path_cache: DeployPathCache | None = None,
    exclude_unreachable: bool = True,
    method: str = "permute",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Pick assignment maximizing encirclement structure under reachability.

    For 3v3, ``permute`` evaluates all bijections (exact joint structure objective).
    Falls back to marginal OT matrix when no fully reachable permutation exists.
    """
    import time
    from itertools import permutations

    t0 = time.perf_counter()
    p = np.asarray(pursuer_positions, dtype=np.float64).reshape(3, 3)
    s = np.asarray(slots, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    reachable = build_pair_reachability_matrix(
        p, s, obstacles,
        safety_margin=safety_margin, uav_radius=uav_radius,
        path_cache=path_cache,
    )
    cost_matrix, diag = build_structure_assignment_cost_matrix(
        p, s, e, obstacles, previous_assignment, cfg,
        safety_margin=safety_margin, uav_radius=uav_radius,
        switch_penalty=switch_penalty, path_cache=path_cache,
        exclude_unreachable=exclude_unreachable,
    )

    best_assign: np.ndarray | None = None
    best_cost = float(INF_COST)
    best_score = -float(INF_COST)

    if method == "permute":
        for perm in permutations(range(3)):
            if exclude_unreachable and not all(reachable[i, perm[i]] for i in range(3)):
                continue
            assign = np.array(perm, dtype=np.int64)
            score = assignment_structure_score(p, s, assign, e, cfg)
            cost = -cfg.structure_scale * score
            for i in range(3):
                cost += _travel_tiebreak(
                    i, int(perm[i]), p[i, :2], s[int(perm[i]), :2], obstacles, path_cache,
                    safety_margin=safety_margin, uav_radius=uav_radius, cfg=cfg,
                )
            if previous_assignment is not None:
                prev = np.asarray(previous_assignment, dtype=np.int64).reshape(3)
                for i in range(3):
                    if int(prev[i]) != int(perm[i]):
                        cost += switch_penalty
            if cost < best_cost:
                best_cost = cost
                best_score = score
                best_assign = assign

    if best_assign is None:
        from marl_uav.framework.role_allocation import default_ot_epsilon, entropic_ot_assignment

        eps = default_ot_epsilon(cost_matrix, 0.05, 0.25)
        best_assign = entropic_ot_assignment(
            cost_matrix, epsilon=eps, num_iters=25,
            prev_assignment=previous_assignment, inertia_margin=0.05,
        )
        best_score = assignment_structure_score(p, s, best_assign, e, cfg)
        if exclude_unreachable and any(
            not reachable[i, int(best_assign[i])] for i in range(3)
        ):
            if previous_assignment is not None:
                best_assign = np.asarray(previous_assignment, dtype=np.int64).reshape(3).copy()
            else:
                best_assign = np.arange(3, dtype=np.int64)
            best_score = assignment_structure_score(p, s, best_assign, e, cfg)

    ms = (time.perf_counter() - t0) * 1000.0
    assigned_metrics = compute_pursuit_structure_metrics_3v1(
        np.stack([s[int(best_assign[i]), :].copy() for i in range(3)], axis=0),
        e,
    )
    diag.update({
        "assignment_method": method,
        "assigned_structure_score": float(best_score),
        "assigned_C_cov": float(assigned_metrics["C_cov"]),
        "assigned_C_col": float(assigned_metrics["C_col"]),
        "assigned_D_ang": float(assigned_metrics["D_ang"]),
        "assignment_time_ms": ms,
        "assignment_objective": "structure",
    })
    return best_assign, diag
