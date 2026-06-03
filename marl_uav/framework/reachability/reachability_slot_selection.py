"""Reachability-aware slot generation and assignment for obstacle-rich SCE.

LOS is only a visibility predicate; it does not tell us whether a pursuer can
track a dynamically moving slot without skimming obstacles.  This module first
selects three actual encirclement slots from many candidates, then assigns
pursuers to those actual slots using path/obstacle-aware pair costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np

from marl_uav.envs.tasks.pursuit_evasion_3v1_task import compute_pursuit_structure_metrics_3v1
from marl_uav.framework.geometry.obstacle_geometry import Obstacle, has_line_of_sight, path_length
from marl_uav.framework.planning.path_cache import DeployPathCache, path_clearance_stats
from marl_uav.framework.planning.path_tracking import points_min_boundary_clearance

INF_COST = 1.0e12


@dataclass
class SlotCandidate:
    pos: np.ndarray
    theta: float
    angle_index: int
    radius: float
    min_obstacle_clearance: float
    is_inside_obstacle: bool
    obstacle_risk: float
    los_to_evader_blocked: bool


@dataclass
class ReachabilitySlotConfig:
    num_candidate_slots: int = 12
    slot_radius_candidates: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0)
    slot_min_clearance: float = 0.8
    slot_boundary_margin: float = 0.8
    allow_los_blocked_slots: bool = True
    max_candidates: int = 48
    preserve_all_angles: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ReachabilitySlotConfig:
        d = dict(raw or {})
        radii = d.get("slot_radius_candidates", (2.5, 3.0, 3.5, 4.0))
        return cls(
            num_candidate_slots=max(int(d.get("num_candidate_slots", 12)), 3),
            slot_radius_candidates=tuple(float(r) for r in radii),
            slot_min_clearance=float(d.get("slot_min_clearance", 0.8)),
            slot_boundary_margin=float(d.get("slot_boundary_margin", 0.8)),
            allow_los_blocked_slots=bool(d.get("allow_los_blocked_slots", True)),
            max_candidates=max(int(d.get("max_candidates", 48)), 3),
            preserve_all_angles=bool(d.get("preserve_all_angles", True)),
        )


@dataclass
class AssignmentCostConfig:
    w_path: float = 1.0
    w_time: float = 0.5
    w_risk: float = 2.0
    w_clearance: float = 2.0
    w_los: float = 0.3
    w_switch: float = 0.5
    w_turn: float = 1.0
    turn_min_radius: float = 1.0
    turn_min_clearance: float = 0.6
    infeasible_cost: float = 1.0e6

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AssignmentCostConfig:
        d = dict(raw or {})
        return cls(
            w_path=float(d.get("w_path", 1.0)),
            w_time=float(d.get("w_time", 0.5)),
            w_risk=float(d.get("w_risk", 2.0)),
            w_clearance=float(d.get("w_clearance", 2.0)),
            w_los=float(d.get("w_los", 0.3)),
            w_switch=float(d.get("w_switch", 0.5)),
            w_turn=float(d.get("w_turn", 1.0)),
            turn_min_radius=float(d.get("turn_min_radius", 1.0)),
            turn_min_clearance=float(d.get("turn_min_clearance", 0.6)),
            infeasible_cost=float(d.get("infeasible_cost", 1.0e6)),
        )


@dataclass
class StructureSelectionConfig:
    w_D_ang: float = 1.0
    w_C_cov: float = 1.0
    w_C_col: float = 1.0
    w_gap: float = 1.5
    w_assignment_feasibility: float = 2.0
    w_slot_risk: float = 0.0
    structure_violation_weight: float = 5.0
    min_D_ang: float = 0.35
    min_C_cov: float = 0.35
    max_C_col: float = 0.70
    preserve_assignment_until_unreachable: bool = True
    max_slot_combinations: int = 500

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> StructureSelectionConfig:
        d = dict(raw or {})
        return cls(
            w_D_ang=float(d.get("w_D_ang", 1.0)),
            w_C_cov=float(d.get("w_C_cov", 1.0)),
            w_C_col=float(d.get("w_C_col", 1.0)),
            w_gap=float(d.get("w_gap", 1.5)),
            w_assignment_feasibility=float(d.get("w_assignment_feasibility", 2.0)),
            # Kept for backward-compatible configs, but defaulted to zero:
            # the risky object is the arrival path/turn, not an abstract slot
            # after the slot has passed hard obstacle/boundary filters.
            w_slot_risk=float(d.get("w_slot_risk", 0.0)),
            structure_violation_weight=float(d.get("structure_violation_weight", 5.0)),
            min_D_ang=float(d.get("min_D_ang", 0.35)),
            min_C_cov=float(d.get("min_C_cov", 0.35)),
            max_C_col=float(d.get("max_C_col", 0.70)),
            preserve_assignment_until_unreachable=bool(d.get("preserve_assignment_until_unreachable", True)),
            max_slot_combinations=max(int(d.get("max_slot_combinations", 500)), 1),
        )


def _slot_clearance_and_risk(
    xy: np.ndarray,
    obstacles: list[Obstacle],
    *,
    uav_radius: float,
) -> tuple[float, bool, float]:
    if not obstacles:
        return float("inf"), False, 0.0
    p = np.asarray(xy, dtype=np.float64).reshape(2)
    best = float("inf")
    risk = 0.0
    inside = False
    for obs in obstacles:
        if obs.kind != "circle":
            continue
        c = np.asarray(obs.center, dtype=np.float64).reshape(2)
        clearance = float(np.linalg.norm(p - c)) - float(obs.radius) - float(uav_radius)
        best = min(best, clearance)
        if clearance < 0.0:
            inside = True
        risk += 1.0 / max(clearance + 1.0, 0.05)
    return best, inside, float(risk)


def generate_candidate_slots(
    evader_pos: np.ndarray,
    pursuer_positions: np.ndarray,
    obstacles: list[Obstacle],
    cfg: ReachabilitySlotConfig | dict[str, Any] | None,
    *,
    world_xy: float,
    safety_margin: float,
    uav_radius: float,
) -> list[SlotCandidate]:
    """Generate a ring of obstacle-aware candidate slots around the evader.

    Fixed three-slot geometry is brittle in obstacle fields: one ideal slot may
    sit behind a pillar or inside a narrow passage.  We sample multiple radii and
    angles, filter physically unsafe candidates, and leave structure choice to a
    later combinatorial stage.
    """
    del pursuer_positions
    scfg = cfg if isinstance(cfg, ReachabilitySlotConfig) else ReachabilitySlotConfig.from_dict(cfg)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    out: list[SlotCandidate] = []
    phase = float(np.arctan2(e[1], e[0])) if np.linalg.norm(e[:2]) > 1e-9 else 0.0
    angles = phase + np.linspace(0.0, 2.0 * np.pi, scfg.num_candidate_slots, endpoint=False)
    w = float(world_xy)
    min_clear = float(scfg.slot_min_clearance)

    for radius in scfg.slot_radius_candidates:
        for angle_idx, theta in enumerate(angles):
            xy = e[:2] + float(radius) * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
            if np.any(np.abs(xy) > w - scfg.slot_boundary_margin):
                continue
            clearance, inside, risk = _slot_clearance_and_risk(xy, obstacles, uav_radius=uav_radius)
            los_blocked = not has_line_of_sight(
                xy, e[:2], obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            )
            if inside:
                continue
            if clearance < min_clear and not scfg.allow_los_blocked_slots:
                continue
            if los_blocked and not scfg.allow_los_blocked_slots:
                continue
            pos = np.array([xy[0], xy[1], e[2]], dtype=np.float32)
            out.append(SlotCandidate(
                pos=pos,
                theta=float(theta),
                angle_index=int(angle_idx),
                radius=float(radius),
                min_obstacle_clearance=float(clearance),
                is_inside_obstacle=bool(inside),
                obstacle_risk=float(risk + max(0.0, min_clear - clearance)),
                los_to_evader_blocked=bool(los_blocked),
            ))

    def _cand_key(c: SlotCandidate) -> tuple[float, float]:
        return (c.obstacle_risk, abs(c.radius - float(np.median(scfg.slot_radius_candidates))))

    if not scfg.preserve_all_angles:
        out.sort(key=_cand_key)
        return out[: scfg.max_candidates]

    # Keep at least the safest candidate for every scanned angle before applying
    # the global cap.  This prevents an obstacle on one side from causing the
    # selector to stop considering later, structurally better flight directions.
    by_angle: dict[int, list[SlotCandidate]] = {}
    for cand in out:
        by_angle.setdefault(int(cand.angle_index), []).append(cand)
    for vals in by_angle.values():
        vals.sort(key=_cand_key)

    balanced: list[SlotCandidate] = []
    for idx in sorted(by_angle):
        balanced.append(by_angle[idx][0])
    extra: list[SlotCandidate] = []
    for idx in sorted(by_angle):
        extra.extend(by_angle[idx][1:])
    extra.sort(key=_cand_key)
    balanced.extend(extra)
    return balanced[: max(int(scfg.max_candidates), min(len(by_angle), len(balanced)))]


def _path_risk_integral(
    path: list[np.ndarray] | np.ndarray,
    obstacles: list[Obstacle],
    *,
    uav_radius: float,
) -> float:
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2 or not obstacles:
        return 0.0
    total = 0.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-9:
            continue
        for t in np.linspace(0.0, 1.0, 5):
            p = p0 + t * (p1 - p0)
            clear, _mean = path_clearance_stats([p], obstacles, uav_radius=uav_radius)
            if np.isfinite(clear):
                total += seg_len / max(clear + 0.5, 0.05)
    return float(total)


def _path_turn_risk(
    path: list[np.ndarray] | np.ndarray,
    obstacles: list[Obstacle],
    *,
    uav_radius: float,
    min_turn_radius: float,
    min_turn_clearance: float,
    world_xy: float | None = None,
    boundary_margin: float = 0.0,
) -> float:
    """Penalty for sharp, low-clearance corners that a real UAV cannot cut safely."""
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    total = 0.0
    for p0, p1, p2 in zip(pts[:-2], pts[1:-1], pts[2:]):
        v0 = p1 - p0
        v1 = p2 - p1
        l0 = float(np.linalg.norm(v0))
        l1 = float(np.linalg.norm(v1))
        if l0 < 1e-6 or l1 < 1e-6:
            continue
        d0 = v0 / l0
        d1 = v1 / l1
        angle = float(np.arccos(np.clip(np.dot(d0, d1), -1.0, 1.0)))
        if angle < 0.15:
            continue
        corner_clear, _ = path_clearance_stats([p1], obstacles, uav_radius=uav_radius)
        boundary_clear = points_min_boundary_clearance(
            [p1], world_xy=world_xy, boundary_margin=boundary_margin,
        )
        clearance_pen = max(0.0, float(min_turn_clearance) - float(corner_clear)) / max(float(min_turn_clearance), 1e-6)
        boundary_pen = max(0.0, float(min_turn_clearance) - float(boundary_clear)) / max(float(min_turn_clearance), 1e-6)
        radius_pen = max(0.0, float(min_turn_radius) - min(l0, l1) * 0.5) / max(float(min_turn_radius), 1e-6)
        total += (angle / np.pi) * (1.0 + clearance_pen + boundary_pen + radius_pen)
    return float(total)


def _best_assignment_for_slots(cost: np.ndarray) -> tuple[np.ndarray, float]:
    n_agents, n_slots = cost.shape
    best = None
    best_cost = float(INF_COST)
    for perm in permutations(range(n_slots), n_agents):
        c = float(sum(cost[i, perm[i]] for i in range(n_agents)))
        if c < best_cost:
            best_cost = c
            best = np.array(perm, dtype=np.int64)
    if best is None:
        best = np.arange(n_agents, dtype=np.int64)
    return best, best_cost


def _slot_set_reachability_cost(pair_cost: np.ndarray, feasible: np.ndarray) -> float:
    """Cost of making selected actual slots reachable before assigning roles."""
    selected = np.asarray(pair_cost, dtype=np.float64)
    selected_feasible = np.asarray(feasible, dtype=bool)
    per_slot: list[float] = []
    for j in range(selected.shape[1]):
        vals = selected[:, j][selected_feasible[:, j]]
        if vals.size == 0:
            return float(INF_COST)
        per_slot.append(float(np.min(vals)))
    return float(np.mean(per_slot)) if per_slot else float(INF_COST)


def _structure_loss(slots: np.ndarray, evader_pos: np.ndarray, cfg: StructureSelectionConfig) -> tuple[float, dict[str, float]]:
    metrics = compute_pursuit_structure_metrics_3v1(slots, evader_pos)
    gap = float(metrics.get("phi_max", 2.0 * np.pi))
    gap_loss = max(0.0, gap - 2.0 * np.pi / 3.0) / (4.0 * np.pi / 3.0)
    violation = (
        max(0.0, cfg.min_C_cov - float(metrics["C_cov"]))
        + max(0.0, cfg.min_D_ang - float(metrics["D_ang"]))
        + max(0.0, float(metrics["C_col"]) - cfg.max_C_col)
    )
    loss = (
        cfg.w_C_cov * (1.0 - float(metrics["C_cov"]))
        + cfg.w_D_ang * (1.0 - float(metrics["D_ang"]))
        + cfg.w_C_col * float(metrics["C_col"])
        + cfg.w_gap * gap_loss
        + cfg.structure_violation_weight * violation
    )
    return float(loss), {
        "candidate_C_cov": float(metrics["C_cov"]),
        "candidate_C_col": float(metrics["C_col"]),
        "candidate_D_ang": float(metrics["D_ang"]),
        "candidate_max_escape_gap": gap,
        "candidate_structure_violation": float(violation),
    }


def _candidate_descriptor(c: SlotCandidate) -> dict[str, float]:
    return {"angle_index": int(c.angle_index), "radius": float(c.radius)}


def _match_previous_candidate_descriptors(
    candidates: list[SlotCandidate],
    descriptors: list[dict[str, Any]] | None,
) -> np.ndarray | None:
    if not descriptors or len(descriptors) != 3:
        return None
    matched: list[int] = []
    used: set[int] = set()
    for desc in descriptors:
        angle = int(desc.get("angle_index", -1))
        radius = float(desc.get("radius", np.nan))
        best = None
        best_key = (float("inf"), float("inf"))
        for idx, cand in enumerate(candidates):
            if idx in used or int(cand.angle_index) != angle:
                continue
            rdiff = abs(float(cand.radius) - radius) if np.isfinite(radius) else 0.0
            key = (rdiff, float(cand.obstacle_risk))
            if key < best_key:
                best_key = key
                best = idx
        if best is None:
            return None
        matched.append(int(best))
        used.add(int(best))
    return np.asarray(matched, dtype=np.int64)


def select_reachability_aware_slots(
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    obstacles: list[Obstacle],
    previous_assignment: np.ndarray | None,
    path_cache: DeployPathCache,
    bounds: tuple[float, float, float, float],
    planner_cfg: dict[str, Any],
    *,
    world_xy: float,
    pursuer_speed: float,
    safety_margin: float,
    uav_radius: float,
    slot_cfg: dict[str, Any] | None,
    assignment_cfg: dict[str, Any] | None,
    structure_cfg: dict[str, Any] | None,
    switch_penalty: float,
    previous_candidate_descriptors: list[dict[str, Any]] | None = None,
    previous_slot_descriptors: list[dict[str, Any]] | None = None,
    previous_slot_assignment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select actual slots first, then assign pursuers to those slots.

    The separation is intentional: obstacle-rich SCE should first decide which
    three physical locations form a good reachable encirclement structure.  Only
    after those actual slots are fixed do we solve the pursuer-slot assignment
    with distance, path risk, clearance, LOS, switch, and turn costs.
    """
    pcfg = AssignmentCostConfig.from_dict(assignment_cfg)
    scfg = StructureSelectionConfig.from_dict(structure_cfg)
    candidates = generate_candidate_slots(
        evader_pos, pursuer_pos, obstacles, slot_cfg,
        world_xy=world_xy, safety_margin=safety_margin, uav_radius=uav_radius,
    )
    if len(candidates) < 3:
        return _fallback_slots(evader_pos, world_xy), np.arange(3, dtype=np.int64), {
            "fallback_slot_selection": True,
            "fallback_reason": "too_few_candidates",
            "num_slot_candidates": len(candidates),
            "num_scanned_slot_angles": len({int(c.angle_index) for c in candidates}),
        }

    p = np.asarray(pursuer_pos, dtype=np.float64).reshape(3, 3)
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    n = len(candidates)
    pair_cost = np.full((3, n), pcfg.infeasible_cost, dtype=np.float64)
    pair_feasible = np.zeros((3, n), dtype=bool)
    time_to_slot = np.full((3, n), np.inf, dtype=np.float64)
    path_clear = np.full((3, n), -np.inf, dtype=np.float64)
    path_risk = np.full((3, n), np.inf, dtype=np.float64)
    los_blocked = np.zeros((3, n), dtype=bool)

    path_cache.ensure_static_graph(obstacles, bounds, planner_cfg, safety_margin=safety_margin, uav_radius=uav_radius)
    timeout_ms = float(planner_cfg.get("planner_timeout_ms", 5.0))
    for i in range(3):
        for j, cand in enumerate(candidates):
            start = p[i, :2]
            goal = np.asarray(cand.pos[:2], dtype=np.float64)
            los_blocked[i, j] = not has_line_of_sight(
                start, goal, obstacles, safety_margin=safety_margin, uav_radius=uav_radius,
            )
            path, _ms, timed_out = path_cache.static_graph.query_path(
                start, goal, obstacles, planner_cfg,
                safety_margin=safety_margin, uav_radius=uav_radius, timeout_ms=timeout_ms,
            )
            if timed_out or path is None:
                continue
            plen = path_length(path)
            tslot = plen / max(float(pursuer_speed), 1e-6)
            min_clear, _mean_clear = path_clearance_stats(path, obstacles, uav_radius=uav_radius)
            risk = _path_risk_integral(path, obstacles, uav_radius=uav_radius)
            turn_risk = _path_turn_risk(
                path,
                obstacles,
                uav_radius=uav_radius,
                min_turn_radius=pcfg.turn_min_radius,
                min_turn_clearance=pcfg.turn_min_clearance,
                world_xy=world_xy,
                boundary_margin=ReachabilitySlotConfig.from_dict(slot_cfg).slot_boundary_margin,
            )
            clear_pen = max(0.0, float(ReachabilitySlotConfig.from_dict(slot_cfg).slot_min_clearance) - min_clear)
            pair_cost[i, j] = (
                pcfg.w_path * (plen / max(float(world_xy), 1e-6))
                + pcfg.w_time * (tslot / 80.0)
                + pcfg.w_risk * risk / max(float(world_xy), 1e-6)
                + pcfg.w_clearance * clear_pen
                + pcfg.w_los * float(los_blocked[i, j])
                + pcfg.w_turn * turn_risk
            )
            if previous_assignment is not None:
                prev = np.asarray(previous_assignment, dtype=np.int64).reshape(-1)
                if prev.shape[0] == 3 and int(prev[i]) != j:
                    pair_cost[i, j] += pcfg.w_switch * switch_penalty
            pair_feasible[i, j] = True
            time_to_slot[i, j] = tslot
            path_clear[i, j] = min_clear
            path_risk[i, j] = risk

    best_targets = None
    best_candidate_idx = None
    best_assign: np.ndarray | None = None
    best_score = float(INF_COST)
    best_struct_diag: dict[str, float] = {}
    checked = 0
    lock_candidate_idx = _match_previous_candidate_descriptors(
        candidates,
        previous_slot_descriptors if previous_slot_descriptors is not None else previous_candidate_descriptors,
    )
    lock_reachable = None
    lock_preserved = False
    lock_assignment = (
        np.asarray(previous_slot_assignment, dtype=np.int64).reshape(-1)
        if previous_slot_assignment is not None
        else np.arange(3, dtype=np.int64)
    )
    if lock_assignment.shape[0] != 3 or np.any(lock_assignment < 0) or np.any(lock_assignment > 2):
        lock_assignment = np.arange(3, dtype=np.int64)
    if scfg.preserve_assignment_until_unreachable and lock_candidate_idx is not None:
        lock_reachable = np.array(
            [bool(pair_feasible[i, int(lock_candidate_idx[int(lock_assignment[i])])]) for i in range(3)],
            dtype=bool,
        )
        # Stability rule: keep each pursuer on its current structural slot as
        # long as a fully scanned planner can still reach that slot.  Role
        # switching is a last resort, because frequent switching destroys the
        # encirclement manifold before it can form.
        if bool(np.all(lock_reachable)):
            slots = np.stack([candidates[int(k)].pos for k in lock_candidate_idx], axis=0).astype(np.float64)
            struct_loss, struct_diag = _structure_loss(slots, e, scfg)
            best_targets = slots.astype(np.float32)
            best_candidate_idx = lock_candidate_idx.copy()
            best_assign = lock_assignment.copy()
            lock_costs = [pair_cost[i, int(lock_candidate_idx[int(lock_assignment[i])])] for i in range(3)]
            best_score = float(struct_loss + np.mean(lock_costs))
            best_struct_diag = {
                **struct_diag,
                "actual_slot_set_score": float(best_score),
                "actual_slot_reachability_cost": float(np.mean(lock_costs)),
                "actual_slot_risk_cost": 0.0,
                "assignment_lock_preserved": 1.0,
            }
            lock_preserved = True

    if not lock_preserved:
        for combo in combinations(range(n), 3):
            checked += 1
            combo_idx = np.array(combo, dtype=np.int64)
            if not np.all(np.any(pair_feasible[:, combo_idx], axis=0)):
                continue
            combo_cost = pair_cost[:, combo_idx]
            combo_feasible = pair_feasible[:, combo_idx]
            reach_cost = _slot_set_reachability_cost(combo_cost, combo_feasible)
            if reach_cost >= pcfg.infeasible_cost:
                continue
            slots = np.stack([candidates[k].pos for k in combo_idx], axis=0).astype(np.float64)
            struct_loss, struct_diag = _structure_loss(slots, e, scfg)
            feasible_frac = float(np.mean(combo_feasible))
            # Actual slot selection optimizes structure plus arrival-path
            # reachability.  LOS/clearance around the slot is still logged and can
            # hard-filter impossible locations, but it should not masquerade as the
            # collision risk of flying to that slot.
            slot_risk = 0.0
            total = (
                struct_loss
                + scfg.w_assignment_feasibility * reach_cost
                + scfg.w_slot_risk * slot_risk
                + (1.0 - feasible_frac)
            )
            if total < best_score:
                best_score = total
                best_targets = slots.astype(np.float32)
                best_candidate_idx = combo_idx.copy()
                best_assign = None
                best_struct_diag = {
                    **struct_diag,
                    "actual_slot_set_score": float(total),
                    "actual_slot_reachability_cost": float(reach_cost),
                    "actual_slot_risk_cost": float(slot_risk),
                    "assignment_lock_preserved": 0.0,
                }

    fallback = False
    if best_targets is None:
        fallback = True
        # Reachable fallback: pick candidates with the best arrival-path cost.
        reachable_slots = [j for j in range(n) if np.any(pair_feasible[:, j])]
        if len(reachable_slots) >= 3:
            reachable_slots.sort(key=lambda j: float(np.min(pair_cost[:, j][pair_feasible[:, j]])))
            idx = np.array(reachable_slots[:3], dtype=np.int64)
            best_targets = np.stack([candidates[j].pos for j in idx], axis=0).astype(np.float32)
            best_candidate_idx = idx.copy()
        else:
            best_targets = _fallback_slots(e, world_xy)
            best_candidate_idx = np.array([], dtype=np.int64)

    if best_candidate_idx is not None and best_candidate_idx.size == 3:
        selected_cost = pair_cost[:, best_candidate_idx]
        if best_assign is None:
            best_assign, final_assign_cost = _best_assignment_for_slots(selected_cost)
        else:
            final_assign_cost = float(sum(selected_cost[i, int(best_assign[i])] for i in range(3)))
    else:
        selected_cost = np.linalg.norm(p[:, None, :2] - best_targets[None, :, :2], axis=2)
        if best_assign is None:
            best_assign, final_assign_cost = _best_assignment_for_slots(selected_cost)
        else:
            final_assign_cost = float(sum(selected_cost[i, int(best_assign[i])] for i in range(3)))

    finite_time = time_to_slot[np.isfinite(time_to_slot)]
    finite_clear = path_clear[np.isfinite(path_clear)]
    finite_risk = path_risk[np.isfinite(path_risk)]
    diag: dict[str, Any] = {
        "num_slot_candidates": int(n),
        "num_scanned_slot_angles": int(len({int(c.angle_index) for c in candidates})),
        "num_evaluated_slot_combinations": int(checked),
        "candidate_slot_positions": [c.pos.astype(float).tolist() for c in candidates],
        "candidate_slot_angle_index": [int(c.angle_index) for c in candidates],
        "candidate_slot_clearance": [float(c.min_obstacle_clearance) for c in candidates],
        "candidate_slot_risk": [float(c.obstacle_risk) for c in candidates],
        "candidate_slot_los_blocked": [bool(c.los_to_evader_blocked) for c in candidates],
        "candidate_slot_reachable": [bool(x) for x in np.any(pair_feasible, axis=0)],
        "fallback_slot_selection": bool(fallback),
        "fallback_slot_selection_rate": 1.0 if fallback else 0.0,
        "slot_reachable_rate": float(np.mean(np.any(pair_feasible, axis=0))),
        "unreachable_slot_rate": float(1.0 - np.mean(np.any(pair_feasible, axis=0))),
        "mean_time_to_slot": float(np.mean(finite_time)) if finite_time.size else float("nan"),
        "max_time_to_slot": float(np.max(finite_time)) if finite_time.size else float("nan"),
        "path_clearance_min": float(np.min(finite_clear)) if finite_clear.size else float("nan"),
        "path_clearance_mean": float(np.mean(finite_clear)) if finite_clear.size else float("nan"),
        "path_risk_integral": float(np.mean(finite_risk)) if finite_risk.size else float("nan"),
        "slot_behind_obstacle_rate": float(np.mean([c.los_to_evader_blocked for c in candidates])),
        "los_blocked_slot_rate": float(np.mean(los_blocked)),
        "reachability_assignment_cost_matrix": pair_cost.copy(),
        "selected_assignment_cost_matrix": selected_cost.copy(),
        "final_assignment_cost": float(final_assign_cost),
        "pair_feasible_matrix": pair_feasible.copy(),
        "actual_slot_positions": np.asarray(best_targets, dtype=np.float32).tolist(),
        **best_struct_diag,
    }
    if best_candidate_idx is not None and best_candidate_idx.size:
        diag["selected_candidate_indices"] = [int(x) for x in best_candidate_idx.reshape(-1)]
        diag["selected_candidate_descriptors"] = [
            _candidate_descriptor(candidates[int(x)]) for x in best_candidate_idx.reshape(-1)
        ]
        if np.asarray(best_assign).shape[0] == 3 and best_candidate_idx.size == 3:
            assigned_idx = [int(best_candidate_idx[int(best_assign[i])]) for i in range(3)]
            diag["assigned_candidate_indices"] = assigned_idx
            diag["assigned_candidate_descriptors"] = [
                _candidate_descriptor(candidates[int(x)]) for x in assigned_idx
            ]
    diag["assignment_lock_preserved"] = bool(lock_preserved)
    if lock_reachable is not None:
        diag["locked_assignment_reachable"] = [bool(x) for x in lock_reachable.reshape(-1)]
        if not lock_preserved:
            diag["assignment_lock_released_reason"] = "assigned_slot_unreachable_after_full_scan"
    return np.asarray(best_targets, dtype=np.float32), np.asarray(best_assign, dtype=np.int64), diag


def _fallback_slots(evader_pos: np.ndarray, world_xy: float) -> np.ndarray:
    e = np.asarray(evader_pos, dtype=np.float64).reshape(3)
    r = min(3.0, max(1.0, float(world_xy) * 0.2))
    out = np.zeros((3, 3), dtype=np.float32)
    for k in range(3):
        a = 2.0 * np.pi * k / 3.0
        out[k, :2] = e[:2] + r * np.array([np.cos(a), np.sin(a)], dtype=np.float64)
        out[k, :2] = np.clip(out[k, :2], -float(world_xy) + 0.8, float(world_xy) - 0.8)
        out[k, 2] = e[2]
    return out
