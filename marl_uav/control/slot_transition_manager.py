"""Commanded-slot transition manager used by deployable SCE controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_uav.framework.planning.visibility_path_planner import plan_path


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= float(max_norm) or n <= 1e-12:
        return v.copy()
    return v * (float(max_norm) / n)


@dataclass
class SlotTransitionState:
    commanded_slot_pos: np.ndarray
    commanded_slot_vel: np.ndarray
    previous_raw_slot_pos: np.ndarray
    last_jump_step: int = -10**9
    previous_jump_step: int = -10**9
    unstable_hold_until_step: int = -10**9
    transition_start_distance: float = 0.0
    mode: str = "INIT"
    safe_hold_reason: str = ""


class SlotTransitionManager:
    """Convert raw assigned slots into valid, smooth commanded slots.

    This is the production SCE counterpart of the ABCD slot-tracking benchmark
    controller: raw slots remain visible in diagnostics, while the action layer
    only receives a boundary-valid, obstacle-clear, transition-safe command.
    """

    def __init__(
        self,
        *,
        world_xy: float,
        uav_radius: float,
        safety_margin: float,
        dt: float,
        slot_ref_vmax: float,
        slot_ref_amax: float,
        jump_detection_threshold: float,
        frequent_jump_min_interval_steps: int = 20,
        high_freq_factor: float = 1.25,
        planner_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.world_xy = float(world_xy)
        self.uav_radius = float(uav_radius)
        self.safety_margin = float(safety_margin)
        self.dt = max(float(dt), 1e-9)
        self.slot_ref_vmax = max(float(slot_ref_vmax), 1e-9)
        self.slot_ref_amax = max(float(slot_ref_amax), 1e-9)
        self.jump_detection_threshold = max(float(jump_detection_threshold), 0.0)
        self.frequent_jump_min_interval_steps = max(int(frequent_jump_min_interval_steps), 1)
        self.high_freq_factor = max(float(high_freq_factor), 0.0)
        self.planner_cfg = dict(planner_cfg or {})
        self.state: SlotTransitionState | None = None

    def reset(self, raw_slot_pos: np.ndarray, obstacles: list[Any]) -> dict[str, Any]:
        raw = np.asarray(raw_slot_pos, dtype=np.float64).reshape(3)
        proxy, diag = self._project_to_valid(raw, obstacles)
        self.state = SlotTransitionState(
            commanded_slot_pos=proxy.copy(),
            commanded_slot_vel=np.zeros(3, dtype=np.float64),
            previous_raw_slot_pos=raw.copy(),
            transition_start_distance=0.0,
            mode="RAW_TRACKING" if diag["raw_slot_valid"] else "PROXY_TRACKING",
        )
        validity = self._validity_diag(proxy, obstacles)
        return self._output(
            raw=raw,
            proxy=proxy,
            mode=self.state.mode,
            jump_detected=False,
            jump_distance=0.0,
            minimum_reach_time=0.0,
            jump_interval_steps=0,
            raw_slot_too_unstable=False,
            raw_slot_valid=bool(diag["raw_slot_valid"]),
            commanded_slot_valid=bool(validity["commanded_slot_valid"]),
            proxy_slot_used=bool(diag["proxy_slot_used"]),
            transition_progress=1.0,
            commanded_step_norm=0.0,
            commanded_validity=validity,
            commanded_transition_segment_safe=True,
            safe_hold_active=False,
            safe_hold_reason="",
            slot_planner_called=False,
            slot_planner_success=False,
            slot_planner_path_valid=False,
            slot_replan_reason="",
            raw_diag=diag,
        )

    def update(
        self,
        *,
        raw_slot_pos: np.ndarray,
        previous_commanded_slot_pos: np.ndarray | None = None,
        uav_pos: np.ndarray | None = None,
        obstacles: list[Any],
        step: int,
    ) -> dict[str, Any]:
        del uav_pos
        if self.state is None:
            return self.reset(raw_slot_pos, obstacles)

        raw = np.asarray(raw_slot_pos, dtype=np.float64).reshape(3)
        prev_cmd = (
            np.asarray(previous_commanded_slot_pos, dtype=np.float64).reshape(3)
            if previous_commanded_slot_pos is not None
            else self.state.commanded_slot_pos.copy()
        )
        proxy, diag = self._project_to_valid(raw, obstacles)
        raw_delta = float(np.linalg.norm(raw[:2] - self.state.previous_raw_slot_pos[:2]))
        jump_detected = bool(raw_delta > self.jump_detection_threshold)
        previous_jump_step = int(self.state.last_jump_step)
        jump_interval_steps = int(step) - previous_jump_step if previous_jump_step > -10**8 else 10**9
        minimum_reach_time = raw_delta / self.slot_ref_vmax
        jump_interval_time = float(jump_interval_steps) * self.dt if jump_interval_steps < 10**8 else float("inf")
        raw_slot_too_unstable = bool(
            jump_detected
            and previous_jump_step > -10**8
            and jump_interval_time < max(minimum_reach_time * self.high_freq_factor, self.dt)
        )
        if jump_detected:
            self.state.previous_jump_step = previous_jump_step
            self.state.last_jump_step = int(step)
            self.state.transition_start_distance = float(np.linalg.norm(proxy[:2] - prev_cmd[:2]))
            if raw_slot_too_unstable:
                self.state.unstable_hold_until_step = max(
                    int(self.state.unstable_hold_until_step),
                    int(step) + self.frequent_jump_min_interval_steps,
                )

        frequent_jump = bool(
            jump_detected
            and previous_jump_step > -10**8
            and int(step) - previous_jump_step < self.frequent_jump_min_interval_steps
        )
        rejoin_directly = bool(
            np.linalg.norm(proxy[:2] - prev_cmd[:2]) <= max(self.slot_ref_vmax * self.dt, 0.03) + 1e-9
        )
        hold_unstable = bool(raw_slot_too_unstable or int(step) < int(self.state.unstable_hold_until_step))

        if (
            bool(diag["raw_slot_valid"])
            and not jump_detected
            and self.state.mode not in ("JUMP_INTERPOLATION", "PROXY_INTERPOLATION", "HOLD_UNSTABLE", "SAFE_HOLD")
            and rejoin_directly
        ):
            commanded = proxy.copy()
            cmd_vel_xy = (commanded[:2] - prev_cmd[:2]) / self.dt
            mode = "RAW_TRACKING"
        elif hold_unstable or (frequent_jump and self.state.mode in ("JUMP_INTERPOLATION", "HOLD_UNSTABLE", "SAFE_HOLD")):
            commanded = prev_cmd.copy()
            cmd_vel_xy = np.zeros(2, dtype=np.float64)
            mode = "HOLD_UNSTABLE"
        else:
            commanded_xy, cmd_vel_xy = self._limited_step(prev_cmd[:2], self.state.commanded_slot_vel[:2], proxy[:2])
            commanded = prev_cmd.copy()
            commanded[:2] = commanded_xy
            commanded[2] = proxy[2]
            mode = "JUMP_INTERPOLATION" if jump_detected or self.state.mode == "JUMP_INTERPOLATION" else "PROXY_INTERPOLATION"
            if float(np.linalg.norm(commanded[:2] - proxy[:2])) <= max(0.03, self.slot_ref_vmax * self.dt):
                commanded = proxy.copy()
                mode = "PROXY_TRACKING" if diag["proxy_slot_used"] else "RAW_TRACKING"

        commanded_validity = self._validity_diag(commanded, obstacles)
        transition_safe = self._segment_safe(prev_cmd, commanded, obstacles)
        safe_hold_active = False
        safe_hold_reason = ""
        slot_planner_called = False
        slot_planner_success = False
        slot_planner_path_valid = False
        slot_replan_reason = ""
        prev_valid = self._validity_diag(prev_cmd, obstacles)

        if not bool(commanded_validity["commanded_slot_valid"]):
            safe_hold_reason = str(commanded_validity["commanded_slot_invalid_reason"])
        elif not transition_safe:
            safe_hold_reason = "COMMAND_TRANSITION_BLOCKED"
        elif not bool(diag.get("proxy_slot_valid", False)):
            safe_hold_reason = "NO_VALID_PROXY_SLOT"

        if safe_hold_reason in (
            "COMMAND_TRANSITION_BLOCKED",
            "COMMANDED_SLOT_INSIDE_OBSTACLE",
            "COMMANDED_SLOT_TOO_CLOSE_OBSTACLE",
        ) and bool(diag.get("proxy_slot_valid", False)) and bool(self.planner_cfg.get("enabled", False)):
            slot_planner_called = True
            planned_command, path_valid, planner_reason = self._planned_command_step(prev_cmd, proxy, obstacles)
            slot_replan_reason = planner_reason
            if path_valid:
                planned_validity = self._validity_diag(planned_command, obstacles)
                planned_transition_safe = self._segment_safe(prev_cmd, planned_command, obstacles)
                if bool(planned_validity["commanded_slot_valid"]) and planned_transition_safe:
                    commanded = planned_command
                    cmd_vel_xy = (commanded[:2] - prev_cmd[:2]) / self.dt
                    mode = "PATH_PROXY_INTERPOLATION"
                    safe_hold_reason = ""
                    commanded_validity = planned_validity
                    transition_safe = True
                    slot_planner_success = True
                    slot_planner_path_valid = True

        if safe_hold_reason:
            if bool(prev_valid["commanded_slot_valid"]):
                commanded = prev_cmd.copy()
                cmd_vel_xy = np.zeros(2, dtype=np.float64)
                commanded_validity = self._validity_diag(commanded, obstacles)
                transition_safe = True
            mode = "SAFE_HOLD"
            safe_hold_active = True

        commanded_valid = bool(commanded_validity["commanded_slot_valid"]) and bool(transition_safe)
        step_norm = float(np.linalg.norm(commanded[:2] - prev_cmd[:2]))
        start_dist = max(float(self.state.transition_start_distance), 1e-9)
        remaining = float(np.linalg.norm(commanded[:2] - proxy[:2]))
        transition_progress = 1.0 if start_dist <= 1e-8 else float(np.clip(1.0 - remaining / start_dist, 0.0, 1.0))

        self.state.commanded_slot_pos = commanded.copy()
        self.state.commanded_slot_vel = np.array([cmd_vel_xy[0], cmd_vel_xy[1], 0.0], dtype=np.float64)
        self.state.previous_raw_slot_pos = raw.copy()
        self.state.mode = mode
        self.state.safe_hold_reason = safe_hold_reason
        return self._output(
            raw=raw,
            proxy=proxy,
            mode=mode,
            jump_detected=jump_detected,
            jump_distance=raw_delta if jump_detected else 0.0,
            minimum_reach_time=minimum_reach_time if jump_detected else 0.0,
            jump_interval_steps=jump_interval_steps if jump_detected and jump_interval_steps < 10**8 else 0,
            raw_slot_too_unstable=raw_slot_too_unstable,
            raw_slot_valid=bool(diag["raw_slot_valid"]),
            commanded_slot_valid=bool(commanded_valid),
            proxy_slot_used=bool(diag["proxy_slot_used"]),
            transition_progress=transition_progress,
            commanded_step_norm=step_norm,
            commanded_validity=commanded_validity,
            commanded_transition_segment_safe=bool(transition_safe),
            safe_hold_active=bool(safe_hold_active),
            safe_hold_reason=safe_hold_reason,
            slot_planner_called=slot_planner_called,
            slot_planner_success=slot_planner_success,
            slot_planner_path_valid=slot_planner_path_valid,
            slot_replan_reason=slot_replan_reason,
            raw_diag=diag,
        )

    def _limited_step(
        self,
        current_xy: np.ndarray,
        current_vel_xy: np.ndarray,
        target_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = np.asarray(target_xy, dtype=np.float64).reshape(2) - np.asarray(current_xy, dtype=np.float64).reshape(2)
        desired_vel = _clip_norm(delta / self.dt, self.slot_ref_vmax)
        dv = _clip_norm(desired_vel - np.asarray(current_vel_xy, dtype=np.float64).reshape(2), self.slot_ref_amax * self.dt)
        next_vel = _clip_norm(np.asarray(current_vel_xy, dtype=np.float64).reshape(2) + dv, self.slot_ref_vmax)
        step = next_vel * self.dt
        if float(np.linalg.norm(step)) > float(np.linalg.norm(delta)):
            return np.asarray(target_xy, dtype=np.float64).reshape(2).copy(), delta / self.dt
        return np.asarray(current_xy, dtype=np.float64).reshape(2) + step, next_vel

    def _project_to_valid(self, raw_slot: np.ndarray, obstacles: list[Any]) -> tuple[np.ndarray, dict[str, Any]]:
        raw = np.asarray(raw_slot, dtype=np.float64).reshape(3)
        proxy = raw.copy()
        limit = max(self.world_xy - self.uav_radius - self.safety_margin, 0.0)
        before = proxy[:2].copy()
        proxy[0] = float(np.clip(proxy[0], -limit, limit))
        proxy[1] = float(np.clip(proxy[1], -limit, limit))
        raw_outside = bool(np.max(np.abs(raw[:2])) > limit + 1e-6)
        raw_clear = self._clearance(raw[:2], obstacles)
        raw_valid = bool((not raw_outside) and raw_clear >= self.safety_margin)

        for _ in range(6):
            changed = False
            for obs in obstacles:
                clear = self._clearance_one(proxy[:2], obs)
                if clear >= self.safety_margin:
                    continue
                center = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
                rel = proxy[:2] - center
                dist = float(np.linalg.norm(rel))
                normal = np.array([1.0, 0.0], dtype=np.float64) if dist <= 1e-9 else rel / dist
                radius = float(getattr(obs, "radius", 0.0))
                if getattr(obs, "kind", "circle") == "aabb" and getattr(obs, "half_extents", None) is not None:
                    radius = float(np.linalg.norm(np.asarray(obs.half_extents, dtype=np.float64).reshape(2)))
                proxy[:2] = center + normal * (radius + self.uav_radius + self.safety_margin)
                proxy[0] = float(np.clip(proxy[0], -limit, limit))
                proxy[1] = float(np.clip(proxy[1], -limit, limit))
                changed = True
            if not changed:
                break

        proxy_valid = self._is_valid(proxy, obstacles)
        return proxy, {
            "raw_slot_valid": bool(raw_valid),
            "raw_slot_outside_boundary": bool(raw_outside),
            "raw_slot_inside_obstacle": bool(raw_clear < 0.0),
            "raw_slot_too_close_obstacle": bool(raw_clear < self.safety_margin),
            "proxy_slot_valid": bool(proxy_valid),
            "proxy_slot_used": bool(np.linalg.norm(proxy[:2] - raw[:2]) > 1e-9 or not raw_valid),
            "proxy_slot_adjusted": bool(np.linalg.norm(proxy[:2] - before) > 1e-9 or np.linalg.norm(proxy[:2] - raw[:2]) > 1e-9),
            "proxy_slot_min_obstacle_clearance": float(self._clearance(proxy[:2], obstacles)),
        }

    def _output(
        self,
        *,
        raw: np.ndarray,
        proxy: np.ndarray,
        mode: str,
        jump_detected: bool,
        raw_slot_valid: bool,
        commanded_slot_valid: bool,
        proxy_slot_used: bool,
        transition_progress: float,
        commanded_step_norm: float,
        jump_distance: float,
        minimum_reach_time: float,
        jump_interval_steps: int,
        raw_slot_too_unstable: bool,
        commanded_validity: dict[str, Any],
        commanded_transition_segment_safe: bool,
        safe_hold_active: bool,
        safe_hold_reason: str,
        slot_planner_called: bool,
        slot_planner_success: bool,
        slot_planner_path_valid: bool,
        slot_replan_reason: str,
        raw_diag: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.state
        assert state is not None
        commanded = state.commanded_slot_pos.copy()
        return {
            "commanded_slot_pos": commanded,
            "commanded_slot_vel": state.commanded_slot_vel.copy(),
            "proxy_slot_pos": proxy.copy(),
            "slot_transition_mode": str(mode),
            "jump_detected": bool(jump_detected),
            "jump_distance": float(jump_distance),
            "minimum_reach_time": float(minimum_reach_time),
            "jump_interval_steps": int(jump_interval_steps),
            "raw_slot_too_unstable": bool(raw_slot_too_unstable),
            "raw_slot_valid": bool(raw_slot_valid),
            "commanded_slot_valid": bool(commanded_slot_valid),
            "commanded_slot_inside_obstacle": bool(commanded_validity.get("commanded_slot_inside_obstacle", False)),
            "commanded_slot_outside_boundary": bool(commanded_validity.get("commanded_slot_outside_boundary", False)),
            "commanded_slot_too_close_obstacle": bool(commanded_validity.get("commanded_slot_too_close_obstacle", False)),
            "commanded_transition_segment_safe": bool(commanded_transition_segment_safe),
            "commanded_slot_invalid_reason": str(commanded_validity.get("commanded_slot_invalid_reason", "")),
            "safe_hold_active": bool(safe_hold_active),
            "safe_hold_reason": str(safe_hold_reason),
            "slot_planner_called": bool(slot_planner_called),
            "slot_planner_success": bool(slot_planner_success),
            "slot_planner_path_valid": bool(slot_planner_path_valid),
            "slot_replan_reason": str(slot_replan_reason),
            "proxy_slot_valid": bool(raw_diag.get("proxy_slot_valid", commanded_slot_valid)),
            "proxy_slot_used": bool(proxy_slot_used),
            "proxy_slot_adjusted": bool(raw_diag.get("proxy_slot_adjusted", proxy_slot_used)),
            "raw_slot_inside_obstacle": bool(raw_diag.get("raw_slot_inside_obstacle", False)),
            "raw_slot_too_close_obstacle": bool(raw_diag.get("raw_slot_too_close_obstacle", False)),
            "raw_slot_outside_boundary": bool(raw_diag.get("raw_slot_outside_boundary", False)),
            "transition_progress": float(transition_progress),
            "commanded_slot_step_norm": float(commanded_step_norm),
            "commanded_slot_lag_to_raw": float(np.linalg.norm(commanded[:2] - raw[:2])),
        }

    def _is_valid(self, slot: np.ndarray, obstacles: list[Any]) -> bool:
        return bool(self._validity_diag(slot, obstacles)["commanded_slot_valid"])

    def _validity_diag(self, slot: np.ndarray, obstacles: list[Any]) -> dict[str, Any]:
        p = np.asarray(slot, dtype=np.float64).reshape(3)
        limit = self.world_xy - self.uav_radius - self.safety_margin
        outside = bool(np.max(np.abs(p[:2])) > limit + 1e-6)
        clear = self._clearance(p[:2], obstacles)
        inside = bool(clear < 0.0)
        too_close = bool(clear < self.safety_margin - 1e-6)
        reason = ""
        if outside:
            reason = "COMMANDED_SLOT_OUTSIDE_BOUNDARY"
        elif inside:
            reason = "COMMANDED_SLOT_INSIDE_OBSTACLE"
        elif too_close:
            reason = "COMMANDED_SLOT_TOO_CLOSE_OBSTACLE"
        return {
            "commanded_slot_valid": bool(not outside and not too_close),
            "commanded_slot_outside_boundary": outside,
            "commanded_slot_inside_obstacle": inside,
            "commanded_slot_too_close_obstacle": too_close,
            "commanded_slot_min_obstacle_clearance": float(clear),
            "commanded_slot_invalid_reason": reason,
        }

    def _segment_safe(self, start: np.ndarray, end: np.ndarray, obstacles: list[Any]) -> bool:
        a = np.asarray(start, dtype=np.float64).reshape(3)[:2]
        b = np.asarray(end, dtype=np.float64).reshape(3)[:2]
        limit = self.world_xy - self.uav_radius - self.safety_margin
        for t in np.linspace(0.0, 1.0, 9):
            p = a + float(t) * (b - a)
            if np.max(np.abs(p)) > limit + 1e-6:
                return False
            if self._clearance(p, obstacles) < self.safety_margin - 1e-6:
                return False
        return True

    def _planned_command_step(self, start: np.ndarray, target: np.ndarray, obstacles: list[Any]) -> tuple[np.ndarray, bool, str]:
        start3 = np.asarray(start, dtype=np.float64).reshape(3)
        target3 = np.asarray(target, dtype=np.float64).reshape(3)
        cfg = dict(self.planner_cfg)
        cfg.setdefault("type", "visibility_graph")
        cfg.setdefault("grid_resolution", 0.25)
        bounds = (-self.world_xy, -self.world_xy, self.world_xy, self.world_xy)
        path = plan_path(
            start3[:2],
            target3[:2],
            obstacles,
            bounds=bounds,
            cfg=cfg,
            safety_margin=self.safety_margin,
            uav_radius=self.uav_radius,
        )
        if path is None and cfg.get("type") != "grid_astar":
            retry_cfg = dict(cfg)
            retry_cfg["type"] = "grid_astar"
            path = plan_path(
                start3[:2],
                target3[:2],
                obstacles,
                bounds=bounds,
                cfg=retry_cfg,
                safety_margin=self.safety_margin,
                uav_radius=self.uav_radius,
            )
        if path is None:
            return start3.copy(), False, "NO_PATH"
        pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            return target3.copy(), True, "ALREADY_AT_TARGET"
        budget = max(self.slot_ref_vmax * self.dt, 1e-6)
        cursor = pts[0].copy()
        for nxt in pts[1:]:
            seg = nxt - cursor
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-9:
                cursor = nxt.copy()
                continue
            if budget <= seg_len:
                out = start3.copy()
                out[:2] = cursor + seg * (budget / seg_len)
                out[2] = target3[2]
                return out, True, "PATH_COMMAND_STEP"
            budget -= seg_len
            cursor = nxt.copy()
        return target3.copy(), True, "PATH_COMMAND_TARGET"

    def _clearance(self, position_xy: np.ndarray, obstacles: list[Any]) -> float:
        if not obstacles:
            return float("inf")
        return float(min(self._clearance_one(position_xy, obs) for obs in obstacles))

    def _clearance_one(self, position_xy: np.ndarray, obs: Any) -> float:
        p = np.asarray(position_xy, dtype=np.float64).reshape(2)
        c = np.asarray(getattr(obs, "center", [0.0, 0.0]), dtype=np.float64).reshape(2)
        if getattr(obs, "kind", "circle") == "aabb" and getattr(obs, "half_extents", None) is not None:
            half = np.asarray(obs.half_extents, dtype=np.float64).reshape(2)
            q = np.maximum(np.abs(p - c) - half, 0.0)
            return float(np.linalg.norm(q) - self.uav_radius)
        return float(np.linalg.norm(p - c) - float(getattr(obs, "radius", 0.0)) - self.uav_radius)
