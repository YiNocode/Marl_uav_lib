"""Control-decision latency metrics for debug browser and real-time analysis."""

from __future__ import annotations

from typing import Any


def should_record_control_timing(env: Any | None) -> bool:
    """Record per-step decision latency when debug browser is active or task.debug."""
    if env is None:
        return False
    if getattr(getattr(env, "task", None), "debug", False):
        return True
    try:
        from marl_uav.utils.debug_browser import get_debug_browser_hub

        return get_debug_browser_hub() is not None
    except Exception:
        return False


def nominal_control_hz(env: Any | None) -> float | None:
    backend = getattr(env, "backend", None) if env is not None else None
    if backend is not None and hasattr(backend, "control_hz"):
        return float(getattr(backend, "control_hz"))
    return None


def publish_control_timing(env: Any, **parts: float | None) -> dict[str, float] | None:
    """Merge timing fragments onto ``env._last_control_timing`` for debug frames."""
    if not should_record_control_timing(env):
        return None

    out: dict[str, float] = {}
    prev = getattr(env, "_last_control_timing", None)
    if isinstance(prev, dict):
        out.update({k: float(v) for k, v in prev.items()})
    task_state = getattr(env, "task_state", None)
    task_timing = getattr(task_state, "last_control_timing", None) if task_state is not None else None
    if isinstance(task_timing, dict):
        out.update({k: float(v) for k, v in task_timing.items()})

    for key, val in parts.items():
        if val is None:
            continue
        out[key] = float(val)

    total = out.get("total_decision_latency")
    if total is not None and total > 0.0:
        out["control_frequency"] = 1.0 / total

    nominal = nominal_control_hz(env)
    if nominal is not None:
        out["nominal_control_hz"] = nominal

    env._last_control_timing = out
    if task_state is not None:
        task_state.last_control_timing = dict(out)
    return out


def control_timing_for_frame(env: Any | None) -> dict[str, float] | None:
    """Serialize the latest timing block for the debug browser JSON frame."""
    if env is None:
        return None
    raw = getattr(env, "_last_control_timing", None)
    if not isinstance(raw, dict):
        task_state = getattr(env, "task_state", None)
        raw = getattr(task_state, "last_control_timing", None) if task_state is not None else None
    if not isinstance(raw, dict) or not raw:
        return None

    keys = (
        "manifold_update_time",
        "slot_assignment_time",
        "action_mapping_time",
        "total_decision_latency",
        "control_frequency",
        "nominal_control_hz",
    )
    out = {k: float(raw[k]) for k in keys if k in raw and raw[k] is not None}
    return out or None
