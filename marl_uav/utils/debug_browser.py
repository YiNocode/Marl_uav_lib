"""Browser-based real-time debug visualization for pursuit simulations.

Uses stdlib HTTP + Server-Sent Events (no extra dependencies). When enabled,
simulation steps publish JSON frames that a canvas frontend consumes to draw
trajectories, deformable manifold curves, slot targets, and algorithm params.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from marl_uav.utils.debug_recorder import EpisodeRecorder, validate_episode_document
from marl_uav.utils.control_timing import control_timing_for_frame
from marl_uav.utils.debug_viz import (
    _active_viz_profile,
    build_controller_targets,
    filter_algorithm_fields,
    resolve_viz_profile,
)

_STATIC_DIR = Path(__file__).resolve().parent / "debug_browser_static"
_HUB_LOCK = threading.Lock()
_HUB: "DebugBrowserHub | None" = None

_MARKER_EVENTS = frozenset({"episode_start", "episode_end"})
_VISUAL_CARRY_KEYS = (
    "scene_id",
    "viz",
    "world_xy",
    "z_min",
    "z_max",
    "positions",
    "agent_labels",
    "pursuer_ids",
    "evader_id",
    "obstacles",
    "manifold",
    "role",
    "controller_targets",
    "kinematics",
    "speed_bounds",
    "algorithm",
    "dream_manifold",
    "pursuit_structure",
    "control_timing",
)


def _to_list(arr: Any) -> list[Any] | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).tolist()


def _build_agent_kinematics(
    backend_state: Any,
    agent_labels: list[str],
) -> dict[str, Any]:
    """Per-agent linear velocity from backend state index 2 (body frame, m/s)."""
    lin_vel = np.asarray(backend_state.states[:, 2, :], dtype=np.float32)
    agents: list[dict[str, Any]] = []
    for idx in range(lin_vel.shape[0]):
        v = lin_vel[idx]
        label = agent_labels[idx] if idx < len(agent_labels) else f"A{idx}"
        agents.append(
            {
                "label": label,
                "linear": v.tolist(),
                "speed_xy": float(np.linalg.norm(v[:2])),
                "speed_3d": float(np.linalg.norm(v)),
            }
        )
    return {"agents": agents, "frame": "body"}


class DebugBrowserHub:
    """Thread-safe pub/sub hub for debug frames."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        playback_speed: float = 1.0,
        step_dt: float = 1.0 / 60.0,
        start_paused: bool = True,
        record_dir: Path | str | None = None,
        run_meta: dict[str, Any] | None = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self._enabled = True
        self._episode_idx = 0
        self._total_episodes = 1
        self._latest_json: str | None = None
        self._subscribers: list[queue.Queue[str]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._meta: dict[str, Any] = dict(run_meta or {})
        self._control_lock = threading.Lock()
        self._playback_speed = max(float(playback_speed), 0.0)
        self._step_dt = max(float(step_dt), 1e-6)
        self._start_paused = bool(start_paused)
        self._paused = bool(start_paused)
        self._awaiting_start = bool(start_paused)
        self._start_acknowledged = False
        self._start_gate = threading.Event()
        if not start_paused:
            self._start_gate.set()
        self._episode_run_started = False
        self._latest_visual: dict[str, Any] | None = None
        self._completed_episodes = 0
        self._captured_episodes = 0
        self._recorder: EpisodeRecorder | None = None
        if record_dir is not None:
            self._recorder = EpisodeRecorder(Path(record_dir), run_meta=self._meta)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def set_run_plan(self, *, total_episodes: int) -> None:
        self._total_episodes = max(int(total_episodes), 1)

    def get_run_stats(self) -> dict[str, Any]:
        with self._control_lock:
            completed = int(self._completed_episodes)
            captured = int(self._captured_episodes)
            total = int(self._total_episodes)
        return {
            "completed_episodes": completed,
            "captured_episodes": captured,
            "capture_rate": float(captured / completed) if completed else 0.0,
            "total_episodes_planned": total,
        }

    def record_episode_result(self, captured: bool) -> None:
        with self._control_lock:
            self._completed_episodes += 1
            if captured:
                self._captured_episodes += 1

    def get_control_state(self) -> dict[str, Any]:
        with self._control_lock:
            waiting_for_start = bool(
                self._start_paused and not self._episode_run_started and not self._start_gate.is_set()
            )
            if waiting_for_start:
                self._awaiting_start = True
                self._paused = True
            state = {
                "playback_speed": float(self._playback_speed),
                "step_dt": float(self._step_dt),
                "paused": bool(self._paused),
                "awaiting_start": bool(self._awaiting_start),
                "needs_start_click": waiting_for_start,
                "episode_idx": int(self._episode_idx),
                "total_episodes": int(self._total_episodes),
            }
        state["run_stats"] = self.get_run_stats()
        return state

    def set_playback_speed(self, speed: float) -> None:
        with self._control_lock:
            self._playback_speed = max(float(speed), 0.0)

    def set_paused(self, paused: bool) -> None:
        with self._control_lock:
            self._paused = bool(paused)

    def set_step_dt(self, step_dt: float) -> None:
        with self._control_lock:
            self._step_dt = max(float(step_dt), 1e-6)

    def start_run(self) -> None:
        with self._control_lock:
            self._start_acknowledged = True
            self._start_gate.set()
            self._awaiting_start = False
            self._paused = False
        print("[debug-browser] simulation started")

    def arm_start_gate(self) -> None:
        """Sync UI flags right before the sim thread blocks for Start."""
        with self._control_lock:
            if not self._start_paused:
                return
            if self._start_gate.is_set():
                self._awaiting_start = False
                self._paused = False
            else:
                self._awaiting_start = True
                self._paused = True

    def apply_control_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("start"):
            self.start_run()
        if "playback_speed" in payload:
            self.set_playback_speed(float(payload["playback_speed"]))
        if "paused" in payload:
            self.set_paused(bool(payload["paused"]))
            if not bool(payload["paused"]) and not payload.get("start"):
                with self._control_lock:
                    if self._episode_run_started:
                        self._awaiting_start = False
        if "step_dt" in payload:
            self.set_step_dt(float(payload["step_dt"]))
        return self.get_control_state()

    def _is_blocked(self) -> bool:
        with self._control_lock:
            return bool(self._paused or self._awaiting_start)

    def wait_if_blocked(self) -> None:
        """Block while waiting for the browser Start click."""
        if not self._start_paused:
            with self._control_lock:
                self._episode_run_started = True
            return
        self.arm_start_gate()
        while not self._start_gate.is_set():
            time.sleep(0.05)
        with self._control_lock:
            self._episode_run_started = True
            self._awaiting_start = False
            self._paused = False

    def wait_after_step(self) -> None:
        """Block between env steps so browser playback matches playback_speed."""
        while True:
            if self._is_blocked():
                time.sleep(0.05)
                continue
            with self._control_lock:
                speed = float(self._playback_speed)
                step_dt = float(self._step_dt)
            if speed <= 0.0:
                time.sleep(0.05)
                continue
            time.sleep(step_dt / speed)
            return

    def set_meta(self, meta: dict[str, Any]) -> None:
        self._meta = dict(meta)
        if self._recorder is not None:
            self._recorder.set_run_meta(self._meta)

    @property
    def record_dir(self) -> Path | None:
        if self._recorder is None:
            return None
        return self._recorder.record_dir

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        handler_cls = _make_handler(self)
        self._server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="debug-browser-http",
            daemon=True,
        )
        self._thread.start()
        print(f"[debug-browser] open {self.url}")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        self._thread = None

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=64)
        with _HUB_LOCK:
            if self._latest_json is not None:
                try:
                    q.put_nowait(self._latest_json)
                except queue.Full:
                    pass
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with _HUB_LOCK:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _enrich_marker_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = str(payload.get("event", ""))
        if event not in _MARKER_EVENTS or self._latest_visual is None:
            return payload
        enriched = dict(payload)
        for key in _VISUAL_CARRY_KEYS:
            if key not in enriched and key in self._latest_visual:
                enriched[key] = self._latest_visual[key]
        return enriched

    def publish(self, frame: dict[str, Any]) -> None:
        if not self._enabled:
            return
        payload = dict(frame)
        payload.setdefault("ts_ms", int(time.time() * 1000))
        if self._meta:
            payload.setdefault("run_meta", self._meta)
        payload["run_stats"] = self.get_run_stats()
        if payload.get("positions"):
            self._latest_visual = dict(payload)
        elif str(payload.get("event", "")) in _MARKER_EVENTS:
            payload = self._enrich_marker_payload(payload)
        data = json.dumps(payload, ensure_ascii=False)
        if self._recorder is not None:
            try:
                self._recorder.on_frame(payload)
            except Exception:
                pass
        with _HUB_LOCK:
            self._latest_json = data
            dead: list[queue.Queue[str]] = []
            for sub in self._subscribers:
                try:
                    sub.put_nowait(data)
                except queue.Full:
                    try:
                        sub.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        sub.put_nowait(data)
                    except queue.Full:
                        dead.append(sub)
            for sub in dead:
                self._subscribers.remove(sub)

    def next_episode(self) -> int:
        self._episode_idx += 1
        with self._control_lock:
            early_start = self._start_gate.is_set()
            self._start_acknowledged = False
            self._episode_run_started = False
            if self._start_paused:
                if early_start:
                    self._awaiting_start = False
                    self._paused = False
                else:
                    self._start_gate.clear()
                    self._awaiting_start = True
                    self._paused = True
            else:
                self._start_gate.set()
        return self._episode_idx


def configure_debug_browser(
    *,
    enabled: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
    meta: dict[str, Any] | None = None,
    autostart: bool = True,
    playback_speed: float = 1.0,
    step_dt: float = 1.0 / 60.0,
    start_paused: bool = True,
    record_dir: Path | str | None = None,
) -> DebugBrowserHub | None:
    """Enable or disable the global debug browser hub."""
    global _HUB
    with _HUB_LOCK:
        if not enabled:
            if _HUB is not None:
                _HUB.stop()
                _HUB = None
            return None
        if _HUB is None:
            _HUB = DebugBrowserHub(
                host=host,
                port=port,
                playback_speed=playback_speed,
                step_dt=step_dt,
                start_paused=start_paused,
                record_dir=record_dir,
                run_meta=dict(meta or {}),
            )
        else:
            _HUB.host = host
            _HUB.port = port
            _HUB.set_playback_speed(playback_speed)
            _HUB.set_step_dt(step_dt)
            if start_paused:
                _HUB.set_paused(True)
                with _HUB._control_lock:
                    _HUB._awaiting_start = True
                    _HUB._start_acknowledged = False
                    _HUB._episode_run_started = False
                    _HUB._start_gate.clear()
            else:
                _HUB._start_gate.set()
            _HUB._start_paused = bool(start_paused)
        if meta:
            _HUB.set_meta(meta)
        if autostart:
            _HUB.start()
        return _HUB


def get_debug_browser_hub() -> DebugBrowserHub | None:
    with _HUB_LOCK:
        return _HUB


def _build_role_assignment_info(env: Any, backend_state: Any) -> dict[str, Any]:
    task = getattr(env, "task", None)
    task_state = getattr(env, "task_state", None)
    if task is None or task_state is None or not hasattr(task, "_assigned_targets_from_state"):
        return {}
    lin_pos = backend_state.states[:, 3, :]
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64)
    pursuer_pos = np.asarray(lin_pos[pursuer_ids], dtype=np.float32)
    evader_pos = np.asarray(lin_pos[int(task_state.evader_id)], dtype=np.float32)
    slot_targets, assignment, assigned_targets = task._assigned_targets_from_state(
        pursuer_pos,
        evader_pos,
        task_state=task_state,
    )
    out: dict[str, Any] = {
        "slot_targets": _to_list(slot_targets),
        "role_assignment": np.asarray(assignment, dtype=np.int64).tolist(),
        "assigned_targets": _to_list(assigned_targets),
    }
    if getattr(task, "role_assignment_mode", "") == "entropic_ot":
        dist_mat = np.linalg.norm(
            pursuer_pos[:, None, :] - slot_targets[None, :, :],
            axis=2,
        ).astype(np.float64)
        from marl_uav.framework.role_allocation import default_ot_epsilon, sinkhorn_transport_plan

        eps = default_ot_epsilon(
            dist_mat,
            float(getattr(task, "ot_epsilon", 0.05)),
            getattr(task, "ot_epsilon_scale", None),
        )
        plan = sinkhorn_transport_plan(
            dist_mat,
            epsilon=eps,
            num_iters=int(getattr(task, "ot_sinkhorn_iterations", 25)),
        )
        out["ot"] = {
            "epsilon": float(eps),
            "cost_matrix": dist_mat.tolist(),
            "transport_plan": plan.tolist(),
            "inertia_margin": float(getattr(task, "assignment_inertia_margin", 0.0)),
        }
    return out


def build_debug_frame(
    env: Any,
    info: dict[str, Any],
    *,
    event: str,
    dream_manifold: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one simulation snapshot for the browser frontend."""
    task = getattr(env, "task", None)
    task_state = getattr(env, "task_state", None)
    backend_state = getattr(env, "prev_backend_state", None)
    if backend_state is None or task_state is None:
        return {"event": event, "error": "env not ready"}

    lin_pos = np.asarray(backend_state.states[:, 3, :], dtype=np.float32)
    pursuer_ids = np.asarray(task_state.pursuer_ids, dtype=np.int64).tolist()
    evader_id = int(task_state.evader_id)

    positions: list[list[float]] = []
    agent_labels: list[str] = []
    for idx in range(lin_pos.shape[0]):
        if idx in pursuer_ids:
            agent_labels.append(f"P{pursuer_ids.index(idx)}")
        elif idx == evader_id:
            agent_labels.append("E")
        else:
            agent_labels.append(f"A{idx}")
        positions.append(lin_pos[idx].tolist())

    ps = info.get("pursuit_structure") or {}
    viz = _active_viz_profile(extra)
    frame: dict[str, Any] = {
        "event": event,
        "scene_id": "pursuit_3v1",
        "viz": viz,
        "step": int(getattr(env, "step_count", 0)),
        "episode_return": float(info.get("episode_return", getattr(env, "_episode_return", 0.0))),
        "episode_len": int(info.get("episode_len", getattr(env, "_episode_len", 0))),
        "world_xy": float(getattr(task, "world_xy", 20.0)),
        "z_min": float(getattr(task, "z_min", 0.0)),
        "z_max": float(getattr(task, "z_max", 5.0)),
        "positions": positions,
        "agent_labels": agent_labels,
        "pursuer_ids": pursuer_ids,
        "evader_id": evader_id,
        "capture": bool(info.get("capture", False)),
        "termination_reason": str(info.get("termination_reason", "running")),
        "rewards": info.get("rewards"),
    }
    frame["kinematics"] = _build_agent_kinematics(backend_state, agent_labels)
    timing = control_timing_for_frame(env)
    if timing:
        frame["control_timing"] = timing
    hub = get_debug_browser_hub()
    if hub is not None and isinstance(hub._meta.get("speed_bounds"), dict):
        frame["speed_bounds"] = dict(hub._meta["speed_bounds"])
    if viz.get("structure_metrics") and isinstance(ps, dict):
        frame["pursuit_structure"] = {k: float(v) for k, v in ps.items()}

    ref_curve = info.get("reference_manifold_curve")
    ref_targets = info.get("reference_manifold_targets")
    if viz.get("manifold_curve") or viz.get("slot_targets"):
        manifold: dict[str, Any] = {}
        if viz.get("manifold_curve") and ref_curve is not None:
            manifold["curve"] = _to_list(ref_curve)
        if viz.get("slot_targets") and ref_targets is not None:
            manifold["slot_targets"] = _to_list(ref_targets)
        if manifold:
            frame["manifold"] = manifold

    if viz.get("role_allocation") and backend_state is not None:
        role = _build_role_assignment_info(env, backend_state)
        if role and not viz.get("ot_details"):
            role.pop("ot", None)
        if role:
            frame["role"] = role

    if dream_manifold and viz.get("dream_manifold"):
        frame["dream_manifold"] = {k: float(v) for k, v in dream_manifold.items()}

    hub = get_debug_browser_hub()
    controller_cfg = (hub._meta.get("controller") if hub else None) or {}
    ctrl_targets = build_controller_targets(env, viz=viz, controller_cfg=controller_cfg)
    if ctrl_targets:
        frame["controller_targets"] = ctrl_targets

    if task is not None:
        algo_raw = {
            "method": viz.get("method"),
            "task_name": getattr(task, "__class__", type("T", (), {})).__name__,
            "role_assignment_mode": str(getattr(task, "role_assignment_mode", "")),
            "manifold_target_phase": float(getattr(task, "manifold_target_phase", 0.0)),
            "manifold_target_radius_scale": float(getattr(task, "manifold_target_radius_scale", 1.0)),
            "manifold_contraction_rate": float(getattr(task, "manifold_contraction_rate", 0.0)),
            "manifold_structure_gate_scale": float(getattr(task, "manifold_structure_gate_scale", 0.0)),
            "target_radius_xy": float(getattr(task_state, "latest_target_radius_xy", 0.0)),
            "mean_radius_xy": float(getattr(task_state, "prev_mean_radius_xy", 0.0)),
            "structure_hold_steps": int(getattr(task_state, "structure_hold_steps", 0)),
            "ot_epsilon": float(getattr(task, "ot_epsilon", 0.0)),
            "ot_epsilon_scale": getattr(task, "ot_epsilon_scale", None),
            "ot_sinkhorn_iterations": int(getattr(task, "ot_sinkhorn_iterations", 0)),
            "assignment_inertia_margin": float(getattr(task, "assignment_inertia_margin", 0.0)),
            "capture_dist": float(getattr(task, "capture_dist", 0.0)),
        }
        if hub is None or hub._meta.get("execution_kind") != "rl":
            algo_raw["pursuer_speed_xy"] = float(getattr(task, "pursuer_speed_xy", 0.0))
            algo_raw["evader_speed_xy"] = float(getattr(task, "evader_speed_xy", 0.0))
        filtered = filter_algorithm_fields(algo_raw, viz)
        if filtered:
            frame["algorithm"] = filtered

    if viz.get("obstacles"):
        obstacle_xy = info.get("obstacle_xy")
        obstacle_r = info.get("obstacle_r")
        if obstacle_xy is None and task_state is not None:
            obstacle_xy = getattr(task_state, "obstacle_xy", None)
            obstacle_r = getattr(task_state, "obstacle_r", None)
        if obstacle_xy is not None:
            frame["obstacles"] = {
                "xy": _to_list(obstacle_xy),
                "r": _to_list(obstacle_r),
            }

    if extra:
        frame.update(extra)
    return frame


def publish_env_frame(
    env: Any,
    info: dict[str, Any],
    *,
    event: str,
    dream_manifold: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    hub = get_debug_browser_hub()
    if hub is None:
        return
    frame = build_debug_frame(env, info, event=event, dream_manifold=dream_manifold, extra=extra)
    hub.publish(frame)


def publish_episode_marker(event: str, **fields: Any) -> None:
    hub = get_debug_browser_hub()
    if hub is None:
        return
    if event == "episode_end" and "capture" in fields:
        hub.record_episode_result(bool(fields["capture"]))
    viz = dict(hub._meta.get("viz") or {})
    payload = {
        "event": event,
        "scene_id": fields.get("scene_id", "pursuit_3v1"),
        "viz": viz,
        **fields,
        "run_stats": hub.get_run_stats(),
    }
    hub.publish(payload)


_MIME_BY_SUFFIX = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
}


def _content_type_for(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def _make_handler(hub: DebugBrowserHub):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._serve_file(_STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            if path.startswith("/assets/"):
                rel = path[len("/assets/") :]
                target = (_STATIC_DIR / rel).resolve()
                if not str(target).startswith(str(_STATIC_DIR.resolve())):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._serve_file(target, _content_type_for(target))
                return
            if path == "/events":
                self._serve_sse()
                return
            if path == "/health":
                self._send_json({"ok": True, "url": hub.url})
                return
            if path == "/control":
                self._send_json(hub.get_control_state())
                return
            if path == "/api/recordings":
                self._send_recordings_list()
                return
            if path.startswith("/api/recordings/"):
                name = path[len("/api/recordings/") :]
                self._send_recording_file(name)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send_recordings_list(self) -> None:
            if hub._recorder is None:
                self._send_json({"record_dir": None, "episodes": [], "run_stats": None})
                return
            manifest = hub._recorder._manifest
            self._send_json(
                {
                    "record_dir": str(hub._recorder.record_dir),
                    "episodes": hub._recorder.list_episodes(),
                    "run_stats": manifest.get("run_stats"),
                }
            )

        def _send_recording_file(self, filename: str) -> None:
            if hub._recorder is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            safe_name = Path(filename).name
            try:
                doc = hub._recorder.load_episode(safe_name)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_json(doc)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/control":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            state = hub.apply_control_update(payload if isinstance(payload, dict) else {})
            self._send_json(state)

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_sse(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            sub = hub.subscribe()
            try:
                while True:
                    try:
                        msg = sub.get(timeout=15.0)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    chunk = f"data: {msg}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(sub)

    return _Handler
