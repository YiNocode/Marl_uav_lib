"""Persist debug-browser episode trajectories to versioned JSON files.

Recording format is intentionally decoupled from simulators / algorithms so the
frontend can replay episodes from disk without a live backend.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

DEBUG_EPISODE_FORMAT = "marl_uav.debug_episode"
DEBUG_EPISODE_VERSION = 1


class EpisodeRecorder:
    """Write one JSON file per episode under ``record_dir``."""

    def __init__(self, record_dir: Path, *, run_meta: dict[str, Any] | None = None) -> None:
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._run_meta = dict(run_meta or {})
        self._lock = threading.Lock()
        self._pending_reset: dict[str, Any] | None = None
        self._episode_frames: list[dict[str, Any]] = []
        self._episode_info: dict[str, Any] = {}
        self._manifest: dict[str, Any] = {
            "format": DEBUG_EPISODE_FORMAT,
            "version": DEBUG_EPISODE_VERSION,
            "created_ms": int(time.time() * 1000),
            "run_meta": self._run_meta,
            "episodes": [],
        }
        self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.record_dir / "manifest.json"

    def set_run_meta(self, meta: dict[str, Any]) -> None:
        with self._lock:
            self._run_meta.update(meta)
            self._manifest["run_meta"] = dict(self._run_meta)
            self._write_manifest_unlocked()

    def on_frame(self, frame: dict[str, Any]) -> None:
        event = str(frame.get("event", ""))
        if event == "reset":
            with self._lock:
                self._pending_reset = dict(frame)
            return
        if event == "episode_start":
            with self._lock:
                self._episode_frames = []
                self._episode_info = {
                    "index": int(frame.get("episode", 0)),
                    "seed": frame.get("seed"),
                    "total_episodes": int(frame.get("total_episodes", 0)),
                }
                if self._pending_reset is not None:
                    self._episode_frames.append(dict(self._pending_reset))
                    self._pending_reset = None
            return
        if event in ("step",):
            with self._lock:
                if self._episode_info:
                    self._episode_frames.append(dict(frame))
            return
        if event == "episode_end":
            self._finalize_episode(frame)
            return

    def _finalize_episode(self, summary: dict[str, Any]) -> None:
        with self._lock:
            if not self._episode_info:
                return
            ep_idx = int(self._episode_info.get("index", 0))
            seed = self._episode_info.get("seed")
            seed_suffix = f"_seed{seed}" if seed is not None else ""
            filename = f"episode_{ep_idx:04d}{seed_suffix}.json"
            payload = {
                "format": DEBUG_EPISODE_FORMAT,
                "version": DEBUG_EPISODE_VERSION,
                "scene_id": self._infer_scene_id(self._episode_frames),
                "meta": dict(self._run_meta),
                "episode": dict(self._episode_info),
                "summary": {
                    "episode_return": summary.get("episode_return"),
                    "episode_len": summary.get("episode_len"),
                    "capture": summary.get("capture"),
                },
                "frames": list(self._episode_frames),
            }
            out_path = self.record_dir / filename
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            entry = {
                "file": filename,
                "episode": ep_idx,
                "seed": seed,
                "num_frames": len(self._episode_frames),
                "capture": bool(summary.get("capture", False)),
                "episode_return": summary.get("episode_return"),
                "episode_len": summary.get("episode_len"),
            }
            episodes = list(self._manifest.get("episodes", []))
            episodes = [e for e in episodes if int(e.get("episode", -1)) != ep_idx]
            episodes.append(entry)
            episodes.sort(key=lambda e: int(e.get("episode", 0)))
            self._manifest["episodes"] = episodes
            if summary.get("run_stats"):
                self._manifest["run_stats"] = dict(summary["run_stats"])
            self._write_manifest_unlocked()
            self._episode_frames = []
            self._episode_info = {}

    @staticmethod
    def _infer_scene_id(frames: list[dict[str, Any]]) -> str:
        for frame in frames:
            if frame.get("scene_id"):
                return str(frame["scene_id"])
            if frame.get("positions") and frame.get("pursuer_ids") is not None:
                return "pursuit_3v1"
        return "generic"

    def list_episodes(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._manifest.get("episodes", []))

    def load_episode(self, filename: str) -> dict[str, Any]:
        path = self.record_dir / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self) -> None:
        with self._lock:
            self._write_manifest_unlocked()

    def _write_manifest_unlocked(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2)


def validate_episode_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Normalize an on-disk or uploaded episode document for replay."""
    if doc.get("format") != DEBUG_EPISODE_FORMAT:
        raise ValueError(f"Unsupported format: {doc.get('format')!r}")
    version = int(doc.get("version", 0))
    if version != DEBUG_EPISODE_VERSION:
        raise ValueError(f"Unsupported version: {version}")
    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Episode document has no frames")
    scene_id = str(doc.get("scene_id") or EpisodeRecorder._infer_scene_id(frames))
    step_dt = float((doc.get("meta") or {}).get("step_dt", 1.0 / 60.0))
    return {
        "scene_id": scene_id,
        "meta": dict(doc.get("meta") or {}),
        "episode": dict(doc.get("episode") or {}),
        "summary": dict(doc.get("summary") or {}),
        "frames": frames,
        "step_dt": step_dt,
    }
