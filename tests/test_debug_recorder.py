"""Tests for debug episode file recording."""

from __future__ import annotations

import json
from pathlib import Path

from marl_uav.utils.debug_recorder import (
    DEBUG_EPISODE_FORMAT,
    EpisodeRecorder,
    validate_episode_document,
)


def test_episode_recorder_writes_json_and_manifest(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, run_meta={"step_dt": 0.016, "config": "demo.yaml"})
    rec.on_frame({"event": "reset", "step": 0, "scene_id": "pursuit_3v1", "positions": [[0, 0, 1]]})
    rec.on_frame({"event": "episode_start", "episode": 1, "seed": 101, "total_episodes": 2})
    rec.on_frame({"event": "step", "step": 1, "positions": [[1, 0, 1]]})
    rec.on_frame({"event": "episode_end", "episode_return": 1.5, "episode_len": 1, "capture": False})

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == DEBUG_EPISODE_FORMAT
    assert len(manifest["episodes"]) == 1

    ep_path = tmp_path / manifest["episodes"][0]["file"]
    doc = json.loads(ep_path.read_text(encoding="utf-8"))
    norm = validate_episode_document(doc)
    assert norm["scene_id"] == "pursuit_3v1"
    assert len(norm["frames"]) == 2
    assert norm["frames"][0]["event"] == "reset"
    assert norm["frames"][1]["event"] == "step"
