/**
 * File replay source — reads episode JSON from disk/API only.
 * No dependency on live simulation or algorithm backend.
 */

import { LocalPlayback } from "../core/playback.js";

function normalizeEpisodeDoc(raw) {
  if (raw?.format !== "marl_uav.debug_episode") {
    throw new Error(`Unsupported format: ${raw?.format || "unknown"}`);
  }
  const frames = raw.frames || [];
  if (!frames.length) throw new Error("Episode has no frames");
  return {
    scene_id: raw.scene_id || frames[0]?.scene_id || "pursuit_3v1",
    meta: raw.meta || {},
    episode: raw.episode || {},
    summary: raw.summary || {},
    frames,
    step_dt: Number(raw.meta?.step_dt || 1 / 60),
    viz: raw.meta?.viz || frames[0]?.viz || {},
  };
}

export function createFileSource({ bus }) {
  const playback = new LocalPlayback({
    onFrame: (frame, meta) => bus.emit("frame", { frame, source: "replay", replay: meta }),
    onStatus: (status) => bus.emit("replay-status", status),
  });

  return {
    mode: "replay",
    playback,

    async loadFromUrl(url) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Failed to load ${url}`);
      const raw = await resp.json();
      const doc = normalizeEpisodeDoc(raw);
      playback.loadDocument(doc);
      bus.emit("episode-loaded", { doc, source: "api" });
      return doc;
    },

    async loadFromFile(file) {
      const text = await file.text();
      const raw = JSON.parse(text);
      const doc = normalizeEpisodeDoc(raw);
      playback.loadDocument(doc);
      bus.emit("episode-loaded", { doc, source: "file" });
      return doc;
    },

    async listServerRecordings() {
      const resp = await fetch("/api/recordings");
      if (!resp.ok) return { episodes: [], record_dir: null };
      return resp.json();
    },
  };
}
