import { createBus } from "./core/bus.js";
import { createViewport, fitViewport, mergeBounds, attachViewportInteractions } from "./core/viewport.js";
import { createTrailStore } from "./core/trails.js";
import { registerScene, getScene, resolveSceneId } from "./scene/registry.js";
import { pursuit3v1Scene, resetObstacleCache } from "./scene/pursuit_3v1.js";
import { createLiveSource } from "./sources/live.js?v=debug-browser-20260602e";
import { createFileSource } from "./sources/file.js";
import { mountHeader } from "./ui/header.js?v=debug-browser-20260602e";

registerScene(pursuit3v1Scene);

export function createApp() {
  const bus = createBus();
  const view = createViewport();
  const trails = createTrailStore();

  const canvas = document.getElementById("view");
  const wrap = document.getElementById("canvas-wrap");
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");

  let latest = null;
  let latestVisual = null;
  let liveEpisode = 0;
  let lastHandledBoundaryKey = "";
  let mode = "live";
  let ui = null;
  let sidebarInteracting = false;

  function sameEpisode(a, b) {
    return Number(a?.episode ?? 0) === Number(b?.episode ?? 0);
  }

  function boundaryKey(event, episode) {
    return `${episode ?? ""}|${event ?? ""}`;
  }

  function shouldApplyLiveFrame(frame) {
    const ep = Number(frame.episode ?? 0);
    if (ep <= 0) return true;
    if (ep > liveEpisode) liveEpisode = ep;
    if (ep < liveEpisode) return false;
    return true;
  }

  sidebar.addEventListener("pointerdown", () => {
    sidebarInteracting = true;
  });
  window.addEventListener("pointerup", () => {
    if (!sidebarInteracting) return;
    sidebarInteracting = false;
    refreshSidebar();
  });
  window.addEventListener("pointercancel", () => {
    sidebarInteracting = false;
  });

  function frameForDraw() {
    if (!latest) return latestVisual;
    if (latest.positions?.length) return latest;
    if (latestVisual?.positions?.length && sameEpisode(latestVisual, latest)) {
      return latestVisual;
    }
    return latest;
  }

  function refreshSidebar() {
    const frame = frameForDraw();
    if (!frame) return;
    const scene = getScene(resolveSceneId(frame));
    const scrollTop = sidebar.scrollTop;
    sidebar.innerHTML = scene.renderSidebar?.(frameForDraw() || latest || frame, {
      mode,
      replayIndex: file.playback.index,
    }) || "";
    sidebar.scrollTop = scrollTop;
  }

  const live = createLiveSource({
    bus,
    onControlState: (state) => ui?.applyControlState?.(state),
  });

  const file = createFileSource({ bus });

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = wrap.clientWidth * dpr;
    canvas.height = wrap.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function draw(forceFit = false) {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    const frame = frameForDraw();
    ctx.clearRect(0, 0, w, h);
    if (!frame) {
      ctx.fillStyle = "#8b949e";
      ctx.font = "14px sans-serif";
      ctx.fillText(mode === "live" ? "等待仿真数据…" : "请加载 episode 文件", 24, 40);
      return;
    }
    const scene = getScene(resolveSceneId(frame));
    if (forceFit || view.autoFit) {
      const b = scene.collectBounds?.(frame, trails);
      if (b) fitViewport(view, b, w, h);
    }
    scene.draw?.(ctx, { frame, trails, view, w, h });
    if (!sidebarInteracting) refreshSidebar();
  }

  bus.on("live-boundary", ({ event, episode }) => {
    const key = boundaryKey(event, episode);
    if (key === lastHandledBoundaryKey) return;
    lastHandledBoundaryKey = key;
    const ep = Number(episode ?? 0);
    if (ep > 0) liveEpisode = ep;
    if (event === "episode_start") {
      latestVisual = null;
      resetObstacleCache();
      trails.clear();
      view.autoFit = true;
    }
  });

  bus.on("frame", ({ frame, source, replay }) => {
    if (source === "live" && !shouldApplyLiveFrame(frame)) return;

    latest = frame;
    const epMatch = !latestVisual || sameEpisode(latestVisual, frame);
    if (frame?.positions?.length && (epMatch || frame.event !== "episode_end")) {
      latestVisual = frame;
    }
    if (source === "live" && frame?.event === "episode_start") {
      const ep = Number(frame.episode ?? 0);
      if (ep > 0) liveEpisode = ep;
      if (frame.positions?.length) {
        latestVisual = frame;
      }
    }
    if (source === "live" || !replay?.seek) trails.onFrame(frame);
    if (source === "replay" && replay?.seek) trails.clear();
    if (source === "replay" && replay?.seek) {
      // rebuild trails up to current index when scrubbing
      trails.clear();
      for (let i = 0; i <= file.playback.index; i++) trails.onFrame(file.playback.frames[i]);
    }
    draw();
    ui?.onFrame?.(frame, source);
  });

  bus.on("status", (s) => ui?.onStatus?.(s));
  bus.on("replay-status", (s) => ui?.onReplayStatus?.(s));
  bus.on("episode-loaded", ({ doc }) => {
    resetObstacleCache();
    trails.clear();
    view.autoFit = true;
    ui?.onEpisodeLoaded?.(doc);
    draw(true);
  });

  attachViewportInteractions({
    wrap,
    canvas,
    view,
    onChange: (fit) => draw(fit),
  });

  bus.on("viewport", ({ fit, autoFit: af }) => {
    if (af !== undefined) view.autoFit = af;
    ui?.setAutoFit?.(view.autoFit);
    draw(fit || view.autoFit);
  });

  bus.on("draw-request", ({ fit, autoFit: af }) => {
    if (af !== undefined) view.autoFit = af;
    draw(fit);
  });

  ui = mountHeader({
    bus,
    live,
    file,
    onModeChange(next) {
      mode = next;
      if (next === "live") {
        file.playback.pause();
        live.start();
      } else {
        live.stop();
      }
      latest = null;
      latestVisual = null;
      trails.clear();
      overlay.classList.add("hidden");
      draw();
    },
    onStartLive: () => live.syncControl({ start: true, paused: false }),
    onPauseLive: (paused) => live.syncControl({ paused }),
    onSpeedLive: (speed) => live.syncControl({ playback_speed: speed }),
    setOverlay(text, visible) {
      overlay.textContent = text || "";
      overlay.classList.toggle("hidden", !visible);
    },
  });

  window.addEventListener("resize", resize);

  ui.setMode("live");
  live.start();
  resize();

  return { bus, live, file };
}
