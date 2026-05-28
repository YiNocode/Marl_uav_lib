import { createBus } from "./core/bus.js";
import { createViewport, fitViewport, mergeBounds, attachViewportInteractions } from "./core/viewport.js";
import { createTrailStore } from "./core/trails.js";
import { registerScene, getScene, resolveSceneId } from "./scene/registry.js";
import { pursuit3v1Scene } from "./scene/pursuit_3v1.js";
import { createLiveSource } from "./sources/live.js";
import { createFileSource } from "./sources/file.js";
import { mountHeader } from "./ui/header.js";

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
  let mode = "live";
  let ui = null;

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
    ctx.clearRect(0, 0, w, h);
    if (!latest) {
      ctx.fillStyle = "#8b949e";
      ctx.font = "14px sans-serif";
      ctx.fillText(mode === "live" ? "等待仿真数据…" : "请加载 episode 文件", 24, 40);
      return;
    }
    const scene = getScene(resolveSceneId(latest));
    if (forceFit || view.autoFit) {
      const b = scene.collectBounds?.(latest, trails);
      if (b) fitViewport(view, b, w, h);
    }
    scene.draw?.(ctx, { frame: latest, trails, view, w, h });
    sidebar.innerHTML = scene.renderSidebar?.(latest, {
      mode,
      replayIndex: file.playback.index,
    }) || "";
  }

  bus.on("frame", ({ frame, source, replay }) => {
    latest = frame;
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
