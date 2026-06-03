/** Header toolbar & mode-specific controls. */

export function mountHeader({ bus, live, file, onModeChange, onStartLive, onPauseLive, onSpeedLive, setOverlay }) {
  const header = document.getElementById("app-header");
  header.innerHTML = `
    <h1>MARL UAV Viz</h1>
    <div class="mode-tabs">
      <button type="button" data-mode="live" class="active">实时</button>
      <button type="button" data-mode="replay">回放</button>
    </div>
    <div id="conn-status" class="status">—</div>
    <div id="episode-status" class="status"></div>
    <div id="capture-rate-status" class="status"></div>
    <div id="step-status" class="status"></div>
    <div class="toolbar" id="live-tools">
      <button type="button" id="btn-start" class="primary">开始</button>
      <button type="button" id="btn-pause" class="hidden">暂停</button>
      <label>倍速 <input id="speed-slider" type="range" min="0.05" max="4" step="0.05" value="0.25" />
        <span id="speed-val">0.25×</span></label>
      <button type="button" id="btn-fit">适应视图</button>
      <button type="button" id="btn-autofit" class="active">自动跟随</button>
    </div>
    <div class="toolbar hidden" id="replay-tools">
      <select id="recording-select"><option value="">— 服务端录制 —</option></select>
      <button type="button" id="btn-load-recording">加载</button>
      <label class="button-like">本地文件 <input id="file-input" type="file" accept=".json,application/json" hidden /></label>
      <button type="button" id="btn-browse">浏览…</button>
      <button type="button" id="btn-r-play">播放</button>
      <button type="button" id="btn-r-pause">暂停</button>
      <label>倍速 <input id="r-speed" type="range" min="0.05" max="4" step="0.05" value="0.25" />
        <span id="r-speed-val">0.25×</span></label>
      <div class="replay-bar"><input id="scrub" type="range" min="0" max="0" value="0" /><span id="scrub-label">0/0</span></div>
    </div>
  `;

  const conn = header.querySelector("#conn-status");
  const epStatus = header.querySelector("#episode-status");
  const captureStatus = header.querySelector("#capture-rate-status");
  const stepStatus = header.querySelector("#step-status");
  const liveTools = header.querySelector("#live-tools");
  const replayTools = header.querySelector("#replay-tools");
  const btnStart = header.querySelector("#btn-start");
  const btnPause = header.querySelector("#btn-pause");
  const speedSlider = header.querySelector("#speed-slider");
  const speedVal = header.querySelector("#speed-val");
  const recordingSelect = header.querySelector("#recording-select");
  const fileInput = header.querySelector("#file-input");

  let mode = "live";
  let awaitingStart = true;
  let autoFit = true;
  let lastFrame = null;
  let lastControlState = null;
  let startRequested = false;
  let liveRunArmed = false;
  let lastLiveStep = null;
  let lastLiveEpisode = 0;

  function resolveLiveStep(frame) {
    const ep = Number(frame.episode ?? 0);
    if (ep > 0 && ep !== lastLiveEpisode) {
      lastLiveEpisode = ep;
      if (frame.event === "episode_start") lastLiveStep = 0;
    }
    if (frame.step !== undefined && frame.step !== null && frame.step !== "") {
      lastLiveStep = Number(frame.step);
      return lastLiveStep;
    }
    if (frame.event === "episode_end" && frame.episode_len !== undefined) {
      lastLiveStep = Number(frame.episode_len);
      return lastLiveStep;
    }
    if (frame.event === "episode_start") return 0;
    return lastLiveStep;
  }

  function syncStartControls(state = null) {
    if (state) lastControlState = state;
    if (state?.run_armed) liveRunArmed = true;
    const st = lastControlState || {};
    const completedEpisodes = Number(
      st?.run_stats?.completed_episodes ?? lastFrame?.run_stats?.completed_episodes ?? 0,
    );
    const atEpisodeHead =
      !startRequested &&
      mode === "live" &&
      lastFrame &&
      (lastFrame.event === "episode_start" || lastFrame.event === "reset") &&
      Number(lastFrame.step ?? 0) === 0;
    const serverWaiting = !!(st.awaiting_start || st.needs_start_click);
    const needsManualStart = serverWaiting || (atEpisodeHead && completedEpisodes === 0);
    awaitingStart = !startRequested && needsManualStart;
    btnStart.classList.toggle("hidden", !awaitingStart);
    btnPause.classList.toggle("hidden", awaitingStart);
    setOverlay(awaitingStart ? "已加载初始帧，点击「开始」运行仿真" : "", awaitingStart);
    if (st.paused !== undefined && !awaitingStart) {
      btnPause.textContent = st.paused ? "继续" : "暂停";
    }
  }

  function updateRunStats(stats) {
    if (!stats || stats.completed_episodes === undefined) {
      captureStatus.textContent = "";
      return;
    }
    const captured = Number(stats.captured_episodes || 0);
    const completed = Number(stats.completed_episodes || 0);
    const pct = (Number(stats.capture_rate || 0) * 100).toFixed(1);
    const planned = Number(stats.total_episodes_planned || 0);
    const planSuffix = planned > 0 ? ` · 计划 ${planned}` : "";
    captureStatus.textContent = `捕获率 ${captured}/${completed} (${pct}%)${planSuffix}`;
  }

  function setMode(next) {
    mode = next;
    if (next === "live") startRequested = false;
    header.querySelectorAll(".mode-tabs button").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === next);
    });
    liveTools.classList.toggle("hidden", next !== "live");
    replayTools.classList.toggle("hidden", next !== "replay");
    onModeChange(next);
    if (next === "replay") refreshRecordings();
  }

  async function refreshRecordings() {
    const data = await file.listServerRecordings();
    recordingSelect.innerHTML = `<option value="">— 服务端录制 (${data.episodes?.length || 0}) —</option>`;
    (data.episodes || []).forEach((ep) => {
      const opt = document.createElement("option");
      opt.value = ep.file;
      opt.textContent = `#${ep.episode} seed=${ep.seed ?? "?"} frames=${ep.num_frames} cap=${ep.capture ? "Y" : "N"}`;
      recordingSelect.appendChild(opt);
    });
    if (data.run_stats) updateRunStats(data.run_stats);
  }

  header.querySelectorAll(".mode-tabs button").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    startRequested = true;
    liveRunArmed = true;
    syncStartControls();
    try {
      const state = await onStartLive();
      if (!state) {
        startRequested = false;
        conn.textContent = "控制失败";
        conn.className = "status err";
        alert("无法发送开始指令。请确认 run_debug_browser.py 正在运行，然后重试。");
        syncStartControls();
        return;
      }
      syncStartControls(state);
      conn.textContent = "运行中";
      conn.className = "status ok";
    } catch (err) {
      startRequested = false;
      conn.textContent = "控制失败";
      conn.className = "status err";
      alert(err?.message || String(err));
      syncStartControls();
    } finally {
      btnStart.disabled = false;
    }
  });
  btnPause.addEventListener("click", () => {
    const paused = btnPause.textContent === "暂停";
    onPauseLive(paused);
    btnPause.textContent = paused ? "继续" : "暂停";
  });

  let speedTimer = null;
  speedSlider.addEventListener("input", () => {
    speedVal.textContent = `${Number(speedSlider.value).toFixed(2)}×`;
    clearTimeout(speedTimer);
    speedTimer = setTimeout(() => onSpeedLive(Number(speedSlider.value)), 120);
  });

  header.querySelector("#btn-fit").addEventListener("click", () => bus.emit("viewport", { fit: true, autoFit: false }));
  header.querySelector("#btn-autofit").addEventListener("click", (ev) => {
    autoFit = !autoFit;
    ev.target.classList.toggle("active", autoFit);
    bus.emit("viewport", { autoFit });
  });

  header.querySelector("#btn-browse").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    try {
      await file.loadFromFile(f);
    } catch (e) {
      alert(e.message);
    }
  });

  header.querySelector("#btn-load-recording").addEventListener("click", async () => {
    const name = recordingSelect.value;
    if (!name) return;
    try {
      await file.loadFromUrl(`/api/recordings/${encodeURIComponent(name)}`);
    } catch (e) {
      alert(e.message);
    }
  });

  const rSpeed = header.querySelector("#r-speed");
  const rSpeedVal = header.querySelector("#r-speed-val");
  const scrub = header.querySelector("#scrub");
  const scrubLabel = header.querySelector("#scrub-label");

  rSpeed.addEventListener("input", () => {
    rSpeedVal.textContent = `${Number(rSpeed.value).toFixed(2)}×`;
    file.playback.setSpeed(Number(rSpeed.value));
  });

  header.querySelector("#btn-r-play").addEventListener("click", () => file.playback.play());
  header.querySelector("#btn-r-pause").addEventListener("click", () => file.playback.pause());

  scrub.addEventListener("input", () => {
    file.playback.pause();
    file.playback.seek(Number(scrub.value));
  });

  bus.on("viewport", ({ fit, autoFit: af }) => {
    if (af !== undefined) autoFit = af;
    bus.emit("draw-request", { fit, autoFit });
  });

  return {
    setMode,
    applyControlState(state) {
      if (!state) return;
      if (state.playback_speed !== undefined) {
        speedSlider.value = String(state.playback_speed);
        speedVal.textContent = `${Number(state.playback_speed).toFixed(2)}×`;
      }
      syncStartControls(state);
      if (state.episode_idx && state.total_episodes) {
        epStatus.textContent = `Episode ${state.episode_idx}/${state.total_episodes}`;
      }
      if (state.run_stats) updateRunStats(state.run_stats);
    },
    onFrame(frame, source) {
      lastFrame = frame;
      if (source === "live" && frame.event === "sim_error") {
        startRequested = false;
        conn.textContent = "仿真错误";
        conn.className = "status err";
        setOverlay(String(frame.error || "仿真运行失败，请查看终端日志"), true);
        syncStartControls();
        return;
      }
      if (source === "live" && frame.event === "step" && Number(frame.step ?? 0) > 0) {
        startRequested = false;
      }
      syncStartControls();
      const stepVal = source === "live" ? resolveLiveStep(frame) : frame.step;
      stepStatus.textContent = `${source} step=${stepVal ?? "—"} event=${frame.event ?? "—"}`;
      if (frame.event === "episode_start") {
        startRequested = false;
        lastLiveStep = 0;
        epStatus.textContent = `Episode ${frame.episode}/${frame.total_episodes || "?"}`;
        if (source === "live") {
          const epNum = Number(frame.episode ?? 0);
          if (liveRunArmed && epNum > 1) {
            live.syncControl({ start: true, paused: false }).then((state) => {
              if (state) syncStartControls(state);
            });
          } else {
            live.fetchControl().then((state) => {
              if (state) syncStartControls(state);
            });
          }
        }
      }
      if (frame.run_stats) updateRunStats(frame.run_stats);
      if (source === "replay" && frame.event === "episode_end" && frame.capture !== undefined) {
        captureStatus.textContent = `本集捕获: ${frame.capture ? "是" : "否"}`;
      }
    },
    onStatus(s) {
      if (s.live === "connected") {
        conn.textContent = mode === "live" ? "已连接" : conn.textContent;
        conn.className = "status ok";
      } else if (s.live === "error") {
        conn.textContent = "连接断开";
        conn.className = "status err";
      }
    },
    onReplayStatus(st) {
      scrub.max = String(Math.max(st.total - 1, 0));
      scrub.value = String(st.index);
      scrubLabel.textContent = `${st.index + 1}/${st.total}`;
      epStatus.textContent = st.episode?.index
        ? `Episode ${st.episode.index}${st.episode.total_episodes ? `/${st.episode.total_episodes}` : ""}`
        : "";
      stepStatus.textContent = `replay ${st.index + 1}/${st.total}`;
    },
    setAutoFit(v) {
      autoFit = v;
      header.querySelector("#btn-autofit").classList.toggle("active", v);
    },
    updateRunStats,
    onEpisodeLoaded(doc) {
      if (doc?.summary?.capture !== undefined && mode === "replay") {
        captureStatus.textContent = `本集捕获: ${doc.summary.capture ? "是" : "否"}`;
      }
    },
  };
}
