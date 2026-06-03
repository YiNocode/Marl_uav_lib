/** Live SSE frame source — streams from running simulation backend. */

function frameKey(frame) {
  return [
    frame.episode ?? "",
    frame.event ?? "",
    frame.step ?? "",
    frame.episode_len ?? "",
    frame.ts_ms ?? "",
  ].join("|");
}

function isBoundaryEvent(event) {
  return event === "episode_start" || event === "episode_end" || event === "reset";
}

function boundaryKey(frame) {
  return `${frame.episode ?? ""}|${frame.event ?? ""}`;
}

export function createLiveSource({ bus, onControlState }) {
  let es = null;
  let latestTimer = null;
  let latestKey = "";
  let lastBoundaryKey = "";
  let reconnectTimer = null;
  let reconnectDelayMs = 500;

  function emitFrame(frame, source) {
    const key = frameKey(frame);
    const force = frame.event === "step" || frame.event === "reset";
    if (!force && key === latestKey) return;
    latestKey = key;
    bus.emit("frame", { frame, source });
  }

  function resetStreamState(frame) {
    if (!frame || !isBoundaryEvent(frame.event)) return;
    const bk = boundaryKey(frame);
    if (bk === lastBoundaryKey) return;
    lastBoundaryKey = bk;
    latestKey = "";
    bus.emit("live-boundary", { event: frame.event, episode: frame.episode });
  }

  function scheduleReconnect() {
    if (reconnectTimer !== null) return;
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      if (es) {
        es.close();
        es = null;
      }
      reconnectDelayMs = Math.min(reconnectDelayMs * 1.5, 5000);
      connectEventSource();
    }, reconnectDelayMs);
  }

  function connectEventSource() {
    es = new EventSource("/events");
    es.onopen = () => {
      reconnectDelayMs = 500;
      bus.emit("status", { live: "connected" });
      fetchControl();
      fetchLatest().catch(() => {});
    };
    es.onerror = () => {
      bus.emit("status", { live: "error" });
      scheduleReconnect();
    };
    es.onmessage = (ev) => {
      try {
        const frame = JSON.parse(ev.data);
        if (isBoundaryEvent(frame.event)) resetStreamState(frame);
        emitFrame(frame, "live");
        bus.emit("status", {
          live: `sse ep=${frame.episode ?? "?"} step=${frame.step ?? "?"} ${frame.event ?? "?"}`,
        });
      } catch (e) {
        console.error(e);
      }
    };
  }

  async function syncControl(payload) {
    const resp = await fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) return null;
    const state = await resp.json();
    onControlState?.(state);
    return state;
  }

  async function fetchControl() {
    const resp = await fetch("/control");
    if (!resp.ok) return null;
    const state = await resp.json();
    onControlState?.(state);
    return state;
  }

  async function fetchLatest() {
    const resp = await fetch(`/latest?_=${Date.now()}`, { cache: "no-store" });
    if (!resp.ok) {
      bus.emit("status", { live: "latest-error" });
      return null;
    }
    const frame = await resp.json();
    if (!frame || frame.event === "no_frame") {
      bus.emit("status", { live: "connected/no-frame" });
      return frame;
    }
    if (isBoundaryEvent(frame.event)) resetStreamState(frame);
    emitFrame(frame, "live");
    bus.emit("status", {
      live: `poll ep=${frame.episode ?? "?"} step=${frame.step ?? "?"} ${frame.event ?? "?"}`,
    });
    return frame;
  }

  return {
    mode: "live",
    start() {
      if (es) return;
      connectEventSource();
      fetchLatest().catch(() => {});
      latestTimer = window.setInterval(() => {
        fetchLatest().catch(() => {});
      }, 200);
    },
    stop() {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      es?.close();
      es = null;
      if (latestTimer) window.clearInterval(latestTimer);
      latestTimer = null;
      latestKey = "";
      lastBoundaryKey = "";
      bus.emit("status", { live: "stopped" });
    },
    syncControl,
    fetchControl,
    fetchLatest,
  };
}
