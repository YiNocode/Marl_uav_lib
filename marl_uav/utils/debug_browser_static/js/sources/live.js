/** Live SSE frame source — streams from running simulation backend. */

export function createLiveSource({ bus, onControlState }) {
  let es = null;

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

  return {
    mode: "live",
    start() {
      if (es) return;
      es = new EventSource("/events");
      es.onopen = () => {
        bus.emit("status", { live: "connected" });
        fetchControl();
      };
      es.onerror = () => bus.emit("status", { live: "error" });
      es.onmessage = (ev) => {
        try {
          const frame = JSON.parse(ev.data);
          bus.emit("frame", { frame, source: "live" });
        } catch (e) {
          console.error(e);
        }
      };
    },
    stop() {
      es?.close();
      es = null;
      bus.emit("status", { live: "stopped" });
    },
    syncControl,
    fetchControl,
  };
}
