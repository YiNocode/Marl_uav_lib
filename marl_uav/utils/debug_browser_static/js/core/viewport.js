/** 2D viewport transform shared by all scene plugins. */

export function createViewport() {
  return {
    cx: 0,
    cy: 0,
    scale: 10,
    panX: 0,
    panY: 0,
    autoFit: true,
  };
}

export function worldToScreen(view, x, y, w, h) {
  return [
    w / 2 + view.panX + (x - view.cx) * view.scale,
    h / 2 + view.panY - (y - view.cy) * view.scale,
  ];
}

export function screenToWorld(view, sx, sy, w, h) {
  return [
    view.cx + (sx - w / 2 - view.panX) / view.scale,
    view.cy - (sy - h / 2 - view.panY) / view.scale,
  ];
}

export function fitViewport(view, bounds, w, h, pad = 36) {
  const spanX = Math.max(bounds.maxX - bounds.minX, 1e-3);
  const spanY = Math.max(bounds.maxY - bounds.minY, 1e-3);
  view.scale = Math.min((w - 2 * pad) / spanX, (h - 2 * pad) / spanY);
  view.cx = (bounds.minX + bounds.maxX) / 2;
  view.cy = (bounds.minY + bounds.maxY) / 2;
  view.panX = 0;
  view.panY = 0;
}

export function mergeBounds(boundsList) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const b of boundsList) {
    if (!b) continue;
    minX = Math.min(minX, b.minX);
    maxX = Math.max(maxX, b.maxX);
    minY = Math.min(minY, b.minY);
    maxY = Math.max(maxY, b.maxY);
  }
  if (!Number.isFinite(minX)) return null;
  return { minX, maxX, minY, maxY };
}

export function attachViewportInteractions({ wrap, canvas, view, onChange }) {
  let dragging = false;
  let dragStart = null;

  wrap.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    view.autoFit = false;
    const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    const [wx, wy] = screenToWorld(view, ev.offsetX, ev.offsetY, wrap.clientWidth, wrap.clientHeight);
    view.scale *= factor;
    const [sx2, sy2] = worldToScreen(view, wx, wy, wrap.clientWidth, wrap.clientHeight);
    view.panX += ev.offsetX - sx2;
    view.panY += ev.offsetY - sy2;
    onChange?.();
  }, { passive: false });

  wrap.addEventListener("mousedown", (ev) => {
    dragging = true;
    dragStart = { x: ev.clientX, y: ev.clientY, panX: view.panX, panY: view.panY };
    wrap.classList.add("dragging");
  });

  window.addEventListener("mousemove", (ev) => {
    if (!dragging || !dragStart) return;
    view.autoFit = false;
    view.panX = dragStart.panX + (ev.clientX - dragStart.x);
    view.panY = dragStart.panY + (ev.clientY - dragStart.y);
    onChange?.();
  });

  window.addEventListener("mouseup", () => {
    dragging = false;
    dragStart = null;
    wrap.classList.remove("dragging");
  });

  wrap.addEventListener("dblclick", () => {
    view.autoFit = true;
    onChange?.(true);
  });
}
