import { worldToScreen } from "../core/viewport.js";

const COLORS = ["#58a6ff", "#3fb950", "#d2a8ff"];
const EVADER = "#f85149";
const MANIFOLD = "#ffa657";
const SLOT = "#79c0ff";
const LINK = "#a371f7";
const CTRL = "#79c0ff";
const PATH_COLORS = ["#58a6ff", "#3fb950", "#d2a8ff"];
const TRACK = "#e3b341";
const OB_HIGHLIGHT = ["#58a6ff", "#3fb950", "#d2a8ff"];
const MANIFOLD_OBS = "#f0883e";
const CANDIDATE = "#8b949e";
const CANDIDATE_BAD = "#f85149";
const CANDIDATE_SELECTED = "#3fb950";
const ACTUAL_VEL = "#f0883e";
const BACKEND_CMD = "#3fb950";

/** Static obstacles persist across step frames when backend omits them. */
let cachedObstacles = null;

function resolveObstacles(frame) {
  if (frame?.obstacles?.xy?.length) {
    cachedObstacles = frame.obstacles;
    return cachedObstacles;
  }
  return cachedObstacles;
}

export function resetObstacleCache() {
  cachedObstacles = null;
}

function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "number") return v.toFixed(d);
  return String(v);
}

function fmtMs(sec) {
  if (sec === null || sec === undefined || Number.isNaN(Number(sec))) return "—";
  return `${(Number(sec) * 1000).toFixed(3)} ms`;
}

function fmtHz(hz) {
  if (hz === null || hz === undefined || Number.isNaN(Number(hz))) return "—";
  return `${Number(hz).toFixed(1)} Hz`;
}

function panel(title, rows) {
  const rowsHtml = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");
  return `<section class="panel"><h2>${title}</h2><div class="kv">${rowsHtml}</div></section>`;
}

function vizOf(frame) {
  return frame?.viz || {};
}

function isManifoldOnlyViz(viz) {
  return Boolean(
    viz?.manifold_only
    || viz?.method === "trajectory_planner"
    || viz?.method === "slot_exec_mappo"
  );
}

function agentColor(label, frame, agentIdx) {
  if (label === "E") return EVADER;
  const pi = frame.pursuer_ids?.indexOf(agentIdx);
  return COLORS[pi] ?? "#8b949e";
}

/** Vertical column chart: altitude rises from z_min baseline. */
function renderHeightChart(frame) {
  const positions = frame?.positions;
  if (!positions?.length) return "";

  const zMin = Number.isFinite(Number(frame.z_min)) ? Number(frame.z_min) : 0.5;
  const zMax = Number.isFinite(Number(frame.z_max)) ? Number(frame.z_max) : 5.0;
  const range = Math.max(zMax - zMin, 0.05);
  const n = positions.length;
  const barW = 34;
  const gap = 14;
  const padL = 40;
  const padR = 10;
  const padT = 14;
  const padB = 28;
  const chartH = 156;
  const totalBarW = n * barW + (n - 1) * gap;
  const w = padL + totalBarW + padR;
  const h = padT + chartH + padB;
  const baseY = padT + chartH;
  const yOf = (z) => baseY - ((z - zMin) / range) * chartH;

  let svg = `<section class="panel height-panel"><h2>高度</h2>`;
  svg += `<svg class="height-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="无人机高度">`;

  svg += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${baseY}" stroke="#484f58" stroke-width="1"/>`;
  svg += `<line x1="${padL}" y1="${baseY}" x2="${padL + totalBarW}" y2="${baseY}" stroke="#484f58" stroke-width="1"/>`;
  svg += `<text x="${padL - 6}" y="${baseY + 3}" fill="#8b949e" font-size="9" text-anchor="end">${fmt(zMin, 1)}</text>`;
  svg += `<text x="${padL - 6}" y="${padT + 4}" fill="#8b949e" font-size="9" text-anchor="end">${fmt(zMax, 1)}</text>`;

  const yMid = yOf((zMin + zMax) / 2);
  svg += `<line x1="${padL}" y1="${yMid}" x2="${padL + totalBarW}" y2="${yMid}" stroke="#21262d" stroke-dasharray="3,5"/>`;

  positions.forEach((pos, i) => {
    const label = frame.agent_labels?.[i] || `A${i}`;
    const z = Number(pos[2] ?? 0);
    const color = agentColor(label, frame, i);
    const oob = z < zMin - 1e-4 || z > zMax + 1e-4;
    const x = padL + i * (barW + gap);
    const clampedZ = Math.max(zMin, Math.min(zMax, z));
    const barHeight = Math.max(((clampedZ - zMin) / range) * chartH, clampedZ > zMin ? 2 : 0);
    const barY = baseY - barHeight;

    svg += `<rect x="${x}" y="${padT}" width="${barW}" height="${chartH}" fill="#21262d" rx="4"/>`;
    if (barHeight > 0) {
      svg += `<rect x="${x}" y="${barY}" width="${barW}" height="${barHeight}" fill="${oob ? "#f85149" : color}" opacity="0.92" rx="4"/>`;
    }
    if (oob) {
      const tipY = z > zMax ? padT - 2 : baseY + 2;
      const tipX = x + barW / 2;
      const dir = z > zMax ? -1 : 1;
      svg += `<path d="M ${tipX - 5} ${tipY + 4 * dir} L ${tipX} ${tipY} L ${tipX + 5} ${tipY + 4 * dir} Z" fill="#f85149"/>`;
    }
    svg += `<text x="${x + barW / 2}" y="${baseY + 16}" fill="${color}" font-size="11" font-weight="600" text-anchor="middle">${label}</text>`;
    const labelY = barHeight > 0 ? barY - 5 : baseY - 8;
    svg += `<text x="${x + barW / 2}" y="${labelY}" fill="${oob ? "#f85149" : "#e6edf3"}" font-size="9" text-anchor="middle" font-family="ui-monospace, monospace">${fmt(z, 2)}</text>`;
  });

  svg += `</svg></section>`;
  return svg;
}

function drawTargetMarkers(ctx, wts, points, { labelPrefix = "T", color = CTRL } = {}) {
  points.forEach((pt, i) => {
    const [sx, sy] = wts(pt[0], pt[1]);
    ctx.fillStyle = color;
    ctx.strokeStyle = "#fff";
    ctx.fillRect(sx - 5, sy - 5, 10, 10);
    ctx.strokeRect(sx - 5, sy - 5, 10, 10);
    ctx.fillStyle = "#8b949e";
    ctx.font = "10px sans-serif";
    ctx.fillText(`${labelPrefix}${i}`, sx + 7, sy - 7);
  });
}

function drawAssignmentLines(ctx, wts, frame, targets) {
  targets.forEach((tgt, pi) => {
    const ppos = frame.positions?.[frame.pursuer_ids?.[pi]];
    if (!ppos || !tgt) return;
    const [x1, y1] = wts(ppos[0], ppos[1]);
    const [x2, y2] = wts(tgt[0], tgt[1]);
    ctx.strokeStyle = LINK;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawArrow(ctx, wts, start, vec, {
  color = "#e6edf3",
  label = "",
  scaleS = 2.0,
  lineDash = [],
  width = 2,
} = {}) {
  if (!start || !vec || vec.length < 2) return;
  const vx = Number(vec[0]);
  const vy = Number(vec[1]);
  if (!Number.isFinite(vx) || !Number.isFinite(vy)) return;
  const mag = Math.hypot(vx, vy);
  if (mag < 1e-5) return;
  const end = [Number(start[0]) + vx * scaleS, Number(start[1]) + vy * scaleS];
  const [x1, y1] = wts(start[0], start[1]);
  const [x2, y2] = wts(end[0], end[1]);
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const head = 8;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(lineDash);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
  ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  if (label) {
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(label, x2 + 5, y2 - 4);
  }
  ctx.restore();
}

function drawWorldAxes(ctx, wts, half) {
  ctx.save();
  ctx.strokeStyle = "#30363d";
  ctx.fillStyle = "#8b949e";
  ctx.lineWidth = 1.3;
  drawArrow(ctx, wts, [-half, 0], [2 * half, 0], { color: "#30363d", label: "X+", scaleS: 1.0, width: 1.3 });
  drawArrow(ctx, wts, [0, -half], [0, 2 * half], { color: "#30363d", label: "Y+", scaleS: 1.0, width: 1.3 });
  const [ox, oy] = wts(0, 0);
  ctx.fillStyle = "#8b949e";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("O", ox + 5, oy + 12);
  ctx.restore();
}

function drawVelocityVectors(ctx, wts, frame) {
  const positions = frame?.positions || [];
  if (!positions.length) return;
  const scaleS = Number(frame?.viz?.velocity_vector_scale_s ?? 2.5);
  const pursuerIds = frame?.pursuer_ids || [];
  const agents = frame?.kinematics?.agents || [];
  const byLabel = new Map(agents.map((a) => [a.label, a]));

  pursuerIds.forEach((agentIdx, pi) => {
    const pos = positions[agentIdx];
    if (!pos) return;
    const label = frame.agent_labels?.[agentIdx] || `P${pi}`;
    const kin = byLabel.get(label);
    if (kin?.linear_world_xy) {
      drawArrow(ctx, wts, pos, kin.linear_world_xy, {
        color: ACTUAL_VEL,
        label: "v",
        scaleS,
        width: 2,
      });
    }
    const cmd = frame.deploy_control?.pursuers?.[pi]?.backend_cmd_world_xy;
    if (cmd) {
      drawArrow(ctx, wts, pos, cmd, {
        color: BACKEND_CMD,
        label: "cmd",
        scaleS,
        lineDash: [4, 3],
        width: 2,
      });
    }
  });

  ctx.save();
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillStyle = ACTUAL_VEL;
  ctx.fillText("orange: actual velocity (world)", 10, 52);
  ctx.fillStyle = BACKEND_CMD;
  ctx.fillText("green: backend cmd ground [vx,vy]", 10, 68);
  ctx.restore();
}

function drawCandidateSlots(ctx, wts, frame) {
  const cs = frame.deploy_control?.candidate_slots;
  const pts = cs?.positions || [];
  if (!pts.length) return;
  const selected = new Set((cs.selected_indices || []).map((x) => Number(x)));
  pts.forEach((pt, i) => {
    const [sx, sy] = wts(pt[0], pt[1]);
    const reachable = cs.reachable?.[i] !== false;
    const losBlocked = cs.los_blocked?.[i] === true;
    const isSelected = selected.has(i);
    ctx.save();
    ctx.globalAlpha = isSelected ? 1.0 : 0.62;
    ctx.strokeStyle = isSelected ? CANDIDATE_SELECTED : (reachable ? CANDIDATE : CANDIDATE_BAD);
    ctx.fillStyle = isSelected ? "rgba(63,185,80,0.22)" : "rgba(139,148,158,0.10)";
    ctx.lineWidth = isSelected ? 2.2 : 1.2;
    ctx.setLineDash(losBlocked ? [3, 3] : []);
    ctx.beginPath();
    ctx.arc(sx, sy, isSelected ? 7 : 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
    if (isSelected) {
      ctx.fillStyle = CANDIDATE_SELECTED;
      ctx.font = "10px sans-serif";
      ctx.fillText(`C${i}`, sx + 8, sy + 3);
    }
    ctx.restore();
  });
}

function limitReasonLabel(reason) {
  const map = {
    path_tangent_cruise: "路径切线满速",
    path_cruise: "路径巡航（满速）",
    tube_cruise: "管道内满速",
    tube_slowdown: "管道边缘减速",
    tube_off_path: "偏离管道",
    approach: "末端接近",
    lookahead_dist: "前瞻距离过近",
    proportional: "比例距离限速",
    yaw_misalign: "偏航未对齐",
    saturated: "已达上限",
    cbf_obstacle: "CBF 障碍物约束",
    turn_safety: "转弯半径避障",
  };
  return map[reason] || reason || "—";
}

function drawManifoldCurve(ctx, wts, frame) {
  const curve = frame?.manifold?.curve;
  if (!curve?.length) return;
  ctx.strokeStyle = MANIFOLD;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  curve.forEach((pt, i) => {
    const [sx, sy] = wts(pt[0], pt[1]);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  });
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1.0;
}

function drawDeployControlOverlay(ctx, wts, frame, viz) {
  const dc = frame.deploy_control;
  if (!dc?.pursuers?.length) return;
  if (isManifoldOnlyViz(viz)) return;

  if (viz.candidate_slots || viz.path_tracking) {
    dc.pursuers.forEach((p, pi) => {
      const slot = p.slot_target_xy;
      const ppos = frame.positions?.[frame.pursuer_ids?.[pi]];
      if (!slot || !ppos) return;
      const [px, py] = wts(ppos[0], ppos[1]);
      const [sx, sy] = wts(slot[0], slot[1]);
      ctx.strokeStyle = PATH_COLORS[pi] || LINK;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(sx, sy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#79c0ff";
      ctx.strokeStyle = PATH_COLORS[pi] || "#fff";
      ctx.fillRect(sx - 5, sy - 5, 10, 10);
      ctx.strokeRect(sx - 5, sy - 5, 10, 10);
      ctx.fillStyle = "#e6edf3";
      ctx.font = "10px sans-serif";
      ctx.fillText(`A${pi}`, sx + 7, sy - 7);
    });
  }

  if (viz.path_tracking && !frame?.manifold?.pursuer_curves?.length) {
    dc.pursuers.forEach((p, pi) => {
      const path = p.assigned_path_xy;
      if (!path?.length) return;
      ctx.strokeStyle = PATH_COLORS[pi] || "#8b949e";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      path.forEach((pt, i) => {
        const [sx, sy] = wts(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      const track = p.track_target_xy;
      if (track) {
        const [tx, ty] = wts(track[0], track[1]);
        ctx.fillStyle = TRACK;
        ctx.strokeStyle = PATH_COLORS[pi] || "#8b949e";
        ctx.beginPath();
        ctx.arc(tx, ty, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#e6edf3";
        ctx.font = "10px sans-serif";
        ctx.fillText(`W${pi}`, tx + 7, ty - 4);
      }
    });
  }

  const obstacles = resolveObstacles(frame);
  if (!obstacles?.xy) return;

  const highlightIdx = new Set();
  (dc.manifold_obstacles || []).forEach((o) => highlightIdx.add(Number(o.index)));
  dc.pursuers.forEach((p, pi) => {
    (p.local_obstacle_indices || []).forEach((idx) => highlightIdx.add(Number(idx)));
    (p.cbf?.active_obstacle_indices || []).forEach((idx) => highlightIdx.add(Number(idx)));

    const ppos = frame.positions?.[frame.pursuer_ids?.[pi]];
    if (!ppos) return;
    const [px, py] = wts(ppos[0], ppos[1]);

    (p.local_obstacle_indices || []).forEach((idx) => {
      const o = obstacles.xy[idx];
      const r = obstacles.r?.[idx] || 0.5;
      if (!o) return;
      const [ox, oy] = wts(o[0], o[1]);
      ctx.strokeStyle = OB_HIGHLIGHT[pi] || "#8b949e";
      ctx.lineWidth = 2;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(ox, oy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = OB_HIGHLIGHT[pi] || "#8b949e";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(ox, oy, Math.max(r * frame._viewScale || 4, 4), 0, Math.PI * 2);
      ctx.stroke();
    });
  });

  (dc.manifold_obstacles || []).forEach((mo) => {
    const idx = Number(mo.index);
    const o = obstacles.xy[idx];
    const r = obstacles.r?.[idx] || 0.5;
    if (!o) return;
    const [ox, oy] = wts(o[0], o[1]);
    const rs = Math.max(r * (frame._viewScale || 4), 4);
    ctx.strokeStyle = MANIFOLD_OBS;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.arc(ox, oy, rs + 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = MANIFOLD_OBS;
    ctx.font = "10px sans-serif";
    ctx.fillText(`M${idx}`, ox + rs + 4, oy - 4);
  });
}

function drawDeployControlOverlayWithView(ctx, wts, frame, viz, view) {
  frame._viewScale = view?.scale || 1;
  drawDeployControlOverlay(ctx, wts, frame, viz);
}

export const pursuit3v1Scene = {
  id: "pursuit_3v1",
  label: "3v1 Pursuit",

  collectBounds(frame, trails) {
    const viz = vizOf(frame);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const add = (x, y) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    };
    const half = (frame?.world_xy || 20) / 2;
    add(-half, -half); add(half, half);
    (frame?.positions || []).forEach((p) => add(p[0], p[1]));
    for (const [, pts] of trails.entries()) pts.forEach(([x, y]) => add(x, y));

    if (viz.manifold_curve) {
      (frame?.manifold?.pursuer_curves || []).forEach((path) => {
        (path || []).forEach((p) => add(p[0], p[1]));
      });
      if (!frame?.manifold?.pursuer_curves?.length) {
        (frame?.manifold?.curve || []).forEach((p) => add(p[0], p[1]));
      }
    }
    if (viz.fixed_ring_curve) {
      (frame?.controller_targets?.ring_curve || []).forEach((p) => add(p[0], p[1]));
    }
    if (viz.slot_targets) {
      const slots = frame?.role?.slot_targets || frame?.manifold?.slot_targets || [];
      slots.forEach((p) => add(p[0], p[1]));
    }
    if (viz.candidate_slots) {
      (frame?.deploy_control?.candidate_slots?.positions || []).forEach((p) => add(p[0], p[1]));
    }
    if (viz.pursuit_targets || viz.fixed_ring_targets) {
      (frame?.controller_targets?.targets || []).forEach((p) => add(p[0], p[1]));
    }
    if (viz.obstacles !== false) {
      const obstacles = resolveObstacles(frame);
      (obstacles?.xy || []).forEach((o, i) => {
        const r = obstacles.r?.[i] || 0.5;
        add(o[0] - r, o[1] - r); add(o[0] + r, o[1] + r);
      });
    }
    if (!Number.isFinite(minX)) return null;
    const margin = Math.max(maxX - minX, maxY - minY) * 0.08 + 0.5;
    return { minX: minX - margin, maxX: maxX + margin, minY: minY - margin, maxY: maxY + margin };
  },

  draw(ctx, { frame, trails, view, w, h }) {
    if (!frame) return;
    const viz = vizOf(frame);
    const half = (frame.world_xy || 20) / 2;
    const wts = (x, y) => worldToScreen(view, x, y, w, h);
    const hasPositions = Boolean(frame.positions?.length);

    drawWorldAxes(ctx, wts, half);

    ctx.strokeStyle = "#30363d";
    const tl = wts(-half, half);
    const br = wts(half, -half);
    ctx.strokeRect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]);

    if (viz.obstacles !== false) {
      const obstacles = resolveObstacles(frame);
      if (obstacles?.xy) {
        ctx.fillStyle = "rgba(139,148,158,0.25)";
        ctx.strokeStyle = "#8b949e";
        obstacles.xy.forEach((o, i) => {
          const r = obstacles.r?.[i] || 0.5;
          const [cx, cy] = wts(o[0], o[1]);
          const rs = Math.max(r * view.scale, 2);
          ctx.beginPath();
          ctx.arc(cx, cy, rs, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        });
      }
    }

    if (hasPositions && viz.fixed_ring_curve && frame.controller_targets?.ring_curve?.length) {
      ctx.strokeStyle = "#a371f7";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      frame.controller_targets.ring_curve.forEach((pt, i) => {
        const [sx, sy] = wts(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (viz.manifold_curve && frame.manifold?.curve) {
      drawManifoldCurve(ctx, wts, frame);
    } else if (viz.manifold_curve && frame.manifold?.pursuer_curves?.length) {
      drawManifoldCurve(ctx, wts, frame);
    }

    if (hasPositions) {
      if (viz.slot_targets) {
        const slotPts = frame.role?.slot_targets || frame.manifold?.slot_targets;
        if (slotPts) drawTargetMarkers(ctx, wts, slotPts, { labelPrefix: "S", color: SLOT });
      }

      if (viz.candidate_slots) {
        drawCandidateSlots(ctx, wts, frame);
      }

      if (viz.role_allocation && frame.role?.assigned_targets) {
        drawAssignmentLines(ctx, wts, frame, frame.role.assigned_targets);
      } else if ((viz.pursuit_targets || viz.fixed_ring_targets) && frame.controller_targets?.targets) {
        drawAssignmentLines(ctx, wts, frame, frame.controller_targets.targets);
        if (viz.fixed_ring_targets) {
          drawTargetMarkers(ctx, wts, frame.controller_targets.targets, { labelPrefix: "R", color: CTRL });
        }
      }

      if (!isManifoldOnlyViz(viz) && (viz.path_tracking || viz.speed_diagnostics)) {
        drawDeployControlOverlayWithView(ctx, wts, frame, viz, view);
      }
      drawVelocityVectors(ctx, wts, frame);
    }

    if (!isManifoldOnlyViz(viz)) {
      for (const [label, pts] of trails.entries()) {
        if (pts.length < 2) continue;
        const isEv = label === "E";
        ctx.strokeStyle = isEv ? EVADER : COLORS[["P0", "P1", "P2"].indexOf(label)] || "#8b949e";
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        pts.forEach(([x, y], i) => {
          const [sx, sy] = wts(x, y);
          if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    if (!hasPositions) return;

    frame.positions.forEach((pos, i) => {
      const label = frame.agent_labels?.[i] || `A${i}`;
      const isEv = label === "E";
      const pi = frame.pursuer_ids?.indexOf(i);
      ctx.fillStyle = isEv ? EVADER : COLORS[pi] || "#8b949e";
      const [sx, sy] = wts(pos[0], pos[1]);
      ctx.beginPath();
      ctx.arc(sx, sy, isEv ? 7 : 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = "#e6edf3";
      ctx.font = "11px sans-serif";
      ctx.fillText(label, sx + 9, sy + 4);
      const kin = frame.kinematics?.agents?.find((a) => a.label === label);
      if (kin) {
        const cap = label === "E"
          ? frame.speed_bounds?.evader_speed_xy_cap
          : frame.speed_bounds?.pursuer_speed_xy_cap;
        const over = cap != null && kin.speed_xy > cap * 1.02;
        ctx.fillStyle = over ? "#f85149" : "#8b949e";
        ctx.font = "10px sans-serif";
        ctx.fillText(`${kin.speed_xy.toFixed(2)} m/s`, sx + 9, sy + 16);
      }
    });

    ctx.fillStyle = "#8b949e";
    ctx.font = "11px sans-serif";
    ctx.fillText(`缩放 ${view.scale.toFixed(2)} px/m`, 10, 18);
    if (viz.method) ctx.fillText(String(viz.method), 10, 34);
  },

  renderSidebar(frame, ctx = {}) {
    const viz = vizOf(frame);
    const ps = frame?.pursuit_structure || {};
    const algo = frame?.algorithm || {};
    const role = frame?.role || {};
    const ot = role.ot || {};
    const dm = frame?.dream_manifold || {};

    let html = renderHeightChart(frame);

    html += panel("仿真", [
      ["模式", ctx.mode === "replay" ? "文件回放" : "实时"],
      ["方法", viz.method || algo.method || "—"],
      ["事件", frame?.event || "—"],
      ["步数", frame?.step ?? ctx.replayIndex ?? "—"],
      ["捕获", frame?.capture ? "是" : "否"],
      ["终止原因", frame?.termination_reason || "—"],
    ]);

    if (frame?.run_stats?.completed_episodes) {
      const rs = frame.run_stats;
      html += panel("累计捕获率", [
        ["成功", `${rs.captured_episodes}/${rs.completed_episodes}`],
        ["比率", `${(Number(rs.capture_rate || 0) * 100).toFixed(1)}%`],
      ]);
    }

    if (frame?.kinematics?.agents?.length) {
      const sb = frame.speed_bounds || {};
      const pursuerCap = sb.pursuer_speed_xy_cap;
      const evaderCap = sb.evader_speed_xy_cap;
      const capNote =
        sb.source === "suite"
          ? `上界 ±${fmt(pursuerCap, 2)} m/s（suite: ${sb.suite_ref || "e1_1_open_space_suite.yaml"}）`
          : sb.source === "config"
            ? `上界 ±${fmt(pursuerCap, 2)} m/s（task.pursuer_speed=${fmt(sb.pursuer_speed_base, 3)}）`
            : "";
      const rows = [];
      if (capNote) rows.push(["追捕速度界", capNote]);
      if (evaderCap != null) rows.push(["逃逸速度界", `±${fmt(evaderCap, 2)} m/s`]);
      frame.kinematics.agents.forEach((a) => {
        const [vx, vy, vz] = a.linear || [0, 0, 0];
        const isEv = a.label === "E";
        const cap = isEv ? evaderCap : pursuerCap;
        const over = cap != null && a.speed_xy > cap * 1.02;
        rows.push([`${a.label} |v_xy|`, `${fmt(a.speed_xy, 2)} m/s${over ? " ⚠" : ""}`]);
        rows.push([`${a.label} v`, `(${fmt(vx, 2)}, ${fmt(vy, 2)}, ${fmt(vz, 2)})`]);
      });
      html += panel("速度 (机体坐标)", rows);
    }

    if (frame.deploy_control?.pursuers?.length) {
      const rows = [];
      frame.deploy_control.pursuers.forEach((p, i) => {
        const ground = p.backend_cmd_ground_xy;
        const world = p.backend_cmd_world_xy;
        if (ground) rows.push([`P${i} cmd ground [vx,vy]`, `(${fmt(ground[0], 3)}, ${fmt(ground[1], 3)})`]);
        if (world) rows.push([`P${i} cmd world [vx,vy]`, `(${fmt(world[0], 3)}, ${fmt(world[1], 3)})`]);
        if (p.backend_cmd_action_layout) rows.push([`P${i} action layout`, p.backend_cmd_action_layout]);
      });
      if (rows.length) {
        rows.unshift(["canvas arrows", "orange actual v, green backend cmd"]);
        html += panel("Velocity / Command Frames", rows);
      }
    }

    const ct = frame?.control_timing;
    if (ct && Object.keys(ct).length) {
      const nominal = ct.nominal_control_hz;
      html += panel("控制时延 / 实时性", [
        ["流形更新", fmtMs(ct.manifold_update_time)],
        ["槽位分配", fmtMs(ct.slot_assignment_time)],
        ["总决策时延", fmtMs(ct.total_decision_latency)],
        ["控制频率", fmtHz(ct.control_frequency)],
        ...(nominal != null ? [["标称控制频率", fmtHz(nominal)]] : []),
      ]);
    }

    if (viz.structure_metrics && Object.keys(ps).length) {
      html += panel("结构指标", [
        ["C_cov", fmt(ps.C_cov, 4)],
        ["C_col", fmt(ps.C_col, 4)],
        ["D_ang", fmt(ps.D_ang, 4)],
      ]);
    }

    if (viz.manifold_curve || viz.slot_targets) {
      html += panel("流形 / 槽位", [
        ["目标半径", fmt(algo.target_radius_xy, 3)],
        ["收缩率", fmt(algo.manifold_contraction_rate, 4)],
        ["相位", fmt(algo.manifold_target_phase, 3)],
      ]);
    }

    if (viz.role_allocation) {
      html += panel("角色分配", [
        ["模式", algo.role_assignment_mode || "—"],
        ["分配", role.role_assignment ? role.role_assignment.join(" → ") : "—"],
      ]);
    }

    if (viz.ot_details && ot.cost_matrix) {
      html += panel("OT", [
        ["ε", fmt(ot.epsilon, 4)],
        ["惯性边界", fmt(ot.inertia_margin, 4)],
      ]);
      html += `<section class="panel"><h2>OT 代价矩阵</h2><table class="ot"><tr><th></th>`;
      ot.cost_matrix[0].forEach((_, j) => { html += `<th>S${j}</th>`; });
      html += `</tr>`;
      ot.cost_matrix.forEach((row, i) => {
        html += `<tr><th>P${i}</th>`;
        row.forEach((v) => { html += `<td>${fmt(v, 2)}</td>`; });
        html += `</tr>`;
      });
      html += `</table></section>`;
    }

    if (viz.dream_manifold && dm.rho !== undefined) {
      html += panel("Dream 流形", [["ρ", fmt(dm.rho, 4)], ["ψ", fmt(dm.psi, 4)]]);
    }

    if (viz.pursuit_targets) {
      html += panel("控制目标", [["类型", "直接追踪逃逸者"]]);
    }
    if (viz.fixed_ring_targets) {
      html += panel("控制目标", [
        ["类型", "固定环"],
        ["环半径", fmt(frame?.controller_targets?.ring_radius, 3)],
      ]);
    }

    if (viz.speed_diagnostics && frame.deploy_control?.pursuers?.length) {
      const dc = frame.deploy_control;
      const rows = [];
      if (dc.lookahead_dist != null) rows.push(["前瞻距离", `${fmt(dc.lookahead_dist, 2)} m`]);
      dc.pursuers.forEach((p, i) => {
        const world = p.world_speed_cmd_xy != null ? `${fmt(p.world_speed_cmd_xy, 2)} 世界 / ` : "";
        rows.push([`P${i} 指令速度`, `${world}${fmt(p.speed_cmd_xy, 2)} 机体 / ${fmt(p.speed_cap_xy, 2)} m/s`]);
        rows.push([`P${i} 限速原因`, limitReasonLabel(p.limit_reason)]);
        rows.push([`P${i} 偏航对齐`, fmt(p.align_factor, 2)]);
        rows.push([`P${i} 前瞻距`, `${fmt(p.track_dist_xy, 2)} m`]);
        const loc = (p.corridor_obstacle_indices || p.local_obstacle_indices || []).map((x) => `#${x}`).join(", ") || "—";
        rows.push([`P${i} 走廊障碍`, loc]);
        if (p.cbf?.cbf_delta_norm > 1e-4) {
          const cbfObs = (p.cbf.active_obstacle_indices || []).map((x) => `#${x}`).join(", ") || "—";
          rows.push([`P${i} CBF 削减`, `${fmt(p.cbf.nominal_speed_xy, 2)}→${fmt(p.cbf.safe_speed_xy, 2)} m/s`]);
          rows.push([`P${i} CBF 障碍`, cbfObs]);
        }
        if (p.turn_safety?.turn_safety_active) {
          rows.push([`P${i} 转弯弧 clearance`, `${fmt(p.turn_safety.turn_arc_min_clearance, 2)} m`]);
          rows.push([`P${i} 边界 clearance`, `${fmt(p.turn_safety.turn_boundary_min_clearance, 2)} m`]);
          rows.push([`P${i} 转角`, `${fmt(Number(p.turn_safety.turn_angle_rad) * 180 / Math.PI, 1)}°`]);
        }
      });
      if (dc.manifold_obstacles?.length) {
        const mo = dc.manifold_obstacles
          .map((o) => `#${Number(o.index)}(w=${fmt(o.influence_weight, 2)})`)
          .join(", ");
        rows.push(["流形变形障碍", mo]);
      }
      html += panel("加速 / 障碍诊断", rows);
    }

    if (viz.candidate_slots && frame.deploy_control?.candidate_slots) {
      const cs = frame.deploy_control.candidate_slots;
      const n = cs.positions?.length || 0;
      const reachable = (cs.reachable || []).filter(Boolean).length;
      const losBlocked = (cs.los_blocked || []).filter(Boolean).length;
      const selected = (cs.selected_indices || []).join(", ") || "n/a";
      html += panel("Reachability Slots", [
        ["Candidates", String(n)],
        ["Reachable", `${reachable}/${n}`],
        ["LOS blocked", `${losBlocked}/${n}`],
        ["Selected C", selected],
        ["Fallback", cs.fallback ? "yes" : "no"],
      ]);
    }

    html += panel("环境", [
      ["捕获距离", fmt(algo.capture_dist, 3)],
      ["世界 XY", fmt(frame?.world_xy, 1)],
    ]);

    html += `<section class="panel"><h2>图例</h2><div class="legend">
      <span style="color:#58a6ff">● P0/P1/P2</span>
      <span style="color:#f85149">● E</span>
      ${viz.manifold_curve
        ? frame?.manifold?.pursuer_curves?.length
          ? '<span style="color:#ffa657">— 围捕流形 P0/P1/P2</span>'
          : '<span style="color:#ffa657">— 围捕流形</span>'
        : ""}
      ${viz.fixed_ring_curve ? '<span style="color:#a371f7">— 固定环</span>' : ""}
      ${viz.slot_targets ? '<span style="color:#79c0ff">□ 槽位</span>' : ""}
      ${viz.path_tracking ? '<span style="color:#e3b341">● 前瞻点 W</span>' : ""}
      ${viz.path_tracking ? '<span>— 虚线=规划路径</span>' : ""}
      ${viz.speed_diagnostics ? '<span style="color:#f0883e">○ 流形障碍 M</span>' : ""}
      ${viz.pursuit_targets || viz.fixed_ring_targets ? '<span style="color:#a371f7">- - 控制目标</span>' : ""}
    </div></section>`;
    return html;
  },
};
