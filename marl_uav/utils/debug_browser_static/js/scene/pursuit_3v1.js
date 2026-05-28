import { worldToScreen } from "../core/viewport.js";

const COLORS = ["#58a6ff", "#3fb950", "#d2a8ff"];
const EVADER = "#f85149";
const MANIFOLD = "#ffa657";
const SLOT = "#79c0ff";
const LINK = "#a371f7";
const CTRL = "#79c0ff";

function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "number") return v.toFixed(d);
  return String(v);
}

function panel(title, rows) {
  const rowsHtml = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");
  return `<section class="panel"><h2>${title}</h2><div class="kv">${rowsHtml}</div></section>`;
}

function vizOf(frame) {
  return frame?.viz || {};
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
      (frame?.manifold?.curve || []).forEach((p) => add(p[0], p[1]));
    }
    if (viz.fixed_ring_curve) {
      (frame?.controller_targets?.ring_curve || []).forEach((p) => add(p[0], p[1]));
    }
    if (viz.slot_targets) {
      const slots = frame?.role?.slot_targets || frame?.manifold?.slot_targets || [];
      slots.forEach((p) => add(p[0], p[1]));
    }
    if (viz.pursuit_targets || viz.fixed_ring_targets) {
      (frame?.controller_targets?.targets || []).forEach((p) => add(p[0], p[1]));
    }
    if (viz.obstacles !== false) {
      (frame?.obstacles?.xy || []).forEach((o, i) => {
        const r = frame.obstacles.r?.[i] || 0.5;
        add(o[0] - r, o[1] - r); add(o[0] + r, o[1] + r);
      });
    }
    if (!Number.isFinite(minX)) return null;
    const margin = Math.max(maxX - minX, maxY - minY) * 0.08 + 0.5;
    return { minX: minX - margin, maxX: maxX + margin, minY: minY - margin, maxY: maxY + margin };
  },

  draw(ctx, { frame, trails, view, w, h }) {
    if (!frame?.positions) return;
    const viz = vizOf(frame);
    const half = (frame.world_xy || 20) / 2;
    const wts = (x, y) => worldToScreen(view, x, y, w, h);

    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(...wts(-half, 0));
    ctx.lineTo(...wts(half, 0));
    ctx.moveTo(...wts(0, -half));
    ctx.lineTo(...wts(0, half));
    ctx.stroke();

    ctx.strokeStyle = "#30363d";
    const tl = wts(-half, half);
    const br = wts(half, -half);
    ctx.strokeRect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]);

    if (viz.obstacles !== false && frame.obstacles?.xy) {
      ctx.fillStyle = "rgba(139,148,158,0.25)";
      ctx.strokeStyle = "#8b949e";
      frame.obstacles.xy.forEach((o, i) => {
        const r = frame.obstacles.r?.[i] || 0.5;
        const [cx, cy] = wts(o[0], o[1]);
        const rs = Math.max(r * view.scale, 2);
        ctx.beginPath();
        ctx.arc(cx, cy, rs, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }

    if (viz.fixed_ring_curve && frame.controller_targets?.ring_curve?.length) {
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
      ctx.strokeStyle = MANIFOLD;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      frame.manifold.curve.forEach((pt, i) => {
        const [sx, sy] = wts(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (viz.slot_targets) {
      const slotPts = frame.role?.slot_targets || frame.manifold?.slot_targets;
      if (slotPts) drawTargetMarkers(ctx, wts, slotPts, { labelPrefix: "S", color: SLOT });
    }

    if (viz.role_allocation && frame.role?.assigned_targets) {
      drawAssignmentLines(ctx, wts, frame, frame.role.assigned_targets);
    } else if ((viz.pursuit_targets || viz.fixed_ring_targets) && frame.controller_targets?.targets) {
      drawAssignmentLines(ctx, wts, frame, frame.controller_targets.targets);
      if (viz.fixed_ring_targets) {
        drawTargetMarkers(ctx, wts, frame.controller_targets.targets, { labelPrefix: "R", color: CTRL });
      }
    }

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

    html += panel("环境", [
      ["捕获距离", fmt(algo.capture_dist, 3)],
      ["世界 XY", fmt(frame?.world_xy, 1)],
    ]);

    html += `<section class="panel"><h2>图例</h2><div class="legend">
      <span style="color:#58a6ff">● P0/P1/P2</span>
      <span style="color:#f85149">● E</span>
      ${viz.manifold_curve ? '<span style="color:#ffa657">— 围捕流形</span>' : ""}
      ${viz.fixed_ring_curve ? '<span style="color:#a371f7">— 固定环</span>' : ""}
      ${viz.slot_targets ? '<span style="color:#79c0ff">□ 槽位</span>' : ""}
      ${viz.pursuit_targets || viz.fixed_ring_targets ? '<span style="color:#a371f7">- - 控制目标</span>' : ""}
    </div></section>`;
    return html;
  },
};
