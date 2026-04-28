"""
3v1 追逃：按缓存文件收集或加载 episode 级详细坐标与围捕结构序列，并做基础统计。

- 若缓存目录中已有有效记录（meta.json + trajectories.npz），则直接加载，不跑仿真。
- 否则运行默认 100 个 episode（可改 --episodes），记录全体 agent 位置轨迹与 pursuit_structure 序列并保存。

加载或收集完成后，默认在缓存目录下 `figures/` 生成：
  图 A：captured vs timeout 的归一化时间上 C_cov / C_col / D_ang 组均值±std；
  图 B：与 eval 一致的 last-K 平均 C_cov–C_col 散点；
  图 C：至多 2 条 captured + 2 条 timeout 典型 3D 轨迹 + 全回合结构曲线，末段逐步标注 cov/col/ang。
加 `--no-plots` 可跳过出图。后续可继续在本文件扩展指标函数。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl_uav.envs.tasks.pursuit_evasion_3v1_task import compute_pursuit_structure_metrics_3v1
from marl_uav.utils.config import load_config

SCHEMA_VERSION = 1
TERMINAL_WINDOWS = (30, 50, 100)
PHASE_NAMES = ("approach", "pre_encirclement", "terminal_capture")
TIME_FRACTIONS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LOCAL_WINDOW_RATIO = 0.1
CONDITIONAL_FAILURE_WINDOW = 100
ESCAPE_THETA_SAMPLES = 72


def _load_eval_build_fns() -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    """从 scripts/eval.py 加载 build_env / build_policy / build_learner（避免重复维护）。"""
    eval_path = _ROOT / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("_marl_eval_shim", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {eval_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_marl_eval_shim"] = mod
    spec.loader.exec_module(mod)
    return mod.build_env, mod.build_policy, mod.build_learner


@dataclass
class EpisodeLogBundle:
    """内存中的 episode 日志包（与磁盘格式对应）。"""

    meta: dict[str, Any]
    trajectories: np.ndarray  # dtype=object, 每项 [T+1, N, 3] float32
    pursuit_series: list[list[dict[str, Any]]]  # 每局一步一个 dict


def default_cache_dir(exp_cfg_path: Path, seed: int) -> Path:
    exp_name = exp_cfg_path.stem
    return _ROOT / "results" / exp_name / "episode_log_cache" / f"seed{seed}"


def cache_is_complete(cache_dir: Path) -> bool:
    meta_p = cache_dir / "meta.json"
    traj_p = cache_dir / "trajectories.npz"
    ps_p = cache_dir / "pursuit_series.json"
    if not meta_p.is_file() or not traj_p.is_file() or not ps_p.is_file():
        return False
    try:
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if int(meta.get("schema_version", 0)) != SCHEMA_VERSION:
        return False
    n = int(meta.get("num_episodes", 0))
    if n <= 0:
        return False
    eps = meta.get("episodes")
    if not isinstance(eps, list) or len(eps) != n:
        return False
    try:
        raw = np.load(traj_p, allow_pickle=True)
        if len(raw["trajectories"]) != n:
            return False
    except (OSError, KeyError, ValueError, TypeError):
        return False
    try:
        with open(ps_p, encoding="utf-8") as f:
            ps = json.load(f)
        pe = ps.get("episodes")
        if not isinstance(pe, list) or len(pe) != n:
            return False
    except (json.JSONDecodeError, OSError):
        return False
    return True


def save_episode_logs(
    cache_dir: Path,
    *,
    meta_header: dict[str, Any],
    per_episode_rows: list[dict[str, Any]],
    trajectories_list: list[np.ndarray],
    pursuit_series: list[list[dict[str, Any]]],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(per_episode_rows)
    if len(trajectories_list) != n or len(pursuit_series) != n:
        raise ValueError("per_episode_rows、trajectories_list、pursuit_series 长度须一致")

    meta = {
        **meta_header,
        "schema_version": SCHEMA_VERSION,
        "num_episodes": n,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": per_episode_rows,
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    lens = np.array([int(r["episode_len"]) for r in per_episode_rows], dtype=np.int32)
    rets = np.array([float(r["episode_return"]) for r in per_episode_rows], dtype=np.float64)
    captured = np.array([bool(r.get("captured", False)) for r in per_episode_rows], dtype=bool)
    timeout = np.array([bool(r.get("timeout", False)) for r in per_episode_rows], dtype=bool)
    collision = np.array([bool(r.get("collision", False)) for r in per_episode_rows], dtype=bool)
    p_oob = np.array([bool(r.get("pursuer_oob", False)) for r in per_episode_rows], dtype=bool)
    obs_term = np.array([bool(r.get("obstacle_termination", False)) for r in per_episode_rows], dtype=bool)

    mean_cov = []
    mean_col = []
    for r in per_episode_rows:
        mean_cov.append(float(r["mean_C_cov"]) if r.get("mean_C_cov") is not None else np.nan)
        mean_col.append(float(r["mean_C_col"]) if r.get("mean_C_col") is not None else np.nan)
    mean_cov_a = np.array(mean_cov, dtype=np.float64)
    mean_col_a = np.array(mean_col, dtype=np.float64)

    traj_obj = np.empty(n, dtype=object)
    for i, arr in enumerate(trajectories_list):
        traj_obj[i] = np.asarray(arr, dtype=np.float32)

    np.savez_compressed(
        cache_dir / "trajectories.npz",
        episode_len=lens,
        episode_return=rets,
        captured=captured,
        timeout=timeout,
        collision=collision,
        pursuer_oob=p_oob,
        obstacle_termination=obs_term,
        mean_C_cov=mean_cov_a,
        mean_C_col=mean_col_a,
        trajectories=traj_obj,
    )

    with open(cache_dir / "pursuit_series.json", "w", encoding="utf-8") as f:
        json.dump({"version": SCHEMA_VERSION, "episodes": pursuit_series}, f, ensure_ascii=False)


def load_episode_logs(cache_dir: Path) -> EpisodeLogBundle:
    with open(cache_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    raw = np.load(cache_dir / "trajectories.npz", allow_pickle=True)
    trajs = raw["trajectories"]
    with open(cache_dir / "pursuit_series.json", encoding="utf-8") as f:
        ps = json.load(f)
    pursuit_eps = ps.get("episodes", [])
    return EpisodeLogBundle(meta=meta, trajectories=trajs, pursuit_series=pursuit_eps)


def collect_episodes_rollout(
    *,
    env: Any,
    mac: Any,
    learner: Any,
    ckpt_path: Path,
    num_episodes: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[list[dict[str, Any]]]]:
    from marl_uav.runners.rollout_worker import RolloutWorker
    from marl_uav.utils.checkpoint import load_checkpoint

    load_checkpoint(ckpt_path, learner)
    mac.set_test_mode(True)
    worker = RolloutWorker(env=env, policy=mac)

    rows: list[dict[str, Any]] = []
    trajs: list[np.ndarray] = []
    pss: list[list[dict[str, Any]]] = []

    for ep in range(num_episodes):
        ep_seed = base_seed * 100_000 + ep
        _, info = worker.collect_episode(seed=ep_seed, record_trajectory=True)
        if "trajectory" not in info:
            raise RuntimeError(
                "collect_episode 未返回 trajectory；请确认环境为 PyFlytAviaryEnv 且 prev_backend_state 可用。"
            )
        traj = np.asarray(info["trajectory"], dtype=np.float32)
        series = info.get("pursuit_structure_series")
        if not isinstance(series, list):
            series = []

        row: dict[str, Any] = {
            "episode": ep,
            "seed_run": ep_seed,
            "episode_len": int(info["episode_len"]),
            "episode_return": float(info["episode_return"]),
            "captured": bool(info.get("capture", False)),
            "timeout": bool(info.get("timeout", False)),
            "collision": bool(info.get("collision", False)),
            "pursuer_oob": bool(info.get("pursuer_oob", False)),
            "obstacle_termination": bool(info.get("obstacle_termination", False)),
            "mean_C_cov": float(info["mean_C_cov"]) if "mean_C_cov" in info else None,
            "mean_C_col": float(info["mean_C_col"]) if "mean_C_col" in info else None,
        }
        rows.append(row)
        trajs.append(traj)
        pss.append([dict(x) for x in series])

        print(
            f"[collect] ep {ep + 1}/{num_episodes} len={row['episode_len']} "
            f"return={row['episode_return']:.3f} cap={row['captured']} to={row['timeout']} "
            f"obs_term={row['obstacle_termination']}"
        )

    return rows, trajs, pss


def print_basic_stats(bundle: EpisodeLogBundle) -> None:
    """基础统计；后续可在此文件追加更多指标。"""
    meta = bundle.meta
    n = int(meta.get("num_episodes", 0))
    eps = meta.get("episodes", [])
    print(f"\n=== Episode log bundle: {n} episodes ===")
    if not eps:
        return

    cap = sum(1 for e in eps if e.get("captured"))
    to = sum(1 for e in eps if e.get("timeout"))
    obs_t = sum(1 for e in eps if e.get("obstacle_termination"))
    lens = [int(e["episode_len"]) for e in eps]
    rets = [float(e["episode_return"]) for e in eps]
    covs = [e.get("mean_C_cov") for e in eps if e.get("mean_C_cov") is not None]
    cols = [e.get("mean_C_col") for e in eps if e.get("mean_C_col") is not None]

    print(f"captured: {cap}/{n} ({100.0 * cap / n:.1f}%)")
    print(f"timeout:  {to}/{n} ({100.0 * to / n:.1f}%)")
    print(f"obstacle_termination (撞柱终局): {obs_t}/{n} ({100.0 * obs_t / n:.1f}%)")
    print(f"episode_len: mean={np.mean(lens):.1f} std={np.std(lens):.1f} min={np.min(lens)} max={np.max(lens)}")
    print(f"episode_return: mean={np.mean(rets):.3f} std={np.std(rets):.3f}")
    if covs:
        print(f"mean_C_cov (last-K mean): mean={np.mean(covs):.4f} std={np.std(covs):.4f}")
    if cols:
        print(f"mean_C_col (last-K mean): mean={np.mean(cols):.4f} std={np.std(cols):.4f}")

    # 轨迹形状抽查
    if bundle.trajectories.size > 0:
        t0 = bundle.trajectories[0]
        print(f"trajectory[0] shape: {np.asarray(t0).shape} dtype={np.asarray(t0).dtype}")
    if bundle.pursuit_series:
        print(f"pursuit_series[0] length: {len(bundle.pursuit_series[0])} steps")


def _metrics_arrays_from_series(series: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not series:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    cov = np.array([float(s["C_cov"]) for s in series], dtype=np.float64)
    col = np.array([float(s["C_col"]) for s in series], dtype=np.float64)
    dang = np.array([float(s["D_ang"]) for s in series], dtype=np.float64)
    return cov, col, dang


def _series_value_array(series: list[dict[str, Any]], key: str) -> np.ndarray:
    if not series:
        return np.zeros(0, dtype=np.float64)
    return np.array([float(s[key]) for s in series], dtype=np.float64)


def _resample_episode_to_unit_interval(
    series: list[dict[str, Any]],
    n_grid: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """返回 (t01, cov, col, dang)，t01 在 [0,1]；单点 episode 退化为常数。"""
    L = len(series)
    if L == 0:
        return None
    cov, col, dang = _metrics_arrays_from_series(series)
    t01 = np.linspace(0.0, 1.0, L) if L > 1 else np.array([0.0, 1.0])
    grid = np.linspace(0.0, 1.0, n_grid)
    if L == 1:
        return grid, np.full(n_grid, cov[0]), np.full(n_grid, col[0]), np.full(n_grid, dang[0])
    return (
        grid,
        np.interp(grid, t01, cov),
        np.interp(grid, t01, col),
        np.interp(grid, t01, dang),
    )


def plot_figure_a_structure_evolution(
    bundle: EpisodeLogBundle,
    out_path: Path,
    *,
    n_grid: int = 101,
) -> bool:
    """图 A：captured vs timeout 在归一化时间上的结构指标平均曲线。"""
    eps = bundle.meta.get("episodes", [])
    n_ep = len(eps)
    if n_ep == 0 or len(bundle.pursuit_series) != n_ep:
        return False

    cap_stack: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    to_stack: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, e in enumerate(eps):
        ser = bundle.pursuit_series[i]
        if not ser:
            continue
        rs = _resample_episode_to_unit_interval(ser, n_grid=n_grid)
        if rs is None:
            continue
        t01, cv, cl, da = rs
        cap = bool(e.get("captured"))
        to = bool(e.get("timeout"))
        if cap and not to:
            cap_stack.append((t01, cv, cl, da))
        elif to and not cap:
            to_stack.append((t01, cv, cl, da))

    if not cap_stack and not to_stack:
        return False

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    titles = (
        r"Coverage $C_{\mathrm{cov}}$",
        r"Collapse $C_{\mathrm{col}}$",
        r"Angular dispersion $D_{\mathrm{ang}}$",
    )

    for ax, title, mi in zip(axes, titles, (1, 2, 3)):
        if cap_stack:
            mat = np.stack([s[mi] for s in cap_stack], axis=0)
            m = np.nanmean(mat, axis=0)
            s = np.nanstd(mat, axis=0)
            t0 = cap_stack[0][0]
            ax.plot(t0, m, color="#2ca02c", lw=2.0, label=f"captured (n={len(cap_stack)})")
            ax.fill_between(t0, m - s, m + s, color="#2ca02c", alpha=0.2)
        if to_stack:
            mat = np.stack([s[mi] for s in to_stack], axis=0)
            m = np.nanmean(mat, axis=0)
            s = np.nanstd(mat, axis=0)
            t0 = to_stack[0][0]
            ax.plot(t0, m, color="#ff7f0e", lw=2.0, label=f"timeout (n={len(to_stack)})")
            ax.fill_between(t0, m - s, m + s, color="#ff7f0e", alpha=0.2)
        ax.set_ylabel("score")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("normalized time τ ∈ [0, 1] (episode start → end)")
    fig.suptitle("Fig A: Pursuit structure vs normalized time (group mean ± std)", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_figure_b_scatter_mean_cov_col(
    bundle: EpisodeLogBundle,
    out_path: Path,
) -> bool:
    """图 B：与 eval 一致的 episode 级 last-K 平均 C_cov vs C_col 散点。"""
    mean_last_steps = 30
    try:
        from marl_uav.runners.rollout_worker import PURSUIT_STRUCTURE_MEAN_LAST_STEPS as mean_last_steps
    except ModuleNotFoundError:
        pass

    rows = bundle.meta.get("episodes", [])
    rows = [r for r in rows if r.get("mean_C_cov") is not None and r.get("mean_C_col") is not None]
    if not rows:
        return False

    xs = np.array([float(r["mean_C_cov"]) for r in rows], dtype=np.float64)
    ys = np.array([float(r["mean_C_col"]) for r in rows], dtype=np.float64)
    lens = np.array([int(r["episode_len"]) for r in rows], dtype=np.float64)
    s = np.clip(22.0 + lens * 5.5, 35.0, 650.0)

    color_list: list[str] = []
    for r in rows:
        if bool(r.get("captured", False)):
            color_list.append("#2ca02c")
        elif bool(r.get("timeout", False)):
            color_list.append("#ff7f0e")
        else:
            color_list.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(xs, ys, s=s, c=color_list, alpha=0.82, edgecolors="0.25", linewidths=0.45, zorder=3)
    ax.set_xlabel(
        rf"$\bar{{C}}_{{\mathrm{{cov}}}}^{{\mathrm{{term}}}}$ (mean over last {mean_last_steps} steps)"
    )
    ax.set_ylabel(
        rf"$\bar{{C}}_{{\mathrm{{col}}}}^{{\mathrm{{term}}}}$ (mean over last {mean_last_steps} steps)"
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="0.25", markersize=9, label="captured"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markeredgecolor="0.25", markersize=9, label="timeout"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markeredgecolor="0.25", markersize=9, label="other"),
    ]
    leg1 = ax.legend(handles=legend_elems, loc="upper right", title="outcome")
    ax.add_artist(leg1)
    ax.set_title(f"Fig B: Episode-mean scatter (n={len(rows)})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _pick_typical_episode_indices(
    eps: list[dict[str, Any]],
    *,
    want_captured: bool,
    n_pick: int,
) -> list[int]:
    pool: list[int] = []
    for i, e in enumerate(eps):
        cap = bool(e.get("captured"))
        to = bool(e.get("timeout"))
        if want_captured and cap and not to:
            pool.append(i)
        if not want_captured and to and not cap:
            pool.append(i)
    if not pool:
        return []
    lens = np.array([int(eps[i]["episode_len"]) for i in pool], dtype=np.float64)
    med = float(np.median(lens))
    pool.sort(key=lambda idx: abs(float(eps[idx]["episode_len"]) - med))
    return pool[: min(n_pick, len(pool))]


def _pursuit_series_from_trajectory(traj_xyz: np.ndarray) -> list[dict[str, Any]]:
    Tp1, n_ag, _ = traj_xyz.shape
    if n_ag < 4:
        return []
    out: list[dict[str, Any]] = []
    for t in range(Tp1):
        p = traj_xyz[t, :3, :]
        ev = traj_xyz[t, 3, :]
        out.append(compute_pursuit_structure_metrics_3v1(p, ev))
    return out


def plot_figure_c_typical_trajectories(
    bundle: EpisodeLogBundle,
    out_path: Path,
    *,
    annotate_last_steps: int | None = None,
) -> bool:
    """图 C：2 条 captured + 2 条 timeout 典型 3D 轨迹，末段标注结构指标。"""
    mean_last_steps = 30
    try:
        from marl_uav.runners.rollout_worker import PURSUIT_STRUCTURE_MEAN_LAST_STEPS as mean_last_steps
    except ModuleNotFoundError:
        pass
    k = annotate_last_steps if annotate_last_steps is not None else max(20, mean_last_steps)
    eps = bundle.meta.get("episodes", [])
    n_ep = len(eps)
    if n_ep == 0 or len(bundle.trajectories) != n_ep:
        return False

    cap_idx = _pick_typical_episode_indices(eps, want_captured=True, n_pick=2)
    to_idx = _pick_typical_episode_indices(eps, want_captured=False, n_pick=2)
    panels: list[tuple[str, int]] = []
    for idx in cap_idx:
        panels.append(("captured", idx))
    for idx in to_idx:
        panels.append(("timeout", idx))
    if not panels:
        return False

    n_p = len(panels)
    nrows = (n_p + 1) // 2
    fig = plt.figure(figsize=(14.0, 5.2 * nrows))
    gs = fig.add_gridspec(nrows, 4, hspace=0.48, wspace=0.35)

    try:
        base_cmap = plt.colormaps["tab10"]
    except (AttributeError, KeyError):
        base_cmap = plt.cm.get_cmap("tab10")

    for pi, (tag, ep_i) in enumerate(panels):
        traj_xyz = np.asarray(bundle.trajectories[ep_i], dtype=np.float32)
        Tp1, n_real, _ = traj_xyz.shape
        series = bundle.pursuit_series[ep_i] if ep_i < len(bundle.pursuit_series) else []
        if not series or len(series) != Tp1:
            series = _pursuit_series_from_trajectory(traj_xyz)

        row = pi // 2
        c0 = (pi % 2) * 2
        ax3d = fig.add_subplot(gs[row, c0], projection="3d")
        axm = fig.add_subplot(gs[row, c0 + 1])

        colors = [base_cmap(i) for i in range(min(n_real, 4))]
        labels = [f"P{i}" for i in range(max(0, n_real - 1))] + ["E"]
        if n_real != 4:
            labels = [f"A{i}" for i in range(n_real)]

        for agent_idx in range(n_real):
            c = colors[agent_idx % len(colors)]
            ax3d.plot(
                traj_xyz[:, agent_idx, 0],
                traj_xyz[:, agent_idx, 1],
                traj_xyz[:, agent_idx, 2],
                "-",
                color=c,
                alpha=0.9,
                lw=1.2,
                label=labels[agent_idx],
            )
        ax3d.set_xlim(-20, 20)
        ax3d.set_ylim(-20, 20)
        ax3d.set_zlim(0.5, 5.0)
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        er = float(eps[ep_i].get("episode_return", 0.0))
        ax3d.set_title(f"{tag} · ep {ep_i} · len={Tp1 - 1} · ret={er:.2f}", fontsize=9)
        ax3d.legend(loc="upper left", fontsize=7)

        if series:
            cov, col, dang = _metrics_arrays_from_series(series)
            t_all = np.arange(len(series), dtype=np.float64)
            axm.plot(t_all, cov, label=r"$C_{\mathrm{cov}}$", color="steelblue", lw=1.4)
            axm.plot(t_all, col, label=r"$C_{\mathrm{col}}$", color="coral", lw=1.4)
            axm.plot(t_all, dang, label=r"$D_{\mathrm{ang}}$", color="seagreen", lw=1.4)
            axm.scatter(t_all, cov, color="steelblue", s=10, alpha=0.5, zorder=4)
            axm.scatter(t_all, col, color="coral", s=10, alpha=0.5, zorder=4)
            axm.scatter(t_all, dang, color="seagreen", s=10, alpha=0.5, zorder=4)
            axm.set_xlim(0, max(len(series) - 1, 1))
            axm.set_ylim(-0.05, 1.05)
            axm.set_xlabel("step")
            axm.set_ylabel("score")
            axm.grid(True, alpha=0.3)
            axm.legend(loc="upper right", fontsize=7)

            start = max(0, len(series) - k)
            axm.axvspan(float(start), float(len(series) - 1), color="0.75", alpha=0.12)
            for j in range(start, len(series)):
                cc, cl, da = float(cov[j]), float(col[j]), float(dang[j])
                axm.annotate(
                    f"t={j}\ncov {cc:.2f}\ncol {cl:.2f}\nang {da:.2f}",
                    xy=(t_all[j], cc),
                    xytext=(6, 4 + (j - start) * 3),
                    textcoords="offset points",
                    fontsize=5,
                    color="0.2",
                    alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85),
                )
            axm.set_title(f"structure (shaded: last {min(k, len(series))} steps)", fontsize=8)
        else:
            axm.text(0.5, 0.5, "no structure series", ha="center", va="center", transform=axm.transAxes)

    fig.suptitle("Fig C: Typical trajectories (up to 2× captured, 2× timeout) + structure", fontsize=12, y=1.01)
    fig.subplots_adjust(top=0.93)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _episode_min_distance_series(traj_xyz: np.ndarray) -> np.ndarray:
    rel = traj_xyz[:, :3, :] - traj_xyz[:, 3:4, :]
    dists = np.linalg.norm(rel, axis=-1)
    return np.min(dists, axis=1).astype(np.float64)


def _xy_velocity_series_from_trajectory(traj_xyz: np.ndarray) -> np.ndarray:
    xy = np.asarray(traj_xyz[:, :, :2], dtype=np.float64)
    if xy.shape[0] <= 1:
        return np.zeros_like(xy, dtype=np.float64)
    dxy = np.diff(xy, axis=0)
    vel = np.zeros_like(xy, dtype=np.float64)
    vel[1:] = dxy
    vel[0] = dxy[0]
    return vel


def _fesc_series_from_trajectory(
    traj_xyz: np.ndarray,
    *,
    n_theta: int = ESCAPE_THETA_SAMPLES,
) -> np.ndarray:
    vel_xy = _xy_velocity_series_from_trajectory(traj_xyz)  # [T, 4, 2]
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float64)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # [K, 2]

    ev = vel_xy[:, 3, :]  # [T, 2]
    pu = vel_xy[:, :3, :]  # [T, 3, 2]
    ev_proj = ev @ dirs.T  # [T, K]
    pu_proj = np.einsum("tad,kd->tak", pu, dirs)  # [T, 3, K]
    # Assumption: compare evader with the fastest pursuer along the same direction.
    score = ev_proj - np.max(pu_proj, axis=1)  # [T, K]
    return np.max(score, axis=1).astype(np.float64)


def _role_stability_series_from_trajectory(traj_xyz: np.ndarray) -> np.ndarray:
    rel_xy = np.asarray(traj_xyz[:, :3, :2] - traj_xyz[:, 3:4, :2], dtype=np.float64)  # [T,3,2]
    theta = np.arctan2(rel_xy[:, :, 1], rel_xy[:, :, 0])  # [T,3]
    T = int(theta.shape[0])
    if T == 0:
        return np.zeros(0, dtype=np.float64)
    ranks = np.zeros((T, 3), dtype=np.int64)
    for t in range(T):
        order = np.argsort(theta[t])
        ranks[t, order] = np.arange(3, dtype=np.int64)
    stability = np.ones(T, dtype=np.float64)
    if T >= 2:
        changes = ranks[1:] != ranks[:-1]  # [T-1,3]
        stability[1:] = 1.0 - np.mean(changes.astype(np.float64), axis=1)
    return stability


def _episode_final_labels(
    row: dict[str, Any],
    traj_xyz: np.ndarray,
    series: list[dict[str, Any]],
) -> dict[str, float]:
    min_dist_series = _episode_min_distance_series(traj_xyz)
    final_min_distance = float(min_dist_series[-1]) if min_dist_series.size else np.nan
    final_phi_max = float(series[-1]["phi_max"]) if series else np.nan
    return {
        "final_min_distance": final_min_distance,
        "remaining_escape_angle_rad": final_phi_max,
        "remaining_escape_angle_deg": float(np.degrees(final_phi_max)) if np.isfinite(final_phi_max) else np.nan,
        "capture_step": float(row["episode_len"]) if bool(row.get("captured", False)) else np.nan,
    }


def _window_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return np.nan
    kk = min(int(k), int(values.size))
    return float(np.mean(values[-kk:]))


def build_episode_analysis_table(bundle: EpisodeLogBundle) -> pd.DataFrame:
    rows = bundle.meta.get("episodes", [])
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        traj_xyz = np.asarray(bundle.trajectories[i], dtype=np.float64)
        series = bundle.pursuit_series[i] if i < len(bundle.pursuit_series) else []
        cov, col, dang = _metrics_arrays_from_series(series)
        phi_max = _series_value_array(series, "phi_max")
        fesc = _fesc_series_from_trajectory(traj_xyz)
        role_stability = _role_stability_series_from_trajectory(traj_xyz)
        record: dict[str, Any] = {
            "episode": int(row["episode"]),
            "episode_len": int(row["episode_len"]),
            "captured": int(bool(row.get("captured", False))),
            "timeout": int(bool(row.get("timeout", False))),
            "collision": int(bool(row.get("collision", False))),
            "pursuer_oob": int(bool(row.get("pursuer_oob", False))),
            "episode_return": float(row["episode_return"]),
        }
        record["outcome"] = (
            "captured"
            if record["captured"]
            else "timeout"
            if record["timeout"]
            else "other"
        )
        record.update(_episode_final_labels(row, traj_xyz, series))
        for k in TERMINAL_WINDOWS:
            record[f"C_cov_last{k}"] = _window_mean(cov, k)
            record[f"C_col_last{k}"] = _window_mean(col, k)
            record[f"D_ang_last{k}"] = _window_mean(dang, k)
            record[f"max_escape_gap_last{k}"] = _window_mean(phi_max, k)
            record[f"F_esc_last{k}"] = _window_mean(fesc, k)
            record[f"role_stability_last{k}"] = _window_mean(role_stability, k)
            record[f"role_instability_last{k}"] = (
                1.0 - record[f"role_stability_last{k}"]
                if np.isfinite(record[f"role_stability_last{k}"])
                else np.nan
            )
            record[f"max_escape_gap_deg_last{k}"] = (
                float(np.degrees(record[f"max_escape_gap_last{k}"]))
                if np.isfinite(record[f"max_escape_gap_last{k}"])
                else np.nan
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return np.nan, np.nan
    xv = x[mask]
    yv = y[mask]
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return np.nan, np.nan
    if method == "pearson":
        stat, pvalue = stats.pearsonr(xv, yv)
    elif method == "spearman":
        stat, pvalue = stats.spearmanr(xv, yv)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(stat), float(pvalue)


def _safe_logistic_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask].astype(int)
    if xv.size < 8 or np.unique(yv).size < 2:
        return {
            "n": float(xv.size),
            "coef": np.nan,
            "intercept": np.nan,
            "odds_ratio_per_std": np.nan,
            "auc": np.nan,
            "accuracy": np.nan,
        }
    x_std = float(np.std(xv))
    if x_std < 1e-8:
        return {
            "n": float(xv.size),
            "coef": np.nan,
            "intercept": np.nan,
            "odds_ratio_per_std": np.nan,
            "auc": np.nan,
            "accuracy": np.nan,
        }
    xz = ((xv - float(np.mean(xv))) / x_std).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(xz, yv)
    prob = model.predict_proba(xz)[:, 1]
    pred = (prob >= 0.5).astype(int)
    coef = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    return {
        "n": float(xv.size),
        "coef": coef,
        "intercept": intercept,
        "odds_ratio_per_std": float(np.exp(coef)),
        "auc": float(roc_auc_score(yv, prob)),
        "accuracy": float(np.mean(pred == yv)),
    }


def summarize_terminal_window_associations(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    terminal_metrics = ("C_cov", "C_col", "D_ang", "max_escape_gap", "F_esc", "role_stability")
    for k in TERMINAL_WINDOWS:
        for metric in terminal_metrics:
            colname = f"{metric}_last{k}"
            x = df[colname].to_numpy(dtype=np.float64)

            sub = df["outcome"].isin(["captured", "timeout"]).to_numpy()
            y_ct = df.loc[sub, "captured"].to_numpy(dtype=np.float64)
            logit = _safe_logistic_summary(x[sub], y_ct)
            records.append(
                {
                    "analysis": "logistic_regression",
                    "target": "captured_vs_timeout",
                    "window": k,
                    "metric": metric,
                    **logit,
                }
            )

            for target in ("final_min_distance", "remaining_escape_angle_rad", "capture_step"):
                y = df[target].to_numpy(dtype=np.float64)
                for method in ("pearson", "spearman"):
                    stat, pvalue = _safe_corr(x, y, method)
                    records.append(
                        {
                            "analysis": method,
                            "target": target,
                            "window": k,
                            "metric": metric,
                            "n": float(np.sum(np.isfinite(x) & np.isfinite(y))),
                            "stat": stat,
                            "pvalue": pvalue,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _time_index(length: int, t01: float) -> int:
    if length <= 0:
        return 0
    return int(np.clip(np.round(float(t01) * max(length - 1, 0)), 0, max(length - 1, 0)))


def _prefix_mean_at_t(values: np.ndarray, t01: float) -> float:
    if values.size == 0:
        return np.nan
    idx = _time_index(int(values.size), t01)
    return float(np.mean(values[: idx + 1]))


def _local_mean_at_t(values: np.ndarray, t01: float, window_ratio: float = LOCAL_WINDOW_RATIO) -> float:
    if values.size == 0:
        return np.nan
    idx = _time_index(int(values.size), t01)
    win = max(3, int(np.round(float(window_ratio) * float(values.size))))
    start = max(0, idx - win + 1)
    return float(np.mean(values[start : idx + 1]))


def summarize_time_auc_curves(
    bundle: EpisodeLogBundle,
    *,
    time_fractions: tuple[float, ...] = TIME_FRACTIONS,
    local_window_ratio: float = LOCAL_WINDOW_RATIO,
) -> pd.DataFrame:
    rows = bundle.meta.get("episodes", [])
    metric_names = ("C_cov", "C_col", "D_ang", "max_escape_gap")
    mode_labels = ("prefix_mean", "local_mean")
    episode_records: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        outcome = (
            "captured"
            if bool(row.get("captured", False))
            else "timeout"
            if bool(row.get("timeout", False))
            else "other"
        )
        if outcome not in ("captured", "timeout"):
            continue
        series = bundle.pursuit_series[i] if i < len(bundle.pursuit_series) else []
        if not series:
            continue
        arrays = {
            "C_cov": _series_value_array(series, "C_cov"),
            "C_col": _series_value_array(series, "C_col"),
            "D_ang": _series_value_array(series, "D_ang"),
            "max_escape_gap": _series_value_array(series, "phi_max"),
        }
        for t01 in time_fractions:
            for metric in metric_names:
                vals = arrays[metric]
                episode_records.append(
                    {
                        "episode": int(row["episode"]),
                        "outcome": outcome,
                        "captured": int(outcome == "captured"),
                        "time_fraction": float(t01),
                        "metric": metric,
                        "mode": "prefix_mean",
                        "value": _prefix_mean_at_t(vals, t01),
                    }
                )
                episode_records.append(
                    {
                        "episode": int(row["episode"]),
                        "outcome": outcome,
                        "captured": int(outcome == "captured"),
                        "time_fraction": float(t01),
                        "metric": metric,
                        "mode": "local_mean",
                        "value": _local_mean_at_t(vals, t01, window_ratio=local_window_ratio),
                    }
                )

    episode_df = pd.DataFrame.from_records(episode_records)
    records: list[dict[str, Any]] = []
    for mode in mode_labels:
        for metric in metric_names:
            for t01 in time_fractions:
                sub = episode_df[
                    (episode_df["mode"] == mode)
                    & (episode_df["metric"] == metric)
                    & (episode_df["time_fraction"] == float(t01))
                ]
                if sub.empty:
                    continue
                x = sub["value"].to_numpy(dtype=np.float64)
                y = sub["captured"].to_numpy(dtype=np.float64)
                logit = _safe_logistic_summary(x, y)
                auc = float(logit["auc"]) if np.isfinite(logit["auc"]) else np.nan
                coef = float(logit["coef"]) if np.isfinite(logit["coef"]) else np.nan
                records.append(
                    {
                        "mode": mode,
                        "metric": metric,
                        "time_fraction": float(t01),
                        "n": float(logit["n"]),
                        "coef": coef,
                        "auc": auc,
                        "oriented_auc": float(auc if coef >= 0.0 else 1.0 - auc) if np.isfinite(auc) and np.isfinite(coef) else np.nan,
                        "odds_ratio_per_std": float(logit["odds_ratio_per_std"]) if np.isfinite(logit["odds_ratio_per_std"]) else np.nan,
                        "accuracy": float(logit["accuracy"]) if np.isfinite(logit["accuracy"]) else np.nan,
                        "window_ratio": float(local_window_ratio),
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_conditional_failures(
    df_episode: pd.DataFrame,
    *,
    window: int = CONDITIONAL_FAILURE_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    use = df_episode[df_episode["outcome"].isin(["captured", "timeout"])].copy()
    ccol_col = f"C_col_last{window}"
    dang_col = f"D_ang_last{window}"
    ccov_col = f"C_cov_last{window}"
    gap_col = f"max_escape_gap_last{window}"
    if use.empty or any(col not in use.columns for col in (ccol_col, dang_col, ccov_col, gap_col)):
        return pd.DataFrame(), pd.DataFrame(), {}

    captured = use[use["captured"] == 1].copy()
    if captured.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    ccol_thr = float(np.nanmedian(captured[ccol_col].to_numpy(dtype=np.float64)))
    dang_thr = float(np.nanmedian(captured[dang_col].to_numpy(dtype=np.float64)))
    ccov_thr = float(np.nanmedian(captured[ccov_col].to_numpy(dtype=np.float64)))
    if not (np.isfinite(ccol_thr) and np.isfinite(dang_thr) and np.isfinite(ccov_thr)):
        return pd.DataFrame(), pd.DataFrame(), {}

    use["good_C_col"] = use[ccol_col] <= ccol_thr
    use["good_D_ang"] = use[dang_col] >= dang_thr
    use["low_C_cov"] = use[ccov_col] <= ccov_thr
    structured = use[use["good_C_col"] & use["good_D_ang"] & use["low_C_cov"]].copy()
    if structured.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "window": float(window),
            "good_c_col_threshold": ccol_thr,
            "good_d_ang_threshold": dang_thr,
            "low_c_cov_threshold": ccov_thr,
        }

    records: list[dict[str, Any]] = []
    for outcome in ("captured", "timeout"):
        sub = structured[structured["outcome"] == outcome]
        if sub.empty:
            continue
        records.append(
            {
                "group": f"structured_failure_subset_{outcome}",
                "n": int(len(sub)),
                "mean_C_col": float(sub[ccol_col].mean()),
                "std_C_col": float(sub[ccol_col].std(ddof=0)),
                "mean_D_ang": float(sub[dang_col].mean()),
                "std_D_ang": float(sub[dang_col].std(ddof=0)),
                "mean_C_cov": float(sub[ccov_col].mean()),
                "std_C_cov": float(sub[ccov_col].std(ddof=0)),
                "mean_max_escape_gap": float(sub[gap_col].mean()),
                "std_max_escape_gap": float(sub[gap_col].std(ddof=0)),
                "mean_max_escape_gap_deg": float(np.degrees(sub[gap_col]).mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            }
        )

    x = structured[gap_col].to_numpy(dtype=np.float64)
    y = structured["captured"].to_numpy(dtype=np.float64)
    logit = _safe_logistic_summary(x, y)
    rho, pvalue = _safe_corr(
        structured[gap_col].to_numpy(dtype=np.float64),
        structured["timeout"].to_numpy(dtype=np.float64),
        "spearman",
    )
    summary = {
        "window": float(window),
        "good_c_col_threshold": ccol_thr,
        "good_d_ang_threshold": dang_thr,
        "low_c_cov_threshold": ccov_thr,
        "structured_subset_n": float(len(structured)),
        "structured_timeout_n": float(int(np.sum(structured["timeout"] == 1))),
        "structured_captured_n": float(int(np.sum(structured["captured"] == 1))),
        "gap_logit_auc": float(logit["auc"]) if np.isfinite(logit["auc"]) else np.nan,
        "gap_logit_coef": float(logit["coef"]) if np.isfinite(logit["coef"]) else np.nan,
        "gap_timeout_spearman": float(rho) if np.isfinite(rho) else np.nan,
        "gap_timeout_spearman_p": float(pvalue) if np.isfinite(pvalue) else np.nan,
    }

    q = structured[gap_col].quantile([0.25, 0.5, 0.75]).to_dict()
    quant_records: list[dict[str, Any]] = []
    for qname, qv in (("q25", q.get(0.25)), ("q50", q.get(0.5)), ("q75", q.get(0.75))):
        if not np.isfinite(qv):
            continue
        sub = structured[structured[gap_col] >= float(qv)]
        quant_records.append(
            {
                "cut": qname,
                "max_escape_gap_min": float(qv),
                "n": int(len(sub)),
                "timeout_rate": float(sub["timeout"].mean()) if len(sub) else np.nan,
                "captured_rate": float(sub["captured"].mean()) if len(sub) else np.nan,
            }
        )

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(quant_records), summary


def _phase_boundaries(
    cov: np.ndarray,
    min_dist: np.ndarray,
) -> tuple[int, int]:
    T = int(min(len(cov), len(min_dist)))
    if T < 3:
        return max(1, T // 3), max(2, 2 * T // 3)

    d0 = float(min_dist[0])
    dmin = float(np.min(min_dist))
    denom = max(d0 - dmin, 1e-6)
    dist_progress = np.clip((d0 - min_dist) / denom, 0.0, 1.0)
    cov_peak = max(float(np.max(cov)), 1e-6)
    cov_progress = np.clip(cov / cov_peak, 0.0, 1.0)

    phase1_mask = (dist_progress >= 1.0 / 3.0) | (cov_progress >= 0.25)
    phase2_mask = (dist_progress >= 2.0 / 3.0) | (cov_progress >= 0.75)

    a_end = int(np.argmax(phase1_mask)) if np.any(phase1_mask) else T // 3
    t_start = int(np.argmax(phase2_mask)) if np.any(phase2_mask) else (2 * T) // 3

    a_end = int(np.clip(a_end, 1, T - 2))
    t_start = int(np.clip(t_start, a_end + 1, T - 1))
    return a_end, t_start


def _phase_slice_values(values: np.ndarray, phase: str, a_end: int, t_start: int) -> np.ndarray:
    if phase == "approach":
        return values[:a_end]
    if phase == "pre_encirclement":
        return values[a_end:t_start]
    if phase == "terminal_capture":
        return values[t_start:]
    raise ValueError(f"Unknown phase: {phase}")


def build_phase_analysis_tables(
    bundle: EpisodeLogBundle,
    df_episode: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_records: list[dict[str, Any]] = []
    trend_records: list[dict[str, Any]] = []
    n_grid = 51

    for i, row in enumerate(bundle.meta.get("episodes", [])):
        series = bundle.pursuit_series[i] if i < len(bundle.pursuit_series) else []
        if not series:
            continue
        traj_xyz = np.asarray(bundle.trajectories[i], dtype=np.float64)
        cov, col, dang = _metrics_arrays_from_series(series)
        min_dist = _episode_min_distance_series(traj_xyz)
        T = int(min(len(cov), len(min_dist)))
        cov = cov[:T]
        col = col[:T]
        dang = dang[:T]
        min_dist = min_dist[:T]
        a_end, t_start = _phase_boundaries(cov, min_dist)

        for phase in PHASE_NAMES:
            cov_p = _phase_slice_values(cov, phase, a_end, t_start)
            col_p = _phase_slice_values(col, phase, a_end, t_start)
            dang_p = _phase_slice_values(dang, phase, a_end, t_start)
            dist_p = _phase_slice_values(min_dist, phase, a_end, t_start)
            if cov_p.size == 0:
                continue
            t_local = np.linspace(0.0, 1.0, cov_p.size)
            grid = np.linspace(0.0, 1.0, n_grid)
            for metric_name, vals in (
                ("C_cov", cov_p),
                ("C_col", col_p),
                ("D_ang", dang_p),
            ):
                if vals.size >= 2 and not np.allclose(vals, vals[0]):
                    slope = float(np.polyfit(t_local, vals, deg=1)[0])
                else:
                    slope = 0.0
                episode_records.append(
                    {
                        "episode": int(row["episode"]),
                        "outcome": df_episode.loc[i, "outcome"],
                        "phase": phase,
                        "phase_steps": int(vals.size),
                        "metric": metric_name,
                        "mean": float(np.mean(vals)),
                        "start": float(vals[0]),
                        "end": float(vals[-1]),
                        "delta": float(vals[-1] - vals[0]),
                        "slope": slope,
                        "mean_min_distance": float(np.mean(dist_p)),
                    }
                )
                interp_vals = np.interp(grid, t_local, vals) if vals.size > 1 else np.full_like(grid, vals[0])
                for j, gv in enumerate(grid):
                    trend_records.append(
                        {
                            "episode": int(row["episode"]),
                            "outcome": df_episode.loc[i, "outcome"],
                            "phase": phase,
                            "metric": metric_name,
                            "t01": float(gv),
                            "value": float(interp_vals[j]),
                        }
                    )

    return pd.DataFrame.from_records(episode_records), pd.DataFrame.from_records(trend_records)


def plot_figure_d_terminal_window_summary(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> bool:
    if summary_df.empty:
        return False
    metrics = ["C_cov", "C_col", "D_ang"]
    windows = list(TERMINAL_WINDOWS)

    logit = summary_df[
        (summary_df["analysis"] == "logistic_regression")
        & (summary_df["target"] == "captured_vs_timeout")
    ]
    if logit.empty:
        return False
    mat_logit = np.full((len(metrics), len(windows)), np.nan, dtype=np.float64)
    mat_auc = np.full_like(mat_logit, np.nan)
    for mi, metric in enumerate(metrics):
        for wi, window in enumerate(windows):
            sel = logit[(logit["metric"] == metric) & (logit["window"] == window)]
            if not sel.empty:
                mat_logit[mi, wi] = float(sel.iloc[0]["coef"])
                mat_auc[mi, wi] = float(sel.iloc[0]["auc"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, mat, title in zip(
        axes,
        (mat_logit, mat_auc),
        ("logistic coef (captured vs timeout)", "AUC (captured vs timeout)"),
    ):
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_xticks(np.arange(len(windows)), [str(x) for x in windows])
        ax.set_yticks(np.arange(len(metrics)), metrics)
        ax.set_xlabel("terminal window K")
        ax.set_title(title)
        for mi in range(len(metrics)):
            for wi in range(len(windows)):
                val = mat[mi, wi]
                if np.isfinite(val):
                    ax.text(wi, mi, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Fig D: Terminal-window logistic summary", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_figure_e_phase_trends(
    phase_trend_df: pd.DataFrame,
    out_path: Path,
) -> bool:
    show_df = phase_trend_df[phase_trend_df["outcome"].isin(["captured", "timeout"])]
    if show_df.empty:
        return False
    metrics = ["C_cov", "C_col", "D_ang"]
    phases = list(PHASE_NAMES)
    colors = {"captured": "#2ca02c", "timeout": "#ff7f0e"}
    fig, axes = plt.subplots(len(metrics), len(phases), figsize=(13.0, 8.0), sharex=True, sharey="row")

    for mi, metric in enumerate(metrics):
        for pi, phase in enumerate(phases):
            ax = axes[mi, pi]
            sub = show_df[(show_df["metric"] == metric) & (show_df["phase"] == phase)]
            for outcome in ("captured", "timeout"):
                grp = sub[sub["outcome"] == outcome]
                if grp.empty:
                    continue
                stat = grp.groupby("t01")["value"].agg(["mean", "std"]).reset_index()
                t = stat["t01"].to_numpy(dtype=np.float64)
                m = stat["mean"].to_numpy(dtype=np.float64)
                s = stat["std"].to_numpy(dtype=np.float64)
                ax.plot(t, m, color=colors[outcome], lw=2.0, label=outcome)
                ax.fill_between(t, m - s, m + s, color=colors[outcome], alpha=0.18)
            if mi == 0:
                ax.set_title(phase.replace("_", "\n"))
            if pi == 0:
                ax.set_ylabel(metric)
            if mi == len(metrics) - 1:
                ax.set_xlabel("phase-normalized time")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            if mi == 0 and pi == len(phases) - 1:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Fig E: Phase-wise metric trends (captured vs timeout)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def best_capture_relevance_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    logit = summary_df[
        (summary_df["analysis"] == "logistic_regression")
        & (summary_df["target"] == "captured_vs_timeout")
    ].copy()
    if logit.empty:
        return pd.DataFrame()

    metric_order = ["C_cov", "D_ang", "C_col", "max_escape_gap"]
    metric_labels = {
        "C_cov": r"$C_{\mathrm{cov}}$",
        "D_ang": r"$D_{\mathrm{ang}}$",
        "C_col": r"$C_{\mathrm{col}}$",
        "max_escape_gap": "max escape gap",
        "F_esc": r"$F_{\mathrm{esc}}$",
        "role_stability": "role stability",
    }
    metric_order = ["C_cov", "D_ang", "C_col", "max_escape_gap", "F_esc", "role_stability"]
    best_rows: list[pd.Series] = []
    for metric in metric_order:
        sel = logit[logit["metric"] == metric].sort_values(["auc", "accuracy"], ascending=False)
        if not sel.empty:
            best_rows.append(sel.iloc[0])
    if not best_rows:
        return pd.DataFrame()

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_df["label"] = best_df["metric"].map(metric_labels)
    return best_df


def plot_figure_f_capture_relevance(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> bool:
    best_df = best_capture_relevance_rows(summary_df)
    if best_df.empty:
        return False

    window_colors = {30: "#9ecae1", 50: "#6baed6", 100: "#2171b5"}

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(best_df), dtype=np.float64)
    colors = [window_colors.get(int(w), "#4c78a8") for w in best_df["window"]]
    ax.bar(x, best_df["auc"], color=colors, alpha=0.92, width=0.66)
    ax.axhline(0.5, color="0.75", lw=1.0, ls="--")
    ax.set_xticks(x, best_df["label"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("best AUC for captured vs timeout")
    ax.set_title("Fig F: Which terminal metric is most related to capture success?")
    ax.grid(True, axis="y", alpha=0.25)

    for xi, (_, row) in enumerate(best_df.iterrows()):
        direction = "capture +" if float(row["coef"]) >= 0.0 else "capture -"
        txt = f"{float(row['auc']):.3f}\nlast{int(row['window'])}\n{direction}"
        ax.text(xi, float(row["auc"]) + 0.02, txt, va="bottom", ha="center", fontsize=9, color="black")

    handles = [
        Line2D([0], [0], color=color, lw=6, label=f"best window = last{window}")
        for window, color in window_colors.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="window")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_figure_g_capture_relevance_comparison(
    summary_a: pd.DataFrame,
    label_a: str,
    summary_b: pd.DataFrame,
    label_b: str,
    out_path: Path,
) -> bool:
    best_a = best_capture_relevance_rows(summary_a)
    best_b = best_capture_relevance_rows(summary_b)
    if best_a.empty or best_b.empty:
        return False

    metric_order = ["C_cov", "D_ang", "C_col", "max_escape_gap", "F_esc", "role_stability"]
    metric_labels = {
        "C_cov": r"$C_{\mathrm{cov}}$",
        "D_ang": r"$D_{\mathrm{ang}}$",
        "C_col": r"$C_{\mathrm{col}}$",
        "max_escape_gap": "max escape gap",
        "F_esc": r"$F_{\mathrm{esc}}$",
        "role_stability": "role stability",
    }
    best_a = best_a.set_index("metric").reindex(metric_order).reset_index()
    best_b = best_b.set_index("metric").reindex(metric_order).reset_index()

    x = np.arange(len(metric_order), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    bars_a = ax.bar(x - width / 2, best_a["auc"], width=width, color="#4c78a8", alpha=0.92, label=label_a)
    bars_b = ax.bar(x + width / 2, best_b["auc"], width=width, color="#f58518", alpha=0.92, label=label_b)

    ax.axhline(0.5, color="0.75", lw=1.0, ls="--")
    ax.set_xticks(x, [metric_labels[m] for m in metric_order])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("best AUC for captured vs timeout")
    ax.set_title("Fig G: Capture-success relevance comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    for bars, best_df in ((bars_a, best_a), (bars_b, best_b)):
        for bar, (_, row) in zip(bars, best_df.iterrows()):
            if not np.isfinite(float(row["auc"])):
                continue
            txt = f"{float(row['auc']):.3f}\nlast{int(row['window'])}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(bar.get_height()) + 0.02,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_figure_h_time_auc_curves(
    auc_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Fig H: AUC vs normalized time",
) -> bool:
    if auc_df.empty:
        return False
    metric_order = ["C_cov", "D_ang", "C_col", "max_escape_gap", "F_esc", "role_stability"]
    metric_labels = {
        "C_cov": r"$C_{\mathrm{cov}}$",
        "D_ang": r"$D_{\mathrm{ang}}$",
        "C_col": r"$C_{\mathrm{col}}$",
        "max_escape_gap": "max escape gap",
        "F_esc": r"$F_{\mathrm{esc}}$",
        "role_stability": "role stability",
    }
    mode_titles = {
        "prefix_mean": "prefix mean up to t",
        "local_mean": f"local window mean (last {int(LOCAL_WINDOW_RATIO * 100)}% episode)",
    }
    colors = {
        "C_cov": "#4c78a8",
        "D_ang": "#54a24b",
        "C_col": "#f58518",
        "max_escape_gap": "#e45756",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for ax, mode in zip(axes, ("prefix_mean", "local_mean")):
        sub_mode = auc_df[auc_df["mode"] == mode]
        for metric in metric_order:
            sub = sub_mode[sub_mode["metric"] == metric].sort_values("time_fraction")
            if sub.empty:
                continue
            t = sub["time_fraction"].to_numpy(dtype=np.float64)
            auc = sub["oriented_auc"].to_numpy(dtype=np.float64)
            ax.plot(t, auc, marker="o", lw=2.0, color=colors[metric], label=metric_labels[metric])
            for tx, av in zip(t, auc):
                ax.text(tx, av + 0.012, f"{av:.2f}", ha="center", va="bottom", fontsize=8, color=colors[metric])
        ax.axhline(0.5, color="0.75", lw=1.0, ls="--")
        ax.set_title(mode_titles[mode])
        ax.set_xlabel("normalized time")
        ax.set_xticks(list(TIME_FRACTIONS))
        ax.set_ylim(0.45, 1.01)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("oriented AUC for captured vs timeout")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_figure_i_time_auc_comparison(
    auc_a: pd.DataFrame,
    label_a: str,
    auc_b: pd.DataFrame,
    label_b: str,
    out_path: Path,
) -> bool:
    if auc_a.empty or auc_b.empty:
        return False
    metric_order = ["C_cov", "D_ang", "C_col", "max_escape_gap"]
    metric_labels = {
        "C_cov": r"$C_{\mathrm{cov}}$",
        "D_ang": r"$D_{\mathrm{ang}}$",
        "C_col": r"$C_{\mathrm{col}}$",
        "max_escape_gap": "max escape gap",
    }
    model_styles = {
        label_a: ("#4c78a8", "-"),
        label_b: ("#f58518", "--"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True, sharey=True)
    axes_flat = axes.reshape(-1)
    for ax, metric in zip(axes_flat, metric_order):
        for label, auc_df in ((label_a, auc_a), (label_b, auc_b)):
            for mode, alpha in (("prefix_mean", 1.0), ("local_mean", 0.55)):
                sub = auc_df[(auc_df["metric"] == metric) & (auc_df["mode"] == mode)].sort_values("time_fraction")
                if sub.empty:
                    continue
                t = sub["time_fraction"].to_numpy(dtype=np.float64)
                auc = sub["oriented_auc"].to_numpy(dtype=np.float64)
                color, linestyle = model_styles[label]
                line_label = f"{label} | {'prefix' if mode == 'prefix_mean' else 'local'}"
                ax.plot(t, auc, marker="o", lw=2.0, ls=linestyle, color=color, alpha=alpha, label=line_label)
                for tx, av in zip(t, auc):
                    ax.text(tx, av + 0.01, f"{av:.2f}", ha="center", va="bottom", fontsize=7, color=color, alpha=alpha)
        ax.axhline(0.5, color="0.75", lw=1.0, ls="--")
        ax.set_title(metric_labels[metric])
        ax.set_xticks(list(TIME_FRACTIONS))
        ax.set_ylim(0.45, 1.01)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("normalized time")
    axes[1, 1].set_xlabel("normalized time")
    axes[0, 0].set_ylabel("oriented AUC")
    axes[1, 0].set_ylabel("oriented AUC")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=True)
    fig.suptitle("Fig I: AUC-vs-time comparison", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_figure_j_conditional_failures(
    df_episode: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    conditional_quantiles: pd.DataFrame,
    out_path: Path,
    *,
    window: int = CONDITIONAL_FAILURE_WINDOW,
) -> bool:
    if conditional_summary.empty:
        return False
    ccol_col = f"C_col_last{window}"
    dang_col = f"D_ang_last{window}"
    ccov_col = f"C_cov_last{window}"
    gap_col = f"max_escape_gap_last{window}"
    captured = df_episode[df_episode["captured"] == 1]
    if captured.empty:
        return False
    ccol_thr = float(captured[ccol_col].median())
    dang_thr = float(captured[dang_col].median())
    ccov_thr = float(captured[ccov_col].median())
    structured_subset = df_episode[
        df_episode["outcome"].isin(["captured", "timeout"])
        & (df_episode[ccol_col] <= ccol_thr)
        & (df_episode[dang_col] >= dang_thr)
        & (df_episode[ccov_col] <= ccov_thr)
    ].copy()
    if structured_subset.empty:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    for outcome, color in (("captured", "#2ca02c"), ("timeout", "#d62728")):
        sub = structured_subset[structured_subset["outcome"] == outcome]
        if sub.empty:
            continue
        ax.scatter(
            sub[ccov_col],
            np.degrees(sub[gap_col]),
            s=46,
            alpha=0.82,
            color=color,
            edgecolors="0.25",
            linewidths=0.4,
            label=f"{outcome} (n={len(sub)})",
        )
    ax.set_xlabel(rf"low terminal $C_{{\mathrm{{cov}}}}$ (last {window})")
    ax.set_ylabel("max escape gap (deg, last window mean)")
    ax.set_title("good C_col + good D_ang + low C_cov")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    if not conditional_quantiles.empty:
        x = np.arange(len(conditional_quantiles), dtype=np.float64)
        vals = conditional_quantiles["timeout_rate"].to_numpy(dtype=np.float64)
        bars = ax.bar(x, vals, width=0.62, color="#9467bd", alpha=0.88)
        labels = [
            f">= {np.degrees(v):.1f}°"
            for v in conditional_quantiles["max_escape_gap_min"].to_numpy(dtype=np.float64)
        ]
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("timeout rate inside structured subset")
        ax.set_title("timeout rate vs escape-gap severity")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(val) + 0.03,
                f"{float(val):.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Fig J: Structured failure analysis", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def save_episode_level_analysis(bundle: EpisodeLogBundle, cache_dir: Path) -> None:
    analysis_dir = cache_dir / "analysis"
    fig_dir = cache_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df_episode = build_episode_analysis_table(bundle)
    df_terminal = summarize_terminal_window_associations(df_episode)
    df_phase_episode, df_phase_trend = build_phase_analysis_tables(bundle, df_episode)
    df_time_auc = summarize_time_auc_curves(bundle)
    df_conditional, df_conditional_quantiles, conditional_meta = summarize_conditional_failures(df_episode)

    df_episode.to_csv(analysis_dir / "episode_terminal_features.csv", index=False, encoding="utf-8-sig")
    df_terminal.to_csv(analysis_dir / "terminal_window_associations.csv", index=False, encoding="utf-8-sig")
    df_phase_episode.to_csv(analysis_dir / "phase_episode_summary.csv", index=False, encoding="utf-8-sig")
    df_phase_trend.to_csv(analysis_dir / "phase_trend_curves.csv", index=False, encoding="utf-8-sig")
    df_time_auc.to_csv(analysis_dir / "time_auc_curves.csv", index=False, encoding="utf-8-sig")
    df_conditional.to_csv(analysis_dir / "conditional_failures_summary.csv", index=False, encoding="utf-8-sig")
    df_conditional_quantiles.to_csv(
        analysis_dir / "conditional_failures_gap_bins.csv", index=False, encoding="utf-8-sig"
    )

    defs = {
        "terminal_windows": list(TERMINAL_WINDOWS),
        "remaining_escape_angle": "final-step phi_max (largest angular gap among three pursuers), in radians/degrees",
        "capture_step": "episode_len for captured episodes; NaN otherwise",
        "phase_rule": {
            "approach_end": "first step where distance-progress >= 1/3 or normalized C_cov >= 0.25",
            "terminal_start": "first step where distance-progress >= 2/3 or normalized C_cov >= 0.75",
            "fallback": "if thresholds fail, use approximate thirds",
        },
        "time_auc_experiment": {
            "time_fractions": list(TIME_FRACTIONS),
            "prefix_mean": "mean from episode start to current normalized time t",
            "local_mean": f"mean over the last {int(LOCAL_WINDOW_RATIO * 100)}% of the episode up to time t",
            "reported_auc": "oriented AUC = auc if logistic coef >= 0 else 1 - auc",
        },
        "conditional_failure_experiment": {
            "window": CONDITIONAL_FAILURE_WINDOW,
            "good_C_col": "C_col_last100 <= median(C_col_last100 over captured episodes)",
            "good_D_ang": "D_ang_last100 >= median(D_ang_last100 over captured episodes)",
            "low_C_cov": "C_cov_last100 <= median(C_cov_last100 over captured episodes)",
            "question": "among episodes with good C_col and good D_ang but still-low C_cov, do timeouts mainly have larger max_escape_gap?",
            **conditional_meta,
        },
    }
    with open(analysis_dir / "analysis_definitions.json", "w", encoding="utf-8") as f:
        json.dump(defs, f, indent=2, ensure_ascii=False)

    ok_d = plot_figure_d_terminal_window_summary(df_terminal, fig_dir / "figD_terminal_window_logistic.png")
    ok_e = plot_figure_e_phase_trends(df_phase_trend, fig_dir / "figE_phase_trends.png")
    ok_f = plot_figure_f_capture_relevance(df_terminal, fig_dir / "figF_capture_relevance.png")
    ok_h = plot_figure_h_time_auc_curves(df_time_auc, fig_dir / "figH_time_auc_curves.png")
    ok_j = plot_figure_j_conditional_failures(
        df_episode,
        df_conditional,
        df_conditional_quantiles,
        fig_dir / "figJ_conditional_failures.png",
    )

    print(f"Saved analysis table: {analysis_dir / 'episode_terminal_features.csv'}")
    print(f"Saved analysis table: {analysis_dir / 'terminal_window_associations.csv'}")
    print(f"Saved analysis table: {analysis_dir / 'phase_episode_summary.csv'}")
    print(f"Saved analysis table: {analysis_dir / 'phase_trend_curves.csv'}")
    print(f"Saved analysis table: {analysis_dir / 'time_auc_curves.csv'}")
    print(f"Saved analysis table: {analysis_dir / 'conditional_failures_summary.csv'}")
    print(f"Saved analysis table: {analysis_dir / 'conditional_failures_gap_bins.csv'}")
    print(f"Saved analysis defs:  {analysis_dir / 'analysis_definitions.json'}")
    print(f"{'Saved' if ok_d else 'Skipped'} Fig D: {fig_dir / 'figD_terminal_window_logistic.png'}")
    print(f"{'Saved' if ok_e else 'Skipped'} Fig E: {fig_dir / 'figE_phase_trends.png'}")
    print(f"{'Saved' if ok_f else 'Skipped'} Fig F: {fig_dir / 'figF_capture_relevance.png'}")
    print(f"{'Saved' if ok_h else 'Skipped'} Fig H: {fig_dir / 'figH_time_auc_curves.png'}")
    print(f"{'Saved' if ok_j else 'Skipped'} Fig J: {fig_dir / 'figJ_conditional_failures.png'}")

    logit = df_terminal[
        (df_terminal["analysis"] == "logistic_regression")
        & (df_terminal["target"] == "captured_vs_timeout")
    ].copy()
    if not logit.empty:
        best_auc = logit.sort_values("auc", ascending=False).head(3)
        print("\n=== Top terminal-window logistic signals (captured vs timeout) ===")
        for _, r in best_auc.iterrows():
            print(
                f"{r['metric']}@last{int(r['window'])}: "
                f"coef={float(r['coef']):+.3f} OR/std={float(r['odds_ratio_per_std']):.3f} "
                f"AUC={float(r['auc']):.3f} n={int(r['n'])}"
            )

    spearman = df_terminal[df_terminal["analysis"] == "spearman"].copy()
    if not spearman.empty:
        best_corr = spearman.assign(abs_stat=spearman["stat"].abs()).sort_values("abs_stat", ascending=False).head(6)
        print("\n=== Strongest terminal-window correlations (Spearman) ===")
        for _, r in best_corr.iterrows():
            print(
                f"{r['metric']}@last{int(r['window'])} vs {r['target']}: "
                f"rho={float(r['stat']):+.3f} p={float(r['pvalue']):.3g} n={int(r['n'])}"
            )
    if conditional_meta:
        subset_n = conditional_meta.get("structured_subset_n", conditional_meta.get("good_subset_n", np.nan))
        print("\n=== Conditional failure analysis (good C_col + good D_ang + low C_cov subset) ===")
        print(
            f"thresholds: C_col <= {float(conditional_meta['good_c_col_threshold']):.4f}, "
            f"D_ang >= {float(conditional_meta['good_d_ang_threshold']):.4f}, "
            f"C_cov <= {float(conditional_meta['low_c_cov_threshold']):.4f}; "
            f"subset n={int(subset_n) if np.isfinite(float(subset_n)) else 0}"
        )
        if not df_conditional.empty:
            for _, r in df_conditional.iterrows():
                print(
                    f"{r['group']}: n={int(r['n'])} "
                    f"mean gap={float(r['mean_max_escape_gap_deg']):.1f} deg "
                    f"mean C_col={float(r['mean_C_col']):.4f} "
                    f"mean D_ang={float(r['mean_D_ang']):.4f} "
                    f"mean C_cov={float(r['mean_C_cov']):.4f}"
                )
        if np.isfinite(float(conditional_meta.get("gap_logit_auc", np.nan))):
            print(
                f"within structured subset, max_escape_gap logistic AUC="
                f"{float(conditional_meta['gap_logit_auc']):.3f}, coef={float(conditional_meta['gap_logit_coef']):+.3f}"
            )


def save_capture_relevance_comparison_figure(
    primary_cache_dir: Path,
    primary_label: str,
    compare_cache_dir: Path,
    compare_label: str,
) -> None:
    primary_csv = primary_cache_dir / "analysis" / "terminal_window_associations.csv"
    compare_csv = compare_cache_dir / "analysis" / "terminal_window_associations.csv"
    if not primary_csv.is_file() or not compare_csv.is_file():
        raise FileNotFoundError("comparison figure needs both analysis/terminal_window_associations.csv files")

    summary_a = pd.read_csv(primary_csv)
    summary_b = pd.read_csv(compare_csv)
    common_root = primary_cache_dir.parent.parent
    out_path = common_root / "comparison_figures" / f"capture_relevance_{primary_label}_vs_{compare_label}.png"
    ok = plot_figure_g_capture_relevance_comparison(summary_a, primary_label, summary_b, compare_label, out_path)
    print(f"{'Saved' if ok else 'Skipped'} Fig G: {out_path}")


def save_time_auc_comparison_figure(
    primary_cache_dir: Path,
    primary_label: str,
    compare_cache_dir: Path,
    compare_label: str,
) -> None:
    primary_csv = primary_cache_dir / "analysis" / "time_auc_curves.csv"
    compare_csv = compare_cache_dir / "analysis" / "time_auc_curves.csv"
    if not primary_csv.is_file() or not compare_csv.is_file():
        raise FileNotFoundError("comparison figure needs both analysis/time_auc_curves.csv files")

    auc_a = pd.read_csv(primary_csv)
    auc_b = pd.read_csv(compare_csv)
    common_root = primary_cache_dir.parent.parent
    out_path = common_root / "comparison_figures" / f"time_auc_{primary_label}_vs_{compare_label}.png"
    ok = plot_figure_i_time_auc_comparison(auc_a, primary_label, auc_b, compare_label, out_path)
    print(f"{'Saved' if ok else 'Skipped'} Fig I: {out_path}")


def save_analysis_figures(bundle: EpisodeLogBundle, cache_dir: Path) -> None:
    fig_dir = cache_dir / "figures"
    paths = {
        "A": fig_dir / "figA_structure_evolution_captured_vs_timeout.png",
        "B": fig_dir / "figB_scatter_mean_cov_col.png",
        "C": fig_dir / "figC_typical_trajectories_structure.png",
    }
    ok_a = plot_figure_a_structure_evolution(bundle, paths["A"])
    ok_b = plot_figure_b_scatter_mean_cov_col(bundle, paths["B"])
    ok_c = plot_figure_c_typical_trajectories(bundle, paths["C"])
    for key, p in paths.items():
        flag = (key == "A" and ok_a) or (key == "B" and ok_b) or (key == "C" and ok_c)
        if flag:
            print(f"Saved Fig {key}: {p}")
        else:
            print(f"Skipped Fig {key} (insufficient data)")
    save_episode_level_analysis(bundle, cache_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3v1 追逃 episode 坐标日志缓存与统计")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "experiment" / "pursuit_evasion_mappo_3v1.yaml"),
        help="顶层实验配置",
    )
    p.add_argument("--seed", type=int, default=6, help="与 checkpoint / 缓存目录一致")
    p.add_argument("--episodes", type=int, default=100, help="无缓存时收集的 episode 数")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint 路径（默认同 eval.py）")
    p.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="缓存目录；默认 results/<exp>/episode_log_cache/seed<seed>/",
    )
    p.add_argument(
        "--compare-cache-dir",
        type=str,
        default="",
        help="可选：第二个缓存目录，用于生成并排对比图",
    )
    p.add_argument(
        "--compare-label",
        type=str,
        default="",
        help="可选：第二个缓存目录在对比图中的显示名称",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="即使已有缓存也重新跑仿真并覆盖",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成 figures/ 下的统计图（图 A/B/C）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg_path = _ROOT / args.config
    exp_cfg = load_config(exp_cfg_path)
    env_cfg_path = _ROOT / exp_cfg.get("env", "configs/env/toy_uav.yaml")
    algo_cfg_path = _ROOT / exp_cfg.get("algo", "configs/algo/ippo.yaml")
    model_cfg_path = _ROOT / exp_cfg.get("model", "configs/model/mlp.yaml")
    task_cfg: dict[str, Any] = exp_cfg.get("task", {}) or {}

    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(exp_cfg_path, args.seed)

    loaded = False
    if not args.force and cache_is_complete(cache_dir):
        print(f"从缓存加载: {cache_dir}")
        bundle = load_episode_logs(cache_dir)
        loaded = True
    else:
        from marl_uav.agents.mac import MAC
        from marl_uav.envs.adapters.pyflyt_aviary_env import PURSUIT_EVASION_3V1_TASK_TYPES

        build_env, build_policy, build_learner = _load_eval_build_fns()
        if args.force and cache_dir.exists():
            print(f"--force: 将重新收集并覆盖 {cache_dir}")
        env = build_env(env_cfg_path, seed=args.seed, task_cfg=task_cfg)
        if not isinstance(env.task, PURSUIT_EVASION_3V1_TASK_TYPES):
            raise ValueError(
                "本脚本仅适用于 pursuit_evasion_3v1 / pursuit_evasion_3v1_ex1 / pursuit_evasion_3v1_ex2，请检查 task.name。"
            )

        if getattr(env, "obs_dim", None) is None or getattr(env, "state_dim", None) is None:
            try:
                env.reset(seed=args.seed)
            except TypeError:
                env.reset()

        policy_core = build_policy(model_cfg_path, env, algo_cfg_path)
        n_actions = (
            env.n_actions
            if getattr(policy_core, "action_space_type", "discrete") == "discrete"
            else int(getattr(policy_core, "action_dim", 0) or 0)
        )
        mac = MAC(obs_dim=env.obs_dim, n_actions=n_actions, n_agents=env.num_agents)
        mac.policy = policy_core
        learner = build_learner(algo_cfg_path, policy_core)

        if args.ckpt:
            ckpt_path = Path(args.ckpt)
        else:
            ckpt_path = _ROOT / "results" / exp_cfg_path.stem / "checkpoints" / str(args.seed) / "best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

        print(f"收集 {args.episodes} 个 episode（含全 agent 坐标与 pursuit_structure）…")
        rows, trajs, pss = collect_episodes_rollout(
            env=env,
            mac=mac,
            learner=learner,
            ckpt_path=ckpt_path,
            num_episodes=int(args.episodes),
            base_seed=int(args.seed),
        )

        meta_header = {
            "config": str(exp_cfg_path.as_posix()),
            "env": str(env_cfg_path.as_posix()),
            "algo": str(algo_cfg_path.as_posix()),
            "model": str(model_cfg_path.as_posix()),
            "task": task_cfg,
            "ckpt": str(ckpt_path.as_posix()),
            "base_seed": int(args.seed),
        }
        save_episode_logs(
            cache_dir,
            meta_header=meta_header,
            per_episode_rows=rows,
            trajectories_list=trajs,
            pursuit_series=pss,
        )
        print(f"已保存缓存到 {cache_dir}")
        bundle = load_episode_logs(cache_dir)

    print_basic_stats(bundle)
    if not args.no_plots:
        save_analysis_figures(bundle, cache_dir)
        if args.compare_cache_dir:
            compare_cache_dir = Path(args.compare_cache_dir)
            compare_bundle = load_episode_logs(compare_cache_dir)
            save_episode_level_analysis(compare_bundle, compare_cache_dir)
            primary_label = cache_dir.parents[1].name
            compare_label = args.compare_label or compare_cache_dir.parents[1].name
            save_capture_relevance_comparison_figure(cache_dir, primary_label, compare_cache_dir, compare_label)
            save_time_auc_comparison_figure(cache_dir, primary_label, compare_cache_dir, compare_label)
    if loaded:
        print(f"\n提示: 使用 --force 可重新跑 {args.episodes} episode 并覆盖缓存。")


if __name__ == "__main__":
    main()
