"""Replay an exported pursuer action trace in PX4 SITL (velocity offboard).

Requires optional dependency: pymavlink

Example (dry-run, validate file only):

    python scripts/playback_action_trace_px4.py \\
        --input results/action_traces/e1_1_open_space_pyflyt_oracle_slot_seed101_pursuer0.json \\
        --dry-run

Example (PX4 SITL must already be running, e.g. make px4_sitl gazebo):

    python scripts/playback_action_trace_px4.py \\
        --input results/action_traces/e1_1_open_space_pyflyt_oracle_slot_seed101_pursuer0.json \\
        --connection udp:127.0.0.1:14540
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "marl_uav.pursuer_action_trace":
        raise ValueError(f"Unsupported trace format: {data.get('format')!r}")
    return data


def _action_steps(trace: dict) -> list[dict]:
    return [s for s in trace.get("steps", []) if "setpoint" in s]


def _enu_to_ned(v_enu: list[float]) -> tuple[float, float, float]:
    """Convert ENU velocity to NED velocity for MAVLink local setpoints."""
    vx, vy, vz = (float(v_enu[0]), float(v_enu[1]), float(v_enu[2]))
    return vy, vx, -vz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay exported action trace on PX4 via OFFBOARD velocity.")
    p.add_argument("--input", type=str, required=True, help="JSON from export_pursuer_action_trace.py")
    p.add_argument("--connection", type=str, default="udp:127.0.0.1:14540")
    p.add_argument("--dry-run", action="store_true", help="Only validate and print trace summary.")
    p.add_argument("--max-steps", type=int, default=0, help="Limit replay steps (0 = all).")
    p.add_argument("--warmup-s", type=float, default=3.0, help="Seconds to stream setpoints before OFFBOARD.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.input)
    if not trace_path.is_absolute():
        trace_path = ROOT / trace_path
    trace = _load_trace(trace_path)
    steps = _action_steps(trace)
    if not steps:
        raise RuntimeError("Trace contains no setpoint steps.")

    control = trace["control"]
    dt_s = float(control["dt_s"])
    init = trace["initial_state"]
    ep = trace["episode"]

    print(f"[px4] trace={trace_path.name}")
    print(
        f"[px4] pursuer={ep['pursuer_index']} steps={len(steps)} "
        f"dt={dt_s:.6f}s capture={ep.get('capture')}"
    )
    print(f"[px4] initial position (ENU m): {init['position_m']}")
    print(f"[px4] first setpoint: {steps[0]['setpoint']}")

    if args.dry_run:
        print("[px4] dry-run OK")
        return

    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise SystemExit(
            "pymavlink is required for live PX4 replay. Install with: pip install pymavlink\n"
            "Or run with --dry-run to validate the exported file."
        ) from exc

    master = mavutil.mavlink_connection(args.connection, source_system=1, source_component=1)
    print(f"[px4] waiting for heartbeat on {args.connection} ...")
    master.wait_heartbeat(timeout=30)
    print(f"[px4] heartbeat from system {master.target_system} component {master.target_component}")

    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    def send_velocity_ned(vn: float, ve: float, vd: float, yaw_rate: float) -> None:
        master.mav.set_position_target_local_ned_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0,
            0,
            0,
            vn,
            ve,
            vd,
            0,
            0,
            0,
            0,
            float(yaw_rate),
        )

    # Warm-up stream before switching mode (PX4 offboard requirement).
    sp0 = steps[0]["setpoint"]
    vn, ve, vd = _enu_to_ned([sp0[0], sp0[1], sp0[3]])
    yaw_rate = float(sp0[2])
    t_end = time.time() + float(args.warmup_s)
    while time.time() < t_end:
        send_velocity_ned(vn, ve, vd, yaw_rate)
        time.sleep(dt_s)

    master.set_mode("OFFBOARD")
    master.armed = True

    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else len(steps)
    for idx, step in enumerate(steps[:max_steps]):
        sp = step["setpoint"]
        vn, ve, vd = _enu_to_ned([sp[0], sp[1], sp[3]])
        yaw_rate = float(sp[2])
        send_velocity_ned(vn, ve, vd, yaw_rate)
        if idx % 60 == 0:
            print(f"[px4] step {idx}/{max_steps} setpoint_enu={sp}")
        time.sleep(dt_s)

    # Hold zero velocity at end.
    for _ in range(30):
        send_velocity_ned(0.0, 0.0, 0.0, 0.0)
        time.sleep(dt_s)

    print("[px4] replay finished")


if __name__ == "__main__":
    main()
