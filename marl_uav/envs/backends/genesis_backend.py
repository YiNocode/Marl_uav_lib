"""Genesis DroneEntity backend for UAV pursuit-evasion tasks.

Genesis is an optional dependency. This module imports it lazily inside the
backend constructor so PyFlyt and toy environments keep working when Genesis is
not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_uav.envs.backends.base_backend import BaseSimBackend, SimBackendState


class GenesisBackend(BaseSimBackend):
    """Thin Genesis scene wrapper with velocity-to-RPM control.

    The high-level MARL tasks emit velocity setpoints in the existing project
    convention ``[vx, vy, yaw_rate, vz]``. For ``action_type="velocity"``, this
    backend translates those setpoints to four motor RPMs with a simple
    stabilizing mixer. For ``action_type="rpm"``, actions are passed through as
    raw motor commands after clipping.
    """

    _initialized = False
    _initialized_device: str | None = None

    def __init__(
        self,
        *,
        num_pursuers: int = 3,
        num_evaders: int = 1,
        world_xy: float = 2.0,
        z_min: float = 0.5,
        z_max: float = 2.0,
        episode_limit: int = 400,
        dt: float = 0.01,
        n_envs: int = 1,
        drone_model: str = "CF2X",
        drone_urdf: str = "urdf/drones/cf2x.urdf",
        max_rpm: float = 25000.0,
        hover_rpm: float = 14475.8,
        headless: bool = True,
        device: str = "gpu",
        gravity: list[float] | tuple[float, float, float] = (0.0, 0.0, -9.81),
        action_type: str = "velocity",
        low_level_control: str = "rpm_pid",
        velocity_low: list[float] | None = None,
        velocity_high: list[float] | None = None,
        rpm_pid: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        try:
            import genesis as gs
        except ImportError as e:
            raise ImportError("Genesis backend requested but genesis is not installed.") from e

        self.gs = gs
        self.num_pursuers = int(num_pursuers)
        self.num_evaders = int(num_evaders)
        self.num_agents = self.num_pursuers + self.num_evaders
        self.world_xy = float(world_xy)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.episode_limit = int(episode_limit)
        self.dt = float(dt)
        self.n_envs = int(n_envs)
        self.drone_model = str(drone_model)
        self.drone_urdf = str(drone_urdf)
        self.max_rpm = float(max_rpm)
        self.hover_rpm = float(hover_rpm)
        self.headless = bool(headless)
        self.device = str(device).lower()
        self.gravity = tuple(float(x) for x in gravity)
        self.action_type = str(action_type).lower()
        self.low_level_control = str(low_level_control).lower()
        self.seed = seed

        if self.action_type not in ("velocity", "rpm"):
            raise ValueError(f"Unsupported Genesis action_type={action_type!r}")

        self.velocity_low = (
            np.asarray(velocity_low, dtype=np.float32).reshape(-1)
            if velocity_low is not None
            else np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        )
        self.velocity_high = (
            np.asarray(velocity_high, dtype=np.float32).reshape(-1)
            if velocity_high is not None
            else np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        if self.velocity_low.size != 4 or self.velocity_high.size != 4:
            raise ValueError("Genesis velocity_low/high must have 4 entries [vx, vy, yaw_rate, vz].")

        pid_cfg = rpm_pid or {}
        self.kx = float(pid_cfg.get("kx", 1800.0))
        self.ky = float(pid_cfg.get("ky", 1800.0))
        self.kz = float(pid_cfg.get("kz", 2600.0))

        self.scene = None
        self.drones: list[Any] = []
        self.elapsed_time = 0.0
        self._fallback_pos = np.zeros((self.num_agents, 3), dtype=np.float32)
        self._fallback_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self._fallback_euler = np.zeros((self.num_agents, 3), dtype=np.float32)
        self._last_setpoints = np.zeros((self.num_agents, 4), dtype=np.float32)

        self._ensure_genesis_initialized()

    def _ensure_genesis_initialized(self) -> None:
        """Initialize Genesis once per process."""
        if GenesisBackend._initialized:
            if GenesisBackend._initialized_device != self.device:
                raise RuntimeError(
                    "Genesis was already initialized with "
                    f"device={GenesisBackend._initialized_device!r}, cannot reinitialize with {self.device!r}."
                )
            return

        backend = self.gs.gpu if self.device == "gpu" else self.gs.cpu
        self.gs.init(backend=backend)
        GenesisBackend._initialized = True
        GenesisBackend._initialized_device = self.device

    def reset(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        seed: int | None = None,
    ) -> SimBackendState:
        """Rebuild the Genesis scene at the requested initial positions."""
        if seed is not None:
            self.seed = int(seed)
            np.random.default_rng(self.seed)

        start_pos = np.asarray(start_pos, dtype=np.float32)
        start_orn = np.asarray(start_orn, dtype=np.float32)
        if start_pos.shape != (self.num_agents, 3):
            raise ValueError(f"start_pos must have shape {(self.num_agents, 3)}, got {start_pos.shape}")
        if start_orn.shape != (self.num_agents, 3):
            raise ValueError(f"start_orn must have shape {(self.num_agents, 3)}, got {start_orn.shape}")

        self._fallback_pos = start_pos.copy()
        self._fallback_vel = np.zeros((self.num_agents, 3), dtype=np.float32)
        self._fallback_euler = start_orn.copy()
        self._last_setpoints = np.zeros((self.num_agents, 4), dtype=np.float32)
        self.elapsed_time = 0.0

        self._build_scene(start_pos)
        return self.get_backend_state()

    def _build_scene(self, start_pos: np.ndarray) -> None:
        """Create a fresh Genesis scene.

        Reset uses scene rebuild because DroneEntity pose reset APIs differ
        across Genesis versions. This keeps the first integration robust and
        isolated from the training loop.
        """
        self.close()
        gs = self.gs
        sim_options = gs.options.SimOptions(dt=self.dt, gravity=self.gravity)
        try:
            self.scene = gs.Scene(sim_options=sim_options, show_viewer=not self.headless)
        except TypeError:
            self.scene = gs.Scene(sim_options=sim_options)

        try:
            self.scene.add_entity(gs.morphs.Plane())
        except Exception:
            pass

        self.drones = []
        for i in range(self.num_agents):
            drone = self.scene.add_entity(
                gs.morphs.Drone(
                    file=self.drone_urdf,
                    model=self.drone_model,
                    pos=tuple(float(x) for x in start_pos[i]),
                )
            )
            self.drones.append(drone)

        if self.n_envs > 1:
            self.scene.build(n_envs=self.n_envs)
        else:
            self.scene.build()

    def step(self, actions: np.ndarray) -> SimBackendState:
        """Send one RPM command per drone and advance the Genesis scene."""
        if self.scene is None or not self.drones:
            raise RuntimeError("Genesis scene is not initialized. Call reset() first.")

        setpoints = np.asarray(actions, dtype=np.float32)
        if setpoints.shape[0] != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} action rows, got {setpoints.shape}")

        self._last_setpoints = setpoints.copy()
        current_vel = self._current_velocity_fallback_safe()
        rpms = self._actions_to_rpms(setpoints, current_vel)

        for drone, drone_rpms in zip(self.drones, rpms):
            cmd = self._format_rpm_command(drone_rpms)
            setter = getattr(drone, "set_propellels_rpm", None)
            if setter is None:
                setter = getattr(drone, "set_propellers_rpm", None)
            if setter is None:
                raise AttributeError(
                    "Genesis Drone entity has neither set_propellels_rpm nor set_propellers_rpm."
                )
            setter(cmd)

        self.scene.step()
        self.elapsed_time += self.dt
        self._integrate_fallback(setpoints)
        return self.get_backend_state()

    def _format_rpm_command(self, rpm: np.ndarray) -> np.ndarray:
        """Match Genesis single-env or batched RPM command shape."""
        rpm = np.asarray(rpm, dtype=np.float32).reshape(4)
        if self.n_envs > 1:
            return np.repeat(rpm.reshape(1, 4), self.n_envs, axis=0)
        return rpm

    def _actions_to_rpms(self, actions: np.ndarray, current_vel: np.ndarray) -> np.ndarray:
        """Convert high-level actions to clipped motor RPMs.

        The velocity mixer is intentionally simple: vertical velocity error
        shifts collective thrust, while target x/y velocities create pitch and
        roll deltas. Yaw is ignored in the first integration to preserve stable
        translational control.
        """
        if self.action_type == "rpm":
            if actions.shape[1] != 4:
                raise ValueError(f"RPM action_type expects action shape [N, 4], got {actions.shape}")
            rpms = actions.astype(np.float32)
        else:
            clipped = np.clip(actions[:, :4], self.velocity_low, self.velocity_high)
            clipped = np.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0)
            target_vx = clipped[:, 0]
            target_vy = clipped[:, 1]
            target_vz = clipped[:, 3]
            curr_vz = current_vel[:, 2]

            base = self.hover_rpm + self.kz * (target_vz - curr_vz)
            pitch_delta = self.kx * target_vx
            roll_delta = self.ky * target_vy
            yaw_delta = np.zeros_like(base)

            rpms = np.stack(
                [
                    base + pitch_delta + roll_delta - yaw_delta,
                    base + pitch_delta - roll_delta + yaw_delta,
                    base - pitch_delta - roll_delta - yaw_delta,
                    base - pitch_delta + roll_delta + yaw_delta,
                ],
                axis=1,
            )

        rpms = np.nan_to_num(rpms, nan=self.hover_rpm, posinf=self.max_rpm, neginf=0.0)
        return np.clip(rpms, 0.0, self.max_rpm).astype(np.float32)

    def get_backend_state(self) -> SimBackendState:
        """Read state from Genesis and package it in the task-compatible layout."""
        states = np.zeros((self.num_agents, 4, 3), dtype=np.float32)
        for i, drone in enumerate(self.drones):
            states[i, 0, :] = self._read_vec(drone, ("get_ang", "get_ang_vel", "get_angular_velocity"), self._zero3())
            states[i, 1, :] = self._read_orientation(drone, self._fallback_euler[i])
            states[i, 2, :] = self._read_vec(drone, ("get_vel", "get_velocity", "get_lin_vel"), self._fallback_vel[i])
            states[i, 3, :] = self._read_vec(drone, ("get_pos", "get_position"), self._fallback_pos[i])
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        contact_array = np.zeros((self.num_agents, self.num_agents), dtype=np.int8)
        return SimBackendState(
            states=states,
            aux_states=[np.zeros((0,), dtype=np.float32) for _ in range(self.num_agents)],
            contact_array=contact_array,
            elapsed_time=float(self.elapsed_time),
        )

    def get_agent_state(self) -> dict[str, list[dict[str, np.ndarray]]]:
        """Return role-grouped state dictionaries for debug consumers."""
        state = self.get_backend_state().states
        agents: list[dict[str, np.ndarray]] = []
        for i in range(self.num_agents):
            agents.append(
                {
                    "pos": state[i, 3, :].copy(),
                    "vel": state[i, 2, :].copy(),
                    "euler": state[i, 1, :].copy(),
                }
            )
        return {
            "pursuers": agents[: self.num_pursuers],
            "evaders": agents[self.num_pursuers :],
        }

    def close(self) -> None:
        """Drop Python references to the current Genesis scene."""
        self.drones = []
        self.scene = None

    def _current_velocity_fallback_safe(self) -> np.ndarray:
        if not self.drones:
            return self._fallback_vel.copy()
        vel = np.zeros_like(self._fallback_vel)
        for i, drone in enumerate(self.drones):
            vel[i] = self._read_vec(drone, ("get_vel", "get_velocity", "get_lin_vel"), self._fallback_vel[i])
        return np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _integrate_fallback(self, setpoints: np.ndarray) -> None:
        """Maintain a finite fallback state if Genesis state getters are unavailable."""
        if self.action_type == "velocity":
            vel = np.clip(setpoints[:, :4], self.velocity_low, self.velocity_high)
            next_vel = np.stack([vel[:, 0], vel[:, 1], vel[:, 3]], axis=1).astype(np.float32)
        else:
            next_vel = self._fallback_vel
        self._fallback_vel = np.nan_to_num(next_vel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._fallback_pos = self._fallback_pos + self._fallback_vel * np.float32(self.dt)
        self._fallback_pos[:, 0] = np.clip(self._fallback_pos[:, 0], -self.world_xy, self.world_xy)
        self._fallback_pos[:, 1] = np.clip(self._fallback_pos[:, 1], -self.world_xy, self.world_xy)
        self._fallback_pos[:, 2] = np.clip(self._fallback_pos[:, 2], self.z_min, self.z_max)

    @staticmethod
    def _zero3() -> np.ndarray:
        return np.zeros((3,), dtype=np.float32)

    def _read_vec(self, obj: Any, names: tuple[str, ...], fallback: np.ndarray) -> np.ndarray:
        for name in names:
            getter = getattr(obj, name, None)
            if getter is None:
                continue
            try:
                return self._to_vec3(getter())
            except Exception:
                continue
        return np.asarray(fallback, dtype=np.float32).reshape(3)

    def _read_orientation(self, obj: Any, fallback: np.ndarray) -> np.ndarray:
        euler = self._read_vec(obj, ("get_euler", "get_rpy"), np.asarray(fallback, dtype=np.float32))
        if np.any(np.isfinite(euler)):
            return euler
        quat_getter = getattr(obj, "get_quat", None) or getattr(obj, "get_quaternion", None)
        if quat_getter is None:
            return np.asarray(fallback, dtype=np.float32).reshape(3)
        try:
            quat = self._to_numpy(quat_getter()).reshape(-1)
            if quat.size >= 4:
                return self._quat_to_euler(quat[:4])
        except Exception:
            pass
        return np.asarray(fallback, dtype=np.float32).reshape(3)

    def _to_vec3(self, value: Any) -> np.ndarray:
        arr = self._to_numpy(value)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim >= 2 and arr.shape[0] == self.n_envs:
            arr = arr[0]
        arr = arr.reshape(-1)
        if arr.size < 3:
            raise ValueError(f"Expected at least 3 values, got shape {arr.shape}")
        return arr[:3].astype(np.float32)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        return np.asarray(value)

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler, accepting either xyzw or wxyz convention."""
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        if abs(q[0]) > abs(q[3]):
            w, x, y, z = q
        else:
            x, y, z, w = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float32)
