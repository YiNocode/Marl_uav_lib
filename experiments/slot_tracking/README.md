# Low-Level Slot-Tracking Benchmark

This benchmark isolates the low-level slot-tracking controller from the full
SCE pipeline. It asks one question: can a UAV safely track an assigned moving
slot under dynamics limits, field boundaries, obstacles, noise, delays, and
multi-agent proximity constraints?

The benchmark does not run high-level SCE slot generation, assignment, MAPPO
training, or the full pursuit-evasion task. Slot trajectories are predefined
and reproducible.

## What It Compares

Controllers:

- `pure_pursuit`: velocity command toward the current slot.
- `pd`: PD tracker with optional slot-velocity feed-forward.
- `apf`: attractive slot tracking plus obstacle and boundary repulsion.
- `nominal_slot_tracker`: dynamic free-space reference tracker,
`v = slot_vel + kp * (slot_pos - uav_pos) - kd * uav_vel`, with configured
speed and acceleration limits.
- `existing`: wrapper around the repository's current
`ObstacleAvoidanceController`. By default it enables `free_space_fallback`,
which bypasses safety/path modules and uses `nominal_slot_tracker` when there
is no nearby obstacle, boundary risk, or inter-agent risk.

All controllers share the same benchmark dynamics, collision checker, obstacle
map, random seed, and slot trajectory. Controller parameters are configured in
YAML and are not silently tuned inside the runner.

## Directory Layout

```text
experiments/slot_tracking/
  run_slot_tracking_benchmark.py
  configs/
    slot_tracking_default.yaml
    slot_tracking_stress.yaml
  scenarios/
    slot_scenarios.py
    obstacle_maps.py
  controllers/
    baseline_pure_pursuit.py
    baseline_pd_tracker.py
    baseline_apf_tracker.py
    wrapped_existing_controller.py
  metrics/
    tracking_metrics.py
    safety_metrics.py
    failure_classifier.py
  analysis/
    summarize_slot_tracking.py
    plot_slot_tracking_results.py
  outputs/
    raw/
    summary/
    figures/
```

## Scenario Groups

- Group A: no-obstacle basic tracking: static, linear, circular, sinusoidal,
random-walk slots across speed levels.
- Group B: boundary stress tests: boundary-parallel motion, corner turns,
temporarily outside-boundary slots, and outward initial velocity.
- Group C: static obstacle tracking: sparse random maps, blocking obstacle,
narrow passage, U-shaped trap, boundary-obstacle combo.
- Group D: dynamic slot deformation and jump tests.
- Group E: robustness tests: observation noise, action delay, reduced actual
max speed, wind disturbance, obstacle dropout.
- Group F: three-UAV shadow-SCE test with predefined moving slots around a
virtual evader, without SCE planning or assignment.

Slot speeds above UAV max speed, and slot paths that are outside the field for
most of the episode, are marked as target-infeasible instead of being counted as
ordinary controller failures.

## Run

Default full benchmark:

```powershell
python experiments/slot_tracking/run_slot_tracking_benchmark.py `
  --config experiments/slot_tracking/configs/slot_tracking_default.yaml `
  --controllers pure_pursuit,pd,apf,nominal_slot_tracker,existing `
  --scenario_group all `
  --num_seeds 50 `
  --output_dir experiments/slot_tracking/outputs/default_run
```

Quick debug run:

```powershell
python experiments/slot_tracking/run_slot_tracking_benchmark.py `
  --config experiments/slot_tracking/configs/slot_tracking_default.yaml `
  --controllers pure_pursuit,pd,apf,nominal_slot_tracker,existing `
  --scenario_group A `
  --num_seeds 3 `
  --max_cases_per_group 2 `
  --output_dir experiments/slot_tracking/outputs/debug_run
```

The runner prints a lightweight progress bar with completed episodes,
percentage, elapsed time, ETA, and the current scenario/controller label. Use
`--no_progress` to disable it when writing logs to a file.

Stress configuration:

```powershell
python experiments/slot_tracking/run_slot_tracking_benchmark.py `
  --config experiments/slot_tracking/configs/slot_tracking_stress.yaml `
  --controllers pure_pursuit,pd,apf,nominal_slot_tracker,existing `
  --scenario_group all `
  --num_seeds 50 `
  --output_dir experiments/slot_tracking/outputs/stress_run
```

Boundary-filter clean configuration:

```powershell
python experiments/slot_tracking/run_slot_tracking_benchmark.py `
  --config experiments/slot_tracking/configs/slot_tracking_B_boundary_clean.yaml `
  --controllers pure_pursuit,pd,apf,nominal_slot_tracker,existing `
  --scenario_group B `
  --num_seeds 5 `
  --output_dir experiments/slot_tracking/outputs/B_boundary_clean
```

`slot_tracking_B_boundary_clean.yaml` starts each UAV near the initial boundary
slot with a valid in-bounds state. This isolates whether the boundary filter
keeps tracking safe, instead of mixing in long approach or reacquisition
failures from distant initial positions.

## Summarize And Plot

```powershell
python experiments/slot_tracking/analysis/summarize_slot_tracking.py `
  --input_dir experiments/slot_tracking/outputs/default_run `
  --output_csv experiments/slot_tracking/outputs/default_run/summary/summary.csv
```

```powershell
python experiments/slot_tracking/analysis/plot_slot_tracking_results.py `
  --input_dir experiments/slot_tracking/outputs/default_run `
  --output_dir experiments/slot_tracking/outputs/default_run/figures
```

## Outputs

Per-step raw trajectory CSV files are written to `outputs/<run>/raw/`.
Columns include:

```text
t, agent_id,
uav_x, uav_y, uav_z,
slot_x, slot_y, slot_z,
uav_vx, uav_vy, uav_vz,
action_x, action_y, action_z,
tracking_error,
nearest_obstacle_distance,
boundary_margin,
decision_time_ms
```

For diagnosis, raw files also include:

```text
v_goal_x, v_goal_y,
v_obstacle_x, v_obstacle_y,
v_boundary_x, v_boundary_y,
v_path_x, v_path_y,
v_smooth_x, v_smooth_y,
v_final_before_clip_x, v_final_before_clip_y,
v_final_after_clip_x, v_final_after_clip_y,
final_action_x, final_action_y,
clip_flag,
speed_saturation_flag,
acceleration_saturation_flag,
double_clip_warning,
cos_to_goal,
progress_projection,
distance_delta,
existing_bypass_reason,
used_existing_safety_modules
```

If an internal existing-controller component is not exposed by the current
project controller, the wrapper logs `NaN` for that component. Free-space
fallback logs zero obstacle/path/boundary velocity because those modules are
intentionally bypassed.

Per-episode metrics are written to:

```text
outputs/<run>/summary/episode_metrics.csv
```

Grouped summary metrics are written to:

```text
outputs/<run>/summary/summary.csv
```

Focused A-group diagnosis is written to:

```text
outputs/<run>/summary/a_group_subscenario_summary.csv
outputs/<run>/summary/a_group_failure_counts.csv
outputs/<run>/summary/infeasible_stress_summary.csv
```

Figures are written to:

```text
outputs/<run>/figures/
```

For failed no-obstacle existing-controller episodes, additional diagnostic
plots are written to:

```text
outputs/<run>/figures/debug_existing/
```

These include trajectory, tracking error, `cos_to_goal`, progress projection,
velocity component norms, and saturation flags.

For failed feasible boundary episodes in `boundary_corner_turn` and
`boundary_parallel_slot`, additional plots are written to:

```text
outputs/<run>/figures/debug_boundary/
```

These include boundary-aware trajectory plots, tracking error, per-boundary
outward action/velocity projections, braking margin, normal/tangential command
components, and command components before/after the boundary filter.

## Metrics

Tracking:

- RMSE, mean, median, P95, max, final error.
- Steady-state error over the final 20% of the episode.
- Slot-lost ratio.
- Time to lock.
- Reacquisition time after slot jumps.

Safety:

- Obstacle collision.
- Boundary violation.
- Inter-agent collision in Group F.
- Minimum obstacle clearance.
- Minimum boundary margin.
- Fraction of time below safety margin.
- Outward velocity ratio near boundaries.

Efficiency and smoothness:

- UAV path length.
- Slot path length.
- Detour ratio.
- Control effort.
- Mean/P95 acceleration norm.
- Mean/P95 jerk norm.
- Speed and acceleration saturation ratios.
- Mean/P95 decision time.

## Success Definition

An episode succeeds only if all configured conditions hold:

- No obstacle collision.
- No boundary violation.
- No inter-agent collision.
- Target is feasible.
- Final error is below `success.final_error_threshold`.
- Steady-state error is below `success.steady_state_error_threshold`.
- Slot-lost ratio is below `success.max_lost_ratio`.

Thresholds live in the YAML config.

## Failure Types

Each failed episode receives exactly one primary failure type:

- `TARGET_INFEASIBLE`
- `NO_PROGRESS`
- `STUCK_NEAR_OBSTACLE`
- `BOUNDARY_FAILURE`
- `COLLISION_OBSTACLE`
- `COLLISION_AGENT`
- `OSCILLATION`
- `LATE_RESPONSE`
- `TRACKING_DIVERGENCE`
- `UNKNOWN_FAILURE`

The rule order is implemented in:

```text
experiments/slot_tracking/metrics/failure_classifier.py
```

## Sanity Tests

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; pytest tests\test_slot_tracking_benchmark.py -q -p no:cacheprovider
```

The tests check that:

- Static slot without obstacles is solved by pure pursuit and PD.
- Obstacle overlap is detected.
- Boundary violation is detected.
- Metrics return finite values.

## Notes

The `existing` controller wrapper calls the current project
`ObstacleAvoidanceController` directly. The wrapper assumes its holonomic branch
returns a world-frame XY velocity command toward the assigned slot, with local
obstacle and boundary safety projection. The benchmark still applies shared
point-mass dynamics and shared safety checks after that command.

In obstacle scenarios, the wrapper uses a configurable line-of-sight shortcut:
if the straight segment from UAV to slot is collision-free and boundary/inter-
agent risk is inactive, it uses `nominal_slot_tracker` rather than invoking the
existing safety/path stack. The existing stack is activated only when actual
risk is detected.

The terminal report includes a focused existing-controller diagnosis:

- Whether existing beats PD in `static_slot`.
- Whether existing beats PD in feasible `linear_slot` speeds `<= 0.8 vmax`.
- Whether existing beats pure pursuit in `circular_slot`.
- Mean `cos_to_goal` for failed existing episodes.
- Fraction of failed existing steps with non-positive progress projection.
- Whether obstacle or boundary velocity is nonzero in free-space A scenarios.
- Whether commands appear to be clipped more than once.
- Top 10 failed existing episodes by P95 error.
