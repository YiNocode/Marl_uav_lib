# Manifold Debug Suite

This directory contains a standalone debugging suite for closed 2D encirclement manifold generation. It checks geometry, obstacle response, boundary safety, and temporal continuity only. It does not evaluate capture rate and does not validate the SCE method.

The adapter first tries `marl_uav.control.manifold_generator.build_shared_manifold_curve` through a tiny synthetic task shim. If that import or call fails, it uses a simple fallback generator: a circular closed curve around the evader with optional obstacle repulsion, boundary-aware radial shrinkage, and temporal smoothing. The fallback is only for validating this debug infrastructure.

## Example Commands

```bash
python debug_manifold/run_debug_manifold.py --case g0_static_no_obstacle
python debug_manifold/run_debug_manifold.py --case g3_single_obstacle --num-steps 200 --K 256
python debug_manifold/run_debug_manifold.py --case all --num-steps 200 --K 256
```

Short aliases `g0` through `g5` are also accepted.

## Outputs

Each run writes to `debug_manifold/outputs/<timestamp>/`:

- `summary.csv`: one row per case.
- `metrics_timeseries.csv`: all timestep metrics.
- `csv/<case>_metrics.csv`: per-case timestep metrics.
- `figures/<case>_overlay.png`: boundary, evader trajectory, obstacles, sampled curves, and final curve.
- `figures/<case>_metrics.png`: selected metric time series.
- `failure_cases/*.png`: automatic snapshots for invalid, infeasible, penetrating, boundary-violating, self-intersecting, target-excluding, or jumpy frames.
- `report.md`: Markdown summary and diagnostic interpretation.

## Default Thresholds

- `closure_error_mean < 1e-3`
- `self_intersection_rate == 0`
- `boundary_violation_rate == 0`
- `obstacle_penetration_rate == 0` for feasible scenes
- `target_inside_rate > 0.99` for feasible scenes
- `winding_number_about_evader` approximately `1`
- `invalid_output_rate == 0`

These thresholds are debugging assumptions, not theoretical guarantees. Edit `DEFAULT_THRESHOLDS` in `metrics.py` to change them.

## Diagnostic Table

| Observed failure | likely cause | what to inspect |
|---|---|---|
| Curve penetrates obstacle | obstacle repulsion too weak or clearance not enforced | inspect min_obstacle_clearance and obstacle_weight |
| Curve self-intersects | local deformation too strong or topology not preserved | inspect self_intersection_count and curvature_p95 |
| Curve jumps suddenly | manifold parameterization discontinuity or hard obstacle influence threshold | inspect pointwise_shift_p95 and hausdorff_shift |
| Boundary violation | generate-then-clip logic or radius too large | inspect min_boundary_margin and boundary_violation_rate |
| Target not inside curve | enclosure constraint broken by obstacle or boundary deformation | inspect winding_number and target_inside |
| INFEASIBLE scene outputs malformed curve | missing feasibility detection | inspect infeasible_rate and invalid_output_rate |

