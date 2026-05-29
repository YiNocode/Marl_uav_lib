# Consolidated TODO (paper + code alignment)

## Theory / writing

- [ ] Migrate `01_abstract_intro_contributions.md` to LaTeX when template is added.
- [ ] Add formal OT/Sinkhorn subsection with citation to `marl_uav/framework/role_allocation.py` (implemented for SCE / `ot_slot`).
- [ ] Remove any draft language implying guaranteed structure preservation.
- [ ] Add weight-sensitivity experiment before claiming tuning robustness.
- [x] Document E1.1 heuristic hierarchy: **slot tier > fixed_ring > pure_pursuit** (`03_experiments.md`, CSV-backed).

## Implementation gaps vs. paper story

- [x] **Entropic OT role allocator** — `marl_uav/framework/role_allocation.py`, `ot_slot` baseline.
- [x] **Hungarian slot baseline** — `hungarian_slot` in E1.1 / E2 suites.
- [x] **Oracle slot baseline** — reference full-method slot targets.
- [ ] Export **SCE**, **mappo**, **dream_mappo_full** into same summary CSV as slot tier for complete E1 table.
- [ ] Demonstrate learned / SCE stack **≥ slot tier** on capture **and** structure (E1.1 bar raised).
- [ ] Migrate `dream_mappo_full` from `nearest` to `entropic_ot` if SCE core validates OT path.
- [ ] E3 narrow-passage scenario configs + benchmark suite.
- [ ] E4 multi-exit scenario configs + benchmark suite.
- [x] E2 unified benchmark runner — `configs/benchmark/e2_obstacles_suite.yaml`, `scripts/benchmark_e2_obstacles.py`.
- [ ] Run E2 multi-seed eval and archive to `save_result/` (expect slot methods to **diverge** from E1 tie).
- [ ] E5 ablation matrix as generated config grid.
- [ ] E7 runtime profiler script (manifold / assign / policy ms); prioritize slot baseline latency as deployability evidence.

## Metrics

- [ ] `collapse index` as named export in benchmark CSV (proxy: `C_col` exists).
- [ ] `angular coverage` explicit column alias for `C_cov`.
- [ ] `endgame-window structural score` composite — define formula or reuse terminal-window means.
- [x] E2 `obstacle_termination_rate` in `benchmark_e2_obstacles.py`.

## Experiments / results

- [x] E1.1 heuristic multi-seed runs archived (`save_result/2.5baseline/`, 5 seeds × 200 ep).
- [ ] E1.1 RL + SCE rows in summary CSV.
- [ ] E2 obstacle-rich runs archived with same schema.
- [ ] Explain / verify **identical** oracle/hungarian/ot metrics in open space (assignment degeneracy); test separation in E2.
- [x] Document actual results paths only; E1.1 heuristic numbers in `03_experiments.md`.

## Documentation

- [ ] Optional: `marl_uav/framework/` thin module aliases (only if imports/tests updated).

## Compatibility note

- Keep filenames `dream_mappo_*` until training registry is updated with aliases.
- Paper comparators: always list **slot tier** before MAPPO variants in tables and prose.
