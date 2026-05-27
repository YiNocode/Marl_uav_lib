# Consolidated TODO (paper + code alignment)

## Theory / writing

- [ ] Migrate `01_abstract_intro_contributions.md` to LaTeX when template is added.
- [ ] Add formal OT/Sinkhorn subsection only after implementation or cite external solver.
- [ ] Remove any draft language implying guaranteed structure preservation.
- [ ] Add weight-sensitivity experiment before claiming tuning robustness.

## Implementation gaps vs. paper story

- [ ] **Entropic OT role allocator** (`role_allocator: entropic_ot`) — current: permutation + inertia in `pursuit_evasion_3v1_task_ex1.py`.
- [ ] E3 narrow-passage scenario configs + benchmark suite.
- [ ] E4 multi-exit scenario configs + benchmark suite.
- [ ] E2 unified benchmark runner (like `e1_1_open_space_suite.yaml`).
- [ ] E5 ablation matrix as generated config grid.
- [ ] E7 runtime profiler script (manifold / assign / policy ms).

## Metrics

- [ ] `collapse index` as named export in benchmark CSV (proxy: `C_col` exists).
- [ ] `angular coverage` explicit column alias for `C_cov`.
- [ ] `endgame-window structural score` composite — define formula or reuse terminal-window means.

## Experiments / results

- [ ] Complete E1.1 multi-seed runs (suite currently lists seed 103 only).
- [ ] Document actual results paths only; never fabricate tables in markdown.

## Documentation

- [ ] Optional: `marl_uav/framework/` thin module aliases (only if imports/tests updated).

## Compatibility note

- Keep filenames `dream_mappo_*` until training registry is updated with aliases.
