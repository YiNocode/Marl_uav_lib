# Experiments (outline)

> Frame all tables as **structure-preserving encirclement framework vs. baselines**, not “MAPPO vs. our MAPPO.”

## E1. Open-space encirclement compatibility

- **Purpose:** Show the framework does not destroy simple capture ability.
- **Repo mapping:** `configs/benchmark/e1_1_open_space_suite.yaml`, scenario `e1_1_open_space`.
- **Methods:**
  - **SCE (core, no RL):** `sce` — deformable manifold + entropic OT + proportional execution (`configs/experiment/e1_1_open_space_pyflyt_sce.yaml`)
  - Framework + MAPPO backend (later): `dream_mappo_full`
  - RL backends / baselines: `mappo`, `reward_shaped_mappo`, `mappo_bc`, `mappo_bc_improve`
  - Heuristics: `pure_pursuit`, `fixed_ring`, `oracle_slot`
- **Runner:** `python scripts/benchmark_e1_1_open_space.py --mode all`
- **Metrics:** `capture_rate`, `capture_time_s`, collision/OOB rates, terminal-window `C_cov`, `C_col`, `D_ang`, `max_escape_gap`, `role_stability` (see `summarize()` in benchmark script).
- **Results:** use CSV under `results/e1_1_open_space_pyflyt/` — **do not invent numbers in prose**.

## E2. Obstacle-rich environment

- **Purpose:** Deformable manifold adapts enclosing geometry around obstacles.
- **Repo mapping:** `pursuit_evasion_3v1_ex2`, `pursuit_evasion_dream_mappo_3v1_ex2.yaml` (PyFlyt / Genesis).
- **Status:** training/eval paths exist; unified benchmark suite **TODO** (mirror E1.1 layout).

## E3. Narrow-passage environment

- **Purpose:** Avoid one-sided collapse under constrained geometry.
- **Status:** **TODO** — dedicated scenario config not present in `configs/benchmark/` snapshot.

## E4. Multi-exit environment

- **Purpose:** Escape-gap suppression at multiple egress directions.
- **Status:** **TODO** — dedicated scenario config not present.

## E5. Ablation study

Required ablations (paper); partial support in code/config:

| Ablation | Repo lever | Status |
|----------|------------|--------|
| Full framework | `dream_mappo_full` / ex1+Dream policy | E1.1 |
| w/o deformable manifold | **TODO** fixed-radius ring policy variant | partial: `fixed_ring` heuristic only |
| w/o transport-based role allocation | `role_assignment_mode: fixed` | supported |
| w/o escape-gap suppression | reduce structure weights / disable encirclement capture gate | **TODO** scripted grid |
| w/o assignment consistency | `assignment_inertia_margin: 0` | supported |
| w/o residual fine-tune | compare `dream_mappo_full` vs `mappo_bc` path | partial |
| RL backend only | `mappo` | E1.1 |
| MAPPO + structure reward only | `reward_shaped_mappo` | E1.1 |

## E6. Structural metrics and failure analysis

- Implemented in E1.1 benchmark summaries and `pursuit_episode_log_stats.py`.
- **TODO:** systematic failure-mode taxonomy (collapse vs. gap vs. role churn).

## E7. Runtime and deployability

- **TODO:** benchmark manifold, assignment, inference times; scalability in \(N\) and slot count; robustness tests (noise, delay, comm dropout) if implemented.

## Reporting checklist (every main table)

- [ ] capture rate (not sole headline)
- [ ] capture time
- [ ] collision / OOB rates
- [ ] max escape gap (rad / deg)
- [ ] angular coverage proxies (`C_cov`, `D_ang`)
- [ ] collapse proxy (`C_col`)
- [ ] role stability / switching
- [ ] endgame-window structural score
