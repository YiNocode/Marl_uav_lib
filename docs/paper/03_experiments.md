# Experiments (outline)

> Frame all tables as **structure-preserving encirclement framework vs. baselines**, not “MAPPO vs. our MAPPO.”  
> **E1.1 headline (heuristic tier):** deployable **slot methods** are the strongest baselines; pure pursuit is insufficient.

## Experiment program overview

| ID | Scenario | Task | Benchmark runner | Primary question |
|----|----------|------|------------------|------------------|
| **E1** | Open space | ex1 | `benchmark_e1_1_open_space.py` | Does the stack preserve encirclement vs. **slot-tier** geometry? |
| **E2** | Obstacle grid | ex2 | `benchmark_e2_obstacles.py` | Does obstacle-aware manifold + slots beat slot baselines when geometry is blocked? |
| **E3** | Narrow passage | **TODO** | — | Avoid one-sided collapse under constrained geometry |
| **E4** | Multi-exit | **TODO** | — | Escape-gap suppression at multiple egress directions |
| **E5** | Ablations | ex1/ex2 | partial configs | Which manifold / OT / structure terms matter? |
| **E6** | Failure analysis | E1/E2 logs | `pursuit_episode_log_stats.py` | Collapse vs. gap vs. role churn vs. obstacle hit |
| **E7** | Runtime | — | **TODO** | Deployability and scaling |

**Shared protocol (E1 / E2 heuristics):** PyFlyt 3v1, continuous velocity setpoints, reward scales zeroed for evaluation-only runs, 5 training seeds × 200 eval episodes per method (`seeds: 101–105`), terminal-window structural metrics (last 30 steps).

---

## E1. Open-space encirclement compatibility

### Purpose

Validate that (i) **weak** chase-only baselines fail predictable structure tests, (ii) **slot-tier** deployable controllers achieve reliable encirclement, and (iii) learned / SCE framework instances must be compared against **slots**, not pure pursuit alone.

### Repo mapping

- Suite: `configs/benchmark/e1_1_open_space_suite.yaml`
- Runner: `python scripts/benchmark_e1_1_open_space.py --mode eval`
- Results: `results/e1_1_open_space_pyflyt/`; archived summary `save_result/2.5baseline/e1_1_open_space_summary_by_method.csv`

### Method ladder (evaluation order)

1. **Weak geometric floor:** `pure_pursuit`
2. **Fixed geometry:** `fixed_ring`
3. **Strong deployable slot tier:** `oracle_slot`, `hungarian_slot`, `ot_slot` (manifold slot targets + proportional control; assignment differs)
4. **SCE core (no RL):** `sce` — full manifold + entropic OT + proportional execution
5. **RL execution backends:** `mappo`, `reward_shaped_mappo`, `mappo_bc`, `mappo_bc_improve`
6. **Framework + MAPPO backend:** `dream_mappo_full`

### E1.1 results — heuristic tier (5 seeds × 200 episodes)

Source: `save_result/2.5baseline/e1_1_open_space_summary_by_method.csv` (terminal window = 30 steps).

| Method | Capture ↑ | Collision ↓ | Timeout ↓ | Capture time (s) ↓ | `C_cov` | `C_col` | `D_ang` | Role stability |
|--------|-----------|-------------|-----------|-------------------|---------|---------|---------|----------------|
| **oracle_slot** | **1.000** | **0.000** | **0.000** | **10.24** | 0.215 | 0.926 | **0.170** | 0.993 |
| **hungarian_slot** | **1.000** | **0.000** | **0.000** | **10.24** | 0.215 | 0.926 | **0.170** | 0.993 |
| **ot_slot** | **1.000** | **0.000** | **0.000** | **10.24** | 0.215 | 0.926 | **0.170** | 0.993 |
| fixed_ring | 0.992 | 0.233 | 0.001 | 10.60 | 0.334 | 0.831 | 0.395 | 0.995 |
| pure_pursuit | 0.594 | 0.820 | 0.210 | 13.53 | 0.030 | 0.996 | 0.003 | 0.987 |

### Analysis (E1.1 structure)

1. **Slot tier dominates.** Oracle, Hungarian, and OT slot controllers tie as **strongest baselines**: perfect capture, **no collision-terminated episodes** in this aggregate, fastest mean capture time, and **best angular regularity** (`D_ang` ≈0.17 vs. 0.40 for fixed ring).
2. **Pure pursuit is not a credible encirclement baseline** (59.4% capture, 82% collision, collapsed coverage). Use it only as a **deliberately weak** chase floor.
3. **Fixed ring is mid-tier:** high capture but **23.3% collision rate** and worse `D_ang` than slots—fixed geometry without deformable manifold + explicit slot assignment is insufficient.
4. **Assignment mode degeneracy in open space:** identical slot-tier rows indicate Hungarian, OT, and oracle nearest matching yield the **same effective assignments** when obstacles are absent; keep all three in the suite for **E2+** separation tests.
5. **Paper bar for learned methods:** `dream_mappo_full`, `sce`, and MAPPO variants should be reported **alongside slot tier**, with structure metrics—not only capture vs. pure pursuit.

### Metrics (all methods)

`capture_rate`, `capture_time_s`, `collision_rate`, `timeout_rate`, `out_of_bounds_rate`, terminal-window `C_cov`, `C_col`, `D_ang`, `max_escape_gap`, `F_esc`, `role_stability` / `role_instability` (`summarize()` in benchmark script).

### Reporting checklist — E1 main table

- [x] capture rate
- [x] capture time
- [x] collision / OOB rates
- [ ] max escape gap (often NaN when fully encircled — report with care)
- [x] angular coverage / regularity proxies
- [x] collapse proxy (`C_col`)
- [x] role stability
- [ ] RL / SCE rows in same CSV (**TODO** export)

---

## E2. Obstacle-rich environment

### Purpose

Test whether **obstacle-aware deformable manifold** and assignment remain effective when slot targets are blocked; slot tier from E1.1 should **separate** under collisions with cylindrical obstacles.

### Repo mapping

- Scenario defaults: `configs/env/e2_obstacle_scenario.yaml`
- Env: `configs/env/e2_pyflyt_3v1_obstacles_heuristic.yaml` (heuristics), `e2_pyflyt_3v1_obstacles.yaml` (RL)
- Suite: `configs/benchmark/e2_obstacles_suite.yaml`
- Runner: `python scripts/benchmark_e2_obstacles.py --mode eval`
- Heuristic configs: `e2_obstacles_pyflyt_{pure_pursuit,oracle_slot,hungarian_slot,ot_slot,fixed_ring}.yaml`

### Status

Benchmark runner and heuristic YAMLs **implemented**; aggregate results **TODO** (mirror E1.1 export to `save_result/`).

### E2-specific metrics

Add **`obstacle_termination_rate`** (pursuer–cylinder contact) alongside E1 structure fields.

---

## E3. Narrow-passage environment

- **Purpose:** Avoid one-sided collapse under constrained geometry.
- **Status:** **TODO** — dedicated scenario config not present in `configs/benchmark/`.

## E4. Multi-exit environment

- **Purpose:** Escape-gap suppression at multiple egress directions.
- **Status:** **TODO** — dedicated scenario config not present.

## E5. Ablation study

Required ablations (paper); partial support in code/config:

| Ablation | Repo lever | Status |
|----------|------------|--------|
| Full framework | `dream_mappo_full` / ex1+Dream policy | E1.1 |
| **Slot tier (strong heuristic)** | `oracle_slot` / `hungarian_slot` / `ot_slot` | **E1.1 complete** |
| w/o deformable manifold | `fixed_ring` vs. slot tier | **E1.1: fixed_ring < slots** |
| w/o transport-based role allocation | `oracle_slot` vs. `hungarian_slot` vs. `ot_slot` | E1 tie; **E2 TODO** |
| w/o escape-gap suppression | reduce structure weights / disable encirclement capture gate | **TODO** scripted grid |
| w/o assignment consistency | `assignment_inertia_margin: 0` | supported |
| w/o residual fine-tune | compare `dream_mappo_full` vs. `mappo_bc` path | partial |
| RL backend only | `mappo` | E1.1 |
| MAPPO + structure reward only | `reward_shaped_mappo` | E1.1 |
| Weak chase floor | `pure_pursuit` | **E1.1 complete** |

## E6. Structural metrics and failure analysis

- Implemented in E1.1 benchmark summaries and `pursuit_episode_log_stats.py`.
- **E1.1 finding:** separate **slot success** from **pure-pursuit collapse** and **fixed-ring collision** modes.
- **TODO:** systematic failure-mode taxonomy exported per episode.

## E7. Runtime and deployability

- Slot baselines (`ot_slot`, `hungarian_slot`) are intended as **deployable** references—profile assignment + control loop latency.
- **TODO:** benchmark manifold, assignment, inference times; scalability in \(N\); robustness tests (noise, delay, comm dropout).

---

## Suggested narrative for Results section (E1)

> In obstacle-free 3v1 pursuit (E1.1), deployable slot controllers with deformable-manifold targets achieve **100% capture** and **zero collision-terminated rollouts**, outperforming fixed-ring (**99.2%**, **23.3%** collision) and pure pursuit (**59.4%**, **82%** collision). Angular regularity and coverage metrics confirm that slot methods maintain encirclement geometry whereas pure pursuit collapses to one-sided chase. We therefore treat **oracle / Hungarian / OT slot** as the primary non-learned comparators for all subsequent scenarios; learned and SCE-full-stack methods must meet this bar before claiming structural advantages.
