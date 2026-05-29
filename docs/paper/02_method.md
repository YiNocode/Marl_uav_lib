# 3. Method (outline)

> Formulas marked **TODO** need derivation from code or theory notes. E1.1 baseline hierarchy is empirically anchored in `save_result/2.5baseline/e1_1_open_space_summary_by_method.csv`.

## 3.1 Problem Formulation

- **Setting:** \(N\) pursuers, one evader, bounded workspace, optional obstacles (ex2).
- **States:** pursuer poses/velocities, evader pose/velocity, environment context (obstacle blocks in ex2).
- **Observations / actions:** per-pursuer observations (ex1: structure-aware 19-d features + role block); continuous velocity setpoints.
- **Execution backend:** learned policy \(\pi_\theta\) — **not** the central algorithmic contribution. Standard centralized-critic PPO / MAPPO training path (`MAPPOLearner`).

## 3.2 Deformable Encirclement Manifold

- **Manifold** \(M_t\): closed curve in the horizontal plane around evader position, with radius field \(\rho(\theta)\) and phase \(\psi\) (policy head / task contraction).
- **Conditioning:** evader-relative geometry; ex2 adds obstacle-aware radial bumps (`_obstacle_aware_radius`, Fourier / bump parameters in `configs/env/e2_obstacle_scenario.yaml`).
- **Tractability:** fixed low-dimensional parameters (e.g., \(\rho\), \(\psi\), contraction rate).
- **Code:** `PursuitEvasion3v1TaskEx1._reference_manifold_targets`, `_reference_manifold_curve`; `manifold_targets_from_pursuit_state` in policy module.

## 3.3 Slot Sampling and Transport-Based Role Allocation

- **Slots:** three (for 3v1) target points sampled on \(M_t\) at fixed angular spacing with deformable radius.
- **Cost matrix** \(C_{ij}\): pursuer \(i\) to slot \(j\) (Euclidean distance in implementation).
- **Entropic OT:** `role_assignment_mode: entropic_ot` (SCE); Sinkhorn plan in `marl_uav/framework/role_allocation.py`; hyperparameters `ot_epsilon`, `ot_epsilon_scale`, `ot_sinkhorn_iterations`. **Deployable baseline:** `ot_slot` (`make_ot_slot_get_actions_fn`).
- **Hungarian (min-cost):** exact bipartite matching on \(C\). **Deployable baseline:** `hungarian_slot`.
- **Oracle / nearest permutation:** full-method slot targets with `role_assignment_mode: nearest` and assignment inertia. **Deployable baseline:** `oracle_slot` (upper reference for “given correct manifold, greedy assignment + P control”).

### 3.3.1 Deployable slot baseline stack (E1.1 strongest tier)

All slot baselines share the same **proportional execution** head (`xy_gain`, `z_gain`, `yaw_gain` in experiment YAML) and **encirclement capture** task settings. They differ only in **how pursuers are matched to manifold slots**:

| Baseline | Assignment | Config |
|----------|------------|--------|
| `oracle_slot` | Nearest-permutation (oracle reference) | `e1_1_open_space_pyflyt_oracle_slot.yaml` |
| `hungarian_slot` | Min-cost Hungarian | `e1_1_open_space_pyflyt_hungarian_slot.yaml` |
| `ot_slot` | Entropic OT + hard matching | `e1_1_open_space_pyflyt_ot_slot.yaml` |

**E1.1 empirical note:** On open space, the three slot variants **produce identical aggregate metrics** (1000 episodes, 5 seeds)—assignment modes collapse to the same effective matching when geometry is unconstrained. They should still be reported separately for reproducibility; **E2+** is where assignment and obstacle-aware manifold choices are expected to diverge.

### 3.3.2 Weaker geometric comparators

| Baseline | Mechanism | E1.1 role |
|----------|-----------|-----------|
| `pure_pursuit` | All pursuers chase evader centroid | **Weak** chase-only floor (59.4% capture) |
| `fixed_ring` | Fixed-radius ring slots, no deformable manifold | **Medium** (99.2% capture, 23.3% collision) |

## 3.4 Topology-Aware Structural Guidance

Structural terms (metrics in `compute_pursuit_structure_metrics_3v1`):

| Term | Metric / proxy | Intent | E1.1 slot tier (mean, last-30 window) |
|------|----------------|--------|----------------------------------------|
| Manifold / coverage | `C_cov` | Encourage full angular coverage | ≈0.21 |
| Collapse / one-sided | `C_col` | Penalize team centroid bias | ≈0.93 |
| Angular regularity | `D_ang` | Penalize uneven spacing | ≈0.17 |
| Escape gap | `phi_max` → `max_escape_gap` | Suppress largest opening | (often NaN when encircled) |
| Assignment consistency | `role_stability` | Penalize slot switching | ≈0.99 |

Pure pursuit exhibits **near-zero** `C_cov` and `D_ang` (≈0.03 / 0.003)—symptomatic of one-sided collapse—not “good regularity.”

In the learned execution backend, analogous terms appear as **structure rewards** (`structure_reward_scale`, radial compress, encirclement capture gate). Wording for paper: *“In our MAPPO backend instantiation, these structural costs are incorporated into the training objective.”*

## 3.5 RL-Based Closed-Loop Execution Backend

- **Optimizer:** unchanged MAPPO / PPO hyperparameters (`configs/algo/dream_mappo.yaml`, `configs/algo/mappo.yaml`).
- **Policy:** `DreamMAPPO` / centralized critic; actor conditions on role features + manifold residuals (`dream_mappo_actor_heads.py`).
- **Inputs:** local obs, role-conditioned reference, structure deltas.
- **Outputs:** continuous action (vx, vy, yaw rate, vz).
- **Statement for paper:** *We do not modify the underlying MAPPO optimizer; it serves as a standard execution-policy training backend.*

Legacy config name `dream_mappo` = **full framework + MAPPO backend**, not a new RL algorithm. **Success criterion in E1:** learned backend should **meet or exceed slot-tier** capture and structure, not merely beat pure pursuit.

## 3.6 Residual Skill-Preserving Fine-Tuning

- Pretrained capture policy via oracle-slot BC (`mappo_bc` experiments)—natural teacher given slot-tier performance.
- Fine-tune with structural rewards + KL / capture guards (`mappo_bc_finetune.yaml`, `mappo_finetune.py`).
- Motivation: preserve interception while learning encirclement structure **at least as strong as deployable slot control**.

## 3.7 Computational Properties

| Component | Status |
|-----------|--------|
| Manifold generation (per step) | **TODO:** profile ms/step |
| Role assignment (Hungarian / OT / nearest) | Slot baselines deployable; **TODO:** compare ms/step |
| Policy inference | **TODO:** profile on target hardware |
| End-to-end control frequency | Available via `control_hz` in env config; loop timing **TODO** |
