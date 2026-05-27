# 3. Method (outline)

> Formulas marked **TODO** need derivation from code or theory notes. Do not copy placeholder math into the camera-ready paper without verification.

## 3.1 Problem Formulation

- **Setting:** \(N\) pursuers, one evader, bounded workspace, optional obstacles (ex2).
- **States:** pursuer poses/velocities, evader pose/velocity, environment context (obstacle blocks in ex2).
- **Observations / actions:** per-pursuer observations (ex1: structure-aware 19-d features + role block); continuous velocity setpoints.
- **Execution backend:** learned policy \(\pi_\theta\) — **not** the central algorithmic contribution. Standard centralized-critic PPO / MAPPO training path (`MAPPOLearner`).

## 3.2 Deformable Encirclement Manifold

- **Manifold** \(M_t\): closed curve in the horizontal plane around evader position, with radius field \(\rho(\theta)\) and phase \(\psi\) (policy head / task contraction).
- **Conditioning:** evader-relative geometry; ex2 adds obstacle-aware radial bumps (`_obstacle_aware_radius`, Fourier / bump parameters in task config).
- **Tractability:** fixed low-dimensional parameters (e.g., \(\rho\), \(\psi\), contraction rate) — **avoid** claiming arbitrary topology expressiveness.
- **Code:** `PursuitEvasion3v1TaskEx1._reference_manifold_targets`, `_reference_manifold_curve`; `manifold_targets_from_pursuit_state` in policy module.

## 3.3 Slot Sampling and Transport-Based Role Allocation

- **Slots:** three (for 3v1) target points sampled on \(M_t\) at fixed angular spacing with deformable radius.
- **Cost matrix** \(C_{ij}\): pursuer \(i\) to slot \(j\) (Euclidean distance in implementation).
- **Target (paper):** entropic OT with temperature \(\varepsilon\), Sinkhorn iterations \(K\) — soft assignment **TODO: implement / document \(K,\varepsilon\)**.
- **Current implementation:** exact permutation search for \(N{=}3\) with **assignment inertia** (`assignment_inertia_margin`) — document as engineering baseline, not OT, until migrated.

## 3.4 Topology-Aware Structural Guidance

Structural terms (metrics in `compute_pursuit_structure_metrics_3v1`):

| Term | Metric / proxy | Intent |
|------|----------------|--------|
| Manifold / coverage | `C_cov` from angular gaps \(\phi_k\) | Encourage full angular coverage |
| Collapse / one-sided | `C_col` = \(\|\frac{1}{N}\sum \hat r_i\|\) | Penalize team centroid bias |
| Angular regularity | `D_ang` | Penalize uneven spacing |
| Escape gap | `phi_max` → `max_escape_gap` | Suppress largest opening |
| Assignment consistency | `role_stability` in episode logs | Penalize slot switching |

In the learned execution backend, analogous terms appear as **structure rewards** (`structure_reward_scale`, radial compress, encirclement capture gate). Wording for paper: *“In our MAPPO backend instantiation, these structural costs are incorporated into the training objective.”*

Optional: inter-UAV separation (`min_pursuer_sep`, collision penalty).

## 3.5 RL-Based Closed-Loop Execution Backend

- **Optimizer:** unchanged MAPPO / PPO hyperparameters (`configs/algo/dream_mappo.yaml`, `configs/algo/mappo.yaml`).
- **Policy:** `DreamMAPPO` / centralized critic; actor conditions on role features + manifold residuals (`dream_mappo_actor_heads.py`).
- **Inputs:** local obs, role-conditioned reference, structure deltas.
- **Outputs:** continuous action (vx, vy, yaw rate, vz).
- **Statement for paper:** *We do not modify the underlying MAPPO optimizer; it serves as a standard execution-policy training backend.*

Legacy config name `dream_mappo` = **full framework + MAPPO backend**, not a new RL algorithm.

## 3.6 Residual Skill-Preserving Fine-Tuning

- Pretrained capture policy via oracle-slot BC (`mappo_bc` experiments).
- Fine-tune with structural rewards + KL / capture guards (`mappo_bc_finetune.yaml`, `mappo_finetune.py`).
- Motivation: preserve interception while learning encirclement structure.

## 3.7 Computational Properties

| Component | Status |
|-----------|--------|
| Manifold generation (per step) | **TODO:** profile ms/step |
| Role assignment | **TODO:** compare permutation vs. future Sinkhorn |
| Policy inference | **TODO:** profile on target hardware |
| End-to-end control frequency | Available via `control_hz` in env config; loop timing **TODO** |
