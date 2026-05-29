# Abstract, Introduction, and Contributions (draft)

> **Status:** Markdown draft for LaTeX migration. E1.1 heuristic baseline numbers sourced from `save_result/2.5baseline/e1_1_open_space_summary_by_method.csv` (5 seeds × 200 episodes). RL / SCE rows pending in the same export pipeline.

## Abstract

Multi-UAV cooperative encirclement in pursuit–evasion must simultaneously maintain interception competence and a coherent enclosing geometry under dynamics, sensing limits, and environmental constraints. We propose a **structure-preserving cooperative encirclement framework** that decouples geometric–topological decision making from low-level control execution. Given the evader state and local environmental context, the framework generates a **deformable closed-curve encirclement manifold**, samples target slots on the manifold, and assigns pursuers to slots through **transport-based role allocation**. Topology-aware structural costs guide manifold tracking, escape-gap suppression, and assignment consistency. A **residual skill-preserving fine-tuning** stage injects structural guidance into a pretrained capture policy without erasing baseline interception behavior. The framework is **policy-backend agnostic**; in this implementation we instantiate a standard actor–critic / MAPPO-style learned closed-loop execution policy as the backend, without modifying the underlying RL optimizer.

On the **E1.1 open-space benchmark** (PyFlyt, 3v1, obstacle-free), **deployable slot controllers**—`oracle_slot`, `hungarian_slot`, and `ot_slot`—form the **strongest geometric baseline tier**: **100% capture**, **0% collision termination**, mean capture time **≈10.2 s**, and favorable terminal-window structure (`D_ang` ≈0.17 vs. **0.40** for `fixed_ring`, **≈0.003** for `pure_pursuit`). Classical **pure pursuit** remains a weak comparator (**59.4%** capture, **82%** collision rate). The full framework must therefore be judged against **slot-aware geometry**, not against pure chase alone. Obstacle-rich (E2) and constrained-geometry scenarios (E3–E4) extend this comparison. <!-- TODO: insert dream_mappo / mappo / sce rows when exported to the same CSV -->

## Introduction

Classical **pure-pursuit** heuristics and undifferentiated chasing collapse to one-sided pursuit: large escape gaps, poor angular coverage (`C_cov` ≈0.03 in E1.1), and frequent inter-pursuer contact. **Fixed-ring** encirclement improves capture (**99.2%**) but still incurs substantial collision-driven failures (**23.3%**) and weaker angular regularity than slot methods. **Deployable slot baselines** close the loop on **deformable-manifold slot targets** with proportional execution; they differ only in **role assignment** (oracle nearest-permutation vs. Hungarian vs. entropic OT). In open space all three slot variants **tie at the top** of the heuristic ladder, establishing a **strong, reproducible geometric ceiling** for E1.

End-to-end multi-agent reinforcement learning (MARL) can improve capture rates but often collapses to one-sided chasing unless structural references and role features are explicit. Prior work that centers on **algorithmic novelty in MAPPO or generic reward shaping** conflates execution optimization with the structural question of *how* a team should enclose a moving target.

We instead treat cooperative encirclement as a **structure-preserving** problem: specify a desirable enclosing geometry, allocate roles online, and let a learned policy execute role-conditioned references under disturbances. Reinforcement learning enters only as a **closed-loop execution backend** that maps observations and structural references to continuous control commands. **E1.1 shows that a well-instrumented non-learning slot stack already solves open-space encirclement reliably**; the framework’s value proposition is to **match or exceed that tier** while adding deformable / obstacle-aware geometry (E2+), transport-based allocation at scale, and learned robustness under sensing and dynamics noise.

### Contributions

1. **Deformable closed-curve encirclement manifold.** Fixed-dimensional parameterization of a closed curve \(M_t\) around the evader, conditioned on evader state and local environmental context (e.g., obstacle-aware radial deformation in ex2).
2. **Transport-based role allocation.** Assignment of UAVs to manifold slots via a cost matrix and **entropic optimal transport** (SCE / `ot_slot`) or **min-cost Hungarian matching** (`hungarian_slot`); **oracle slot** exposes full-method targets with nearest-permutation assignment for analysis.
3. **Deployable slot baseline family (E1.1).** Documented, benchmarked controllers that isolate **manifold + assignment + proportional execution** without RL—currently the **strongest E1.1 comparators** (`scripts/benchmark_e1_1_open_space.py`, configs under `e1_1_open_space_pyflyt_*_slot.yaml`).
4. **Topology-aware structural costs.** Structural objectives for manifold matching, escape-gap suppression, and assignment consistency; E1.1 terminal-window metrics (`C_cov`, `C_col`, `D_ang`, `role_stability`) separate slot tier from pure pursuit and fixed ring.
5. **Residual skill-preserving fine-tuning.** Adapt a pretrained capture policy with structural guidance while monitoring capture competence (KL / capture-rate guards in `mappo_bc` configs).
6. **Structure-aware evaluation.** Joint reporting of capture outcomes and geometry/role metrics (`scripts/benchmark_e1_1_open_space.py`, E2 mirror `scripts/benchmark_e2_obstacles.py`, `scripts/pursuit_episode_log_stats.py`).

### What we do **not** claim

- A new MAPPO or MARL algorithm as the primary contribution.
- That pure pursuit or fixed ring alone constitute sufficient strong baselines in E1.1 (slot methods dominate).
- Guaranteed topological closure without proof.
- Near-quadratic scaling without runtime evidence.
- Reduced reward-weight sensitivity without ablation sweeps (partial: `configs/search/ex1_reward_grid.yaml`).

### Related positioning (one paragraph)

MAPPO, IPPO, and reward-shaped MAPPO baselines are **necessary comparators** for the execution layer. **Oracle / Hungarian / OT slot** controllers are **equally necessary**: they encode the same geometric intent as the framework’s high-level stack without learned execution. The proposed system should be read as **geometry + roles + structure + (optional) learned execution**, evaluated against **slot-tier** performance, not as “MAPPO vs. pure pursuit.”
