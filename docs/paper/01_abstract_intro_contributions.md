# Abstract, Introduction, and Contributions (draft)

> **Status:** Markdown draft for LaTeX migration. No fabricated results. Align claims with `docs/paper/TODO.md` and code.

## Abstract

Multi-UAV cooperative encirclement in pursuit–evasion must simultaneously maintain interception competence and a coherent enclosing geometry under dynamics, sensing limits, and environmental constraints. We propose a **structure-preserving cooperative encirclement framework** that decouples geometric–topological decision making from low-level control execution. Given the evader state and local environmental context, the framework generates a **deformable closed-curve encirclement manifold**, samples target slots on the manifold, and assigns pursuers to slots through **transport-based role allocation**. Topology-aware structural costs guide manifold tracking, escape-gap suppression, and assignment consistency. A **residual skill-preserving fine-tuning** stage injects structural guidance into a pretrained capture policy without erasing baseline interception behavior. The framework is **policy-backend agnostic**; in this implementation we instantiate a standard actor–critic / MAPPO-style learned closed-loop execution policy as the backend, without modifying the underlying RL optimizer. We evaluate task-level and structure-level metrics across open-space, obstacle-rich, narrow-passage, and multi-exit scenarios. <!-- TODO: insert quantitative summary once all E1–E7 runs are complete -->

## Introduction

Classical geometric controllers and pure-pursuit heuristics struggle to maintain uniform coverage when geometry becomes constrained. End-to-end multi-agent reinforcement learning (MARL) can improve capture rates but often collapses to one-sided chasing, leaving large escape gaps and unstable role assignments. Prior work that centers on **algorithmic novelty in MAPPO or generic reward shaping** conflates execution optimization with the structural question of *how* a team should enclose a moving target.

We instead treat cooperative encirclement as a **structure-preserving** problem: specify a desirable enclosing geometry, allocate roles online, and let a learned policy execute role-conditioned references under disturbances. Reinforcement learning enters only as a **closed-loop execution backend** that maps observations and structural references to continuous control commands.

### Contributions

1. **Deformable closed-curve encirclement manifold.** We introduce a fixed-dimensional parameterization of a closed curve \(M_t\) around the evader, conditioned on evader state and local environmental context (e.g., obstacle-aware radial deformation in ex2), enabling online tractable manifold generation.
2. **Transport-based role allocation.** We formulate assignment of UAVs to manifold slots via a cost matrix and **fixed-budget entropic optimal transport** (target formulation). <!-- TODO: cite Sinkhorn iteration count once implemented; current code uses nearest-permutation assignment with inertia — see 02_method.md §3.3 -->
3. **Topology-aware structural costs.** We design structural objectives for manifold matching, escape-gap suppression, and assignment consistency (plus separation / collision terms where enabled), reducing one-sided collapse and open-gap failures.
4. **Residual skill-preserving fine-tuning.** We adapt a pretrained capture policy with structural guidance while monitoring capture competence (KL / capture-rate guards in `mappo_bc` configs).
5. **Structure-aware evaluation.** We report capture outcomes jointly with coverage, collapse, escape gap, role stability, and endgame-window structural scores (`scripts/benchmark_e1_1_open_space.py`, `scripts/pursuit_episode_log_stats.py`).

### What we do **not** claim

- A new MAPPO or MARL algorithm as the primary contribution.
- Guaranteed topological closure without proof.
- Near-quadratic scaling without runtime evidence.
- Reduced reward-weight sensitivity without ablation sweeps (partial: `configs/search/ex1_reward_grid.yaml`).

### Related positioning (one paragraph)

MAPPO, IPPO, and reward-shaped MAPPO baselines in this repository are **necessary comparators** for the execution layer. The proposed framework should be read as **geometry + roles + structure + (optional) learned execution**, not as “MAPPO + extra rewards.”
