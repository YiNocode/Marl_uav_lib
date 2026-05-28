# Structure-Preserving Cooperative Encirclement (SCE)

**Working title (paper):** *Structure-preserving cooperative encirclement through deformable encirclement manifold generation and transport-based role allocation.*

This repository implements a **geometry–role–structure** pursuit–evasion stack. It does **not** propose a new MAPPO optimizer. MAPPO (and related actor–critic learners) are **execution backends** only.

## Narrative (method pipeline)

```text
Evader state + environment context
  → deformable closed-curve encirclement manifold M_t
  → sampled target slots on M_t
  → role allocation (UAV ↔ slot); target: entropic optimal transport; current code: see below
  → role-conditioned references / topology-aware structural guidance
  → RL-based closed-loop execution policy (MAPPO-style backend in this repo)
```

## Code naming vs. paper naming

| Paper / framework | Repo identifier (keep for compatibility) |
|-------------------|------------------------------------------|
| SCE framework | `framework: structure_preserving_encirclement` (config comments) |
| Full framework instance on E1.1 | `dream_mappo_full` method key, `configs/experiment/e1_1_open_space_pyflyt_dream_mappo_full.yaml` |
| Deformable manifold | `manifold_targets_from_pursuit_state`, task `_reference_manifold_*`, Dream actor heads |
| Role allocation | `role_assignment_mode`: `entropic_ot` (Sinkhorn + hard match), `nearest` (permutation + inertia), `fixed` |
| Structural costs | `structure_*` rewards & metrics `C_cov`, `C_col`, `D_ang`, `phi_max` (escape gap) |
| Execution backend | `algo: dream_mappo` or `algo: mappo` → `MAPPOLearner` / centralized critic PPO |
| Residual skill-preserving fine-tune | `mappo_bc`, `mappo_bc_finetune.yaml`, `marl_uav/utils/mappo_finetune.py` |

## Implementation status (honest)

- **Entropic OT / Sinkhorn:** implemented for `entropic_ot` mode; E1.1 method `sce` uses proportional execution backend. MAPPO backend remains on `dream_mappo_full`.
- **Formal closure guarantees:** not claimed in paper drafts.
- **Runtime scaling claims:** require profiling (TODO).

## Paper drafts

- `docs/paper/01_abstract_intro_contributions.md`
- `docs/paper/02_method.md`
- `docs/paper/03_experiments.md`
- `docs/paper/TODO.md`

## E1.1 benchmark (open-space)

Suite: `configs/benchmark/e1_1_open_space_suite.yaml` — compares **methods** (framework instance, MAPPO baselines, heuristics), not “MAPPO variants” as the scientific object.
