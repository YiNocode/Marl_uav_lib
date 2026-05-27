# MAPPO narrative audit (snapshot)

Generated during restructuring to **structure-preserving cooperative encirclement (SCE)**.  
No LaTeX sources were found in-repo; paper content lives under `docs/paper/`.

## Files where MAPPO was framed as core method (updated or flagged)

| Location | Prior framing | Action |
|----------|---------------|--------|
| `README.md` | "主线 Dream-MAPPO" | Rewritten around SCE + MAPPO backend |
| `configs/experiment/e1_1_open_space_pyflyt_dream_mappo_full.yaml` | "DREAM-MAPPO full method" | Framework description + legacy key kept |
| `configs/algo/dream_mappo.yaml` | "Dream-MAPPO training" | execution_backend comment |
| `configs/model/dream_mappo_centralized.yaml` | "Dream-MAPPO policy" | SCE policy head comment |
| `marl_uav/modules/heads/dream_mappo_actor_heads.py` | Chinese "Dream-MAPPO" module doc | SCE actor head doc |
| Generated `configs/generated/e1_1_open_space/*` | Old descriptions | **Not bulk-edited** — regenerate with `--overwrite-configs` |

## Files correctly keeping MAPPO as baseline / backend (unchanged role)

- `configs/experiment/e1_1_open_space_pyflyt_mappo.yaml` — vanilla MAPPO comparator
- `configs/experiment/e1_1_open_space_pyflyt_reward_shaped_mappo.yaml` — ablation comparator
- `configs/algo/mappo.yaml`, `MAPPOLearner`, toy scripts `run_mappo_toy_uav.py`
- Benchmark suite methods: `mappo`, `mappo_bc`, `reward_shaped_mappo`
- `tests/test_mappo_*.py` — learner correctness tests

## Internal identifiers intentionally NOT renamed

`dream_mappo`, `DreamMAPPO`, `dream_mappo_full`, `guarded_dream_mappo.py` — breaking change if renamed without registry migration.

## High-risk phrases still in generated / train_code trees

Search periodically: `DREAM-MAPPO full method`, `improve MAPPO`, `proposed algorithm`.  
Run: `rg -i "dream-mappo full|improve mappo|proposed algorithm" --glob "*.{yaml,md,py}"`

## Code ↔ paper gaps (must not over-claim)

1. Role allocation: **permutation + inertia**, not Sinkhorn OT (`pursuit_evasion_3v1_task_ex1.py`).
2. E3/E4 scenarios: not in benchmark suites.
3. E7 runtime: no profiler script yet.
