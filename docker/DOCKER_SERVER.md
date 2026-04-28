# Docker Server Training

## 1. Build

GPU server:

```bash
docker build -t marl-uav:gpu .
```

CPU server:

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.3.1-cpu \
  -t marl-uav:cpu .
```

## 2. Train a single experiment

```bash
docker run --rm --gpus all \
  -e TRAIN_CONFIG=configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/save_result:/workspace/save_result \
  marl-uav:gpu train
```

## 3. Run reward-grid search

```bash
docker run --rm --gpus all \
  -e GRID_CONFIG=configs/search/ex1_reward_grid.yaml \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/save_result:/workspace/save_result \
  marl-uav:gpu grid
```

This will:

1. generate grid configs
2. train each run
3. run eval after training
4. write summaries under `configs/generated/reward_grid/`

## 4. Recompute reward-grid summaries

```bash
docker run --rm \
  -e GRID_CONFIG=configs/search/ex1_reward_grid.yaml \
  -v $(pwd)/results:/workspace/results \
  marl-uav:gpu summarize-grid
```

## 5. Run evaluation only

```bash
docker run --rm --gpus all \
  -e EVAL_CONFIG=configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml \
  -e EVAL_SEED=101 \
  -e EVAL_EPISODES=20 \
  -v $(pwd)/results:/workspace/results \
  marl-uav:gpu eval
```

## Notes

- Default image is GPU-oriented. Override `BASE_IMAGE` during build for CPU-only servers.
- `MPLBACKEND=Agg` is set in the image for headless plotting.
- Keep `backend.render: false` in server configs.
- Persist `results/`, `checkpoints/`, and `save_result/` via bind mounts.
