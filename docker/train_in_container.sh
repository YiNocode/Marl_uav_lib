#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
shift || true

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-$TRAIN_CONFIG}"
GRID_CONFIG="${GRID_CONFIG:-configs/search/ex1_reward_grid.yaml}"
EVAL_SEED="${EVAL_SEED:-101}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_NUM_SEEDS="${EVAL_NUM_SEEDS:-1}"

cd /workspace

echo "[container] mode=${MODE}"
echo "[container] pwd=$(pwd)"

case "${MODE}" in
  train)
    echo "[container] train_config=${TRAIN_CONFIG}"
    exec python scripts/train.py --train-config "${TRAIN_CONFIG}" "$@"
    ;;
  eval)
    echo "[container] eval_config=${EVAL_CONFIG}"
    exec python scripts/eval.py \
      --config "${EVAL_CONFIG}" \
      --seed "${EVAL_SEED}" \
      --episodes "${EVAL_EPISODES}" \
      --num-seeds "${EVAL_NUM_SEEDS}" \
      "$@"
    ;;
  grid)
    echo "[container] grid_config=${GRID_CONFIG}"
    exec python scripts/grid_search_ex1_rewards.py \
      --search-config "${GRID_CONFIG}" \
      --overwrite \
      --launch \
      "$@"
    ;;
  summarize-grid)
    echo "[container] grid_config=${GRID_CONFIG}"
    exec python scripts/summarize_ex1_reward_grid.py \
      --search-config "${GRID_CONFIG}" \
      "$@"
    ;;
  bash)
    exec /bin/bash "$@"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    echo "Supported modes: train | eval | grid | summarize-grid | bash" >&2
    exit 2
    ;;
esac
