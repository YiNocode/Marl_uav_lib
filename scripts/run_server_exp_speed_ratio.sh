#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/launch_pursuit_server_runs.py \
  --experiment speed_ratio \
  --seeds 101 102 103 \
  --speed-ratios 1:1 1:2 1:3 \
  --ratio-mode evader:pursuer \
  --base-evader-speed 0.06 \
  --overwrite
