#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PRETRAIN_DELIVERY OUTPUT_ROOT FROZEN_SHA" >&2
  exit 2
fi

PRETRAIN_DELIVERY=$1
OUTPUT_ROOT=$2
FROZEN_SHA=$3
PYTHON=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
VERIFIER_WORKERS=${VERIFIER_WORKERS:-48}
GPU1_UUID=${GPU1_UUID:-GPU-50fb5dae-52a8-5843-bc81-b869586dccde}
GPU3_UUID=${GPU3_UUID:-GPU-b5993142-760d-a6fe-9430-3d0e65203b6d}

ROOT=$(git rev-parse --show-toplevel)
[[ "$FROZEN_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "FROZEN_SHA must be 40 lowercase hex digits" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$FROZEN_SHA" ]] || {
  echo "HEAD does not equal frozen SHA $FROZEN_SHA" >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "frozen worktree is not clean" >&2
  exit 1
}
[[ -x "$PYTHON" ]] || { echo "python runtime is not executable" >&2; exit 1; }
[[ -f "$PRETRAIN_DELIVERY" ]] || { echo "pretrain delivery is missing" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "output root already exists" >&2; exit 1; }

exec "$PYTHON" \
  "$ROOT/overnight_run_07_06/rev_expansion/codex_overnight/analysis/low7_b1_balanced_sweep_driver.py" \
  --out "$OUTPUT_ROOT" \
  --pretrain-delivery "$PRETRAIN_DELIVERY" \
  --gpu1-uuid "$GPU1_UUID" \
  --gpu3-uuid "$GPU3_UUID" \
  --verifier-workers "$VERIFIER_WORKERS" \
  --python "$PYTHON"
