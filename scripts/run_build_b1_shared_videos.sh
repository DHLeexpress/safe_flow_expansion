#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 RUN_DIR LATEST_R19_CKPT OUTPUT_DIR CUDA_PHYSICAL_INDEX [REPLAY_PT]" >&2
  exit 2
fi

RUN_DIR=$1
LATEST_R19_CKPT=$2
OUTPUT_DIR=$3
CUDA_PHYSICAL_INDEX=$4
REPLAY_PT=${5:-}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "fresh output directory required: $OUTPUT_DIR" >&2
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$CUDA_PHYSICAL_INDEX

EXTRA=()
if [[ -n "$REPLAY_PT" ]]; then
  EXTRA+=(--reuse-expansion-replay "$REPLAY_PT")
fi

exec /home/dohyun/miniforge3/envs/cfm_mppi/bin/python \
  scripts/build_b1_shared_videos.py \
  --run-dir "$RUN_DIR" \
  --probe "$RUN_DIR/probe.jsonl" \
  --latest-r19-ckpt "$LATEST_R19_CKPT" \
  --outdir "$OUTPUT_DIR" \
  --device cuda \
  --fps 7 \
  --verifier-workers 16 \
  --expert-search-workers 16 \
  --expert-search-size 2000 \
  "${EXTRA[@]}"
