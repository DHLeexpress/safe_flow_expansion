#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {data|train-eval} RUN_ROOT PHYSICAL_GPU FROZEN_SHA" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
PHASE=$1
RUN_ROOT=$2
PHYSICAL_GPU=$3
FROZEN_SHA=$4
[[ "$PHASE" == "data" || "$PHASE" == "train-eval" ]] || usage
[[ "$PHYSICAL_GPU" =~ ^[0-9]+$ ]] || usage
[[ "$FROZEN_SHA" =~ ^[0-9a-f]{40}$ ]] || usage

REPO_ROOT=$(git rev-parse --show-toplevel)
CURRENT_SHA=$(git rev-parse HEAD)
[[ "$CURRENT_SHA" == "$FROZEN_SHA" ]] || {
  echo "HEAD $CURRENT_SHA != frozen SHA $FROZEN_SHA" >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "frozen worktree is not clean" >&2
  exit 1
}

PYTHON=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
[[ -x "$PYTHON" ]] || {
  echo "python runtime is not executable: $PYTHON" >&2
  exit 1
}
command -v nvidia-smi >/dev/null
if nvidia-smi -i "$PHYSICAL_GPU" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+$'; then
  echo "physical GPU $PHYSICAL_GPU already has a compute process" >&2
  exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONPATH="$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_challenging:$REPO_ROOT/overnight_run_07_06${PYTHONPATH:+:$PYTHONPATH}"

MODULE=afe_restart.stage2_low7_randomized
ENDPOINTS="$RUN_ROOT/endpoints.json"
COMBINED="$RUN_ROOT/combined"
GAMMAS=(0.1 0.2 0.3 0.4 0.5 0.7 1.0)

if [[ "$PHASE" == "data" ]]; then
  [[ ! -e "$RUN_ROOT" ]] || {
    echo "data output root already exists: $RUN_ROOT" >&2
    exit 1
  }
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/shards"
  "$PYTHON" -m "$MODULE" endpoints --pairs 100 --output "$ENDPOINTS" \
    >"$RUN_ROOT/logs/endpoints.log" 2>&1

  pids=()
  labels=()
  for gamma in "${GAMMAS[@]}"; do
    tag=${gamma/./p}
    out="$RUN_ROOT/shards/gamma_$tag"
    "$PYTHON" -m "$MODULE" collect \
      --endpoint-manifest "$ENDPOINTS" \
      --gamma "$gamma" \
      --outdir "$out" \
      --device cuda:0 \
      >"$RUN_ROOT/logs/collect_gamma_$tag.log" 2>&1 &
    pids+=("$!")
    labels+=("$gamma")
  done

  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "gamma ${labels[$index]} collection failed" >&2
      failed=1
    fi
  done
  [[ $failed -eq 0 ]] || exit 1

  manifests=()
  for gamma in "${GAMMAS[@]}"; do
    tag=${gamma/./p}
    manifests+=("$RUN_ROOT/shards/gamma_$tag/manifest.json")
  done
  "$PYTHON" -m "$MODULE" combine \
    --shard-manifests "${manifests[@]}" \
    --outdir "$COMBINED" \
    >"$RUN_ROOT/logs/combine.log" 2>&1
  "$PYTHON" -m "$MODULE" render \
    --manifest "$COMBINED/manifest.json" \
    >"$RUN_ROOT/logs/render.log" 2>&1
  "$PYTHON" -m "$MODULE" video \
    --manifest "$COMBINED/manifest.json" \
    --output "$COMBINED/viz/low7_exact_polytope_replay.mp4" \
    >"$RUN_ROOT/logs/video.log" 2>&1
  echo "data complete: $COMBINED/manifest.json"
  exit 0
fi

[[ -f "$COMBINED/manifest.json" ]] || {
  echo "missing combined data manifest: $COMBINED/manifest.json" >&2
  exit 1
}
PRETRAIN="$RUN_ROOT/pretrain"
EVALUATION="$RUN_ROOT/pretrained_eval_m20"
[[ ! -e "$PRETRAIN" && ! -e "$EVALUATION" ]] || {
  echo "refusing to reuse pretrain/evaluation output" >&2
  exit 1
}
mkdir -p "$RUN_ROOT/logs"
"$PYTHON" -m afe_restart.stage3_low7_pretrain \
  --manifest "$COMBINED/manifest.json" \
  --outdir "$PRETRAIN" \
  --device cuda:0 \
  --epochs 500 \
  --batch-size 512 \
  --validation-pairs 16 \
  >"$RUN_ROOT/logs/pretrain.log" 2>&1

CHECKPOINT="$PRETRAIN/data/checkpoint_candidate.pt"
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
"$PYTHON" -m afe_restart.evaluate_low7_pretrained \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --outdir "$EVALUATION" \
  --M 20 \
  --nfe 12 \
  --device cuda:0 \
  --verifier-workers 16 \
  >"$RUN_ROOT/logs/pretrained_eval_m20.log" 2>&1
echo "train/eval complete: $EVALUATION/EVALUATION_COMPLETE.json"
