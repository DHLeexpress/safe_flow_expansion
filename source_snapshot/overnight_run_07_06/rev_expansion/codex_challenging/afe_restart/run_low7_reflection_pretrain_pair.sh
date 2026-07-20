#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 DATA_MANIFEST OUTPUT_ROOT FROZEN_SHA" >&2
  exit 2
fi

DATA_MANIFEST=$1
OUTPUT_ROOT=$2
FROZEN_SHA=$3
GPU_A=${GPU_A:-1}
GPU_B=${GPU_B:-3}
SEED_A=${SEED_A:-20260717}
SEED_B=${SEED_B:-20260718}
EQ_WEIGHT_A=${EQ_WEIGHT_A:-0}
EQ_WEIGHT_B=${EQ_WEIGHT_B:-0}
GROUP_AVERAGE_A=${GROUP_AVERAGE_A:-0}
GROUP_AVERAGE_B=${GROUP_AVERAGE_B:-0}
M_SELECT=${M_SELECT:-50}
M_CONFIRM=${M_CONFIRM:-100}

REPO_ROOT=$(git rev-parse --show-toplevel)
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
PYTHON=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
[[ -x "$PYTHON" ]] || { echo "python runtime is not executable: $PYTHON" >&2; exit 1; }
[[ -f "$DATA_MANIFEST" ]] || { echo "missing data manifest: $DATA_MANIFEST" >&2; exit 1; }
for gpu in "$GPU_A" "$GPU_B"; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+$'; then
    echo "physical GPU $gpu already has a compute process" >&2
    exit 1
  fi
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTHONPATH="$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_challenging:$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_overnight:$REPO_ROOT/overnight_run_07_06${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "refusing existing OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

run_candidate() {
  local gpu=$1
  local seed=$2
  local equivariance_weight=$3
  local group_average=$4
  local eq_tag=${equivariance_weight//./p}
  local name="seed_${seed}_eq_${eq_tag}_ga_${group_average}"
  local root="$OUTPUT_ROOT/$name"
  mkdir -p "$root"
  local group_args=()
  if [[ "$group_average" == 1 ]]; then
    group_args+=(--reflection-group-average)
  elif [[ "$group_average" != 0 ]]; then
    echo "GROUP_AVERAGE must be 0 or 1" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m afe_restart.stage3_low7_pretrain \
    --manifest "$DATA_MANIFEST" \
    --outdir "$root/pretrain" \
    --device cuda:0 \
    --epochs 500 \
    --batch-size 512 \
    --validation-batch-size 1024 \
    --learning-rate 3e-4 \
    --seed "$seed" \
    --split-seed 31711 \
    --reflection-paired-pretraining \
    --equivariance-weight "$equivariance_weight" \
    "${group_args[@]}" \
    >"$root/pretrain.log" 2>&1
  local checkpoint="$root/pretrain/data/checkpoint_candidate.pt"
  local checksum
  checksum=$(sha256sum "$checkpoint" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_overnight/analysis/low7_balanced_r0_qualification.py" \
    --checkpoint "$checkpoint" \
    --expected-checkpoint-sha256 "$checksum" \
    --outdir "$root/qualification_select" \
    --device cuda:0 \
    --M "$M_SELECT" \
    --seed-bank low7-balanced-r0-selection-v1 \
    --reflection-antithetic \
    --minimum-successes 5 \
    --report-only \
    >"$root/qualification_select.log" 2>&1
}

run_candidate "$GPU_A" "$SEED_A" "$EQ_WEIGHT_A" "$GROUP_AVERAGE_A" &
PID_A=$!
run_candidate "$GPU_B" "$SEED_B" "$EQ_WEIGHT_B" "$GROUP_AVERAGE_B" &
PID_B=$!
wait "$PID_A"
wait "$PID_B"

"$PYTHON" "$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_overnight/analysis/select_low7_balanced_r0.py" \
  --root "$OUTPUT_ROOT" >"$OUTPUT_ROOT/selection.log" 2>&1

readarray -t SELECTED < <("$PYTHON" - "$OUTPUT_ROOT/selection.json" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1]))["selected"]
print(record["checkpoint"])
print(record["checkpoint_sha256"])
PY
)

CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON" \
  "$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_overnight/analysis/low7_balanced_r0_qualification.py" \
  --checkpoint "${SELECTED[0]}" \
  --expected-checkpoint-sha256 "${SELECTED[1]}" \
  --outdir "$OUTPUT_ROOT/confirmation" \
  --device cuda:0 \
  --M "$M_CONFIRM" \
  --seed-bank low7-balanced-r0-disjoint-confirmation-v1 \
  --reflection-antithetic \
  >"$OUTPUT_ROOT/confirmation.log" 2>&1 &
PID_CONFIRM=$!

# This independent-iid audit is deliberately not a qualification gate.  It
# shows the ordinary finite-M fluctuation that antithetic symmetry
# qualification removes without changing the raw temperature-1 law.
CUDA_VISIBLE_DEVICES="$GPU_B" "$PYTHON" \
  "$REPO_ROOT/overnight_run_07_06/rev_expansion/codex_overnight/analysis/low7_balanced_r0_qualification.py" \
  --checkpoint "${SELECTED[0]}" \
  --expected-checkpoint-sha256 "${SELECTED[1]}" \
  --outdir "$OUTPUT_ROOT/iid_audit" \
  --device cuda:0 \
  --M "$M_CONFIRM" \
  --seed-bank low7-balanced-r0-iid-audit-v1 \
  --report-only \
  >"$OUTPUT_ROOT/iid_audit.log" 2>&1 &
PID_IID=$!
wait "$PID_CONFIRM"
wait "$PID_IID"

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
selection = json.load(open(root / "selection.json"))
confirmation = json.load(open(root / "confirmation/qualification.json"))
iid_audit = json.load(open(root / "iid_audit/qualification.json"))
if not confirmation["passed"]:
    raise RuntimeError("selected candidate failed disjoint confirmation")
if confirmation.get("raw_noise_design") != (
    "reflection-antithetic common-random-number pairs"
):
    raise RuntimeError("disjoint confirmation did not use the declared symmetry test")
payload = {
    "status": "LOW7_BALANCED_R0_DELIVERY_COMPLETE",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "selected": selection["selected"],
    "confirmation": str(root / "confirmation/qualification.json"),
    "confirmation_passed": True,
    "iid_audit": str(root / "iid_audit/qualification.json"),
    "iid_audit_is_not_a_gate": True,
    "iid_audit_passed_strict_finite_sample_gate": bool(iid_audit["passed"]),
}
(root / "DELIVERY_COMPLETE.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
