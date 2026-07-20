#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT GPU1_UUID GPU3_UUID" >&2
  exit 2
fi

OUT=$1
GPU1_UUID=$2
GPU3_UUID=$3
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}

[[ ! -e "$OUT" ]] || { echo "output root already exists: $OUT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "python is not executable: $PYTHON" >&2; exit 1; }
command -v nvidia-smi >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
# Do not use grep -q under pipefail: an early grep exit can SIGPIPE ffmpeg and
# falsely fail the launch gate even when libx264 is present.
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264 >/dev/null

CPU_COUNT=$(getconf _NPROCESSORS_ONLN)
(( CPU_COUNT > 1 && CPU_COUNT % 2 == 0 )) || {
  echo "host logical CPU count must be positive and even: $CPU_COUNT" >&2
  exit 1
}
WORKERS=$((CPU_COUNT / 2))

check_gpu() {
  local index=$1 expected=$2 actual pids
  actual=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
  [[ "${actual,,}" == "${expected,,}" ]] || {
    echo "GPU $index UUID mismatch: actual=$actual expected=$expected" >&2
    return 1
  }
  pids=$(nvidia-smi -i "$index" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  [[ -z "$pids" ]] || {
    echo "GPU $index has foreign compute PIDs: $pids" >&2
    return 1
  }
}

check_gpu 1 "$GPU1_UUID"
check_gpu 3 "$GPU3_UUID"

export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" "$HERE/analysis/low7_rbf_v3_support_sweep_driver.py" \
  --out "$OUT" \
  --gpu1-uuid "$GPU1_UUID" \
  --gpu3-uuid "$GPU3_UUID" \
  --verifier-workers "$WORKERS" \
  --python "$PYTHON"
