#!/usr/bin/env bash
# Shared dual-scene AFE2 launcher (canonical protocol: AFE2_FINAL_PROTOCOL.md).
# One trainer, verifier, terminal semantics, acquisition, replay, and update implementation for
# both scene profiles; the profile is the ONLY task-specific input. The identical beta-calibration
# rule runs independently per (scene, checkpoint): continuous log-bisection targets median
# ESS/K=3B/K=0.375. Flat/unbracketable pools stop fail-closed; there is no fallback temperature.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SCENE_PROFILE /absolute/path/to/pretrained.pt EXPECTED_SHA256 /absolute/output/root" >&2
  echo "  SCENE_PROFILE: claude_grid_v1 | codex_radius1_v1" >&2
  exit 2
fi

PROFILE="$1"
case "$PROFILE" in
  claude_grid_v1|codex_radius1_v1) ;;
  *) echo "unknown scene profile: $PROFILE" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
CKPT_DIR=$(cd "$(dirname "$2")" && pwd)
CKPT="$CKPT_DIR/$(basename "$2")"
EXPECTED_CKPT_SHA256=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
if [[ -e "$4" && ! -d "$4" ]]; then
  echo "output root exists and is not a directory: $4" >&2
  exit 2
fi
if [[ -d "$4" && -n "$(find "$4" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "locked AFE2 launcher requires a new or empty output root: $4" >&2
  exit 2
fi
mkdir -p "$4"
OUT=$(cd "$4" && pwd)
PYTHON_BIN=${PYTHON:-python}

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 2
fi
if [[ ! "$EXPECTED_CKPT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "expected checkpoint SHA-256 must contain exactly 64 lowercase hex characters" >&2
  exit 2
fi

for RUN_DIR in "$OUT/beta_calibration" "$OUT/prox_s910" "$OUT/afe_s910"; do
  if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to mix a new run with existing output: $RUN_DIR" >&2
    exit 2
  fi
done

cd "$HERE"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required before launching the two expensive arms" >&2
  exit 2
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe is required to validate the two rendered videos" >&2
  exit 2
fi
FFMPEG_ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null)
if [[ "$FFMPEG_ENCODERS" != *libx264* ]]; then
  echo "ffmpeg is present but the required libx264 encoder is unavailable" >&2
  exit 2
fi
"$PYTHON_BIN" -c 'import matplotlib, numpy, torch' >/dev/null
"$PYTHON_BIN" grid_expand_afe2.py --help >/dev/null

# Immutable non-beta e97eead acquisition/update values, shared by both profiles. The lock rejects
# accidental knob changes; beta is the sole per-(scene,checkpoint) calibration output.
COMMON=(
  --ckpt "$CKPT"
  --expected-ckpt-sha256 "$EXPECTED_CKPT_SHA256"
  --scene-profile "$PROFILE"
  --rounds 10
  --K 64
  --B 8
  --beta-calibration "$OUT/beta_calibration/beta_calibration.json"
  --lam 10
  --T 300
  --reach 0.15
  --M-eval 8
  --batch 128
  --seed 910
  --lock-reference-recipe
)

"$PYTHON_BIN" grid_expand_afe2.py \
  --ckpt "$CKPT" \
  --expected-ckpt-sha256 "$EXPECTED_CKPT_SHA256" \
  --scene-profile "$PROFILE" \
  --rounds 10 \
  --K 64 \
  --B 8 \
  --lam 10 \
  --T 300 \
  --reach 0.15 \
  --M-eval 8 \
  --batch 128 \
  --seed 910 \
  --lock-reference-recipe \
  --calibrate \
  --outdir "$OUT/beta_calibration"

# Deliberately sequential: both arms receive the full GPU and cannot share or mutate runtime
# state. Both reload the exact same checkpoint from disk.
"$PYTHON_BIN" grid_expand_afe2.py \
  "${COMMON[@]}" \
  --arm prox \
  --prox-lr 2e-5 \
  --prox-eta 0.01 \
  --outdir "$OUT/prox_s910"

"$PYTHON_BIN" grid_expand_afe2.py \
  "${COMMON[@]}" \
  --arm afe \
  --afe-lr 1e-4 \
  --afe-steps 250 \
  --outdir "$OUT/afe_s910"

"$PYTHON_BIN" analysis/validate_afe2_pair.py \
  --scene-profile "$PROFILE" \
  --prox "$OUT/prox_s910" \
  --afe "$OUT/afe_s910" \
  --beta-calibration "$OUT/beta_calibration/beta_calibration.json" \
  --out "$OUT/afe2_${PROFILE}_pair_manifest.json"

"$PYTHON_BIN" analysis/afe2_report.py \
  --arms "$OUT/prox_s910" "$OUT/afe_s910" \
  --pair-manifest "$OUT/afe2_${PROFILE}_pair_manifest.json" \
  --out "$OUT/afe2_${PROFILE}_report.png"

"$PYTHON_BIN" video_afe2.py \
  --run "$OUT/prox_s910" \
  --out "$OUT/afe2_${PROFILE}_prox.mp4"

"$PYTHON_BIN" video_afe2.py \
  --run "$OUT/afe_s910" \
  --out "$OUT/afe2_${PROFILE}_afe.mp4"

"$PYTHON_BIN" analysis/validate_afe2_pair.py \
  --scene-profile "$PROFILE" \
  --prox "$OUT/prox_s910" \
  --afe "$OUT/afe_s910" \
  --beta-calibration "$OUT/beta_calibration/beta_calibration.json" \
  --out "$OUT/afe2_${PROFILE}_pair_manifest.json" \
  --report "$OUT/afe2_${PROFILE}_report.png" \
  --prox-video "$OUT/afe2_${PROFILE}_prox.mp4" \
  --afe-video "$OUT/afe2_${PROFILE}_afe.mp4" \
  --delivery-out "$OUT/DELIVERY_COMPLETE.json"

echo "AFE2 pair complete [$PROFILE]: $OUT"
