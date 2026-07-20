#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=/home/dohyun/projects/afe2_runs/low7_fixed_grid_8c80cf0/combined/data/low7_randomized_all_gamma.pt
EXPECTED=4b8e2d9be794584fad232bcc46cf78c2c4f422efb3e0642f503c8a77fcd2e8ec
DEST=$HERE/external_data/low7_randomized_all_gamma.pt

[[ -f "$SOURCE" ]] || { echo "missing canonical dataset: $SOURCE" >&2; exit 1; }
ACTUAL=$(sha256sum "$SOURCE" | awk '{print $1}')
[[ "$ACTUAL" == "$EXPECTED" ]] || {
  echo "dataset hash mismatch: $ACTUAL != $EXPECTED" >&2
  exit 1
}

mkdir -p "$HERE/external_data"
if [[ -e "$DEST" || -L "$DEST" ]]; then
  [[ "$(readlink -f "$DEST")" == "$(readlink -f "$SOURCE")" ]] || {
    echo "refusing to replace existing noncanonical data link: $DEST" >&2
    exit 1
  }
else
  ln -s "$SOURCE" "$DEST"
fi

echo "DATA_LINK_OK $DEST -> $SOURCE"
