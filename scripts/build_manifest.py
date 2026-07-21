#!/usr/bin/env python3
"""Regenerate SOURCE_MANIFEST.json from the tracked package inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.json"
SOURCE_COMMIT = "63ebefa7877c0b923c1c7cdea19228302dd6a0ca"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    relative_paths = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    relative_paths = sorted(
        value for value in relative_paths if value != OUTPUT.name
    )
    files = []
    for relative in relative_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"tracked file is missing: {relative}")
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    OUTPUT.write_text(json.dumps({
        "package": "B1_current_best_static_obstacles",
        "source_commit": SOURCE_COMMIT,
        "files": files,
    }, indent=2, sort_keys=True) + "\n")
    print(f"MANIFEST_WRITTEN files={len(files)}")


if __name__ == "__main__":
    main()
