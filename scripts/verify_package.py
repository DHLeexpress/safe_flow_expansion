#!/usr/bin/env python3
"""Verify the workbook's immutable files, core identities, and local links."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
EXPECTED = {
    "source_commit": "e63ebd80e5fa4f712a1a7cf590ec74c116768873",
    "dataset_sha256": "4b8e2d9be794584fad232bcc46cf78c2c4f422efb3e0642f503c8a77fcd2e8ec",
    "pretrained_checkpoint_sha256": "7ae44f773b3f5fe36579c4101542e495119cf6e348f622f5edbfedaa2855a46c",
    "selected_checkpoint_sha256": "ab6ce3c39671554ef114234c464f23cc18828ea751a4bbb5547beb59793c1b54",
    "scene_sha256": "356d6d48b3af2b017b529562b530f35285c86f9107da512a73de6ef664b03e72",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def verify_manifest() -> int:
    manifest = load_json(MANIFEST)
    failures: list[str] = []
    if manifest.get("source_commit") != EXPECTED["source_commit"]:
        failures.append("manifest source commit mismatch")

    for entry in manifest.get("files", []):
        relative = entry["path"]
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(entry["bytes"]):
            failures.append(
                f"size mismatch: {relative}: {actual_size} != {entry['bytes']}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            failures.append(f"hash mismatch: {relative}")

    recipe = load_json(ROOT / "configs" / "recipe.json")
    pointer = load_json(ROOT / "DATA_POINTER.json")
    identity_checks = {
        "recipe source commit": (recipe["source_git_commit"], EXPECTED["source_commit"]),
        "recipe dataset hash": (
            recipe["demo_reference"]["dataset_sha256"],
            EXPECTED["dataset_sha256"],
        ),
        "data pointer hash": (pointer["sha256"], EXPECTED["dataset_sha256"]),
        "recipe checkpoint hash": (
            recipe["source_checkpoint_sha256"],
            EXPECTED["pretrained_checkpoint_sha256"],
        ),
        "recipe scene hash": (recipe["scene"]["sha256"], EXPECTED["scene_sha256"]),
        "packaged pretrained checkpoint": (
            sha256_file(ROOT / "checkpoints" / "low7_pretrained_checkpoint.pt"),
            EXPECTED["pretrained_checkpoint_sha256"],
        ),
        "packaged selected checkpoint": (
            sha256_file(ROOT / "checkpoints" / "phase_c_selected_r25.pt"),
            EXPECTED["selected_checkpoint_sha256"],
        ),
    }
    for label, (actual, expected) in identity_checks.items():
        if actual != expected:
            failures.append(f"{label}: {actual} != {expected}")

    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for markdown in (ROOT / "README.md", ROOT / "CODE_INDEX.md"):
        for target in link_pattern.findall(markdown.read_text()):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("#"):
                continue
            if not (markdown.parent / target).resolve().exists():
                failures.append(f"broken link in {markdown.name}: {target}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "WORKBOOK_OK "
        f"files={len(manifest['files'])} source={EXPECTED['source_commit'][:8]} "
        f"pretrain={EXPECTED['pretrained_checkpoint_sha256'][:8]} "
        f"selected={EXPECTED['selected_checkpoint_sha256'][:8]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    return verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
