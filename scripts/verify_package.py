#!/usr/bin/env python3
"""Verify the compact B1 static-obstacle package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.json"
EXPECTED = {
    "source_commit": "63ebefa7877c0b923c1c7cdea19228302dd6a0ca",
    "dataset_sha256": "4b8e2d9be794584fad232bcc46cf78c2c4f422efb3e0642f503c8a77fcd2e8ec",
    "pretrained_checkpoint_sha256": "524c9c0a4fd071221ac509b9d8e6fbbfb85fdf1811aa04160317f2a9e2d3ef90",
    "selected_checkpoint_sha256": "60c155472f5ed0e4a1d53581857f09aead7924f8ce11e8e3adf890d5a57fc079",
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


def tracked_files() -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    return {value for value in output if value != MANIFEST.name}


def verify_manifest(failures: list[str]) -> None:
    manifest = load_json(MANIFEST)
    if manifest.get("source_commit") != EXPECTED["source_commit"]:
        failures.append("manifest source commit mismatch")
    entries = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual_tracked = tracked_files()
    if set(entries) != actual_tracked:
        missing = sorted(actual_tracked - set(entries))
        stale = sorted(set(entries) - actual_tracked)
        failures.append(f"manifest inventory mismatch: missing={missing}, stale={stale}")
    for relative, entry in entries.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            failures.append(f"size mismatch: {relative}")
        elif sha256_file(path) != entry["sha256"]:
            failures.append(f"hash mismatch: {relative}")


def verify_identities(failures: list[str]) -> None:
    recipe = load_json(ROOT / "configs" / "b1_current_best_recipe.json")
    teacher = load_json(ROOT / "configs" / "safemppi_static_teacher.json")
    pointer = load_json(ROOT / "DATA_POINTER.json")
    checks = {
        "recipe source": (recipe["source_git_commit"], EXPECTED["source_commit"]),
        "dataset pointer": (pointer["sha256"], EXPECTED["dataset_sha256"]),
        "recipe checkpoint": (
            recipe["source_checkpoint_sha256"],
            EXPECTED["pretrained_checkpoint_sha256"],
        ),
        "scene": (recipe["scene"]["sha256"], EXPECTED["scene_sha256"]),
        "pretrained checkpoint": (
            sha256_file(ROOT / "checkpoints" / "b1_balanced_pretrained.pt"),
            EXPECTED["pretrained_checkpoint_sha256"],
        ),
        "selected checkpoint": (
            sha256_file(ROOT / "checkpoints" / "b1_current_best_r19.pt"),
            EXPECTED["selected_checkpoint_sha256"],
        ),
        "centroid gain": (teacher["planner"]["centroid_gain"], 0.2),
        "centroid smoothing": (teacher["planner"]["centroid_smooth"], 0.25),
        "centroid epsilon": (teacher["planner"]["centroid_eps"], 0.15),
        "safety margin": (teacher["planner"]["safety_margin"], 0.0),
        "extra margin": (teacher["planner"]["barrier_extra_margin"], 0.0),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            failures.append(f"{label}: {actual} != {expected}")
    if not teacher["semantics"]["centroid_active_for_static_obstacles"]:
        failures.append("teacher contract incorrectly disables centroid sampling")


def verify_links(failures: list[str]) -> None:
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    markdown = ROOT / "README.md"
    for target in link_pattern.findall(markdown.read_text()):
        target = target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("#"):
            continue
        if not (markdown.parent / target).resolve().exists():
            failures.append(f"broken README link: {target}")


def main() -> int:
    failures: list[str] = []
    verify_manifest(failures)
    verify_identities(failures)
    verify_links(failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "B1_STATIC_PACKAGE_OK "
        f"files={len(tracked_files())} source={EXPECTED['source_commit'][:8]} "
        f"pretrain={EXPECTED['pretrained_checkpoint_sha256'][:8]} "
        f"selected={EXPECTED['selected_checkpoint_sha256'][:8]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
