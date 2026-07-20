#!/usr/bin/env python3
"""Select a reflection-paired r0 candidate, then require disjoint confirmation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_record(directory: Path) -> dict:
    qualification_path = directory / "qualification_select/qualification.json"
    checkpoint = directory / "pretrain/data/checkpoint_candidate.pt"
    manifest = directory / "pretrain/manifest.json"
    if not all(path.is_file() for path in (qualification_path, checkpoint, manifest)):
        raise FileNotFoundError(f"candidate {directory} is incomplete")
    qualification = json.loads(qualification_path.read_text())
    balances = [
        float(entry["all_routes"]["balance"])
        for entry in qualification["per_gamma"].values()
    ]
    success_balances = [
        float(entry["successful_routes"]["balance"])
        for entry in qualification["per_gamma"].values()
    ]
    successes = sum(
        int(entry["success_count"]) for entry in qualification["per_gamma"].values()
    )
    attempts = sum(int(entry["M"]) for entry in qualification["per_gamma"].values())
    return {
        "name": directory.name,
        "directory": str(directory.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "pretrain_manifest": str(manifest.resolve()),
        "qualification": str(qualification_path.resolve()),
        "passed": bool(qualification["passed"]),
        "minimum_per_gamma_balance": min(balances),
        "mean_per_gamma_balance": sum(balances) / len(balances),
        "minimum_per_gamma_success_balance": min(success_balances),
        "mean_per_gamma_success_balance": sum(success_balances) / len(success_balances),
        "raw_SR": successes / attempts,
    }


def run(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    candidates = [candidate_record(path) for path in sorted(root.glob("seed_*"))]
    if len(candidates) < 2:
        raise RuntimeError("selection requires at least two completed candidate seeds")
    eligible = [item for item in candidates if item["passed"]]
    if not eligible:
        raise RuntimeError("no candidate passed every-gamma raw U/R qualification")
    selected = max(
        eligible,
        key=lambda item: (
            item["minimum_per_gamma_success_balance"],
            item["minimum_per_gamma_balance"],
            item["mean_per_gamma_success_balance"],
            item["mean_per_gamma_balance"],
            item["raw_SR"],
            item["name"],
        ),
    )
    result = {
        "status": "LOW7_BALANCED_R0_SELECTION_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "pass every-gamma all-route and successful-route balance/resolution gate; "
            "maximize minimum successful-route balance, then minimum all-route balance, "
            "then their means, raw SR, and deterministic seed name"
        ),
        "confirmation_required": True,
        "candidates": candidates,
        "selected": selected,
    }
    output = root / "selection.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    result = run(make_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
