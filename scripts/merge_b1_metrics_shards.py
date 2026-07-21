#!/usr/bin/env python3
"""Validate and merge disjoint eval_rounds_m checkpoint shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_rounds(raw: str) -> list[int]:
    if "-" in raw:
        lo, hi = (int(value) for value in raw.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(value) for value in raw.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="append", required=True, type=Path)
    parser.add_argument("--rounds", required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--tag", default="fixedtemp_m200_merged")
    args = parser.parse_args()

    expected_rounds = parse_rounds(args.rounds)
    contracts = [json.loads(path.read_text()) for path in args.contract]
    reference_fields = (
        "M_per_gamma", "NFE", "bank_split", "bank_version", "raw_policy",
        "temperature_map", "trajectories_persisted",
    )
    for field in reference_fields:
        values = [contract[field] for contract in contracts]
        if values[1:] != values[:-1]:
            raise RuntimeError(f"shards disagree on {field}: {values}")

    rows: dict[tuple[int, float], dict] = {}
    checkpoint_hashes: dict[str, str] = {}
    bank_hashes: dict[str, str] = {}
    source_records = []
    for contract_path, contract in zip(args.contract, contracts):
        gamma_key = ",".join(f"{float(gamma):g}" for gamma in contract["gammas"])
        prior_bank_hash = bank_hashes.get(gamma_key)
        if prior_bank_hash is not None and prior_bank_hash != contract["bank_sha256"]:
            raise RuntimeError(
                f"gamma group {gamma_key} uses multiple banks: "
                f"{prior_bank_hash}, {contract['bank_sha256']}"
            )
        bank_hashes[gamma_key] = contract["bank_sha256"]
        metrics_path = Path(contract["metrics"])
        if sha256(metrics_path) != contract["metrics_sha256"]:
            raise RuntimeError(f"metrics hash mismatch: {metrics_path}")
        for key, digest in contract["checkpoint_sha256"].items():
            if key in checkpoint_hashes and checkpoint_hashes[key] != digest:
                raise RuntimeError(f"checkpoint round {key} has conflicting hashes")
            checkpoint_hashes[key] = digest
        for line in metrics_path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (int(row["round"]), float(row["gamma"]))
            if key in rows:
                raise RuntimeError(f"duplicate metric cell: {key}")
            rows[key] = row
        source_records.append({
            "contract": str(contract_path.resolve()),
            "contract_sha256": sha256(contract_path),
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": contract["metrics_sha256"],
            "rounds": contract["rounds"],
            "gammas": contract["gammas"],
            "bank_sha256": contract["bank_sha256"],
        })

    expected = {(round_i, gamma) for round_i in expected_rounds for gamma in GAMMAS}
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise RuntimeError(f"metric cells mismatch; missing={missing}, extra={extra}")
    if sorted(int(key) for key in checkpoint_hashes) != expected_rounds:
        raise RuntimeError("checkpoint hashes do not exactly cover expected rounds")

    args.outdir.mkdir(parents=True, exist_ok=False)
    metrics_out = args.outdir / f"{args.tag}.jsonl"
    payload = "".join(
        json.dumps(rows[key], separators=(",", ":")) + "\n"
        for key in sorted(rows)
    )
    metrics_out.write_text(payload)
    merged = {
        "status": "B1_METRICS_SHARDS_MERGED",
        "rounds": expected_rounds,
        "gammas": list(GAMMAS),
        "cell_count": len(rows),
        "M_per_gamma": contracts[0]["M_per_gamma"],
        "NFE": contracts[0]["NFE"],
        "bank_sha256_by_gamma_group": bank_hashes,
        "bank_split": contracts[0]["bank_split"],
        "bank_version": contracts[0]["bank_version"],
        "temperature_map": contracts[0]["temperature_map"],
        "raw_policy": contracts[0]["raw_policy"],
        "trajectories_persisted": contracts[0]["trajectories_persisted"],
        "checkpoint_sha256": checkpoint_hashes,
        "sources": source_records,
        "metrics": str(metrics_out.resolve()),
        "metrics_sha256": sha256(metrics_out),
    }
    contract_out = args.outdir / f"{args.tag}.contract.json"
    contract_out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(metrics_out)
    print(contract_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
