#!/usr/bin/env python3
"""Replace declared gamma cells in a complete B1 metrics contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> tuple[dict, Path, list[dict]]:
    contract = json.loads(path.read_text())
    metrics = Path(contract["metrics"])
    if sha256(metrics) != contract["metrics_sha256"]:
        raise RuntimeError(f"metrics hash mismatch: {metrics}")
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    return contract, metrics, rows


def resolve_temp(temp_map: dict, gamma: float, round_i: int) -> float:
    value = 1.0
    for from_round, candidate in temp_map.get(f"{gamma:g}", []):
        if round_i >= int(from_round):
            value = float(candidate)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract", required=True, type=Path)
    parser.add_argument(
        "--replace", action="append", default=[], metavar="GAMMA=CONTRACT",
        help="replace every round for one gamma using a single-gamma contract",
    )
    parser.add_argument("--temperature-map", required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--tag", default="fixedtemp_m200_revised")
    args = parser.parse_args()

    base, base_metrics, base_rows = load_contract(args.base_contract)
    rounds = [int(value) for value in base["rounds"]]
    gammas = [float(value) for value in base["gammas"]]
    table = {(int(row["round"]), float(row["gamma"])): row for row in base_rows}
    expected = {(round_i, gamma) for round_i in rounds for gamma in gammas}
    if set(table) != expected:
        raise RuntimeError("base contract does not contain the complete metric grid")

    reference_fields = (
        "M_per_gamma", "NFE", "bank_split", "bank_version", "raw_policy",
        "trajectories_persisted",
    )
    replacement_sources = []
    seen_gammas = set()
    for raw_spec in args.replace:
        raw_gamma, raw_path = raw_spec.split("=", 1)
        gamma = float(raw_gamma)
        if gamma not in gammas or gamma in seen_gammas:
            raise RuntimeError(f"invalid or duplicate replacement gamma: {gamma:g}")
        seen_gammas.add(gamma)
        path = Path(raw_path)
        contract, metrics, rows = load_contract(path)
        for field in reference_fields:
            if contract[field] != base[field]:
                raise RuntimeError(f"replacement disagrees on {field}: {path}")
        if [float(value) for value in contract["gammas"]] != [gamma]:
            raise RuntimeError(f"replacement is not single-gamma {gamma:g}: {path}")
        if [int(value) for value in contract["rounds"]] != rounds:
            raise RuntimeError(f"replacement round coverage mismatch: {path}")
        for round_i in rounds:
            key = str(round_i)
            if contract["checkpoint_sha256"][key] != base["checkpoint_sha256"][key]:
                raise RuntimeError(f"checkpoint mismatch at r{round_i}: {path}")
        replacement_table = {
            (int(row["round"]), float(row["gamma"])): row for row in rows
        }
        replacement_expected = {(round_i, gamma) for round_i in rounds}
        if set(replacement_table) != replacement_expected:
            raise RuntimeError(f"replacement metric cells mismatch: {path}")
        table.update(replacement_table)
        replacement_sources.append({
            "gamma": gamma,
            "contract": str(path.resolve()),
            "contract_sha256": sha256(path),
            "metrics": str(metrics.resolve()),
            "metrics_sha256": contract["metrics_sha256"],
            "bank_sha256": contract["bank_sha256"],
        })

    temp_map = json.loads(args.temperature_map)
    for (round_i, gamma), row in table.items():
        expected_temp = resolve_temp(temp_map, gamma, round_i)
        if abs(float(row["temp"]) - expected_temp) > 1e-12:
            raise RuntimeError(
                f"temperature mismatch at r{round_i}, gamma {gamma:g}: "
                f"{row['temp']} != {expected_temp}"
            )

    args.outdir.mkdir(parents=True, exist_ok=False)
    metrics_out = args.outdir / f"{args.tag}.jsonl"
    metrics_out.write_text("".join(
        json.dumps(table[key], separators=(",", ":")) + "\n"
        for key in sorted(table)
    ))
    output = {
        "status": "B1_METRIC_CELLS_REPLACED",
        "rounds": rounds,
        "gammas": gammas,
        "cell_count": len(table),
        "M_per_gamma": base["M_per_gamma"],
        "NFE": base["NFE"],
        "bank_split": base["bank_split"],
        "bank_version": base["bank_version"],
        "temperature_map": temp_map,
        "raw_policy": base["raw_policy"],
        "trajectories_persisted": base["trajectories_persisted"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "base": {
            "contract": str(args.base_contract.resolve()),
            "contract_sha256": sha256(args.base_contract),
            "metrics": str(base_metrics.resolve()),
            "metrics_sha256": base["metrics_sha256"],
        },
        "replacements": replacement_sources,
        "metrics": str(metrics_out.resolve()),
        "metrics_sha256": sha256(metrics_out),
    }
    contract_out = args.outdir / f"{args.tag}.contract.json"
    contract_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(metrics_out)
    print(contract_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
