#!/usr/bin/env python3
"""Decompose B1 raw-evaluation clearance into all/success/failure groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def group(values: np.ndarray) -> dict[str, float | int | None]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else None,
    }


def analyze(cells: Path, rounds: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    for round_i in rounds:
        pooled_clearance: list[np.ndarray] = []
        pooled_success: list[np.ndarray] = []
        for gamma in GAMMAS:
            cell = cells / f"r{round_i:03d}_g{gamma:.1f}.npz"
            with np.load(cell, allow_pickle=False) as archive:
                clearance = np.asarray(archive["minimum_clearance"], dtype=float)
                success = np.asarray(archive["success"], dtype=bool)
            pooled_clearance.append(clearance)
            pooled_success.append(success)
            rows.append({
                "round": round_i,
                "gamma": gamma,
                "all": group(clearance),
                "success_only": group(clearance[success]),
                "failure_only": group(clearance[~success]),
            })
        clearance = np.concatenate(pooled_clearance)
        success = np.concatenate(pooled_success)
        rows.append({
            "round": round_i,
            "gamma": None,
            "all": group(clearance),
            "success_only": group(clearance[success]),
            "failure_only": group(clearance[~success]),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[0, 19, 20])
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = analyze(args.cells, tuple(args.rounds))
    payload = {
        "definition": (
            "minimum over every state and every obstacle/wall in one trajectory; "
            "the legacy aggregate is the arithmetic mean over all trajectories"
        ),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "round", "gamma", "n_all", "mean_all", "n_success",
            "mean_success", "n_failure", "mean_failure",
        ))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "round": row["round"],
                "gamma": "pooled" if row["gamma"] is None else row["gamma"],
                "n_all": row["all"]["n"],
                "mean_all": row["all"]["mean"],
                "n_success": row["success_only"]["n"],
                "mean_success": row["success_only"]["mean"],
                "n_failure": row["failure_only"]["n"],
                "mean_failure": row["failure_only"]["mean"],
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
