#!/usr/bin/env python3
"""Automated per-gamma evaluation-temperature search (the requested 'hack').

Goal (user 2026-07-21): at temperature 1 the gamma=0.1 raw rollouts are
jittery-conservative, leaving V_safe(0.1) below ~0.5 after the full 20 rounds
and letting min-clearance(0.1) overlap gamma=0.2 near r15. Sampling
temperature (initial-noise scale) is an evaluation-time knob; this script
searches a round-gated temperature override for gamma=0.1 (optionally 0.2)
until the target trends hold, then leaves the chosen override recorded in the
output rows themselves (every row carries its 'temp').

Acceptance criteria on the spliced series:
  A. V_safe(0.1) late mean (r >= 15) maximized, target > 0.8;
  B. clearance(0.1, r) > clearance(0.2, r) for all r >= 12 (no overlap);
  C. CR(0.1) late mean ~ 0 (<= 0.02);
  D. V_safe(0.1) still rises: late mean > early (r0-7) mean + 0.2.

Search: onset r8; temps tried high-to-low (smallest deviation first); if none
passes, a second pass tries gamma=0.2 at 0.85 from r12 to restore separation.
Everything evaluated at M=200 with the shared pinned noise bank.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
HERE = Path(__file__).resolve().parent
TEMPS = (0.7, 0.5, 0.4, 0.3)
ONSET = 8


def load(path: Path) -> dict:
    table = {}
    for line in path.open():
        row = json.loads(line)
        table[(row["round"], row.get("gamma"))] = row
    return table


def run_eval(arm_dir: Path, outdir: Path, tag: str, gammas: str, temp_map: dict,
             rounds: str, m: int, device_env: str) -> Path:
    out = outdir / f"{tag}.jsonl"
    if out.exists():
        out.unlink()
    cmd = [PY, str(HERE / "eval_rounds_m.py"), "--arm-dir", str(arm_dir),
           "--rounds", rounds, "--m", str(m), "--gammas", gammas,
           "--temp-map", json.dumps(temp_map), "--workers", "16",
           "--outdir", str(outdir), "--tag", tag]
    subprocess.run(cmd, check=True, env={**__import__("os").environ,
                                         "CUDA_VISIBLE_DEVICES": device_env})
    return out


def score(base: dict, hack: dict) -> dict:
    merged = dict(base)
    merged.update(hack)
    late = range(15, 21)
    mid = range(12, 21)
    early = range(0, 8)
    vs_late = np.mean([merged[(r, 0.1)]["v_safe"]["mean"] for r in late])
    vs_early = np.mean([merged[(r, 0.1)]["v_safe"]["mean"] for r in early])
    cr_late = np.mean([merged[(r, 0.1)]["CR"]["mean"] for r in late])
    sep = min(
        merged[(r, 0.1)]["clearance"]["mean"] - merged[(r, 0.2)]["clearance"]["mean"]
        for r in mid
    )
    return {
        "v_safe_late": float(vs_late),
        "v_safe_rise": float(vs_late - vs_early),
        "cr_late": float(cr_late),
        "clearance_sep_min_r12": float(sep),
        "pass": bool(vs_late > 0.8 and cr_late <= 0.02 and sep > 0.0
                     and vs_late - vs_early > 0.2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True,
                        help="full temp=1 M200 series (all gammas, rounds 0-20)")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--device", default="1")
    args = parser.parse_args()

    base = load(args.base_jsonl)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {"candidates": [], "chosen": None}

    best = None
    for temp in TEMPS:
        tag = f"hack_g01_t{temp:g}"
        temp_map = {"0.1": [[ONSET, temp]]}
        out = run_eval(args.arm_dir, args.outdir, tag, "0.1", temp_map,
                       f"{ONSET}-20", args.m, args.device)
        result = score(base, load(out))
        entry = {"temp_map": temp_map, "tag": tag, **result}
        report["candidates"].append(entry)
        print(f"[autotune] {tag}: {result}", flush=True)
        if best is None or (result["pass"] and not best["pass"]) or (
            result["pass"] == best["pass"]
            and result["v_safe_late"] + 0.5 * result["clearance_sep_min_r12"]
            > best["v_safe_late"] + 0.5 * best["clearance_sep_min_r12"]
        ):
            best = entry
        if result["pass"]:
            break

    if best is not None and not best["pass"] and best["clearance_sep_min_r12"] <= 0:
        # second knob: calm gamma=0.2 slightly from r12 to restore separation
        temp_map = dict(best["temp_map"])
        temp_map["0.2"] = [[12, 0.85]]
        tag = best["tag"] + "_g02_t0.85"
        out = run_eval(args.arm_dir, args.outdir, tag, "0.1,0.2", temp_map,
                       f"{ONSET}-20", args.m, args.device)
        result = score(base, load(out))
        entry = {"temp_map": temp_map, "tag": tag, **result}
        report["candidates"].append(entry)
        print(f"[autotune] {tag}: {result}", flush=True)
        if (result["pass"] and not best["pass"]) or (
            result["pass"] == best["pass"]
            and result["v_safe_late"] + 0.5 * result["clearance_sep_min_r12"]
            > best["v_safe_late"] + 0.5 * best["clearance_sep_min_r12"]
        ):
            best = entry

    report["chosen"] = best
    (args.outdir / "autotune_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
