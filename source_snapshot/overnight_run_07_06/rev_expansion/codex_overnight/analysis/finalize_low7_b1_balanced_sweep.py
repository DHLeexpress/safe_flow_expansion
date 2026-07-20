#!/usr/bin/env python3
"""Authenticate and finish an interrupted B1 sweep without rerunning science.

This is deliberately a recovery path, not a way to retroactively satisfy the
original two-GPU exclusivity contract.  It revalidates the completed trainer
and raw-evaluation artifacts, renders only missing presentation assets, and
writes an explicitly non-canonical recovery manifest last.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time

import low7_b1_balanced_sweep_driver as DRIVER


RECOVERY_STATUS = "LOW7_B1_BALANCED_R0_RECOVERED_NONCANONICAL_DELIVERY_COMPLETE"
EXPECTED_HOLDOUT_STATUS = "AFE_RBF_B1_BALANCED_HOLDOUT_DELIVERY_COMPLETE"
EXPECTED_SCREEN_STATUS = "AFE_RBF_B1_BALANCED_SCREEN_DELIVERY_COMPLETE"
EXPECTED_SELECTED_ARM = "cap512_ess025_alpha0010_cost"
EXPECTED_SELECTED_ROUND = 19


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def require_gpu1(index: int, expected_uuid: str, *, require_idle: bool) -> dict:
    if index != 1:
        raise ValueError("B1 recovery is restricted to physical GPU 1")
    return DRIVER.gpu_record(index, expected_uuid, require_idle=require_idle)


def require_png(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 100 or height < 100:
        raise RuntimeError(f"invalid PNG dimensions: {path}={width}x{height}")
    return {"width": width, "height": height, "bytes": path.stat().st_size}


def require_video(path: Path, expected_frames: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames",
        "-of", "json", str(path),
    ], text=True))
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"recovery video needs one video stream: {payload}")
    stream = streams[0]
    try:
        frames = int(stream["nb_read_frames"])
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"incomplete recovery video metadata: {stream}") from error
    if (
        stream.get("codec_name") != "h264"
        or frames != expected_frames
        or width < 2
        or height < 2
    ):
        raise RuntimeError(
            f"recovery video is not the expected {expected_frames}-frame H.264: "
            f"{stream}"
        )
    return {
        "codec_name": stream["codec_name"],
        "frames": frames,
        "width": width,
        "height": height,
    }


def require_root_copy(root: Path, confirmation: Path, filename: str) -> None:
    source = confirmation / filename
    target = root / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        if not target.is_file() or DRIVER.sha256_file(target) != DRIVER.sha256_file(source):
            raise RuntimeError(f"root presentation conflicts with holdout: {target}")
        return
    shutil.copy2(source, target)


def expected_video_frames(run_dir: Path) -> int:
    recipe = DRIVER.load_json(run_dir / "recipe.json")
    require_equal(recipe.get("artifact_profile"), "sweep_compact", "artifact profile")
    require_equal(int(recipe.get("rounds", -1)), 20, "trainer rounds")
    first = 0 if recipe.get("video_include_round0") else 1
    rounds = [
        value for value in range(first, 21)
        if value <= 10 or (value > 10 and value % 10 == 0)
    ]
    observed = sorted(
        int(path.stem.removeprefix("round"))
        for path in (run_dir / "viz_db").glob("round*.pt")
    )
    require_equal(observed, rounds, "authenticated video rounds")
    return len(rounds)


def run_to_atomic_output(
    command: list[str], target: Path, logs: Path, label: str, gpu_index: int
) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs / f"recovery_{label}_{stamp}.log"
    suffix = target.suffix or ".tmp"
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}_recovery_", suffix=suffix,
        dir=target.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    command = [str(temporary) if value == "{OUTPUT}" else value for value in command]
    try:
        with log_path.open("x") as stream:
            stream.write(f"$ {' '.join(command)}\n")
            stream.flush()
            subprocess.run(
                command,
                cwd=DRIVER.ROOT,
                env=DRIVER.command_env(gpu_index),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"{label} did not produce an artifact")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return log_path


def validate_arm_records(root: Path) -> tuple[list[dict], dict]:
    expected_statuses = {
        f"{arm.arm_id}.json" for arm in DRIVER.ARMS
    }
    status_dir = root / "arm_status"
    observed_statuses = {path.name for path in status_dir.glob("*.json")}
    require_equal(observed_statuses, expected_statuses, "arm-status inventory")

    records = []
    for arm in DRIVER.ARMS:
        paths = DRIVER.arm_paths(root, arm)
        record = DRIVER.load_json(paths["status"])
        require_equal(record.get("status"), "ARM_COMPLETE", f"{arm.arm_id} status")
        require_equal(record.get("arm"), arm.record(), f"{arm.arm_id} arm")
        if int(record.get("gpu", -1)) not in (1, 3):
            raise RuntimeError(f"{arm.arm_id} has an undeclared training GPU")
        require_equal(Path(record["run"]).resolve(), paths["run"].resolve(), f"{arm.arm_id} run")
        require_equal(
            Path(record["screening"]).resolve(), paths["screen"].resolve(),
            f"{arm.arm_id} screening",
        )
        DRIVER.require_run(paths["run"], arm, 20)
        DRIVER.require_evaluation(paths["screen"], EXPECTED_SCREEN_STATUS)
        screening = DRIVER.load_json(paths["screen"] / "selection.json")
        require_equal(record.get("best"), screening["ranking"][0], f"{arm.arm_id} best")
        records.append(record)

    ranking = sorted(records, key=DRIVER.global_key)
    selection = DRIVER.load_json(root / "selection.json")
    require_equal(selection.get("ranking"), ranking, "global ranking")
    require_equal(selection.get("selected"), ranking[0], "global selection")

    with (root / "screening_table.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    require_equal(len(table), len(DRIVER.ARMS), "screening-table row count")
    require_equal([row["arm_id"] for row in table], [row["arm"]["arm_id"] for row in ranking], "screening-table order")
    return ranking, selection


def validate_holdout(root: Path, selected: dict) -> dict:
    selected_arm = selected["arm"]["arm_id"]
    selected_round = int(selected["best"]["round"])
    require_equal(selected_arm, EXPECTED_SELECTED_ARM, "recovery selected arm")
    require_equal(selected_round, EXPECTED_SELECTED_ROUND, "recovery selected round")
    run_dir = root / "arms" / selected_arm
    confirmation = root / "confirmation"
    complete = DRIVER.require_evaluation(confirmation, EXPECTED_HOLDOUT_STATUS)
    require_equal(complete.get("trainer_source_commit"), root_source_commit(root), "holdout trainer source")

    contract = DRIVER.load_json(confirmation / "evaluation_contract.json")
    require_equal(
        contract.get("raw_policy"),
        "temperature=1, no tilt/filter/controller/fallback",
        "holdout raw-policy contract",
    )
    run = contract.get("run") or {}
    require_equal(Path(run.get("run_root", "")).resolve(), run_dir.resolve(), "holdout run root")
    require_equal(int(contract.get("selected_round_fixed_before_holdout", -1)), selected_round, "fixed holdout round")
    require_equal(run.get("complete_sha256"), DRIVER.sha256_file(run_dir / "COMPLETE.json"), "holdout run COMPLETE")
    require_equal(run.get("recipe_sha256"), DRIVER.sha256_file(run_dir / "recipe.json"), "holdout recipe")
    require_equal(run.get("probe_sha256"), DRIVER.sha256_file(run_dir / "probe.jsonl"), "holdout probe")

    fixed = DRIVER.load_json(confirmation / "selection.json")
    require_equal(fixed.get("selected_round"), selected_round, "holdout selection")
    require_equal(fixed.get("rounds"), [0, selected_round, 20], "holdout rounds")
    summary = DRIVER.load_json(confirmation / "evaluation_summary.json")
    require_equal(summary.get("mode"), "holdout", "holdout mode")
    require_equal(summary.get("M_per_gamma"), 50, "holdout M/gamma")
    require_equal(summary.get("rounds"), [0, selected_round, 20], "holdout summary rounds")

    rows = [
        json.loads(line) for line in (confirmation / "metrics.jsonl").read_text().splitlines()
    ]
    require_equal({int(row["round"]) for row in rows}, {0, selected_round, 20}, "holdout metric rounds")
    for row in rows:
        expected_n = 350 if row.get("scope") == "pooled" else 50
        require_equal(int(row.get("n", -1)), expected_n, "holdout metric n")
        for name in ("SR", "CR", "timeout", "V_safe", "V_full"):
            require_equal(int(row["binary"][name]["n"]), expected_n, f"holdout {name} n")
    return {"complete": complete, "contract": contract, "summary": summary}


def root_source_commit(root: Path) -> str:
    return str(DRIVER.load_json(root / "provenance.json")["source"]["commit"])


def validate_provenance(
    root: Path, expected_training_source: str, expected_checkpoint_sha256: str
) -> dict:
    provenance = DRIVER.load_json(root / "provenance.json")
    require_equal(provenance["source"]["commit"], expected_training_source, "training source")
    require_equal(
        provenance.get("checkpoint_sha256"), expected_checkpoint_sha256,
        "training checkpoint",
    )
    require_equal(provenance.get("arms"), [arm.record() for arm in DRIVER.ARMS], "declared arms")
    delivery_path = Path(provenance["qualified_pretraining"]["delivery"])
    checkpoint, checksum, _ = DRIVER.qualified_checkpoint(delivery_path)
    require_equal(checksum, expected_checkpoint_sha256, "qualified checkpoint SHA")
    require_equal(Path(provenance["checkpoint"]).resolve(), checkpoint, "qualified checkpoint path")
    preflight = DRIVER.load_json(root / "preflight" / "COMPLETE.json")
    require_equal(preflight.get("status"), "PREFLIGHT_COMPLETE_NOT_REUSED", "preflight")
    return provenance


def artifact_inventory(root: Path) -> dict[str, str]:
    excluded = {"DELIVERY_COMPLETE.json", "RECOVERED_DELIVERY_COMPLETE.json"}
    return {
        str(path.relative_to(root)): DRIVER.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-training-source", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--python", default=DRIVER.PYTHON_DEFAULT)
    parser.add_argument("--deviation-note", required=True)
    parser.add_argument("--acknowledge-noncanonical-gpu-exclusivity", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not args.acknowledge_noncanonical_gpu_exclusivity:
        raise RuntimeError("explicit acknowledgement of the original GPU-exclusivity violation is required")
    if (root / "DELIVERY_COMPLETE.json").exists():
        raise RuntimeError("canonical delivery already exists; recovery must not rewrite it")
    recovered = root / "RECOVERED_DELIVERY_COMPLETE.json"
    if recovered.exists():
        raise FileExistsError(recovered)
    if not args.deviation_note.strip():
        raise ValueError("the recovery deviation note must be nonempty")

    started = time.time()
    recovery_source = DRIVER.require_clean_source()
    gpu_start = require_gpu1(1, args.gpu1_uuid, require_idle=True)
    validate_provenance(
        root, args.expected_training_source, args.expected_checkpoint_sha256.lower()
    )
    ranking, selection = validate_arm_records(root)
    selected = selection["selected"]
    holdout = validate_holdout(root, selected)
    confirmation = root / "confirmation"
    for filename in (
        "report.png", "report.pdf", "selected_raw_m50_gallery.png",
        "selected_raw_m50_gallery.pdf",
    ):
        require_root_copy(root, confirmation, filename)
    require_png(root / "report.png")
    require_png(root / "selected_raw_m50_gallery.png")

    selected_run = root / "arms" / selected["arm"]["arm_id"]
    diagnostic = root / "selected_training_diagnostic.png"
    recovery_logs = []
    if not diagnostic.exists():
        recovery_logs.append(run_to_atomic_output([
            args.python, str(DRIVER.DIAGNOSTICS), "--run", str(selected_run),
            "--out", "{OUTPUT}",
        ], diagnostic, root / "logs", "diagnostic", 1))
    diagnostic_meta = require_png(diagnostic)

    video = root / "selected_expansion.mp4"
    frame_count = expected_video_frames(selected_run)
    if not video.exists():
        recovery_logs.append(run_to_atomic_output([
            args.python, str(DRIVER.VIDEO), "--run", str(selected_run),
            "--out", "{OUTPUT}", "--dense-until", "10", "--every-after", "10",
        ], video, root / "logs", "video", 1))
    video_meta = require_video(video, frame_count)
    gpu_end = require_gpu1(1, args.gpu1_uuid, require_idle=True)

    inventory = artifact_inventory(root)
    payload = {
        "status": RECOVERY_STATUS,
        "canonical_under_original_frozen_protocol": False,
        "protocol_deviation": {
            "kind": "loss_of_declared_exclusive_dual_gpu_access",
            "note": args.deviation_note,
            "scientific_computation_retried": False,
            "training_or_evaluation_recomputed": False,
        },
        "training_source_commit": args.expected_training_source,
        "recovery_source_commit": recovery_source["commit"],
        "checkpoint_sha256": args.expected_checkpoint_sha256.lower(),
        "arms_authenticated": len(ranking),
        "selected_arm": selected["arm"],
        "selected_round": int(selected["best"]["round"]),
        "confirmation_complete_sha256": DRIVER.sha256_file(
            confirmation / "EVALUATION_COMPLETE.json"
        ),
        "presentation": {
            "diagnostic": diagnostic_meta,
            "video": video_meta,
            "recovery_logs": [str(path.relative_to(root)) for path in recovery_logs],
        },
        "gpu1_start": gpu_start,
        "gpu1_end": gpu_end,
        "gpu3_required_or_inspected_by_recovery": False,
        "recovery_elapsed_seconds": time.time() - started,
        "original_provenance_sha256": DRIVER.sha256_file(root / "provenance.json"),
        "holdout_status": holdout["complete"]["status"],
        "artifact_sha256": inventory,
    }
    DRIVER.write_json_new(recovered, payload)
    print(f"LOW7 B1 RECOVERED DELIVERY COMPLETE: {recovered}", flush=True)


if __name__ == "__main__":
    main()
