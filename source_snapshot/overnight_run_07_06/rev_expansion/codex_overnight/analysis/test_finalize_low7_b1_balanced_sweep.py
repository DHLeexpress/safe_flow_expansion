from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import finalize_low7_b1_balanced_sweep as FINALIZE


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_recovery_checks_only_physical_gpu1(monkeypatch) -> None:
    calls = []

    def fake(index, uuid, *, require_idle):
        calls.append((index, uuid, require_idle))
        return {"physical_index": index, "uuid": uuid}

    monkeypatch.setattr(FINALIZE.DRIVER, "gpu_record", fake)
    assert FINALIZE.require_gpu1(1, "gpu-one", require_idle=True)["physical_index"] == 1
    assert calls == [(1, "gpu-one", True)]
    with pytest.raises(ValueError, match="physical GPU 1"):
        FINALIZE.require_gpu1(3, "gpu-three", require_idle=True)
    assert calls == [(1, "gpu-one", True)]


def test_expected_video_frames_uses_compact_recipe_without_round0(tmp_path) -> None:
    write_json(tmp_path / "recipe.json", {
        "rounds": 20, "artifact_profile": "sweep_compact",
    })
    viz = tmp_path / "viz_db"
    viz.mkdir()
    for round_i in (*range(1, 11), 20):
        (viz / f"round{round_i}.pt").touch()
    assert FINALIZE.expected_video_frames(tmp_path) == 11


def test_root_presentation_is_copy_or_verify_never_overwrite(tmp_path) -> None:
    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    (confirmation / "report.pdf").write_bytes(b"authenticated")
    FINALIZE.require_root_copy(tmp_path, confirmation, "report.pdf")
    assert (tmp_path / "report.pdf").read_bytes() == b"authenticated"
    (tmp_path / "report.pdf").write_bytes(b"conflict")
    with pytest.raises(RuntimeError, match="conflicts"):
        FINALIZE.require_root_copy(tmp_path, confirmation, "report.pdf")
    assert (tmp_path / "report.pdf").read_bytes() == b"conflict"


def test_png_header_validation(tmp_path) -> None:
    png = tmp_path / "plot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 640, 480))
    assert FINALIZE.require_png(png)["width"] == 640
    png.write_bytes(b"not a png")
    with pytest.raises(RuntimeError, match="not a PNG"):
        FINALIZE.require_png(png)


def test_arm_inventory_and_ranking_are_recomputed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(FINALIZE.DRIVER, "require_run", lambda *args: {})
    monkeypatch.setattr(FINALIZE.DRIVER, "require_evaluation", lambda *args: {})
    records = []
    for index, arm in enumerate(FINALIZE.DRIVER.ARMS):
        paths = FINALIZE.DRIVER.arm_paths(tmp_path, arm)
        paths["run"].mkdir(parents=True)
        paths["screen"].mkdir(parents=True)
        best = {
            "J": index / 100.0, "SR": 0.5, "CR": 0.5, "timeout": 0.0,
            "minimum_clearance": 0.01, "round": 10,
        }
        record = {
            "status": "ARM_COMPLETE", "arm": arm.record(), "gpu": 1,
            "elapsed_seconds": 1.0, "best": best,
            "run": str(paths["run"]), "screening": str(paths["screen"]),
        }
        write_json(paths["status"], record)
        write_json(paths["screen"] / "selection.json", {"ranking": [best]})
        records.append(record)
    ranking = sorted(records, key=FINALIZE.DRIVER.global_key)
    write_json(tmp_path / "selection.json", {
        "ranking": ranking, "selected": ranking[0],
    })
    with (tmp_path / "screening_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm_id",))
        writer.writeheader()
        writer.writerows({"arm_id": row["arm"]["arm_id"]} for row in ranking)

    observed, selection = FINALIZE.validate_arm_records(tmp_path)
    assert observed == ranking
    assert selection["selected"] == ranking[0]
    write_json(tmp_path / "arm_status" / "unexpected.json", {})
    with pytest.raises(RuntimeError, match="arm-status inventory"):
        FINALIZE.validate_arm_records(tmp_path)


def test_video_probe_requires_h264_and_exact_frames(monkeypatch, tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(FINALIZE.subprocess, "check_output", lambda *args, **kwargs: json.dumps({
        "streams": [{
            "codec_name": "h264", "width": 1994, "height": 1008,
            "nb_read_frames": "11",
        }]
    }))
    assert FINALIZE.require_video(video, 11)["frames"] == 11
    with pytest.raises(RuntimeError, match="expected 12-frame"):
        FINALIZE.require_video(video, 12)
