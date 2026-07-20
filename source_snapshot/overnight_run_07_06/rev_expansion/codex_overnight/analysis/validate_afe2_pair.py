"""Validate that two AFE2 runs differ only in the declared update arm."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from matplotlib import image as mpl_image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from afe2_scene_profiles import assert_scene_snapshot
import afe2_calibration as BC


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path):
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_png(path: Path) -> None:
    try:
        pixels = np.asarray(mpl_image.imread(path))
    except Exception as exc:
        raise RuntimeError(f"report is not a decodable image: {path}") from exc
    if pixels.ndim not in (2, 3) or min(pixels.shape[:2]) < 100:
        raise RuntimeError(f"report image has invalid dimensions: {path}={pixels.shape}")


def _validate_mp4(path: Path, expected_frames: int = 10) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate delivered videos")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"video does not have exactly one video stream: {path}")
    stream = streams[0]
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", 0)) <= 0
        or int(stream.get("height", 0)) <= 0
        or int(stream.get("nb_read_frames", -1)) != expected_frames
    ):
        raise RuntimeError(f"video decode/frame-count validation failed: {path}={stream}")


def _rounds(path: Path) -> list[int]:
    return [
        json.loads(line, parse_constant=_reject_constant)["round"]
        for line in path.read_text().splitlines()
        if line
    ]


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in path.read_text().splitlines()
        if line
    ]


def _common_recipe(recipe: dict) -> dict:
    common = copy.deepcopy(recipe)
    common.pop("arm", None)
    common.pop("update", None)
    return common


def _validate_beta_calibration(recipe: dict) -> None:
    calibration = recipe.get("beta_calibration") or {}
    if calibration.get("checkpoint_contract") != recipe.get(
        "source_checkpoint_contract"
    ):
        raise RuntimeError("beta calibration embeds a different checkpoint contract")
    expected_fields = {
        "checkpoint_sha256": recipe.get("source_checkpoint_sha256"),
        "checkpoint_model_sha256": recipe.get("source_checkpoint_model_sha256"),
        "checkpoint_contract_sha256": recipe.get(
            "source_checkpoint_contract_sha256"
        ),
        "scene_sha256": recipe.get("scene", {}).get("sha256"),
        "source_git_commit": recipe.get("source_git_commit"),
        "lam": recipe.get("lam"),
        "K": recipe.get("K"),
        "B": recipe.get("B"),
        "seed": recipe.get("seed"),
    }
    try:
        chosen = BC.validate_success(calibration, expected_fields)
    except ValueError as exc:
        raise RuntimeError("locked run has an invalid continuous-ESS beta calibration") from exc
    if float(recipe.get("beta")) != chosen:
        raise RuntimeError("run beta is not the declared calibration choice")
    digest = str(recipe.get("beta_calibration_sha256") or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("run has no valid beta calibration artifact digest")


def _validate_checkpoint_contract(recipe: dict) -> None:
    contract = recipe.get("source_checkpoint_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("run has no profile-bound checkpoint eligibility contract")
    digest = recipe.get("source_checkpoint_contract_sha256")
    if digest != _canonical_json_sha256(contract):
        raise RuntimeError("run checkpoint eligibility contract digest mismatch")
    if contract.get("checkpoint_file_sha256") != recipe.get("source_checkpoint_sha256"):
        raise RuntimeError("run checkpoint contract has the wrong file identity")
    if contract.get("checkpoint_model_state_sha256") != recipe.get(
        "source_checkpoint_model_sha256"
    ):
        raise RuntimeError("run checkpoint contract has the wrong model-state identity")


def _validate_viz_round(db, recipe, scene, q_y, run_name: str, round_i: int) -> None:
    """Validate one promoted seven-gamma visualization database semantically."""

    prefix = f"{run_name} viz round {round_i}"
    if int(db.get("round", -1)) != round_i:
        raise RuntimeError(f"{prefix} identity mismatch")
    db_scene = db.get("scene")
    if db_scene is None:
        raise RuntimeError(f"{prefix} has no serialized scene")
    assert_scene_snapshot(db_scene)
    if db_scene["sha256"] != scene["sha256"]:
        raise RuntimeError(f"{prefix} has the wrong scene")
    if not np.array_equal(np.asarray(db["goal"]), np.asarray(scene["goal"])):
        raise RuntimeError(f"{prefix} goal mismatch")
    if not np.array_equal(np.asarray(db["x0"])[:2], np.asarray(scene["start_state"])[:2]):
        raise RuntimeError(f"{prefix} start mismatch")

    expected_gammas = [float(value) for value in recipe["reference_recipe"]["gammas"]]
    gamma_keys = {round(value, 8) for value in expected_gammas}
    allowed_status = {"reached", "nvp", "timeout", "collision", "oob"}
    episodes = list(db.get("eps") or [])
    episode_gammas = [round(float(row["gamma"]), 8) for row in episodes]
    if len(episodes) != len(expected_gammas) or set(episode_gammas) != gamma_keys:
        raise RuntimeError(f"{prefix} lacks the exact gamma sweep")
    viz_rows = list(db.get("viz") or [])
    if any(round(float(row.get("gamma", float("nan"))), 8) not in gamma_keys for row in viz_rows):
        raise RuntimeError(f"{prefix} has an undeclared gamma")

    K = int(recipe["reference_recipe"]["K"])
    B = int(recipe["reference_recipe"]["B"])
    for gamma in expected_gammas:
        key = round(gamma, 8)
        episode = next(row for row in episodes if round(float(row["gamma"]), 8) == key)
        if episode.get("status") not in allowed_status:
            raise RuntimeError(f"{prefix} gamma {gamma} has invalid status")
        path = np.asarray(episode.get("path"), dtype=float)
        if (
            path.ndim != 2
            or path.shape[1] != 2
            or len(path) < 1
            or not np.isfinite(path).all()
            or not np.allclose(path[0], np.asarray(scene["start_state"])[:2])
            or int(episode.get("steps", -1)) != len(path) - 1
        ):
            raise RuntimeError(f"{prefix} gamma {gamma} has an invalid path")
        steps = [row for row in viz_rows if round(float(row["gamma"]), 8) == key]
        if not steps or sorted(int(row["t"]) for row in steps) != list(range(len(steps))):
            raise RuntimeError(f"{prefix} gamma {gamma} has an incomplete step stream")
        expected_step_rows = int(episode["steps"]) + int(episode["status"] == "nvp")
        if len(steps) != expected_step_rows:
            raise RuntimeError(f"{prefix} gamma {gamma} step/path mismatch")
        for step in steps:
            segments = np.asarray(step.get("segsK"))
            drawn = [int(value) for value in step.get("drawn", [])]
            if (
                segments.shape != (K, 10, 2)
                or not np.isfinite(segments).all()
                or len(drawn) != B
                or len(set(drawn)) != B
                or min(drawn) < 0
                or max(drawn) >= K
                or len(step.get("y", [])) != B
                or len(step.get("exec_y", [])) != B
                or len(step.get("terminal_rescue", [])) != B
                or len(step.get("terminal_tau", [])) != B
                or int(step.get("sel", -1)) not in {-1, *drawn}
            ):
                raise RuntimeError(f"{prefix} gamma {gamma} violates K/B semantics")
        if episode["status"] == "nvp" and int(steps[-1].get("sel", -2)) != -1:
            raise RuntimeError(f"{prefix} gamma {gamma} NVP has a selected plan")

    train_ids = np.asarray(db.get("train_ids", []), dtype=np.int64)
    if train_ids.size and (
        train_ids.min() < 0
        or train_ids.max() >= len(q_y)
        or not bool(np.all(q_y[train_ids] == 1))
    ):
        raise RuntimeError(f"{prefix} trains outside full-H D+")


def _validate_complete_identity(complete: dict, recipe: dict, run_name: str) -> None:
    if complete.get("scene_sha256") != recipe.get("scene", {}).get("sha256"):
        raise RuntimeError(f"{run_name} completion marker has the wrong scene")
    if complete.get("checkpoint_sha256") != recipe.get("source_checkpoint_sha256"):
        raise RuntimeError(f"{run_name} completion marker has the wrong checkpoint")
    if complete.get("source_git_commit") != recipe.get("source_git_commit"):
        raise RuntimeError(f"{run_name} completion marker has the wrong source commit")
    if complete.get("checkpoint_model_sha256") != recipe.get(
        "source_checkpoint_model_sha256"
    ):
        raise RuntimeError(f"{run_name} completion marker has the wrong model-state hash")
    if complete.get("checkpoint_contract_sha256") != recipe.get(
        "source_checkpoint_contract_sha256"
    ):
        raise RuntimeError(f"{run_name} completion marker has the wrong checkpoint contract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox", type=Path, required=True)
    parser.add_argument("--scene-profile", required=True,
                        help="expected scene profile name for this pair (explicit, no default)")
    parser.add_argument("--afe", type=Path, required=True)
    parser.add_argument("--beta-calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--prox-video", type=Path)
    parser.add_argument("--afe-video", type=Path)
    parser.add_argument("--delivery-out", type=Path)
    args = parser.parse_args()
    expected_profile = str(args.scene_profile)

    roots = {"prox": args.prox.resolve(), "afe": args.afe.resolve()}
    recipes = {name: _load_json(root / "recipe.json") for name, root in roots.items()}
    if recipes["prox"].get("arm") != "prox" or recipes["afe"].get("arm") != "afe":
        raise RuntimeError("run directories do not contain the expected prox/afe arms")
    if not all(recipe.get("reference_recipe_locked") is True for recipe in recipes.values()):
        raise RuntimeError("both runs must enforce --lock-reference-recipe")
    if _common_recipe(recipes["prox"]) != _common_recipe(recipes["afe"]):
        raise RuntimeError("AFE2 pair differs outside the declared update arm")
    for recipe in recipes.values():
        _validate_checkpoint_contract(recipe)
        _validate_beta_calibration(recipe)
    calibration_path = args.beta_calibration.resolve()
    calibration = _load_json(calibration_path)
    calibration_sha256 = _sha256(calibration_path)
    for name, recipe in recipes.items():
        if calibration != recipe.get("beta_calibration"):
            raise RuntimeError(f"{name} recipe does not embed the supplied beta calibration")
        if calibration_sha256 != recipe.get("beta_calibration_sha256"):
            raise RuntimeError(f"{name} beta calibration artifact hash mismatch")
    if any(recipe.get("source_git_tracked_dirty") is not False for recipe in recipes.values()):
        raise RuntimeError("both runs must come from a tracked-clean source tree")
    if any(recipe.get("source_git_untracked_runtime_sources") != [] for recipe in recipes.values()):
        raise RuntimeError("both runs must come from committed runtime source files")
    if any(not recipe.get("source_git_commit") for recipe in recipes.values()):
        raise RuntimeError("both runs must record their exact source commit")

    expected_rounds = list(range(11))
    probes = {name: root / "probe.jsonl" for name, root in roots.items()}
    for name, path in probes.items():
        observed = _rounds(path)
        if observed != expected_rounds:
            raise RuntimeError(
                f"{name} probe rounds are {observed}; expected {expected_rounds}"
            )
        if (roots[name] / "INCOMPLETE.json").exists():
            raise RuntimeError(f"{name} run is explicitly marked INCOMPLETE")
        complete_path = roots[name] / "COMPLETE.json"
        if not complete_path.is_file():
            raise RuntimeError(f"{name} run lacks trainer-written COMPLETE.json")
        complete = _load_json(complete_path)
        if complete.get("status") != "COMPLETE" or complete.get("completed_round") != 10:
            raise RuntimeError(f"{name} COMPLETE.json is not a ten-round completion marker")
        for relative, expected_hash in complete.get("artifact_sha256", {}).items():
            artifact = roots[name] / relative
            if not artifact.is_file() or _sha256(artifact) != expected_hash:
                raise RuntimeError(f"{name} completion artifact mismatch: {relative}")
        required = {
            "recipe.json",
            "probe.jsonl",
            "final.pt",
            "dstore.pt",
            *{f"ckpt_{round_i}.pt" for round_i in expected_rounds},
            *{f"viz_db/round{round_i}.pt" for round_i in expected_rounds[1:]},
        }
        if set(complete.get("artifact_sha256", {})) != required:
            raise RuntimeError(f"{name} COMPLETE.json has an incomplete artifact inventory")

    scene = recipes["prox"]["scene"]
    assert_scene_snapshot(scene)
    if scene["profile"]["name"] != expected_profile:
        raise RuntimeError(
            f"pair scene profile {scene['profile']['name']!r} != expected {expected_profile!r}")
    for name, root in roots.items():
        complete = _load_json(root / "COMPLETE.json")
        _validate_complete_identity(complete, recipes[name], name)
        dstore = torch.load(root / "dstore.pt", map_location="cpu", weights_only=False)
        q_y = np.asarray(dstore["q_y"], dtype=np.int8)
        q_exec = np.asarray(dstore["q_exec"], dtype=np.int8)
        q_exec_y = np.asarray(dstore["q_exec_y"], dtype=np.int8)
        q_rescue = np.asarray(dstore["q_terminal_rescue"], dtype=np.int8)
        q_tau = np.asarray(dstore["q_terminal_tau"], dtype=np.int16)
        q_exec_margin = np.asarray(dstore["q_exec_margin"], dtype=float)
        q_terminal_reason = list(dstore["q_terminal_reason"])
        last_probe = _records(root / "probe.jsonl")[-1]
        if len(q_y) != int(last_probe["n_D"]) or int(q_y.sum()) != int(last_probe["n_Dpos"]):
            raise RuntimeError(f"{name} dstore counts disagree with the final probe")
        executed = np.flatnonzero(q_exec == 1)
        for qid in executed:
            prefix_witness = (
                q_exec_y[qid] == 1
                and q_rescue[qid] == 1
                and q_tau[qid] >= 1
                and q_terminal_reason[qid] == "ok"
                and np.isfinite(q_exec_margin[qid])
                and q_exec_margin[qid] > 0.0
            )
            if q_y[qid] != 1 and not prefix_witness:
                raise RuntimeError(f"{name} executed query {qid} has no certificate witness")
        for round_i in expected_rounds[1:]:
            db = torch.load(
                root / "viz_db" / f"round{round_i}.pt",
                map_location="cpu",
                weights_only=False,
            )
            _validate_viz_round(db, recipes[name], scene, q_y, name, round_i)

    manifest = {
        "status": "VALIDATED_MATCHED_AFE2_PAIR",
        "scene_profile": scene["profile"]["name"],
        "scene_sha256": scene["sha256"],
        "source_checkpoint_sha256": recipes["prox"]["source_checkpoint_sha256"],
        "source_checkpoint_model_sha256": recipes["prox"][
            "source_checkpoint_model_sha256"
        ],
        "source_checkpoint_contract_sha256": recipes["prox"][
            "source_checkpoint_contract_sha256"
        ],
        "source_git_commit": recipes["prox"].get("source_git_commit"),
        "reference_recipe": recipes["prox"]["reference_recipe"],
        "beta_calibration": {
            "path": str(calibration_path),
            "sha256": calibration_sha256,
            "chosen": calibration["chosen"],
            "ess_target": calibration["ess_target"],
            "solver": calibration["solver"],
            "sigma_pool_sha256": calibration["sigma_pool_sha256"],
        },
        "runs": {
            name: {
                "root": str(root),
                "recipe_sha256": _sha256(root / "recipe.json"),
                "probe_sha256": _sha256(root / "probe.jsonl"),
                "complete_sha256": _sha256(root / "COMPLETE.json"),
                "rounds": expected_rounds,
            }
            for name, root in roots.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"validated matched AFE2 {scene['profile']['name']} pair: {args.out}")
    delivery_values = (args.report, args.prox_video, args.afe_video, args.delivery_out)
    if any(value is not None for value in delivery_values):
        if not all(value is not None for value in delivery_values):
            raise RuntimeError("report, both videos, and delivery-out must be supplied together")
        artifacts = {
            "beta_calibration": calibration_path,
            "pair_manifest": args.out.resolve(),
            "report": args.report.resolve(),
            "prox_video": args.prox_video.resolve(),
            "afe_video": args.afe_video.resolve(),
        }
        for name, path in artifacts.items():
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"delivery artifact missing or empty: {name}={path}")
        _validate_png(artifacts["report"])
        _validate_mp4(artifacts["prox_video"])
        _validate_mp4(artifacts["afe_video"])
        delivery = {
            "status": "DELIVERY_COMPLETE",
            "scene_sha256": scene["sha256"],
            "source_checkpoint_sha256": recipes["prox"]["source_checkpoint_sha256"],
            "source_checkpoint_model_sha256": recipes["prox"][
                "source_checkpoint_model_sha256"
            ],
            "source_checkpoint_contract_sha256": recipes["prox"][
                "source_checkpoint_contract_sha256"
            ],
            "source_git_commit": recipes["prox"]["source_git_commit"],
            "artifacts": {
                name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
                for name, path in artifacts.items()
            },
        }
        args.delivery_out.parent.mkdir(parents=True, exist_ok=True)
        args.delivery_out.write_text(json.dumps(delivery, indent=2, sort_keys=True) + "\n")
        print(f"validated complete AFE2 delivery: {args.delivery_out}")


if __name__ == "__main__":
    main()
