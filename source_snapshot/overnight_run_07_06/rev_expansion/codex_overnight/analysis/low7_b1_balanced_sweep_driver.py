#!/usr/bin/env python3
"""Fail-closed two-GPU B1 sweep from a disjoint-qualified balanced r0."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "grid_expand_afe_rbf.py"
EVALUATOR = ROOT / "paper_results" / "low7_support_sweep_eval.py"
DIAGNOSTICS = ROOT / "analysis" / "afe_rbf_sweep_diagnostics.py"
VIDEO = ROOT / "video_afe2.py"
SCENE = "low7_radius1_canonical_v1"
PYTHON_DEFAULT = "/home/dohyun/miniforge3/envs/cfm_mppi/bin/python"
SEED = 910


@dataclass(frozen=True)
class Arm:
    gp_cap: int
    ess_target: float
    alpha: float
    execution_rule: str

    @property
    def arm_id(self) -> str:
        ess = f"{int(round(100 * self.ess_target)):03d}"
        alpha = {0.0: "0000", 0.001: "0001", 0.01: "0010"}[self.alpha]
        execution = (
            "cost" if self.execution_rule == "nominal_hp_safemppi_cost" else "margin"
        )
        return f"cap{self.gp_cap}_ess{ess}_alpha{alpha}_{execution}"

    def record(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "gp_cap": self.gp_cap,
            "adaptive_ess_target": self.ess_target,
            "negative_alpha": self.alpha,
            "execution_rule": self.execution_rule,
        }


ARMS = tuple(
    Arm(cap, ess, alpha, execution)
    for cap, ess, alpha, execution in itertools.product(
        (512, 768),
        (0.25, 0.5),
        (0.0, 0.001, 0.01),
        ("nominal_hp_max_step_margin", "nominal_hp_safemppi_cost"),
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def write_json_new(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require_clean_source() -> dict:
    repository = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True
    ).strip())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=repository).returncode != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repository
        ).returncode != 0
    )
    runtime_untracked = [
        value
        for value in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repository,
            text=True,
        ).splitlines()
        if value.endswith((".py", ".sh"))
    ]
    if dirty or runtime_untracked:
        raise RuntimeError(
            f"B1 sweep requires committed clean source: dirty={dirty}, "
            f"runtime_untracked={runtime_untracked}"
        )
    return {"repository": str(repository), "commit": commit}


def gpu_record(index: int, expected_uuid: str, *, require_idle: bool) -> dict:
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [value.strip() for value in line.split(",")]
    if len(fields) != 6 or fields[0] != str(index):
        raise RuntimeError(f"malformed GPU record: {line}")
    if fields[1].lower() != expected_uuid.lower():
        raise RuntimeError(f"GPU {index} UUID mismatch: {line}")
    pids = subprocess.check_output([
        "nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if require_idle and pids:
        raise RuntimeError(f"GPU {index} has foreign compute PIDs: {pids}")
    return {
        "physical_index": index,
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "active_compute_pids": pids,
    }


def qualified_checkpoint(delivery_path: Path) -> tuple[Path, str, dict]:
    delivery_path = delivery_path.resolve()
    delivery = load_json(delivery_path)
    if (
        delivery.get("status") != "LOW7_BALANCED_R0_DELIVERY_COMPLETE"
        or delivery.get("confirmation_passed") is not True
    ):
        raise RuntimeError("balanced-r0 delivery did not pass disjoint confirmation")
    selected = delivery.get("selected") or {}
    checkpoint = Path(selected.get("checkpoint", "")).resolve()
    expected = str(selected.get("checkpoint_sha256", "")).lower()
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise RuntimeError("balanced-r0 selected checkpoint linkage failed")
    confirmation_path = Path(delivery.get("confirmation", "")).resolve()
    confirmation = load_json(confirmation_path)
    if not confirmation.get("passed") or int(confirmation.get("M_per_gamma", -1)) != 100:
        raise RuntimeError("balanced-r0 confirmation is not the declared M=100/gamma gate")
    if confirmation.get("raw_noise_design") != (
        "reflection-antithetic common-random-number pairs"
    ):
        raise RuntimeError("balanced-r0 confirmation lacks the exact symmetry noise design")
    if confirmation.get("checkpoint", {}).get("file_sha256") != expected:
        raise RuntimeError("balanced-r0 confirmation/checkpoint SHA mismatch")
    per_gamma = confirmation.get("per_gamma", {})
    if len(per_gamma) != 7:
        raise RuntimeError("balanced-r0 confirmation lacks seven gamma cells")
    for gamma, row in per_gamma.items():
        routes = row.get("all_routes", {})
        successful_routes = row.get("successful_routes", {})
        route_interval = routes.get("u_fraction_wilson95", (-1.0, -1.0))
        successful_interval = successful_routes.get(
            "u_fraction_wilson95", (-1.0, -1.0)
        )
        if float(routes.get("balance", -1.0)) < 0.8:
            raise RuntimeError(f"balanced-r0 gamma {gamma} failed route balance")
        if float(routes.get("resolved_fraction", -1.0)) < 0.95:
            raise RuntimeError(f"balanced-r0 gamma {gamma} failed route resolution")
        if int(row.get("success_count", -1)) < 10:
            raise RuntimeError(f"balanced-r0 gamma {gamma} has too few successes")
        if float(successful_routes.get("balance", -1.0)) < 0.8:
            raise RuntimeError(
                f"balanced-r0 gamma {gamma} failed successful-route balance"
            )
        if float(successful_routes.get("resolved_fraction", -1.0)) < 0.95:
            raise RuntimeError(
                f"balanced-r0 gamma {gamma} failed successful-route resolution"
            )
        if not (float(route_interval[0]) <= 0.5 <= float(route_interval[1])):
            raise RuntimeError(f"balanced-r0 gamma {gamma} rejects equal all-route mass")
        if not (
            float(successful_interval[0]) <= 0.5 <= float(successful_interval[1])
        ):
            raise RuntimeError(
                f"balanced-r0 gamma {gamma} rejects equal successful-route mass"
            )
    return checkpoint, expected, {
        "delivery": str(delivery_path),
        "delivery_sha256": sha256_file(delivery_path),
        "confirmation": str(confirmation_path),
        "confirmation_sha256": sha256_file(confirmation_path),
        "selected": selected,
    }


def command_env(gpu_index: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return env


def run_logged(command: list[str], log_path: Path, gpu_index: int) -> None:
    with log_path.open("x") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=command_env(gpu_index),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def trainer_command(args, arm: Arm, run_dir: Path, *, preflight: bool) -> list[str]:
    return [
        args.python, str(TRAINER),
        "--protocol-profile", (
            "b1_balanced_r0_preflight" if preflight else "b1_balanced_r0_sweep"
        ),
        "--ckpt", str(args.checkpoint),
        "--expected-ckpt-sha256", args.checkpoint_sha256,
        "--balanced-r0-delivery", str(args.pretrain_delivery.resolve()),
        "--scene-profile", SCENE,
        "--outdir", str(run_dir),
        "--rounds", "1" if preflight else "20",
        "--rollout-replicas", "8", "--K", "16", "--B", "4", "--T", "300",
        "--M-eval", "0", "--batch", "128", "--afe-steps", "0",
        "--afe-lr", "1e-5", "--gp-cap", str(arm.gp_cap), "--gp-lam", "1e-2",
        "--acquisition-mode", "sequential",
        "--adaptive-ess-target", f"{arm.ess_target:g}",
        "--adaptive-beta-contexts-per-gamma", "64",
        "--adaptive-beta-equalize-gammas",
        "--replay-window", "2",
        "--replay-sampling", "round_gamma_replica_context",
        "--replay-update-mode", "one_epoch_without_replacement",
        "--replay-loss-weighting", "gamma_episode_context_query_equal_mass",
        "--gp-replay-window", "2",
        "--gp-replay-sampling", "round_gamma_replica_context",
        "--lengthscale-multiplier", "1.0",
        "--negative-alpha", f"{arm.alpha:g}",
        "--execution-rule", arm.execution_rule,
        "--conditioning-schema", "low7_closest_boundary_tie_mean",
        "--freeze-visual-encoder", "--skip-training-probes",
        "--calibration-replicas", "8", "--calibration-control-steps", "4",
        "--sweep-compact-artifacts", "--compact-checkpoint-every", "1",
        "--route-metric-steps", "10", "--nvp-audit-all-k",
        "--verifier-workers", str(args.verifier_workers), "--seed", str(SEED),
    ]


def require_run(run_dir: Path, arm: Arm, rounds: int) -> dict:
    complete = load_json(run_dir / "COMPLETE.json")
    if complete.get("status") != "COMPLETE" or int(complete.get("completed_round", -1)) != rounds:
        raise RuntimeError(f"B1 trainer completion failed: {run_dir}")
    recipe = load_json(run_dir / "recipe.json")
    exact = {
        "algorithm": "afe_rbf_low7_b1_balanced_r0_sweep_v1",
        "rounds": rounds,
        "rollout_replicas": 8,
        "K": 16,
        "B": 4,
        "gp_cap": arm.gp_cap,
        "adaptive_ess_target": arm.ess_target,
        "negative_alpha": arm.alpha,
        "execution_rule": arm.execution_rule,
        "replay_window": 2,
        "gp_replay_window": 2,
        "replay_update_mode": "one_epoch_without_replacement",
        "replay_loss_weighting": "gamma_episode_context_query_equal_mass",
    }
    mismatches = {
        key: (recipe.get(key), expected)
        for key, expected in exact.items() if recipe.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"B1 recipe mismatch: {mismatches}")
    rows = [json.loads(line) for line in (run_dir / "probe.jsonl").read_text().splitlines()]
    if [int(row["round"]) for row in rows] != list(range(rounds + 1)):
        raise RuntimeError("B1 probe rounds are not contiguous")
    for row in rows[1:]:
        if float(row.get("replay_epoch_coverage", -1.0)) != 1.0:
            raise RuntimeError("B1 did not use every eligible W2 positive")
        if int(row.get("replay_duplicate_draws", -1)) != 0:
            raise RuntimeError("B1 duplicated a positive replay query")
        if int(row.get("optimizer_draws", -1)) != int(row.get("replay_eligible", -2)):
            raise RuntimeError("B1 positive optimizer draws differ from eligible D+")
        if float(row.get("rel_param_change", {}).get("E_g", -1.0)) != 0.0:
            raise RuntimeError("B1 frozen visual encoder changed")
        if arm.alpha > 0.0 and int(row.get("negative_replay_eligible", 0)) > 0:
            if row.get("signed_active") is not True:
                raise RuntimeError("B1 nonzero-alpha arm ignored available NVP evidence")
    return {"complete": complete, "recipe": recipe, "probe": rows}


def require_evaluation(outdir: Path, expected_status: str) -> dict:
    complete = load_json(outdir / "EVALUATION_COMPLETE.json")
    if complete.get("status") != expected_status:
        raise RuntimeError(
            f"B1 evaluation status {complete.get('status')!r} != {expected_status!r}"
        )
    for relative, expected in complete.get("artifact_sha256", {}).items():
        path = outdir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"B1 evaluation artifact hash failed: {path}")
    return complete


def arm_paths(out: Path, arm: Arm) -> dict[str, Path]:
    return {
        "run": out / "arms" / arm.arm_id,
        "screen": out / "screening" / arm.arm_id,
        "train_log": out / "logs" / f"{arm.arm_id}.train.log",
        "screen_log": out / "logs" / f"{arm.arm_id}.screen.log",
        "status": out / "arm_status" / f"{arm.arm_id}.json",
    }


def run_arm(args, arm: Arm, gpu_index: int) -> dict:
    paths = arm_paths(args.out, arm)
    started = time.time()
    run_logged(trainer_command(args, arm, paths["run"], preflight=False), paths["train_log"], gpu_index)
    require_run(paths["run"], arm, 20)
    run_logged([
        args.python, str(EVALUATOR), "--study", "b1", "--mode", "screen",
        "--run-root", str(paths["run"]), "--outdir", str(paths["screen"]),
        "--verifier-workers", str(args.verifier_workers),
    ], paths["screen_log"], gpu_index)
    require_evaluation(
        paths["screen"], "AFE_RBF_B1_BALANCED_SCREEN_DELIVERY_COMPLETE"
    )
    selection = load_json(paths["screen"] / "selection.json")
    record = {
        "status": "ARM_COMPLETE",
        "arm": arm.record(),
        "gpu": gpu_index,
        "elapsed_seconds": time.time() - started,
        "best": selection["ranking"][0],
        "run": str(paths["run"]),
        "screening": str(paths["screen"]),
    }
    write_json_new(paths["status"], record)
    return record


def run_queue(args, gpu_index: int, arms: list[Arm]) -> list[dict]:
    return [run_arm(args, arm, gpu_index) for arm in arms]


def global_key(record: dict):
    score = record["best"]
    return (
        -float(score["J"]),
        -float(score["SR"]),
        float(score["CR"]),
        float(score["timeout"]),
        -float(score["minimum_clearance"]),
        int(score["round"]),
        record["arm"]["arm_id"],
    )


def run_preflight(args) -> list[dict]:
    probes = (
        (1, Arm(512, 0.25, 0.01, "nominal_hp_safemppi_cost")),
        (3, Arm(768, 0.5, 0.0, "nominal_hp_max_step_margin")),
    )
    def one(gpu_index, arm):
        root = args.out / "preflight" / arm.arm_id
        root.mkdir()
        started = time.time()
        run_logged(
            trainer_command(args, arm, root / "run", preflight=True),
            root / "train.log",
            gpu_index,
        )
        require_run(root / "run", arm, 1)
        return {
            "gpu": gpu_index,
            "arm": arm.record(),
            "elapsed_seconds": time.time() - started,
            "scientific_reuse": False,
        }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, gpu, arm) for gpu, arm in probes]
        return [future.result() for future in futures]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pretrain-delivery", type=Path, required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--gpu3-uuid", required=True)
    parser.add_argument("--verifier-workers", type=int, default=48)
    parser.add_argument("--python", default=PYTHON_DEFAULT)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"B1 sweep output root must be absent: {args.out}")
    if args.verifier_workers < 1:
        raise ValueError("verifier worker count must be positive")
    cpu_count = os.cpu_count() or 0
    if 4 * args.verifier_workers > cpu_count - 16:
        raise RuntimeError(
            f"four co-scheduled arms need headroom on {cpu_count} CPUs; "
            f"requested {args.verifier_workers} workers each"
        )
    source = require_clean_source()
    args.checkpoint, args.checkpoint_sha256, pretrain = qualified_checkpoint(
        args.pretrain_delivery
    )
    gpus = {
        "1": gpu_record(1, args.gpu1_uuid, require_idle=True),
        "3": gpu_record(3, args.gpu3_uuid, require_idle=True),
    }
    for relative in ("preflight", "arms", "screening", "logs", "arm_status"):
        (args.out / relative).mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json_new(args.out / "provenance.json", {
        "source": source,
        "qualified_pretraining": pretrain,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "gpus": gpus,
        "host_cpu_count": cpu_count,
        "controlled_co_scheduling": "two independent arms per physical GPU",
        "verifier_workers_per_arm": args.verifier_workers,
        "arms": [arm.record() for arm in ARMS],
    })
    preflight = run_preflight(args)
    write_json_new(args.out / "preflight" / "COMPLETE.json", {
        "status": "PREFLIGHT_COMPLETE_NOT_REUSED", "runs": preflight
    })

    # Four independent queues: two on each H100.  This overlaps the CPU-bound
    # verifier phase of one arm with GPU sampling/update from another.
    ordered = sorted(ARMS, key=lambda arm: (-arm.gp_cap, arm.arm_id))
    slots: list[tuple[int, list[Arm]]] = [(1, []), (1, []), (3, []), (3, [])]
    for index, arm in enumerate(ordered):
        slots[index % len(slots)][1].append(arm)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_queue, args, gpu, queue) for gpu, queue in slots]
        records = [record for future in futures for record in future.result()]
    if len(records) != len(ARMS):
        raise RuntimeError("B1 sweep did not complete all 24 arms")
    selected = min(records, key=global_key)
    selected_arm = next(
        arm for arm in ARMS if arm.arm_id == selected["arm"]["arm_id"]
    )
    selected_round = int(selected["best"]["round"])

    table = sorted(records, key=global_key)
    fields = (
        "rank", "arm_id", "gp_cap", "adaptive_ess_target", "negative_alpha",
        "execution_rule", "best_round", "J", "SR", "CR", "timeout",
        "minimum_clearance", "elapsed_seconds",
    )
    with (args.out / "screening_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(table, 1):
            writer.writerow({
                "rank": rank,
                **{key: value for key, value in record["arm"].items() if key != "arm_id"},
                "arm_id": record["arm"]["arm_id"],
                "best_round": record["best"]["round"],
                **{key: record["best"][key] for key in (
                    "J", "SR", "CR", "timeout", "minimum_clearance"
                )},
                "elapsed_seconds": record["elapsed_seconds"],
            })
    write_json_new(args.out / "selection.json", {
        "rule": "maximize J=mean_gamma 2*min(success_U,success_R)/M; then SR, CR, timeout, clearance, earlier round",
        "selected": selected,
        "ranking": table,
    })

    selected_paths = arm_paths(args.out, selected_arm)
    holdout = args.out / "confirmation"
    run_logged([
        args.python, str(EVALUATOR), "--study", "b1", "--mode", "holdout",
        "--run-root", str(selected_paths["run"]), "--outdir", str(holdout),
        "--selected-round", str(selected_round),
        "--verifier-workers", str(min(128, cpu_count - 16)),
    ], args.out / "logs" / "confirmation.log", 1)
    require_evaluation(
        holdout, "AFE_RBF_B1_BALANCED_HOLDOUT_DELIVERY_COMPLETE"
    )
    for filename in (
        "report.png", "report.pdf", "selected_raw_m50_gallery.png",
        "selected_raw_m50_gallery.pdf",
    ):
        shutil.copy2(holdout / filename, args.out / filename)
    run_logged([
        args.python, str(DIAGNOSTICS), "--run", str(selected_paths["run"]),
        "--out", str(args.out / "selected_training_diagnostic.png"),
    ], args.out / "logs" / "diagnostic.log", 1)
    run_logged([
        args.python, str(VIDEO), "--run", str(selected_paths["run"]),
        "--out", str(args.out / "selected_expansion.mp4"),
        "--dense-until", "10", "--every-after", "10",
    ], args.out / "logs" / "video.log", 1)

    write_json_new(args.out / "SWEEP_COMPLETE.json", {
        "status": "LOW7_B1_BALANCED_R0_SWEEP_COMPLETE",
        "source_commit": source["commit"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "arms_completed": len(records),
        "selected_arm": selected_arm.record(),
        "selected_round": selected_round,
        "elapsed_seconds": time.time() - started,
    })
    artifact_hashes = {
        str(path.relative_to(args.out)): sha256_file(path)
        for path in sorted(args.out.rglob("*"))
        if path.is_file() and path.name != "DELIVERY_COMPLETE.json"
    }
    write_json_new(args.out / "DELIVERY_COMPLETE.json", {
        "status": "LOW7_B1_BALANCED_R0_DELIVERY_COMPLETE",
        "source_commit": source["commit"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "arms_completed": len(records),
        "selected_arm": selected_arm.record(),
        "selected_round": selected_round,
        "artifact_sha256": artifact_hashes,
    })
    print(f"LOW7 B1 BALANCED-R0 SWEEP COMPLETE: {args.out}", flush=True)


if __name__ == "__main__":
    main()
