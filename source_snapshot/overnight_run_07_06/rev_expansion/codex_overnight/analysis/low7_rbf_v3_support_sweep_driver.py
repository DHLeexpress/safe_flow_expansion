#!/usr/bin/env python3
"""Two-GPU six-arm optimizer-dose/demo-support qualification driver."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "grid_expand_afe_rbf.py"
EVALUATOR = ROOT / "paper_results" / "low7_support_sweep_eval.py"
DIAGNOSTICS = ROOT / "analysis" / "afe_rbf_sweep_diagnostics.py"
VIDEO = ROOT / "video_afe2.py"
SCENE = "low7_radius1_canonical_v1"
SOURCE_RECIPE = Path(
    "/home/dohyun/projects/afe2_runs/low7_rbf_v2_lineage_mass_giant_dad39e6/run/recipe.json"
)
REFERENCE_EVALUATION = Path(
    "/home/dohyun/projects/afe2_runs/low7_rbf_v2_lineage_mass_giant_dad39e6/evaluation"
)
PYTHON_DEFAULT = "/home/dohyun/miniforge3/envs/cfm_mppi/bin/python"
SEED = 910


class Arm:
    def __init__(self, steps: int, demo_frac: float):
        self.steps = int(steps)
        self.demo_frac = float(demo_frac)

    @property
    def arm_id(self):
        demo = {0.0: "000", 0.125: "0125", 0.25: "0250"}[self.demo_frac]
        return f"opt{self.steps:03d}_demo{demo}"

    def record(self):
        return {
            "arm_id": self.arm_id,
            "optimizer_steps_per_round": self.steps,
            "demo_frac_objective_mass": self.demo_frac,
            "rollout_replicas": 8,
        }


GPU_QUEUES = {
    1: [Arm(16, 0.0), Arm(16, 0.25), Arm(32, 0.125)],
    3: [Arm(16, 0.125), Arm(32, 0.0), Arm(32, 0.25)],
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


def write_json_new(path: Path, value: Any):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_record():
    repository = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True
    ).strip())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=repository).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repository).returncode != 0
    )
    runtime_untracked = [
        value for value in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repository, text=True,
        ).splitlines() if value.endswith((".py", ".sh"))
    ]
    if dirty or runtime_untracked:
        raise RuntimeError(
            f"support sweep requires committed clean source: dirty={dirty}, "
            f"runtime_untracked={runtime_untracked}"
        )
    sources = (
        Path(__file__).resolve(), ROOT / "run_low7_rbf_v3_support_sweep.sh",
        TRAINER, ROOT / "afe_demo_support.py", ROOT / "afe_core.py",
        ROOT / "afe_rbf_core.py", ROOT / "afe_execution.py", ROOT / "afe_context.py",
        ROOT / "grid_expand_afe2.py", EVALUATOR, DIAGNOSTICS, VIDEO,
    )
    return {
        "repository": str(repository), "git_commit": commit,
        "tracked_dirty": False, "untracked_runtime_sources": [],
        "runtime_source_sha256": {
            str(path.relative_to(repository)): sha256_file(path) for path in sources
        },
    }


def gpu_record(index: int, expected_uuid: str, require_idle=True):
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [value.strip() for value in line.split(",")]
    if len(fields) != 7 or fields[0] != str(index) or fields[1].lower() != expected_uuid.lower():
        raise RuntimeError(f"GPU {index} identity mismatch: {line}")
    pids = subprocess.check_output([
        "nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    if require_idle and pids:
        raise RuntimeError(f"GPU {index} is shared by foreign compute PIDs: {pids}")
    return {
        "physical_index": index, "uuid": fields[1], "name": fields[2],
        "driver_version": fields[3], "memory_total_mib": int(fields[4]),
        "memory_used_mib": int(fields[5]), "utilization_percent": int(fields[6]),
        "active_compute_pids": pids, "process_device": "cuda:0",
    }


def resolve_checkpoint(recipe_path: Path):
    recipe = load_json(recipe_path.resolve())
    if recipe.get("scene", {}).get("profile", {}).get("name") != SCENE:
        raise RuntimeError("reviewed source recipe is not the canonical giant-obstacle scene")
    checkpoint = Path(recipe["source_checkpoint"]).resolve()
    expected = str(recipe["source_checkpoint_sha256"]).lower()
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise RuntimeError("reviewed recipe checkpoint path/hash authentication failed")
    return checkpoint, expected, {
        "reviewed_recipe": str(recipe_path.resolve()),
        "reviewed_recipe_sha256": sha256_file(recipe_path.resolve()),
        "checkpoint": str(checkpoint), "checkpoint_sha256": expected,
        "checkpoint_model_sha256": recipe["source_checkpoint_model_sha256"],
        "scene_sha256": recipe["scene"]["sha256"],
    }


def trainer_command(args, arm: Arm, outdir: Path, preflight=False):
    return [
        args.python, str(TRAINER),
        "--protocol-profile", "v3_support_preflight" if preflight else "v3_support_sweep",
        "--ckpt", str(args.checkpoint),
        "--expected-ckpt-sha256", args.checkpoint_sha256,
        "--scene-profile", SCENE,
        "--outdir", str(outdir),
        "--rounds", "1" if preflight else "100",
        "--rollout-replicas", "8", "--K", "16", "--B", "4", "--T", "300",
        "--M-eval", "0", "--batch", "128", "--afe-steps", "0",
        "--afe-lr", "1e-5", "--gp-cap", "512", "--gp-lam", "1e-2",
        "--acquisition-mode", "sequential", "--adaptive-ess-target", "0.5",
        "--adaptive-beta-contexts-per-gamma", "64", "--adaptive-beta-equalize-gammas",
        "--replay-window", "2", "--replay-sampling", "round_gamma_replica_context",
        "--replay-update-mode", "fixed_macro_steps_exact_epoch",
        "--replay-loss-weighting", "gamma_episode_context_query_equal_mass",
        "--gp-replay-window", "2", "--gp-replay-sampling", "round_gamma_replica_context",
        "--lengthscale-multiplier", "1.0", "--negative-alpha", "0",
        "--execution-rule", "nominal_hp_max_step_margin",
        "--conditioning-schema", "low7_closest_boundary", "--freeze-visual-encoder",
        "--skip-training-probes", "--calibration-replicas", "8",
        "--calibration-control-steps", "4", "--sweep-compact-artifacts",
        "--compact-checkpoint-every", "1", "--route-metric-steps", "10",
        "--verifier-workers", str(args.verifier_workers), "--seed", str(SEED),
        "--optimizer-steps-per-round", str(arm.steps),
        "--demo-frac", f"{arm.demo_frac:g}",
    ]


def command_env(gpu_index: int):
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    return env


def run_logged(command, log: Path, gpu_index: int):
    with log.open("x") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        subprocess.run(
            command, cwd=ROOT, env=command_env(gpu_index),
            stdout=stream, stderr=subprocess.STDOUT, check=True,
        )


def require_complete(run_dir: Path, rounds: int):
    complete = load_json(run_dir / "COMPLETE.json")
    if complete.get("status") != "COMPLETE" or int(complete.get("completed_round", -1)) != rounds:
        raise RuntimeError(f"trainer completion failed for {run_dir}")
    for relative, expected in complete.get("artifact_sha256", {}).items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"trainer artifact hash failed: {path}")
    return complete


def require_evaluation_complete(evaluation_dir: Path, expected_status: str):
    complete = load_json(evaluation_dir / "EVALUATION_COMPLETE.json")
    if complete.get("status") != expected_status:
        raise RuntimeError(
            f"evaluation completion failed for {evaluation_dir}: {complete.get('status')}"
        )
    for relative, expected in complete.get("artifact_sha256", {}).items():
        path = evaluation_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"evaluation artifact hash failed: {path}")
    return complete


def _support_recipe_contract(recipe, arm: Arm, rounds: int, workers: int):
    exact = {
        "algorithm": "afe_rbf_low7_v3_optimizer_demo_support_v1",
        "protocol_profile": "v3_support_preflight" if rounds == 1 else "v3_support_sweep",
        "rounds": rounds,
        "rollout_replicas": 8,
        "K": 16, "B": 4, "T": 300, "batch": 128,
        "afe_lr": 1.0e-5, "negative_alpha": 0.0,
        "optimizer_steps_per_round": arm.steps,
        "demo_frac": arm.demo_frac,
        "replay_window": 2, "gp_replay_window": 2,
        "replay_update_mode": "fixed_macro_steps_exact_epoch",
        "replay_loss_weighting": "gamma_episode_context_query_equal_mass",
        "execution_rule": "nominal_hp_max_step_margin",
        "nvp_all_k_audit": False,
        "verifier_workers": workers,
    }
    mismatches = {
        key: {"actual": recipe.get(key), "expected": expected}
        for key, expected in exact.items() if recipe.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"support arm recipe mismatch: {mismatches}")
    for flag in ("no_curriculum", "no_anchor", "no_prox", "no_fallback"):
        if recipe.get(flag) is not True:
            raise RuntimeError(f"support recipe does not preserve {flag}=true")
    demo = recipe.get("demo_reference") or {}
    if int(demo.get("pair_leakage", -1)) != 0:
        raise RuntimeError("support demo provenance does not prove zero split leakage")
    if recipe.get("scene", {}).get("profile", {}).get("name") != SCENE:
        raise RuntimeError("support arm changed the giant-obstacle scene")


def validate_support_run(run_dir: Path, arm: Arm, rounds: int, workers: int):
    complete = require_complete(run_dir, rounds)
    recipe = load_json(run_dir / "recipe.json")
    _support_recipe_contract(recipe, arm, rounds, workers)
    rows = read_metric_rows(run_dir / "probe.jsonl")
    if [int(row["round"]) for row in rows] != list(range(rounds + 1)):
        raise RuntimeError(f"support probe is not contiguous through round {rounds}")
    for row in rows[1:]:
        if int(row.get("optimizer_steps", -1)) != arm.steps:
            raise RuntimeError(f"r{row['round']} has the wrong optimizer-step dose")
        if float(row.get("replay_epoch_coverage", -1.0)) != 1.0:
            raise RuntimeError(f"r{row['round']} lacks full unique D+ coverage")
        if int(row.get("replay_duplicate_draws", -1)) != 0:
            raise RuntimeError(f"r{row['round']} duplicated positive replay queries")
        if int(row.get("optimizer_draws", -1)) != int(row.get("replay_eligible", -2)):
            raise RuntimeError(f"r{row['round']} optimizer draws differ from eligible W2 D+")
        if len(row.get("replay_batch_sizes", ())) != arm.steps:
            raise RuntimeError(f"r{row['round']} macro-batch count differs from dose")
        if len(row.get("replay_macro_mass", ())) != arm.steps:
            raise RuntimeError(f"r{row['round']} macro loss-mass count differs from dose")
        if not np.isclose(float(row.get("replay_raw_mass_sum", np.nan)), 1.0, atol=1e-9):
            raise RuntimeError(f"r{row['round']} hierarchical replay mass does not sum to one")
        if row.get("nvp_audit", {}).get("count", 0) != 0:
            raise RuntimeError(f"r{row['round']} performed forbidden all-K NVP audit")
        if arm.demo_frac == 0.0:
            if row.get("demo_cfm") is not None or int(row.get("demo_examples", -1)) != 0:
                raise RuntimeError(f"r{row['round']} no-demo arm accessed demo objective")
        else:
            if row.get("demo_cfm") is None:
                raise RuntimeError(f"r{row['round']} demo arm lacks its normalized loss")
            if int(row.get("demo_original_count", -1)) != int(row.get("demo_reflected_count", -2)):
                raise RuntimeError(f"r{row['round']} demo original/reflection counts differ")
            sampling = row.get("demo_sampling", {})
            counts = sampling.get("gamma_counts", {})
            if not counts or max(counts.values()) - min(counts.values()) > 1:
                raise RuntimeError(f"r{row['round']} demo gamma sampling is unbalanced")
            trajectory = sampling.get("trajectory_balance_by_gamma", {})
            if not trajectory or max(
                value["draw_count_spread"] for value in trajectory.values()
            ) > 1:
                raise RuntimeError(f"r{row['round']} demo trajectory sampling is unbalanced")
        if float(row.get("rel_param_change", {}).get("E_g", np.nan)) != 0.0:
            raise RuntimeError(f"r{row['round']} frozen visual encoder changed")
    return complete, recipe, rows


def run_preflight(args, arm: Arm, gpu_index: int):
    gpu_record(
        gpu_index, args.gpu1_uuid if gpu_index == 1 else args.gpu3_uuid,
        require_idle=True,
    )
    root = args.out / "preflight" / arm.arm_id
    root.mkdir(parents=True)
    started = time.perf_counter()
    run_logged(trainer_command(args, arm, root / "run", preflight=True), root / "train.log", gpu_index)
    _, _, rows = validate_support_run(
        root / "run", arm, 1, args.verifier_workers
    )
    row = rows[-1]
    return {
        "arm": arm.record(), "gpu": gpu_index,
        "elapsed_seconds": time.perf_counter() - started,
        "gather_seconds": row["t_gather"], "update_seconds": row["t_update"],
        "optimizer_steps": row["optimizer_steps"],
        "replay_eligible": row["replay_eligible"],
        "replay_epoch_coverage": row["replay_epoch_coverage"],
        "replay_duplicate_draws": row["replay_duplicate_draws"],
        "scientific_reuse": False,
    }


def arm_paths(out: Path, arm: Arm):
    return {
        "run": out / "arms" / arm.arm_id,
        "evaluation": out / "screening" / arm.arm_id,
        "train_log": out / "logs" / f"{arm.arm_id}.train.log",
        "eval_log": out / "logs" / f"{arm.arm_id}.screen.log",
        "status": out / "arm_status" / f"{arm.arm_id}.json",
    }


def run_arm(args, arm: Arm, gpu_index: int):
    paths = arm_paths(args.out, arm)
    gpu_record(
        gpu_index, args.gpu1_uuid if gpu_index == 1 else args.gpu3_uuid,
        require_idle=True,
    )
    started = time.perf_counter()
    run_logged(trainer_command(args, arm, paths["run"]), paths["train_log"], gpu_index)
    complete, recipe, probe_rows = validate_support_run(
        paths["run"], arm, 100, args.verifier_workers
    )
    evaluate = [
        args.python, str(EVALUATOR), "--run-root", str(paths["run"]),
        "--outdir", str(paths["evaluation"]), "--mode", "screen",
        "--reference-r0-evaluation", str(REFERENCE_EVALUATION),
        "--verifier-workers", str(args.verifier_workers),
    ]
    run_logged(evaluate, paths["eval_log"], gpu_index)
    evaluation_complete = require_evaluation_complete(
        paths["evaluation"], "AFE_RBF_V3_SUPPORT_SCREEN_DELIVERY_COMPLETE"
    )
    selection = load_json(paths["evaluation"] / "selection.json")
    record = {
        "status": "ARM_COMPLETE", "arm": arm.record(), "gpu": gpu_index,
        "elapsed_seconds": time.perf_counter() - started,
        "trainer_complete_sha256": sha256_file(paths["run"] / "COMPLETE.json"),
        "evaluation_complete_sha256": sha256_file(paths["evaluation"] / "EVALUATION_COMPLETE.json"),
        "best_round": selection["best_round"],
        "best_score": selection["ranking"][0],
        "paths": {key: str(value) for key, value in paths.items()},
        "trainer_algorithm": complete["algorithm"],
        "recipe_sha256": sha256_file(paths["run"] / "recipe.json"),
        "probe_sha256": sha256_file(paths["run"] / "probe.jsonl"),
        "checkpoint_rounds": sum(
            relative.startswith("ckpt_")
            for relative in complete["artifact_sha256"]
        ),
        "all_k_nvp_audit_disabled": recipe["nvp_all_k_audit"] is False,
    }
    write_json_new(paths["status"], record)
    return record


def run_gpu_queue(args, gpu_index: int):
    output = []
    for arm in GPU_QUEUES[gpu_index]:
        output.append(run_arm(args, arm, gpu_index))
    return output


def read_metric_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def round_score(rows, round_i):
    gammas = [row for row in rows if row["scope"] == "gamma" and row["round"] == round_i]
    pooled = next(row for row in rows if row["scope"] == "pooled" and row["round"] == round_i)
    return {
        "round": round_i,
        "J": sum(row["successful_route_coverage"]["C"] for row in gammas) / 7,
        "SR": pooled["binary"]["SR"]["estimate"],
        "CR": pooled["binary"]["CR"]["estimate"],
        "timeout": pooled["binary"]["timeout"]["estimate"],
        "V_safe": pooled["binary"]["V_safe"]["estimate"],
        "V_full": pooled["binary"]["V_full"]["estimate"],
        "minimum_clearance": pooled["minimum_clearance"]["mean"],
    }


def global_key(row):
    return (-row["J"], -row["SR"], row["CR"], row["timeout"],
            -row["minimum_clearance"], row["round"], row["arm_id"])


TRAINING_SCALARS = (
    "beta_used", "beta_next", "ess_med", "uplift_med", "sig_span_med",
    "cfm", "demo_cfm", "mixed_objective", "demo_positive_gradient_cosine",
    "grad_clipped_fraction", "preclip_grad_norm_mean", "replay_weight_ess",
    "replay_weight_ess_fraction", "replay_macro_mass_max_residual",
    "replay_epoch_coverage", "replay_duplicate_draws", "optimizer_steps",
    "optimizer_draws", "replay_eligible", "n_D", "n_Dpos", "rep_cos",
    "t_gather", "t_update",
)


def compact_training_row(arm_id: str, row):
    return {
        "arm_id": arm_id,
        "round": int(row["round"]),
        **{key: row.get(key) for key in TRAINING_SCALARS},
        "rel_param_change": row.get("rel_param_change"),
        "replay_batch_size_min": row.get("replay_batch_size_min"),
        "replay_batch_size_max": row.get("replay_batch_size_max"),
        "replay_mass_diagnostics": row.get("replay_mass_diagnostics"),
        "demo_sampling": row.get("demo_sampling"),
        "route_modes_early": row.get("route_modes_early"),
        "per_gamma": row.get("per_gamma"),
        "nvp_audit": row.get("nvp_audit"),
    }


def assemble_screening(args, arm_records):
    table = []
    all_points = []
    endpoint_details = {}
    training_endpoints = {}
    training_rows = []
    for record in sorted(arm_records, key=lambda value: value["arm"]["arm_id"]):
        arm_id = record["arm"]["arm_id"]
        rows = read_metric_rows(args.out / "screening" / arm_id / "metrics.jsonl")
        probes = read_metric_rows(args.out / "arms" / arm_id / "probe.jsonl")
        probe_by_round = {int(row["round"]): row for row in probes}
        training_rows.extend(
            compact_training_row(arm_id, row) for row in probes[1:]
        )
        selection = load_json(args.out / "screening" / arm_id / "selection.json")
        for round_i in selection["ranking"]:
            score = round_score(rows, int(round_i["round"]))
            all_points.append({"arm_id": arm_id, **record["arm"], **score})
        best = {"arm_id": arm_id, **record["arm"], **round_score(rows, selection["best_round"])}
        final = round_score(rows, 100)
        table.append({**best, **{f"final_{key}": value for key, value in final.items() if key != "round"}, "final_round": 100})
        endpoint_details[arm_id] = {
            label: [
                row for row in rows if int(row["round"]) == round_i
            ]
            for label, round_i in (
                ("pretrained_r0", 0), ("best", int(selection["best_round"])),
                ("final_r100", 100),
            )
        }
        training_endpoints[arm_id] = {
            "best": compact_training_row(
                arm_id, probe_by_round[int(selection["best_round"])]
            ) if int(selection["best_round"]) > 0 else None,
            "final_r100": compact_training_row(arm_id, probe_by_round[100]),
        }
    selected = min(table, key=global_key)
    pareto = []
    for candidate in all_points:
        dominated = any(
            other["SR"] >= candidate["SR"] and other["J"] >= candidate["J"]
            and (other["SR"] > candidate["SR"] or other["J"] > candidate["J"])
            for other in all_points
        )
        if not dominated:
            pareto.append(candidate)
    pareto.sort(key=lambda row: (row["SR"], row["J"], row["arm_id"], row["round"]))
    write_json_new(args.out / "screening_summary.json", {
        "selection_rule": "J, SR, CR, timeout, clearance, earlier round, arm_id exact tie",
        "selected": selected, "arms": table, "pareto_SR_J": pareto,
    })
    write_json_new(args.out / "screening_endpoint_metrics.json", endpoint_details)
    write_json_new(args.out / "training_endpoint_metrics.json", training_endpoints)
    with (args.out / "training_metrics.jsonl").open("x") as stream:
        for row in training_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    fields = list(table[0])
    with (args.out / "screening_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(table)
    with (args.out / "metrics.jsonl").open("x") as stream:
        for row in all_points:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    table_fig, table_axis = plt.subplots(figsize=(17, 3.3))
    table_axis.axis("off")
    labels = ("arm", "updates", "demo mass", "best r", "best J", "best SR", "best CR", "best TO", "r100 J", "r100 SR")
    cells = [[
        row["arm_id"], row["optimizer_steps_per_round"],
        f"{row['demo_frac_objective_mass']:.3f}", row["round"],
        f"{row['J']:.3f}", f"{row['SR']:.3f}", f"{row['CR']:.3f}",
        f"{row['timeout']:.3f}", f"{row['final_J']:.3f}", f"{row['final_SR']:.3f}",
    ] for row in table]
    rendered = table_axis.table(
        cellText=cells, colLabels=labels, loc="center", cellLoc="center"
    )
    rendered.auto_set_font_size(False); rendered.set_fontsize(9); rendered.scale(1, 1.45)
    table_axis.set_title("Raw temperature-1 M10/gamma screening (selection uses J first)", pad=12)
    table_fig.tight_layout()
    table_fig.savefig(args.out / "screening_table.png", dpi=180)
    table_fig.savefig(args.out / "screening_table.pdf")
    plt.close(table_fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.get_cmap("tab10")
    for index, arm in enumerate(sorted({row["arm_id"] for row in all_points})):
        points = sorted((row for row in all_points if row["arm_id"] == arm), key=lambda row: row["round"])
        axes[0].plot([p["round"] for p in points], [p["J"] for p in points], "o-", ms=3, label=arm, color=colors(index))
        axes[1].plot([p["SR"] for p in points], [p["J"] for p in points], "o-", ms=3, label=arm, color=colors(index))
    axes[0].set(xlabel="round", ylabel="J", ylim=(-.03, 1.03), title="Successful balanced-route coverage")
    axes[1].set(xlabel="raw SR", ylabel="J", xlim=(-.03, 1.03), ylim=(-.03, 1.03), title="SR–J trade-off")
    for axis in axes: axis.grid(alpha=.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Six-arm raw temperature-1 M10/gamma screening")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.out / "screening_curves.png", dpi=180)
    fig.savefig(args.out / "screening_curves.pdf")
    plt.close(fig)
    return selected


def ffprobe(path: Path, frames: int):
    payload = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames",
        "-of", "json", str(path),
    ], text=True))
    stream = payload["streams"][0]
    if int(stream["nb_read_frames"]) != frames:
        raise RuntimeError(f"selected video frame count != {frames}: {stream}")
    return stream


def run_confirmation(args, selected):
    arm_id = selected["arm_id"]
    run_dir = args.out / "arms" / arm_id
    confirmation = args.out / "confirmation"
    gpu_record(1, args.gpu1_uuid, require_idle=True)
    run_logged([
        args.python, str(EVALUATOR), "--run-root", str(run_dir),
        "--outdir", str(confirmation), "--mode", "holdout",
        "--selected-round", str(selected["round"]),
        "--verifier-workers", str(args.verifier_workers),
    ], args.out / "logs" / "confirmation.log", 1)
    require_evaluation_complete(
        confirmation, "AFE_RBF_V3_SUPPORT_HOLDOUT_DELIVERY_COMPLETE"
    )
    holdout_selection = load_json(confirmation / "selection.json")
    if (
        int(holdout_selection.get("selected_round", -1)) != int(selected["round"])
        or "did not select" not in holdout_selection.get("selection_source", "")
    ):
        raise RuntimeError("M50 confirmation changed or obscured the Stage A selection")
    holdout_rows = read_metric_rows(confirmation / "metrics.jsonl")
    required_rounds = {0, int(selected["round"]), 100}
    if {int(row["round"]) for row in holdout_rows} != required_rounds:
        raise RuntimeError("M50 confirmation lacks r0/selected/r100 endpoint cells")
    for row in holdout_rows:
        expected_n = 350 if row["scope"] == "pooled" else 50
        if any(int(row["binary"][key]["n"]) != expected_n for key in (
            "SR", "CR", "timeout", "V_safe", "V_full"
        )):
            raise RuntimeError("M50 confirmation cell has the wrong sample count")
    run_logged([
        args.python, str(DIAGNOSTICS), "--run", str(run_dir),
        "--out", str(args.out / "selected_training_diagnostic.png"),
    ], args.out / "logs" / "selected_diagnostic.log", 1)
    run_logged([
        args.python, str(VIDEO), "--run", str(run_dir),
        "--out", str(args.out / "selected_expansion_diagnostic.mp4"),
        "--dense-until", "10", "--every-after", "10",
    ], args.out / "logs" / "selected_video.log", 1)
    video = ffprobe(args.out / "selected_expansion_diagnostic.mp4", 19)
    for filename in ("report.png", "report.pdf", "selected_raw_m50_gallery.png", "selected_raw_m50_gallery.pdf"):
        shutil.copy2(confirmation / filename, args.out / filename)
    return {
        "selected_arm": arm_id, "selected_round": selected["round"],
        "confirmation": str(confirmation), "video": video,
        "evaluation_complete_sha256": sha256_file(
            confirmation / "EVALUATION_COMPLETE.json"
        ),
        "holdout_noise_bank": load_json(
            confirmation / "evaluation_contract.json"
        )["noise_bank"],
    }


def inventory(out: Path):
    return {
        str(path.relative_to(out)): sha256_file(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name not in {"DELIVERY_COMPLETE.json", "SWEEP_COMPLETE.json"}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-recipe", type=Path, default=SOURCE_RECIPE)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--gpu3-uuid", required=True)
    parser.add_argument("--verifier-workers", type=int, required=True)
    parser.add_argument("--python", default=PYTHON_DEFAULT)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"support sweep output root must be absent: {args.out}")
    cpu_count = os.cpu_count()
    if cpu_count is None or args.verifier_workers * 2 != cpu_count:
        raise RuntimeError(
            f"two pipelines must each receive half of {cpu_count} CPUs; got {args.verifier_workers}"
        )
    source = source_record()
    checkpoint, checkpoint_sha, provenance = resolve_checkpoint(args.checkpoint_recipe)
    args.checkpoint = checkpoint
    args.checkpoint_sha256 = checkpoint_sha
    gpu = {
        "1": gpu_record(1, args.gpu1_uuid, require_idle=True),
        "3": gpu_record(3, args.gpu3_uuid, require_idle=True),
    }
    for relative in ("preflight", "arms", "screening", "logs", "arm_status"):
        (args.out / relative).mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json_new(args.out / "provenance.json", {
        "source": source, "checkpoint_and_scene": provenance, "gpus": gpu,
        "host_cpu_count": cpu_count, "verifier_workers_per_pipeline": args.verifier_workers,
        "gpu_queues": {
            str(index): [arm.record() for arm in queue]
            for index, queue in GPU_QUEUES.items()
        },
    })
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_preflight, args, Arm(16, 0.0), 1),
            pool.submit(run_preflight, args, Arm(32, 0.0), 3),
        ]
        preflight = [future.result() for future in futures]
    write_json_new(args.out / "preflight" / "timing_summary.json", {
        "status": "TIMING_ONLY_NOT_REUSED", "arms": preflight,
    })
    # Both short processes must be gone before the scientific queues begin.
    gpu_record(1, args.gpu1_uuid, require_idle=True)
    gpu_record(3, args.gpu3_uuid, require_idle=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_gpu_queue, args, index) for index in (1, 3)]
        arm_records = [record for future in futures for record in future.result()]
    selected = assemble_screening(args, arm_records)
    confirmation = run_confirmation(args, selected)
    elapsed = time.time() - started
    write_json_new(args.out / "SWEEP_COMPLETE.json", {
        "status": "LOW7_RBF_V3_SUPPORT_SWEEP_COMPLETE",
        "source_commit": source["git_commit"], "checkpoint_sha256": checkpoint_sha,
        "scene_sha256": provenance["scene_sha256"], "arms_completed": 6,
        "selected": selected, "confirmation": confirmation, "elapsed_seconds": elapsed,
    })
    artifacts = inventory(args.out)
    write_json_new(args.out / "DELIVERY_COMPLETE.json", {
        "status": "LOW7_RBF_V3_SUPPORT_DELIVERY_COMPLETE",
        "source_commit": source["git_commit"], "arms_completed": 6,
        "selected_arm": selected["arm_id"], "selected_round": selected["round"],
        "elapsed_seconds": elapsed, "artifact_sha256": artifacts,
    })
    print(f"LOW7 RBF V3 SUPPORT SWEEP COMPLETE: {args.out}", flush=True)


if __name__ == "__main__":
    main()
