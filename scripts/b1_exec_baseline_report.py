#!/usr/bin/env python3
"""Gate, assemble, render, and authenticate the B1 execution-rule baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKBOOK = Path(__file__).resolve().parents[1]
SNAP = (
    WORKBOOK
    / "source_snapshot/overnight_run_07_06/rev_expansion/codex_overnight"
)
for entry in (WORKBOOK / "scripts", SNAP.parents[1], SNAP.parent, SNAP):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import eval_rounds_m as EVAL  # noqa: E402
from afe2_scene_profiles import build_scene, get_scene_profile  # noqa: E402


GAMMAS = tuple(EVAL.GAMMAS)
PLOT_GAMMAS = (0.1, 0.5, 1.0)
DISPLAY_INDICES = tuple(range(10))
Z95 = 1.959963984540054


def gamma_slug(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model_state(checkpoint: Path) -> dict:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint has no state_dict: {checkpoint}")
    return state


def compare_model_states(left: Path, right: Path) -> dict:
    import torch

    a = model_state(left)
    b = model_state(right)
    if set(a) != set(b):
        missing_left = sorted(set(b) - set(a))
        missing_right = sorted(set(a) - set(b))
        raise RuntimeError(
            f"r0 model keys differ: left_missing={missing_left}, "
            f"right_missing={missing_right}"
        )
    for name in sorted(a):
        x, y = a[name], b[name]
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            if x != y:
                raise RuntimeError(f"r0 non-tensor model value differs: {name}")
            continue
        if x.dtype != y.dtype or x.shape != y.shape:
            raise RuntimeError(
                f"r0 model tensor metadata differs at {name}: "
                f"{x.dtype}/{tuple(x.shape)} != {y.dtype}/{tuple(y.shape)}"
            )
        if not torch.equal(x, y):
            unequal = torch.logical_not(torch.eq(x, y)).nonzero()
            index = tuple(int(v) for v in unequal[0].tolist())
            raise RuntimeError(
                f"r0 model tensor differs at {name}{index}: "
                f"{x[index].item()!r} != {y[index].item()!r}"
            )
    return {"tensor_count": len(a), "bitwise_identical": True}


def first_array_difference(left: Path, right: Path) -> str | None:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return f"array keys differ: {sorted(a.files)} != {sorted(b.files)}"
        for name in sorted(a.files):
            x, y = a[name], b[name]
            if x.dtype != y.dtype or x.shape != y.shape:
                return (
                    f"{name} metadata differs: {x.dtype}/{x.shape} "
                    f"!= {y.dtype}/{y.shape}"
                )
            equal = (
                np.array_equal(x, y, equal_nan=True)
                if x.dtype.kind in "fc"
                else np.array_equal(x, y)
            )
            if equal:
                continue
            if x.ndim == 0:
                return f"{name} differs: {x.item()!r} != {y.item()!r}"
            mismatch = ~((x == y) | (np.isnan(x) & np.isnan(y))) if x.dtype.kind in "fc" else x != y
            index = tuple(int(v) for v in np.argwhere(mismatch)[0])
            if name == "paths" and len(index) >= 3:
                return (
                    f"trajectory={index[0]} timestep={index[1]} tensor=paths"
                    f"[{index[2]}]: {x[index]!r} != {y[index]!r}"
                )
            return f"{name}{index}: {x[index]!r} != {y[index]!r}"
    return None


def gate_r0(args) -> int:
    if args.out.exists():
        raise FileExistsError(args.out)
    model = compare_model_states(args.margin_ckpt, args.cost_ckpt)
    margin_contract = load_json(args.margin_shard / "margin.contract.json")
    cost_contract = load_json(args.cost_shard / "cost.contract.json")
    if margin_contract["bank_sha256"] != cost_contract["bank_sha256"]:
        raise RuntimeError("r0 common-bank tensor hashes differ")
    margin_bank = args.margin_shard / "margin_common_bank.npy"
    cost_bank = args.cost_shard / "cost_common_bank.npy"
    if margin_bank.read_bytes() != cost_bank.read_bytes():
        raise RuntimeError("r0 common-bank NPY bytes differ")
    if sha256_file(margin_bank) != margin_contract["bank_file_sha256"]:
        raise RuntimeError("margin r0 bank file hash does not validate")
    if sha256_file(cost_bank) != cost_contract["bank_file_sha256"]:
        raise RuntimeError("cost r0 bank file hash does not validate")

    cells = {}
    for gamma in GAMMAS:
        name = f"r000_g{gamma_slug(gamma)}.npz"
        left = args.margin_shard / "cells" / name
        right = args.cost_shard / "cells" / name
        if sha256_file(left) != sha256_file(right):
            detail = first_array_difference(left, right)
            raise RuntimeError(
                f"r0 scientific archive differs at gamma={gamma:g}: "
                f"{detail or 'container bytes differ despite equal parsed arrays'}"
            )
        cells[f"{gamma:g}"] = sha256_file(left)

    margin_rows = load_jsonl(args.margin_shard / "margin.jsonl")
    cost_rows = load_jsonl(args.cost_shard / "cost.jsonl")
    for left, right in zip(margin_rows, cost_rows, strict=True):
        for field in (
            "SR",
            "CR",
            "timeout",
            "Validity",
            "minimum_clearance",
            "successful_time_to_goal",
            "route",
            "bank_sha256",
        ):
            if left[field] != right[field]:
                raise RuntimeError(
                    f"r0 parsed metric differs at gamma={left['gamma']:g}, "
                    f"field={field}: {left[field]!r} != {right[field]!r}"
                )
    payload = {
        "status": "B1_EXEC_RULE_R0_EQUIVALENCE_PASS",
        "model_state": model,
        "bank_tensor_sha256": margin_contract["bank_sha256"],
        "bank_file_sha256": sha256_file(margin_bank),
        "cell_archive_sha256": cells,
        "cells": len(cells),
        "M_per_gamma": 200,
        "temperature": 1.0,
        "parsed_metrics_exact": True,
        "scientific_archives_byte_identical": True,
    }
    write_json(args.out, payload)
    print(args.out)
    return 0


def merge_recipe_diff(margin_path: Path, existing_path: Path, fresh_path: Path) -> dict:
    margin = load_json(margin_path)
    existing = load_json(existing_path)
    fresh = load_json(fresh_path)

    def walk(a, b, prefix=""):
        rows = []
        if type(a) is not type(b):
            return [{"path": prefix, "margin": a, "cost": b}]
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                path = f"{prefix}.{key}" if prefix else key
                if key not in a:
                    rows.append({"path": path, "margin": None, "cost": b[key]})
                elif key not in b:
                    rows.append({"path": path, "margin": a[key], "cost": None})
                else:
                    rows.extend(walk(a[key], b[key], path))
        elif isinstance(a, list):
            if a != b:
                rows.append({"path": prefix, "margin": a, "cost": b})
        elif a != b:
            rows.append({"path": prefix, "margin": a, "cost": b})
        return rows

    existing_diff = walk(margin, existing)
    fresh_diff = walk(margin, fresh)
    scientific_fields = (
        "scene",
        "source_checkpoint_sha256",
        "source_checkpoint_model_sha256",
        "source_checkpoint_contract_sha256",
        "seed",
        "rollout_replicas",
        "gammas",
        "K",
        "B",
        "T",
        "kernel",
        "base_lengthscale",
        "lengthscale",
        "lengthscale_multiplier",
        "gp_cap",
        "gp_lam",
        "acquisition_mode",
        "adaptive_ess_target",
        "adaptive_beta_contexts_per_gamma",
        "adaptive_beta_equalize_gammas",
        "gp_replay_window",
        "gp_replay_sampling",
        "replay_window",
        "replay_sampling",
        "replay_update_mode",
        "replay_loss_weighting",
        "replay_epochs",
        "optimizer_steps_formula",
        "optimizer_steps_per_round",
        "negative_alpha",
        "batch",
        "afe_lr",
        "afe_steps",
        "freeze_visual_encoder",
        "conditioning_schema",
        "raw_condition_dim",
        "no_curriculum",
        "no_anchor",
        "no_prox",
        "no_fallback",
        "demo_frac",
        "demo_reference",
        "calibration_replicas",
        "calibration_control_steps",
        "nvp_all_k_audit",
        "route_diagnostics",
        "verifier_workers",
        "nfe",
        "reach",
        "training_probes",
    )
    normalized_differences = [
        {
            "field": field,
            "margin": margin.get(field),
            "cost": fresh.get(field),
        }
        for field in scientific_fields
        if margin.get(field) != fresh.get(field)
    ]
    execution_pass = (
        margin.get("execution_rule") == "nominal_hp_max_step_margin"
        and fresh.get("execution_rule") == "nominal_hp_safemppi_cost"
    )
    rounds_pass = (
        int(margin.get("rounds", -1)) >= 15
        and int(fresh.get("rounds", -1)) >= 15
    )
    parity_pass = (
        not normalized_differences and execution_pass and rounds_pass
    )
    return {
        "status": (
            "B1_EXEC_RULE_RECIPE_PARITY_PASS"
            if parity_pass
            else "B1_EXEC_RULE_RECIPE_PARITY_FAIL"
        ),
        "preferred_existing_arm_qualified": parity_pass,
        "normalized_scientific_fields": list(scientific_fields),
        "normalized_scientific_differences": normalized_differences,
        "execution_rule_gate": execution_pass,
        "round_availability_gate": rounds_pass,
        "source_lineage_audit": {
            "cost_source": "63ebefa7877c0b923c1c7cdea19228302dd6a0ca",
            "margin_source": "39fb3e63542e6b23efe3321f311505e553c0bec6",
            "cost_is_ancestor_of_margin": True,
            "trainer_diff_scope": (
                "adds only b1_balanced_r0_margin50 profile declaration, "
                "50-round validation, and its algorithm label; existing "
                "b1_balanced_r0_sweep behavior is unchanged"
            ),
        },
        "derived_outcome_policy": {
            "beta_and_calibration_counts": (
                "downstream outputs of the unchanged calibration mechanism "
                "under the alternate execution rule, not recipe settings"
            ),
            "wall_times": "non-scientific runtime fields",
        },
        "existing_arm": {
            "recipe": str(existing_path),
            "differences": existing_diff,
            "selected_for_evaluation": parity_pass,
        },
        "fresh_arm": {
            "recipe": str(fresh_path),
            "differences": fresh_diff,
            "forbidden_differences": normalized_differences,
        },
    }


def copy_cells(shards: list[Path], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True)
    hashes = {}
    for shard in shards:
        for source in sorted((shard / "cells").glob("*.npz")):
            target = destination / source.name
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(source, target)
            digest = sha256_file(target)
            if digest != sha256_file(source):
                raise RuntimeError(f"copy hash mismatch: {source}")
            hashes[str(target)] = digest
    if len(hashes) != 112:
        raise RuntimeError(
            f"expected 112 scientific cells, found {len(hashes)} at {destination}"
        )
    return hashes


def canonical_jsonl(shards: list[Path], arm: str, destination: Path) -> list[dict]:
    rows = []
    for shard in shards:
        rows.extend(load_jsonl(shard / f"{arm}.jsonl"))
    rows.sort(key=lambda row: (row["round"], row["gamma"]))
    keys = [(row["round"], row["gamma"]) for row in rows]
    expected = [(r, g) for r in range(16) for g in GAMMAS]
    if keys != expected:
        raise RuntimeError(f"{arm} cell keys do not match r0-r15 x seven gammas")
    with destination.open("x") as stream:
        for row in rows:
            row = dict(row)
            row["archive"] = (
                f"scientific/{arm}/"
                f"r{row['round']:03d}_g{gamma_slug(row['gamma'])}.npz"
            )
            stream.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return rows


def load_cell(out: Path, arm: str, round_i: int, gamma: float):
    path = (
        out
        / "scientific"
        / arm
        / f"r{round_i:03d}_g{gamma_slug(gamma)}.npz"
    )
    return np.load(path, allow_pickle=False)


def wilson(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.bool_)
    n = len(values)
    p = float(values.mean())
    denom = 1.0 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denom
    half = (
        Z95
        * math.sqrt(p * (1.0 - p) / n + Z95 * Z95 / (4 * n * n))
        / denom
    )
    return p, max(0.0, center - half), min(1.0, center + half)


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    half = (
        Z95 * float(values.std(ddof=1)) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return mean, mean - half, mean + half


METRICS = (
    ("collision_rate_event", "Collision rate", wilson, (0.0, 1.0)),
    ("validity_event", "Validity", wilson, (0.0, 1.0)),
    ("minimum_clearance", "Min. clearance [m]", mean_interval, None),
    (
        "successful_time_to_goal",
        "Time-to-goal [s]",
        mean_interval,
        None,
    ),
)


def curve_values(out: Path, arm: str, gamma: float | None, field: str, interval):
    means, lows, highs = [], [], []
    for round_i in range(16):
        values = []
        selected = GAMMAS if gamma is None else (gamma,)
        for value in selected:
            with load_cell(out, arm, round_i, value) as cell:
                array = cell[field]
                if field == "successful_time_to_goal":
                    array = array[np.isfinite(array)]
                values.append(array)
        merged = np.concatenate(values)
        mean, low, high = interval(merged)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    return np.asarray(means), np.asarray(lows), np.asarray(highs)


def render_curves(out: Path) -> None:
    colors = {
        gamma: plt.get_cmap("plasma")(0.08 + 0.84 * index / 6)
        for index, gamma in enumerate(GAMMAS)
    }
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.2), squeeze=False)
    rounds = np.arange(16)
    for row_index, (arm, label) in enumerate(
        (
            ("margin", "max-step-margin"),
            ("cost", "native-SafeMPPI-cost"),
        )
    ):
        for col_index, (field, title, interval, ylim) in enumerate(METRICS):
            axis = axes[row_index, col_index]
            for gamma in GAMMAS:
                mean, low, high = curve_values(out, arm, gamma, field, interval)
                axis.plot(rounds, mean, color=colors[gamma], lw=1.45, alpha=0.78)
                axis.fill_between(
                    rounds, low, high, color=colors[gamma], alpha=0.09, lw=0
                )
            mean, low, high = curve_values(out, arm, None, field, interval)
            axis.plot(rounds, mean, color="black", lw=3.0)
            axis.fill_between(rounds, low, high, color="black", alpha=0.12, lw=0)
            axis.set_title(title)
            axis.grid(alpha=0.24)
            axis.set_xlim(-0.3, 15.3)
            if ylim:
                axis.set_ylim(*ylim)
            if row_index == 1:
                axis.set_xlabel("Expansion round")
        axes[row_index, 0].set_ylabel(label, fontsize=15)
    handles = [
        plt.Line2D([0], [0], color=colors[g], lw=2.2, label=rf"$\gamma={g:g}$")
        for g in GAMMAS
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=3, label="pooled"))
    fig.legend(handles=handles, ncol=8, loc="upper center", frameon=False)
    fig.suptitle("Raw temperature-1, M=200/gamma", y=0.955, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"paired_execution_rule_curves.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def draw_gallery_axis(axis, env, cell, gamma: float, title: str, ylabel: str):
    for obstacle in env.obstacles.detach().cpu().numpy():
        axis.add_patch(
            plt.Circle(obstacle[:2], obstacle[2], color="#bdbdbd", zorder=1)
        )
    color = plt.get_cmap("plasma")({0.1: 0.08, 0.5: 0.52, 1.0: 0.92}[gamma])
    for index in DISPLAY_INDICES:
        length = int(cell["path_length"][index])
        path = cell["paths"][index, :length]
        axis.plot(path[:, 0], path[:, 1], color=color, lw=1.25, alpha=0.7)
        dots = path[::8]
        axis.plot(
            dots[:, 0],
            dots[:, 1],
            ".",
            color=color,
            markersize=2.2,
            alpha=0.72,
        )
        if not bool(cell["success"][index]):
            axis.plot(
                path[-1, 0],
                path[-1, 1],
                marker="x",
                color="#cc3311",
                markersize=9,
                markeredgewidth=2,
            )
    start = env.x0.detach().cpu().numpy()[:2]
    goal = env.goal.detach().cpu().numpy()
    axis.plot(*start, "ks", markersize=5)
    axis.plot(*goal, "*", color="gold", markeredgecolor="black", markersize=12)
    axis.set_xlim(-0.3, 5.3)
    axis.set_ylim(-0.3, 5.3)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title)
    axis.set_ylabel(ylabel, fontsize=14)


def render_gallery(out: Path) -> dict:
    env = build_scene(get_scene_profile(EVAL.SCENE))
    rows = (
        ("margin", 0, "pretrained r0"),
        ("margin", 15, "max-step-margin r15"),
        ("cost", 15, "native-SafeMPPI-cost r15"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(14.4, 14.0), squeeze=False)
    outcomes = {}
    for row_index, (arm, round_i, label) in enumerate(rows):
        outcomes[label] = {}
        for col_index, gamma in enumerate(PLOT_GAMMAS):
            with load_cell(out, arm, round_i, gamma) as cell:
                draw_gallery_axis(
                    axes[row_index, col_index],
                    env,
                    cell,
                    gamma,
                    rf"$\gamma={gamma:g}$" if row_index == 0 else "",
                    label if col_index == 0 else "",
                )
                outcomes[label][f"{gamma:g}"] = [
                    cell["status"][index].decode("ascii")
                    for index in DISPLAY_INDICES
                ]
    fig.subplots_adjust(
        left=0.13,
        right=0.99,
        bottom=0.015,
        top=0.98,
        wspace=0.04,
        hspace=0.055,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            out / f"paired_execution_rule_gallery.{suffix}",
            dpi=280 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    manifest = {
        "display_indices": list(DISPLAY_INDICES),
        "display_rule": (
            "fixed indices 0--9 from the declared M=200 bank; no visual "
            "curation and no success filtering"
        ),
        "gammas": list(PLOT_GAMMAS),
        "outcomes": outcomes,
        "failure_marker": "red X at terminal collision or failure",
        "state_dot_stride": 8,
    }
    write_json(out / "paired_execution_rule_gallery.json", manifest)
    return manifest


def probe_rows(arm_dir: Path) -> dict[int, dict]:
    return {row["round"]: row for row in load_jsonl(arm_dir / "probe.jsonl")}


def nested(row: dict, path: str, default=np.nan):
    value: Any = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def viz_diagnostics(arm_dir: Path) -> dict[str, dict[int, float]]:
    import torch
    import afe_execution as execution

    env = build_scene(get_scene_profile(EVAL.SCENE))
    output = {
        "progress": {},
        "margin": {},
        "cost": {},
        "episode_length": {},
    }
    for round_i in range(1, 16):
        path = arm_dir / "viz_db" / f"round{round_i}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        progress, margins, costs = [], [], []
        for record in payload["viz"]:
            selected = int(record["sel"])
            if selected < 0:
                continue
            queried = list(record["drawn"])
            if selected not in queried:
                continue
            local = queried.index(selected)
            progress.append(float(record["step_progress"][local]))
            margins.append(float(record["nominal_hp_step_margin"][local]))
            value = record["safemppi_cost"][local]
            if value is None:
                state = np.asarray(record["state"], dtype=np.float64)
                positions = np.asarray(
                    record["segsK"][selected], dtype=np.float64
                )
                dt = float(env.dt)
                controls = []
                position = state[:2].copy()
                velocity = state[2:4].copy()
                for next_position in positions:
                    acceleration = (
                        2.0
                        * (next_position - position - dt * velocity)
                        / (dt * dt)
                    )
                    controls.append(acceleration)
                    position = next_position
                    velocity = velocity + dt * acceleration
                value = execution.safemppi_plan_costs(
                    state,
                    np.asarray(controls)[None],
                    positions[None],
                    env,
                )[0]
            costs.append(float(value))
        if progress:
            output["progress"][round_i] = float(np.median(progress))
            output["margin"][round_i] = float(np.median(margins))
            output["cost"][round_i] = float(np.median(costs))
        output["episode_length"][round_i] = float(
            np.mean([episode["steps"] for episode in payload["eps"]])
        )
    return output


def render_training_diagnostic(out: Path, margin_dir: Path, cost_dir: Path) -> None:
    arms = {
        "max-step-margin": (probe_rows(margin_dir), viz_diagnostics(margin_dir)),
        "native-SafeMPPI-cost": (probe_rows(cost_dir), viz_diagnostics(cost_dir)),
    }
    colors = {
        "max-step-margin": "#0072b2",
        "native-SafeMPPI-cost": "#d55e00",
    }
    specs = (
        ("beta_used", "Beta", None),
        ("ess_med", "Realized ESS/K", None),
        ("uplift_med", "Uncertainty uplift", None),
        ("n_D", "D", None),
        ("n_Dpos", "D+", None),
        ("positive_fraction", "Positive fraction", None),
        ("nvp_rate", "NVP rate", None),
        ("nvp_classes", "NVP classifications", None),
        ("optimizer_draws", "Optimizer draws", None),
        ("n_distinct", "Unique eligible positives", None),
        ("cfm", "CFM loss", None),
        ("negative_cfm", "Negative loss activity", None),
        ("route_balance", "Executed U/R balance", None),
        ("progress", "Executed one-step progress", "viz"),
        ("margin", r"Executed nominal $H_P$ margin", "viz"),
        ("cost", "Executed native SafeMPPI cost", "viz"),
        ("episode_length", "Episode length", "viz"),
        ("positive_contexts", "Contexts/round", None),
    )
    fig, axes = plt.subplots(5, 4, figsize=(20, 20), squeeze=False)
    for axis, (field, title, source) in zip(axes.flat, specs):
        for label, (probe, viz) in arms.items():
            if source == "viz":
                mapping = viz[field]
                rounds = sorted(mapping)
                values = [mapping[r] for r in rounds]
            else:
                rounds = list(range(16))
                values = []
                for round_i in rounds:
                    row = probe[round_i]
                    if field == "positive_fraction":
                        values.append(
                            row.get("n_Dpos", 0) / max(row.get("n_D", 0), 1)
                        )
                    elif field == "nvp_rate":
                        count = nested(row, "nvp_audit.count", 0)
                        values.append(count / 56.0)
                    elif field == "nvp_classes":
                        values.append(
                            nested(
                                row,
                                "nvp_audit.class_counts.selected_B_acquisition_miss",
                                0,
                            )
                        )
                    elif field == "route_balance":
                        values.append(
                            nested(row, "route_modes_early.executed.balance")
                        )
                    elif field == "positive_contexts":
                        values.append(
                            nested(
                                row,
                                "replay_mass_diagnostics.positive_contexts",
                                0,
                            )
                        )
                    else:
                        values.append(row.get(field, np.nan))
            axis.plot(
                rounds,
                values,
                marker="o",
                markersize=2.5,
                lw=1.7,
                color=colors[label],
                label=label,
            )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlim(-0.3, 15.3)
    for axis in axes[-1]:
        axis.set_xlabel("Expansion round")
    for axis in axes.flat[len(specs) :]:
        axis.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Frozen B1 gathering diagnostics (execution-only fields available at "
        "compact viz rounds r1-r10)",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "training_diagnostic.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def summary_value(out: Path, arm: str, round_i: int, gamma: float | None) -> dict:
    arrays = defaultdict(list)
    selected = GAMMAS if gamma is None else (gamma,)
    for value in selected:
        with load_cell(out, arm, round_i, value) as cell:
            for field in (
                "success",
                "collision_rate_event",
                "timeout_event",
                "validity_event",
                "minimum_clearance",
                "successful_time_to_goal",
                "route_label",
            ):
                arrays[field].append(cell[field])
    merged = {field: np.concatenate(values) for field, values in arrays.items()}
    route = merged["route_label"]
    u = int(np.count_nonzero(route == 1))
    r = int(np.count_nonzero(route == -1))
    resolved = u + r
    return {
        "SR": float(merged["success"].mean()),
        "CR": float(merged["collision_rate_event"].mean()),
        "timeout": float(merged["timeout_event"].mean()),
        "Validity": float(merged["validity_event"].mean()),
        "minimum_clearance": float(merged["minimum_clearance"].mean()),
        "successful_time_to_goal": (
            float(np.nanmean(merged["successful_time_to_goal"]))
            if np.isfinite(merged["successful_time_to_goal"]).any()
            else None
        ),
        "route": {
            "U": u,
            "R": r,
            "ambiguous": int(np.count_nonzero(route == 0)),
            "balance": 2.0 * min(u, r) / resolved if resolved else 0.0,
        },
    }


def scientific_comparison(out: Path) -> dict:
    directions = {
        "SR": "max",
        "CR": "min",
        "timeout": "min",
        "Validity": "max",
        "minimum_clearance": "max",
        "successful_time_to_goal": "min",
    }
    result = {}
    for arm in ("margin", "cost"):
        result[arm] = {}
        for gamma in (*GAMMAS, None):
            key = "pooled" if gamma is None else f"{gamma:g}"
            series = {
                round_i: summary_value(out, arm, round_i, gamma)
                for round_i in range(16)
            }
            entry = {
                f"r{round_i}": series[round_i]
                for round_i in (0, 5, 10, 15)
            }
            entry["descriptive_extrema"] = {}
            for metric, direction in directions.items():
                available = [
                    (round_i, row[metric])
                    for round_i, row in series.items()
                    if row[metric] is not None
                ]
                best = (
                    max(available, key=lambda item: item[1])
                    if direction == "max"
                    else min(available, key=lambda item: item[1])
                )
                worst = (
                    min(available, key=lambda item: item[1])
                    if direction == "max"
                    else max(available, key=lambda item: item[1])
                )
                entry["descriptive_extrema"][metric] = {
                    "best_round": best[0],
                    "best_value": best[1],
                    "worst_round": worst[0],
                    "worst_value": worst[1],
                    "selection_use": False,
                }
            result[arm][key] = entry
    return {
        "status": "B1_EXEC_RULE_SCIENTIFIC_COMPARISON_COMPLETE",
        "temperature": 1.0,
        "M_per_gamma": 200,
        "checkpoint_selection": "none; r0-r15 are all reported",
        "arms": result,
    }


def build(args) -> int:
    out = args.out
    if (out / "EVALUATION_COMPLETE.json").exists():
        raise FileExistsError(out / "EVALUATION_COMPLETE.json")
    gate = load_json(args.r0_gate)
    if gate.get("status") != "B1_EXEC_RULE_R0_EQUIVALENCE_PASS":
        raise RuntimeError("r0 gate did not pass")

    recipe = merge_recipe_diff(
        args.margin_arm / "recipe.json",
        args.existing_cost_arm / "recipe.json",
        args.cost_arm / "recipe.json",
    )
    if recipe["status"] != "B1_EXEC_RULE_RECIPE_PARITY_PASS":
        raise RuntimeError(
            "fresh cost recipe has forbidden differences: "
            f"{recipe['fresh_arm']['forbidden_differences']}"
        )
    write_json(out / "recipe_diff.json", recipe)

    scientific = out / "scientific"
    scientific.mkdir(exist_ok=True)
    if any(scientific.iterdir()):
        raise RuntimeError("scientific output directory must be empty before assembly")
    margin_shards = [args.margin_r0, args.margin_full]
    cost_shards = [args.cost_r0, args.cost_full]
    margin_hashes = copy_cells(margin_shards, scientific / "margin")
    cost_hashes = copy_cells(cost_shards, scientific / "cost")

    banks = [
        args.margin_r0 / "margin_common_bank.npy",
        args.margin_full / "margin_common_bank.npy",
        args.cost_r0 / "cost_common_bank.npy",
        args.cost_full / "cost_common_bank.npy",
    ]
    first_bank = banks[0].read_bytes()
    if any(path.read_bytes() != first_bank for path in banks[1:]):
        raise RuntimeError("common-bank files differ across evaluation shards")
    shutil.copyfile(banks[0], scientific / "common_bank.npy")

    margin_json = out / "margin_temp1_m200.jsonl"
    cost_json = out / "cost_temp1_m200.jsonl"
    canonical_jsonl(margin_shards, "margin", margin_json)
    canonical_jsonl(cost_shards, "cost", cost_json)

    render_curves(out)
    gallery = render_gallery(out)
    render_training_diagnostic(out, args.margin_arm, args.cost_arm)
    comparison = scientific_comparison(out)
    write_json(out / "scientific_comparison.json", comparison)

    declared = [
        margin_json,
        cost_json,
        out / "paired_execution_rule_curves.png",
        out / "paired_execution_rule_curves.pdf",
        out / "paired_execution_rule_gallery.png",
        out / "paired_execution_rule_gallery.pdf",
        out / "paired_execution_rule_gallery.json",
        out / "training_diagnostic.png",
        out / "recipe_diff.json",
        out / "scientific_comparison.json",
        scientific / "common_bank.npy",
        args.r0_gate,
    ]
    artifact_hashes = {
        str(path.relative_to(out)): sha256_file(path) for path in declared
    }
    checkpoint_hashes = {
        arm: {
            str(round_i): sha256_file(directory / f"ckpt_{round_i}.pt")
            for round_i in range(16)
        }
        for arm, directory in (
            ("margin", args.margin_arm),
            ("cost", args.cost_arm),
        )
    }
    evaluation = {
        "status": "B1_EXEC_RULE_TEMP1_M200_EVALUATION_COMPLETE",
        "base_frozen_sha": "5c0b753bae650bd189ce943a03d93871f29eb871",
        "automation_sha": args.automation_sha,
        "trainer_sha": "39fb3e63542e6b23efe3321f311505e553c0bec6",
        "recipe_parity": recipe["status"],
        "r0_equivalence": gate["status"],
        "arms": {
            "margin": str(args.margin_arm),
            "cost": str(args.cost_arm),
        },
        "temperature": 1.0,
        "M_per_gamma": 200,
        "gammas": list(GAMMAS),
        "rounds": list(range(16)),
        "cells_per_arm": 112,
        "bank_split": "b1_exec_rule_temp1_m200_v1",
        "common_bank_file_sha256": sha256_file(scientific / "common_bank.npy"),
        "checkpoint_sha256": checkpoint_hashes,
        "scientific_archive_sha256": {
            "margin": margin_hashes,
            "cost": cost_hashes,
        },
        "artifact_sha256": artifact_hashes,
        "gallery": gallery,
        "parameter_changes_after_results": False,
        "checkpoint_or_temperature_selection": False,
    }
    write_json(out / "EVALUATION_COMPLETE.json", evaluation)
    evaluation_sha = sha256_file(out / "EVALUATION_COMPLETE.json")
    delivery_artifacts = {
        **artifact_hashes,
        "EVALUATION_COMPLETE.json": evaluation_sha,
    }
    delivery = {
        "status": "B1_EXEC_RULE_DELIVERY_COMPLETE",
        "evaluation_complete_sha256": evaluation_sha,
        "artifact_sha256": delivery_artifacts,
        "validation": {
            relative: sha256_file(out / relative) == digest
            for relative, digest in delivery_artifacts.items()
        },
    }
    if not all(delivery["validation"].values()):
        raise RuntimeError("declared artifact hash failed validation")
    write_json(out / "DELIVERY_COMPLETE.json", delivery)
    print(out / "DELIVERY_COMPLETE.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("gate-r0")
    gate.add_argument("--margin-ckpt", type=Path, required=True)
    gate.add_argument("--cost-ckpt", type=Path, required=True)
    gate.add_argument("--margin-shard", type=Path, required=True)
    gate.add_argument("--cost-shard", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    gate.set_defaults(func=gate_r0)

    builder = subparsers.add_parser("build")
    builder.add_argument("--out", type=Path, required=True)
    builder.add_argument("--margin-arm", type=Path, required=True)
    builder.add_argument("--cost-arm", type=Path, required=True)
    builder.add_argument("--existing-cost-arm", type=Path, required=True)
    builder.add_argument("--margin-r0", type=Path, required=True)
    builder.add_argument("--cost-r0", type=Path, required=True)
    builder.add_argument("--margin-full", type=Path, required=True)
    builder.add_argument("--cost-full", type=Path, required=True)
    builder.add_argument("--r0-gate", type=Path, required=True)
    builder.add_argument("--automation-sha", required=True)
    builder.set_defaults(func=build)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
