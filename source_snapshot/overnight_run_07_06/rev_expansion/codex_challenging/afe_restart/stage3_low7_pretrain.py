#!/usr/bin/env python3
"""Matched fresh pretraining for paired, full-space low7 demonstrations.

This is additive to :mod:`stage3_pretrain`.  It changes exactly two scientific
inputs: each context carries the closest-boundary vector, and complete expert
trajectories use uniformly sampled start/goal pairs.  Targets remain the exact
pre-execution H=10 plans selected from full-verifier-positive SafeMPPI outputs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import grid_hp_expt as HP
import grid_feats as GF

from .deps import sha256_file, write_dependency_manifest
from .policy import model_state_hash
from .schemas import QueryContext, query_content_hash
from .stage3_pretrain import seeded_cfm_loss
from .scene import make_id_scene
from .stage2_low7_randomized import _scene_geometry_sha256


DATA_SCHEMA = "afe_planned_demo_v3_low7_uniform_pairs"
TRAIN_SCHEMA = "afe_fresh_pretrain_v2_low7_uniform_pairs"
REFLECTION_TRAIN_SCHEMA = "afe_fresh_pretrain_v3_low7_reflection_paired"
EQUIVARIANT_TRAIN_SCHEMA = "afe_fresh_pretrain_v4_low7_reflection_equivariant"
GROUP_AVERAGED_TRAIN_SCHEMA = "afe_fresh_pretrain_v5_low7_reflection_group_average"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _canonical_gamma(value: float) -> float:
    matches = [gamma for gamma in GAMMAS if abs(float(value) - gamma) <= 5e-7]
    if len(matches) != 1:
        raise ValueError(f"gamma={value!r} is not a declared conditioning level")
    return float(matches[0])


def polar_reflection_indices(n_theta: int = 32) -> torch.Tensor:
    """Return the exact polar-ray permutation induced by ``x <-> y``."""

    if n_theta <= 0:
        raise ValueError("n_theta must be positive")
    theta = -np.pi + (np.arange(n_theta) + 0.5) * 2.0 * np.pi / n_theta
    reflected_source = np.mod(np.pi / 2.0 - theta + np.pi, 2.0 * np.pi) - np.pi
    distance = np.abs(
        np.angle(np.exp(1j * (theta[None, :] - reflected_source[:, None])))
    )
    indices = np.argmin(distance, axis=1)
    if len(set(int(value) for value in indices)) != n_theta:
        raise RuntimeError("x/y polar reflection is not a permutation")
    return torch.as_tensor(indices, dtype=torch.long)


def reflect_low7_batch(
    grid: torch.Tensor,
    low7: torch.Tensor,
    hist: torch.Tensor,
    plans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reflect one low7 batch across the world-frame diagonal ``x=y``."""

    if tuple(grid.shape[1:]) != (3, 32, 32):
        raise ValueError(f"grid must have shape [B,3,32,32], got {tuple(grid.shape)}")
    if tuple(low7.shape[1:]) != (7,):
        raise ValueError(f"low7 must have shape [B,7], got {tuple(low7.shape)}")
    if tuple(hist.shape[1:]) != (16, 2) or tuple(plans.shape[1:]) != (10, 2):
        raise ValueError("history/plan shapes must be [B,16,2] and [B,10,2]")
    if len({len(grid), len(low7), len(hist), len(plans)}) != 1:
        raise ValueError("reflection inputs must have the same batch length")
    indices = polar_reflection_indices(grid.shape[-2]).to(grid.device)
    reflected_grid = grid.index_select(-2, indices)
    reflected_low7 = low7[:, (1, 0, 3, 2, 5, 4, 6)]
    return reflected_grid, reflected_low7, hist.flip(-1), plans.flip(-1)


def reflection_paired_cfm_terms(
    policy: torch.nn.Module,
    grid: torch.Tensor,
    low7: torch.Tensor,
    hist: torch.Tensor,
    plans: torch.Tensor,
    *,
    generator: torch.Generator,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CFM and direct velocity-equivariance losses on exact pairs."""

    reflected = reflect_low7_batch(grid, low7, hist, plans)

    def interleave(original: torch.Tensor, mirror: torch.Tensor) -> torch.Tensor:
        return torch.stack((original, mirror), dim=1).flatten(0, 1)

    paired_grid = interleave(grid, reflected[0])
    paired_low7 = interleave(low7, reflected[1])
    paired_hist = interleave(hist, reflected[2])
    paired_plans = interleave(plans, reflected[3])
    pair_count = len(plans)
    x1 = (paired_plans / float(policy.u_max)).reshape(2 * pair_count, int(policy.d))
    x0_original = torch.randn(
        (pair_count, int(policy.d)),
        device=x1.device,
        dtype=x1.dtype,
        generator=generator,
    )
    x0_reflected = x0_original.reshape(pair_count, -1, 2).flip(-1).reshape_as(
        x0_original
    )
    x0 = interleave(x0_original, x0_reflected)
    tau_original = torch.rand(
        pair_count, device=x1.device, dtype=x1.dtype, generator=generator
    ).clamp_(1.0e-4, 1.0)
    tau = torch.repeat_interleave(tau_original, 2)
    x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
    context = policy.ctx_from(paired_grid, paired_low7, paired_hist)
    prediction = policy(x_tau, tau, context)
    per_sample = ((prediction - (x1 - x0)) ** 2).mean(dim=1)
    original_prediction = prediction[0::2]
    reflected_prediction = prediction[1::2]
    reflected_original_prediction = original_prediction.reshape(
        pair_count, -1, 2
    ).flip(-1).reshape_as(original_prediction)
    per_pair_equivariance = (
        (reflected_prediction - reflected_original_prediction) ** 2
    ).mean(dim=1)
    if sample_weight is None:
        return per_sample.mean(), per_pair_equivariance.mean()
    weights = torch.repeat_interleave(
        sample_weight.to(device=per_sample.device, dtype=per_sample.dtype), 2
    )
    if weights.shape != per_sample.shape or bool((weights <= 0.0).any()):
        raise ValueError("sample_weight must be one positive scalar per source row")
    cfm = (per_sample * weights).sum() / weights.sum()
    pair_weights = sample_weight.to(
        device=per_sample.device, dtype=per_sample.dtype
    )
    equivariance = (
        per_pair_equivariance * pair_weights
    ).sum() / pair_weights.sum()
    return cfm, equivariance


def reflection_paired_cfm_loss(
    policy: torch.nn.Module,
    grid: torch.Tensor,
    low7: torch.Tensor,
    hist: torch.Tensor,
    plans: torch.Tensor,
    *,
    generator: torch.Generator,
    sample_weight: torch.Tensor | None = None,
    equivariance_weight: float = 0.0,
) -> torch.Tensor:
    """CFM plus an explicit reflected-velocity consistency penalty."""

    cfm, equivariance = reflection_paired_cfm_terms(
        policy,
        grid,
        low7,
        hist,
        plans,
        generator=generator,
        sample_weight=sample_weight,
    )
    return cfm + float(equivariance_weight) * equivariance


@dataclass(frozen=True)
class Low7Pool:
    grid: torch.Tensor
    low7: torch.Tensor
    hist: torch.Tensor
    plans: torch.Tensor
    gamma: torch.Tensor
    pair_ids: torch.Tensor
    trajectory_ids: torch.Tensor
    trajectory_weight: torch.Tensor
    trajectory_rows: tuple[Mapping[str, Any], ...]
    query_hashes: tuple[str, ...]
    declared_pair_ids: tuple[int, ...]
    source: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.plans)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _fresh_outdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _tensor(payload: Mapping[str, Any], key: str, shape_tail: tuple[int, ...]) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"dataset field {key!r} must be a tensor")
    if tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{key} must have shape [N,{','.join(map(str, shape_tail))}], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} contains nonfinite values")
    return value.contiguous()


def _tie_mean_boundary_vectors(
    positions: torch.Tensor,
    obstacles: np.ndarray,
    robot_radius: float,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    output = np.empty((len(positions), 2), dtype=np.float32)
    obstacle_array = np.asarray(obstacles, dtype=np.float64)
    for begin in range(0, len(positions), chunk_size):
        points = positions[begin : begin + chunk_size].double().numpy()[:, :2]
        delta = obstacle_array[None, :, :2] - points[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        clearance = distance - obstacle_array[None, :, 2] - float(robot_radius)
        minimum = clearance.min(axis=1)
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(minimum))
        tied = np.abs(clearance - minimum[:, None]) <= tolerance[:, None]
        direction = np.divide(
            delta,
            distance[:, :, None],
            out=np.zeros_like(delta),
            where=distance[:, :, None] > 1.0e-12,
        )
        vectors = direction * clearance[:, :, None] / float(GF.SENSING)
        averaged = (vectors * tied[:, :, None]).sum(axis=1) / tied.sum(
            axis=1, keepdims=True
        )
        averaged[minimum > float(GF.SENSING)] = 0.0
        output[begin : begin + len(points)] = averaged.astype(np.float32)
    return torch.from_numpy(output)


def load_pool(manifest_path: Path, *, tie_average_boundary: bool = False) -> Low7Pool:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != DATA_SCHEMA:
        raise ValueError(f"expected {DATA_SCHEMA}, got {manifest.get('schema_version')!r}")
    pair_count = int(manifest.get("pair_count", -1))
    if pair_count <= 0:
        raise ValueError("combined manifest must declare the immutable endpoint-bank size")
    declared_pair_ids = tuple(range(pair_count))
    dataset_path = Path(manifest["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = (manifest_path.parent / dataset_path).resolve()
    expected_sha = str(manifest["dataset_sha256"])
    if sha256_file(dataset_path) != expected_sha:
        raise RuntimeError("low7 dataset checksum mismatch")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != DATA_SCHEMA:
        raise ValueError("manifest and dataset schema differ")
    grid = _tensor(payload, "grid", (3, 32, 32)).float()
    low7 = _tensor(payload, "low7", (7,)).float()
    hist = _tensor(payload, "hist", (16, 2)).float()
    plans = _tensor(payload, "U", (10, 2)).float()
    count = len(plans)
    for name, tensor in (("grid", grid), ("low7", low7), ("hist", hist)):
        if len(tensor) != count:
            raise ValueError(f"{name}/plan length mismatch")
    gamma = payload["gamma"].double().reshape(-1)
    pair_ids = payload["window_pair_ids"].long().reshape(-1)
    trajectory_ids = payload["window_trajectory_ids"].long().reshape(-1)
    trajectory_weight = payload["trajectory_balanced_weight"].double().reshape(-1)
    if any(len(value) != count for value in (gamma, pair_ids, trajectory_ids, trajectory_weight)):
        raise ValueError("provenance tensor length mismatch")
    observed_pair_ids = {int(value) for value in torch.unique(pair_ids).tolist()}
    if not observed_pair_ids.issubset(set(declared_pair_ids)):
        raise ValueError("dataset contains a pair id outside the immutable endpoint bank")
    if not bool((trajectory_weight > 0).all()):
        raise ValueError("trajectory weights must be positive")
    if not bool(torch.isclose(low7[:, -1].double(), gamma, atol=5e-7).all()):
        raise ValueError("low7[-1] must be the serialized gamma")
    declared = torch.tensor(GAMMAS, dtype=torch.float64)
    if not bool((torch.min(torch.abs(gamma[:, None] - declared[None]), dim=1).values <= 5e-7).all()):
        raise ValueError("dataset contains an undeclared gamma")
    for label in ("target_safe", "target_in_bounds", "target_socp_ok"):
        value = payload.get(label)
        if not isinstance(value, torch.Tensor) or len(value) != count or not bool(value.bool().all()):
            raise ValueError(f"every training target must satisfy {label}")
    fingerprints = payload.get("verifier_spec_fingerprint")
    states = payload.get("verifier_state")
    hashes = tuple(str(value) for value in payload.get("query_hashes", ()))
    if not isinstance(states, torch.Tensor) or tuple(states.shape) != (count, 4):
        raise ValueError("verifier_state must have shape [N,4]")
    if len(fingerprints) != count or len(hashes) != count:
        raise ValueError("query identity arrays have the wrong length")
    # Re-hash every exact generated/verified/training object before optimizing.
    for index in range(count):
        context = QueryContext(
            grid[index].numpy(),
            low7[index].numpy(),
            hist[index].numpy(),
            states[index].double().numpy(),
            str(fingerprints[index]),
        )
        actual = query_content_hash(
            context, _canonical_gamma(float(gamma[index])), plans[index].numpy()
        )
        if actual != hashes[index]:
            raise RuntimeError(f"query identity mismatch at dataset row {index}")
    trajectory_rows = tuple(payload.get("trajectory_rows", ()))
    if not trajectory_rows:
        raise ValueError("dataset has no trajectory provenance")
    row_ids = {int(row["trajectory_id"]) for row in trajectory_rows}
    if row_ids != {int(value) for value in torch.unique(trajectory_ids).tolist()}:
        raise ValueError("trajectory rows do not match window trajectory ids")
    endpoint_path = Path(manifest["endpoint_manifest"])
    if not endpoint_path.is_absolute():
        endpoint_path = (manifest_path.parent / endpoint_path).resolve()
    if sha256_file(endpoint_path) != str(manifest["endpoint_manifest_sha256"]):
        raise RuntimeError("low7 endpoint-manifest checksum mismatch")
    endpoint_payload = json.loads(endpoint_path.read_text())
    boundary_transform = None
    if tie_average_boundary:
        if (
            endpoint_payload.get("schema_version")
            != "afe_low7_fixed_goal_full_grid_endpoint_manifest_v1"
        ):
            raise RuntimeError("tie-mean low7 pretraining requires the fixed-goal grid bank")
        env = make_id_scene(goal=np.asarray((4.7, 4.7), dtype=np.float32))
        if manifest.get("scene", {}).get("geometry_sha256") != _scene_geometry_sha256(env):
            raise RuntimeError("tie-mean transform scene differs from the source dataset")
        transformed = low7.clone()
        transformed[:, 4:6] = _tie_mean_boundary_vectors(
            states,
            env.obstacles.detach().cpu().numpy(),
            float(env.r_robot),
        )
        low7 = transformed
        boundary_transform = {
            "name": "equal-nearest-boundary-vector-mean-v1",
            "source_low7_authenticated_before_transform": True,
            "transformed_low7_sha256": hashlib.sha256(
                low7.numpy().tobytes(order="C")
            ).hexdigest(),
        }
    return Low7Pool(
        grid=grid,
        low7=low7,
        hist=hist,
        plans=plans,
        gamma=gamma,
        pair_ids=pair_ids,
        trajectory_ids=trajectory_ids,
        trajectory_weight=trajectory_weight,
        trajectory_rows=trajectory_rows,
        query_hashes=hashes,
        declared_pair_ids=declared_pair_ids,
        source={
            "manifest": str(manifest_path),
            "dataset": str(dataset_path),
            "sha256": expected_sha,
            "endpoint_manifest": str(endpoint_path),
            "endpoint_manifest_sha256": str(manifest["endpoint_manifest_sha256"]),
            "endpoint_schema": str(endpoint_payload.get("schema_version")),
            "endpoint_sampling": dict(endpoint_payload.get("sampling", {})),
            "conditioning_transform": boundary_transform,
        },
    )


def paired_split(pool: Low7Pool, *, validation_pairs: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Hold out pair ids across every gamma, preventing endpoint leakage."""

    # Split the immutable endpoint bank, not the success-conditioned subset of
    # pairs that happened to yield expert targets.  Failed pairs remain honest
    # missingness and therefore cannot alter train/validation membership.
    pair_ids = list(pool.declared_pair_ids)
    if not 0 < validation_pairs < len(pair_ids):
        raise ValueError("validation_pairs must leave at least one train pair")
    shuffled = np.asarray(pair_ids, dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffled)
    validation = sorted(int(value) for value in shuffled[:validation_pairs])
    train = sorted(int(value) for value in shuffled[validation_pairs:])
    validation_mask = torch.isin(pool.pair_ids, torch.tensor(validation, dtype=torch.long))
    train_mask = torch.isin(pool.pair_ids, torch.tensor(train, dtype=torch.long))
    if bool((train_mask & validation_mask).any()) or not bool((train_mask | validation_mask).all()):
        raise RuntimeError("pair split does not partition the dataset")
    train_rows = torch.where(train_mask)[0]
    validation_rows = torch.where(validation_mask)[0]
    audit = {
        "seed": int(seed),
        "train_pair_ids": train,
        "validation_pair_ids": validation,
        "pair_leakage": 0,
        "train_windows": len(train_rows),
        "validation_windows": len(validation_rows),
        "train_pairs_without_targets": sorted(
            set(train) - {int(value) for value in torch.unique(pool.pair_ids[train_mask]).tolist()}
        ),
        "validation_pairs_without_targets": sorted(
            set(validation)
            - {int(value) for value in torch.unique(pool.pair_ids[validation_mask]).tolist()}
        ),
        "per_gamma": {},
    }
    for gamma in GAMMAS:
        match = torch.isclose(pool.gamma, torch.tensor(gamma, dtype=torch.float64), atol=5e-7)
        audit["per_gamma"][f"{gamma:g}"] = {
            "train_trajectories": len(torch.unique(pool.trajectory_ids[match & train_mask])),
            "validation_trajectories": len(torch.unique(pool.trajectory_ids[match & validation_mask])),
        }
    return train_rows, validation_rows, audit


def _amp(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _objective_weights(pool: Low7Pool, rows: torch.Tensor) -> torch.Tensor:
    """Equal gamma mass, equal trajectory mass within each gamma."""

    result = torch.zeros(len(pool), dtype=torch.float64)
    for gamma in GAMMAS:
        mask = torch.isclose(
            pool.gamma[rows], torch.tensor(gamma, dtype=torch.float64), atol=5e-7
        )
        gamma_rows = rows[mask]
        trajectories = torch.unique(pool.trajectory_ids[gamma_rows])
        if not len(trajectories):
            raise ValueError(f"split has no trajectories for gamma={gamma:g}")
        result[gamma_rows] = pool.trajectory_weight[gamma_rows] / len(trajectories)
    return result


def _cfm_eval(
    model,
    pool: Low7Pool,
    rows: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: torch.device,
    batch: int,
    seed: int,
    amp: bool,
    reflection_paired: bool = False,
    equivariance_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total_objective = 0.0
    total_cfm = 0.0
    total_equivariance = 0.0
    mass = 0.0
    with torch.no_grad():
        for offset in range(0, len(rows), batch):
            index = rows[offset : offset + batch]
            grid = pool.grid[index].to(device)
            low = pool.low7[index].to(device)
            hist = pool.hist[index].to(device)
            plans = pool.plans[index].to(device)
            weight = weights[index].to(device)
            with _amp(device, amp):
                if reflection_paired:
                    cfm, equivariance = reflection_paired_cfm_terms(
                        model,
                        grid,
                        low,
                        hist,
                        plans,
                        generator=generator,
                        sample_weight=weight,
                    )
                    loss = cfm + float(equivariance_weight) * equivariance
                else:
                    cfm = seeded_cfm_loss(
                        model,
                        grid,
                        low,
                        hist,
                        plans,
                        generator=generator,
                        sample_weight=weight,
                    )
                    equivariance = torch.zeros_like(cfm)
                    loss = cfm
            batch_mass = float(weight.sum())
            total_objective += float(loss) * batch_mass
            total_cfm += float(cfm) * batch_mass
            total_equivariance += float(equivariance) * batch_mass
            mass += batch_mass
    return {
        "objective": total_objective / mass,
        "cfm": total_cfm / mass,
        "equivariance": total_equivariance / mass,
    }


def _plot_history(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(epochs, [row["train_cfm"] for row in rows], label="train CFM")
    axis.plot(epochs, [row["validation_cfm"] for row in rows], label="held-out CFM")
    if any(float(row["equivariance_weight"]) > 0.0 for row in rows):
        axis.plot(
            epochs,
            [row["train_equivariance"] for row in rows],
            label="train equivariance",
        )
        axis.plot(
            epochs,
            [row["validation_equivariance"] for row in rows],
            label="held-out equivariance",
        )
    axis.set(xlabel="epoch", ylabel="trajectory-balanced loss")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    outdir = args.outdir.resolve()
    _fresh_outdir(outdir)
    for name in ("data", "logs", "tables", "viz"):
        (outdir / name).mkdir(exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    dependencies = write_dependency_manifest(outdir / "logs/dependencies.json")
    pool = load_pool(
        args.manifest,
        tie_average_boundary=args.reflection_group_average,
    )
    train_rows, validation_rows, split_audit = paired_split(
        pool, validation_pairs=args.validation_pairs, seed=args.split_seed
    )
    train_weight = _objective_weights(pool, train_rows)
    validation_weight = _objective_weights(pool, validation_rows)
    _atomic_json(outdir / "logs/pair_split.json", split_audit)
    model = HP.GridHPFlowPolicy(
        repr_dim=32,
        grid_hw=(32, 32),
        trunk_hidden=tuple(args.trunk_hidden),
        enc_depth=args.enc_depth,
        raw_condition_dim=7,
        conditioning_schema=(
            "low7_closest_boundary_tie_mean"
            if args.reflection_group_average
            else "low7_closest_boundary"
        ),
        reflection_group_average=args.reflection_group_average,
    ).to(device)
    config = model.config()
    if model.ctx_dim != 39 or model.trunk[0].in_features != 91:
        raise RuntimeError("low7 model must have ctx=39 and trunk input=91")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def lr_factor(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        fraction = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    sampler = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    cfm_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_validation_cfm = math.inf
    best_validation_equivariance = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    csv_path = outdir / "tables/training_history.csv"
    fieldnames = (
        "epoch", "train_objective", "train_cfm", "train_equivariance",
        "validation_objective", "validation_cfm", "validation_equivariance",
        "equivariance_weight", "learning_rate", "encoder_gradient_norm",
        "epoch_seconds",
    )
    with csv_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        rows = train_rows[torch.randperm(len(train_rows), generator=sampler)]
        model.train()
        epoch_mass = float(train_weight[rows].sum())
        batch_count = math.ceil(len(rows) / args.batch_size)
        target_mass = epoch_mass / batch_count
        weighted_objective = 0.0
        weighted_cfm = 0.0
        weighted_equivariance = 0.0
        observed_mass = 0.0
        encoder_norm = 0.0
        for offset in range(0, len(rows), args.batch_size):
            index = rows[offset : offset + args.batch_size]
            grid = pool.grid[index].to(device)
            low = pool.low7[index].to(device)
            hist = pool.hist[index].to(device)
            plans = pool.plans[index].to(device)
            weight = train_weight[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device, args.amp):
                if args.reflection_paired_pretraining:
                    cfm, equivariance = reflection_paired_cfm_terms(
                        model,
                        grid,
                        low,
                        hist,
                        plans,
                        generator=cfm_rng,
                        sample_weight=weight,
                    )
                    loss = cfm + args.equivariance_weight * equivariance
                else:
                    cfm = seeded_cfm_loss(
                        model,
                        grid,
                        low,
                        hist,
                        plans,
                        generator=cfm_rng,
                        sample_weight=weight,
                    )
                    equivariance = torch.zeros_like(cfm)
                    loss = cfm
            batch_mass = float(weight.sum())
            (loss * (batch_mass / target_mass)).backward()
            encoder_norm += math.sqrt(
                sum(
                    float((parameter.grad.detach().float() ** 2).sum())
                    for parameter in model.enc_grid.parameters()
                    if parameter.grad is not None
                )
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            weighted_objective += float(loss.detach()) * batch_mass
            weighted_cfm += float(cfm.detach()) * batch_mass
            weighted_equivariance += float(equivariance.detach()) * batch_mass
            observed_mass += batch_mass
        scheduler.step()
        validation = _cfm_eval(
            model,
            pool,
            validation_rows,
            validation_weight,
            device=device,
            batch=args.validation_batch_size,
            seed=args.seed + 10_000,
            amp=args.amp,
            reflection_paired=args.reflection_paired_pretraining,
            equivariance_weight=args.equivariance_weight,
        )
        if validation["objective"] < best_loss:
            best_loss = validation["objective"]
            best_validation_cfm = validation["cfm"]
            best_validation_equivariance = validation["equivariance"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_objective": weighted_objective / observed_mass,
            "train_cfm": weighted_cfm / observed_mass,
            "train_equivariance": weighted_equivariance / observed_mass,
            "validation_objective": validation["objective"],
            "validation_cfm": validation["cfm"],
            "validation_equivariance": validation["equivariance"],
            "equivariance_weight": args.equivariance_weight,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_gradient_norm": encoder_norm / batch_count,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        if epoch % 10 == 0 or epoch == args.epochs - 1 or epoch == best_epoch:
            print(
                f"[low7 pretrain {epoch:03d}/{args.epochs}] "
                f"cfm={row['train_cfm']:.6f} eq={row['train_equivariance']:.6f} "
                f"val={validation['objective']:.6f} best={best_loss:.6f}@{best_epoch}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("training produced no model")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    state_sha = model_state_hash(model)
    query_digest = hashlib.sha256("".join(pool.query_hashes).encode()).hexdigest()
    fixed_goal_grid = (
        pool.source.get("endpoint_schema")
        == "afe_low7_fixed_goal_full_grid_endpoint_manifest_v1"
    )
    extra = {
        "stage_schema": (
            GROUP_AVERAGED_TRAIN_SCHEMA
            if args.reflection_group_average
            else EQUIVARIANT_TRAIN_SCHEMA
            if args.equivariance_weight > 0.0
            else REFLECTION_TRAIN_SCHEMA if args.reflection_paired_pretraining
            else TRAIN_SCHEMA
        ),
        "fresh_from_scratch": True,
        "endpoint_free": True,
        "domain_randomized_start_goal": not fixed_goal_grid,
        "domain_randomized_start": True,
        "fixed_goal": ([4.7, 4.7] if fixed_goal_grid else None),
        "zero_initial_velocity": True,
        "diagonal_start_exclusion": False,
        "source_manifest": str(args.manifest.resolve()),
        "source_query_hash_digest": query_digest,
        "conditioning_transform": pool.source.get("conditioning_transform"),
        "model_state_sha256": state_sha,
        "best_epoch": best_epoch,
        "best_validation_objective": best_loss,
        "best_validation_cfm": best_validation_cfm,
        "best_validation_equivariance": best_validation_equivariance,
        "equivariance_weight": args.equivariance_weight,
        "encoder_trainable_during_pretraining": True,
        "reflection_paired_pretraining": bool(args.reflection_paired_pretraining),
        "reflection_group_average": bool(args.reflection_group_average),
        "reflection_pair_contract": (
            "each source row and its exact x/y reflection share tau and reflected x0; "
            "the pair retains the source row's gamma/trajectory objective mass"
            if args.reflection_paired_pretraining
            else None
        ),
        "expansion_promotion": False,
        "promotion_reason": "awaiting user review of nominal and two OOD pretrained evaluations",
    }
    checkpoint = outdir / "data/checkpoint_candidate.pt"
    phi_candidate = outdir / "data/phi0_frozen_candidate.pt"
    HP.save_hp(model, checkpoint, extra=extra)
    HP.save_hp(
        model,
        phi_candidate,
        extra={**extra, "frozen_feature_snapshot": True, "feature_time": 0.9, "feature_dimension": 32},
    )
    _plot_history(history, outdir / "viz/training_history.png")
    summary = {
        "schema_version": extra["stage_schema"],
        "status": "LOW7_PRETRAINED_AWAITING_OOD_REVIEW",
        "finished_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": {
            "config": config,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state_sha256": state_sha,
        },
        "data": {
            **dict(pool.source),
            "windows": len(pool),
            "trajectories": len(pool.trajectory_rows),
            "unique_pairs": len(torch.unique(pool.pair_ids)),
            "split": split_audit,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "best_epoch": best_epoch,
            "best_validation_objective": best_loss,
            "best_validation_cfm": best_validation_cfm,
            "best_validation_equivariance": best_validation_equivariance,
            "equivariance_weight": args.equivariance_weight,
            "reflection_paired_pretraining": bool(
                args.reflection_paired_pretraining
            ),
            "reflection_group_average": bool(args.reflection_group_average),
            "effective_examples_per_source_window": (
                2 if args.reflection_paired_pretraining else 1
            ),
        },
        "contract": {
            "exact_verified_planned_H10_targets": True,
            "all_train_targets_seen_each_epoch": True,
            "inverse_trajectory_length_weighting": True,
            "equal_gamma_objective_mass": True,
            "pair_level_train_validation_split": True,
            "pair_leakage": 0,
            "absolute_start_goal_inputs": False,
            "closest_boundary_vector_inputs": True,
            "equal_nearest_boundary_tie_mean": bool(
                pool.source.get("conditioning_transform")
            ),
            "gamma_last": True,
            "encoder_frozen_during_pretraining": False,
            "exact_xy_reflection_pairs": bool(args.reflection_paired_pretraining),
            "paired_cfm_source_noise_and_time": bool(
                args.reflection_paired_pretraining
            ),
            "explicit_velocity_equivariance_loss": bool(
                args.equivariance_weight > 0.0
            ),
            "exact_reflection_group_averaged_velocity": bool(
                args.reflection_group_average
            ),
            "expansion_started": False,
        },
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "phi_candidate": str(phi_candidate),
            "phi_candidate_sha256": sha256_file(phi_candidate),
        },
        "dependencies": dependencies,
    }
    _atomic_json(outdir / "manifest.json", summary)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--trunk-hidden", nargs="+", type=int, default=(160, 96))
    parser.add_argument("--enc-depth", type=int, default=3)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--split-seed", type=int, default=31_711)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reflection-paired-pretraining",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train on every verified source row and its exact x/y reflection, "
            "with paired CFM source noise and time"
        ),
    )
    parser.add_argument(
        "--equivariance-weight",
        type=float,
        default=0.0,
        help="weight on direct v(Rx,Rc)=R v(x,c) consistency; requires reflection pairs",
    )
    parser.add_argument(
        "--reflection-group-average",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use the exact x/y group-averaged velocity field; requires "
            "reflection-paired pretraining"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if min(args.epochs, args.batch_size, args.validation_batch_size) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if not math.isfinite(args.equivariance_weight) or args.equivariance_weight < 0.0:
        raise ValueError("equivariance-weight must be finite and non-negative")
    if args.equivariance_weight > 0.0 and not args.reflection_paired_pretraining:
        raise ValueError("equivariance-weight requires reflection-paired-pretraining")
    if args.reflection_group_average and not args.reflection_paired_pretraining:
        raise ValueError("reflection-group-average requires reflection-paired-pretraining")
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
