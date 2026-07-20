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

from .deps import sha256_file, write_dependency_manifest
from .policy import model_state_hash
from .schemas import QueryContext, query_content_hash
from .stage3_pretrain import seeded_cfm_loss


DATA_SCHEMA = "afe_planned_demo_v3_low7_uniform_pairs"
TRAIN_SCHEMA = "afe_fresh_pretrain_v2_low7_uniform_pairs"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _canonical_gamma(value: float) -> float:
    matches = [gamma for gamma in GAMMAS if abs(float(value) - gamma) <= 5e-7]
    if len(matches) != 1:
        raise ValueError(f"gamma={value!r} is not a declared conditioning level")
    return float(matches[0])


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


def load_pool(manifest_path: Path) -> Low7Pool:
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
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
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
                loss = seeded_cfm_loss(
                    model, grid, low, hist, plans, generator=generator, sample_weight=weight
                )
            batch_mass = float(weight.sum())
            total += float(loss) * batch_mass
            mass += batch_mass
    return total / mass


def _plot_history(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(epochs, [row["train_cfm"] for row in rows], label="train CFM")
    axis.plot(epochs, [row["validation_cfm"] for row in rows], label="held-out-pair CFM")
    axis.set(xlabel="epoch", ylabel="trajectory-balanced CFM loss")
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
    pool = load_pool(args.manifest)
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
        conditioning_schema="low7_closest_boundary",
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
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    csv_path = outdir / "tables/training_history.csv"
    fieldnames = (
        "epoch", "train_cfm", "validation_cfm", "learning_rate",
        "encoder_gradient_norm", "epoch_seconds",
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
        weighted_loss = 0.0
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
                loss = seeded_cfm_loss(
                    model, grid, low, hist, plans, generator=cfm_rng, sample_weight=weight
                )
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
            weighted_loss += float(loss.detach()) * batch_mass
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
        )
        if validation < best_loss:
            best_loss = validation
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_cfm": weighted_loss / observed_mass,
            "validation_cfm": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_gradient_norm": encoder_norm / batch_count,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        with csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        if epoch % 10 == 0 or epoch == args.epochs - 1 or epoch == best_epoch:
            print(
                f"[low7 pretrain {epoch:03d}/{args.epochs}] train={row['train_cfm']:.6f} "
                f"val={validation:.6f} best={best_loss:.6f}@{best_epoch}",
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
        "stage_schema": TRAIN_SCHEMA,
        "fresh_from_scratch": True,
        "endpoint_free": True,
        "domain_randomized_start_goal": not fixed_goal_grid,
        "domain_randomized_start": True,
        "fixed_goal": ([4.7, 4.7] if fixed_goal_grid else None),
        "zero_initial_velocity": True,
        "diagonal_start_exclusion": False,
        "source_manifest": str(args.manifest.resolve()),
        "source_query_hash_digest": query_digest,
        "model_state_sha256": state_sha,
        "best_epoch": best_epoch,
        "best_validation_cfm": best_loss,
        "encoder_trainable_during_pretraining": True,
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
        "schema_version": TRAIN_SCHEMA,
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
            "best_validation_cfm": best_loss,
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
            "gamma_last": True,
            "encoder_frozen_during_pretraining": False,
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if min(args.epochs, args.batch_size, args.validation_batch_size) <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
