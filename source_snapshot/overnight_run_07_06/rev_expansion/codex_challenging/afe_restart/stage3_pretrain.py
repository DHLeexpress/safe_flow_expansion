#!/usr/bin/env python3
"""Stage 03: fresh endpoint-free pretraining on verified planned windows.

This loader intentionally accepts only the clean Stage-02 schema.  Every row
is re-hashed and must carry an explicit positive full-verifier label before it
can become a CFM target.  Splits are made by complete source trajectory and
the optimizer sees exactly equal mass from every ``(gamma, R/U)`` stratum,
without reflection, padding, or executed-window reconstruction.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import grid_hp_expt as HP

from .config import GAMMAS, clean_method_absence_manifest
from .deps import (
    assert_no_legacy_expansion_imports,
    sha256_file,
    write_dependency_manifest,
)
from .dynamics import step_state
from .policy import model_state_hash
from .schemas import QueryContext, query_content_hash


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE2_MANIFEST = PACKAGE_ROOT / "stage_results/02_planned_demos/manifest.json"
DEFAULT_OUTDIR = PACKAGE_ROOT / "stage_results/03_pretrain"
SCHEMA_VERSION = "afe_planned_demo_v2_exact_verifier_identity"
PRETRAIN_SCHEMA = "afe_fresh_pretrain_v1"
MODE_NAMES = {0: "R-first", 1: "U-first"}
START = np.asarray((0.5, 0.5), dtype=np.float32)
GOAL = np.asarray((4.5, 4.5), dtype=np.float32)
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


class DatasetContractError(ValueError):
    """Raised before training when a Stage-02 target violates the contract."""


@dataclass(frozen=True)
class PlannedDemoPool:
    """Validated, combined Stage-02 tensors and trajectory provenance."""

    grid: torch.Tensor
    low5: torch.Tensor
    hist: torch.Tensor
    plans: torch.Tensor
    gamma: torch.Tensor
    trajectory_ids: torch.Tensor
    trajectory_steps: torch.Tensor
    direction: torch.Tensor
    trajectory_balanced_weight: torch.Tensor
    query_hashes: tuple[str, ...]
    trajectory_rows: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return int(self.plans.shape[0])

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "grid": self.grid,
            "low5": self.low5,
            "hist": self.hist,
            "U": self.plans,
            "gamma": self.gamma,
            "trajectory_ids": self.trajectory_ids,
            "trajectory_steps": self.trajectory_steps,
            "direction": self.direction,
            "trajectory_balanced_weight": self.trajectory_balanced_weight,
        }


@dataclass(frozen=True)
class GroupSplit:
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    train_trajectory_ids: tuple[int, ...]
    validation_trajectory_ids: tuple[int, ...]
    audit: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_clean_fresh_outdir(path: Path) -> None:
    """Fail closed rather than reuse any artifact from an earlier Stage 3 run.

    A failed mode-diversity gate intentionally leaves only diagnostic candidate
    files.  Reusing an output directory could otherwise leave a previously
    promoted ``checkpoint_best.pt`` visible after the new run fails.  Stage 3
    therefore never resumes and never deletes user data: a nonempty output
    directory is rejected before any dependency log or training artifact is
    written.
    """

    if not path.exists():
        return
    if not path.is_dir():
        raise RuntimeError(f"fresh Stage 3 output path is not a directory: {path}")
    entries = sorted(item.name for item in path.iterdir())
    if entries:
        preview = ", ".join(entries[:6])
        suffix = " ..." if len(entries) > 6 else ""
        raise RuntimeError(
            "fresh Stage 3 refuses a nonempty output directory; choose a new "
            f"--outdir (found: {preview}{suffix})"
        )


def _legacy_path_reason(path: Path) -> str | None:
    lower = str(path.resolve()).lower()
    forbidden = (
        "stage_results/02b_balanced_id",
        "balanced_id_windows_",
        "selected_id_rollouts",
        "giant_ood_id_balanced_v2",
    )
    return next((fragment for fragment in forbidden if fragment in lower), None)


def _resolve_dataset_entries(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage-02 manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetContractError(
            f"manifest schema must be {SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )
    base = manifest_path.parent
    entries: list[dict[str, Any]] = []

    def add(raw: Any, *, expected_gamma: float | None = None, sha256: str | None = None) -> None:
        if isinstance(raw, str):
            path_value = raw
            entry_sha = sha256
        elif isinstance(raw, Mapping):
            path_value = raw.get("path", raw.get("dataset"))
            entry_sha = raw.get("sha256", raw.get("dataset_sha256", sha256))
            if expected_gamma is None and raw.get("gamma") is not None:
                expected_gamma = float(raw["gamma"])
        else:
            raise DatasetContractError(f"invalid dataset manifest entry: {raw!r}")
        if not path_value:
            raise DatasetContractError("dataset manifest entry is missing a path")
        path = Path(path_value)
        if not path.is_absolute():
            path = (base / path).resolve()
        reason = _legacy_path_reason(path)
        if reason is not None:
            raise DatasetContractError(
                f"legacy Stage-2B data are forbidden ({reason!r}): {path}"
            )
        if entry_sha is None:
            raise DatasetContractError(
                f"planned-demo manifest entry requires a SHA-256 checksum: {path}"
            )
        digest = str(entry_sha).lower()
        try:
            if len(digest) != 64:
                raise ValueError
            bytes.fromhex(digest)
        except ValueError as exc:
            raise DatasetContractError(
                f"planned-demo manifest entry has an invalid SHA-256 checksum: {path}"
            ) from exc
        entries.append({"path": path, "sha256": digest, "gamma": expected_gamma})

    if manifest.get("dataset"):
        add(manifest["dataset"], sha256=manifest.get("dataset_sha256"))
    for item in manifest.get("datasets", ()):  # optional sharded manifest
        add(item)
    for gamma, item in manifest.get("per_gamma", {}).items():
        add(item, expected_gamma=float(gamma))
    if not entries:
        raise DatasetContractError(
            "manifest must provide `dataset`, `datasets`, or `per_gamma` planned-demo artifacts"
        )
    unique = {str(item["path"].resolve()) for item in entries}
    if len(unique) != len(entries):
        raise DatasetContractError("the Stage-02 manifest lists a dataset artifact more than once")
    return manifest, entries


def _tensor(payload: Mapping[str, Any], key: str, *, length: int | None = None) -> torch.Tensor:
    if key not in payload or not isinstance(payload[key], torch.Tensor):
        raise DatasetContractError(f"planned-demo payload requires tensor `{key}`")
    value = payload[key].detach().cpu()
    if length is not None and len(value) != length:
        raise DatasetContractError(f"`{key}` has {len(value)} rows, expected {length}")
    return value


def _integer_tensor(
    payload: Mapping[str, Any], key: str, *, length: int | None = None
) -> torch.Tensor:
    value = _tensor(payload, key, length=length)
    if value.dtype not in _INTEGER_DTYPES:
        raise DatasetContractError(f"`{key}` must use an integer dtype, got {value.dtype}")
    return value.long()


def _validate_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DatasetContractError(
            f"dataset schema must be {SCHEMA_VERSION!r}; legacy executed-window tensors are invalid"
        )
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise DatasetContractError("dataset is missing the planned-window contract")
    required_true = (
        "generated_equals_verified_equals_training",
        "only_first_action_executed",
        "all_targets_pre_execution_fully_verified",
        "progress_not_in_safety_label",
    )
    failed = [key for key in required_true if contract.get(key) is not True]
    if failed:
        raise DatasetContractError(
            "executed-composite or unverified targets are forbidden; false/missing contract fields: "
            + ", ".join(failed)
        )
    if int(contract.get("planned_horizon", -1)) != 10:
        raise DatasetContractError("training object must be one generated planned H=10 window")
    if int(contract.get("synthetic_reflections", -1)) != 0:
        raise DatasetContractError("synthetic reflection is forbidden in clean pretraining")
    if int(contract.get("padding", -1)) != 0:
        raise DatasetContractError("trajectory/window padding is forbidden in clean pretraining")
    if int(contract.get("debug_training_targets", -1)) != 0:
        raise DatasetContractError("raw SafeMPPI debug rollouts are forbidden training targets")
    if float(contract.get("debug_target_share", math.nan)) != 0.0:
        raise DatasetContractError("expert dataset must have debug_target_share exactly zero")


def _validate_piece(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_gamma: float | None,
) -> dict[str, Any]:
    _validate_contract(payload)
    plans = _tensor(payload, "U")
    count = len(plans)
    if count == 0:
        raise DatasetContractError(f"planned-demo artifact is empty: {path}")
    grid = _tensor(payload, "grid", length=count).float()
    low5 = _tensor(payload, "low5", length=count).float()
    hist = _tensor(payload, "hist", length=count).float()
    verifier_state = _tensor(payload, "verifier_state", length=count).double()
    raw_verifier_fingerprints = payload.get("verifier_spec_fingerprint")
    if (
        not isinstance(raw_verifier_fingerprints, (tuple, list))
        or len(raw_verifier_fingerprints) != count
    ):
        raise DatasetContractError(
            "verifier_spec_fingerprint must contain one exact digest per target"
        )
    verifier_fingerprints = tuple(
        str(value).lower() for value in raw_verifier_fingerprints
    )
    plans = plans.float()
    gamma_tensor = _tensor(payload, "gamma", length=count).float()
    trajectory_ids = _integer_tensor(payload, "window_trajectory_ids", length=count)
    source_trajectory_ids = _integer_tensor(
        payload, "source_trajectory_ids", length=count
    )
    window_seeds = _integer_tensor(payload, "window_seeds", length=count)
    steps = _integer_tensor(payload, "window_steps", length=count)
    direction = _integer_tensor(payload, "window_direction", length=count)
    trajectory_weight = _tensor(
        payload, "trajectory_balanced_weight", length=count
    ).double()
    raw_plan_kinds = payload.get("window_plan_kind")
    if not isinstance(raw_plan_kinds, (tuple, list)) or len(raw_plan_kinds) != count:
        raise DatasetContractError("window_plan_kind must identify every expert target")
    plan_kinds = tuple(str(value) for value in raw_plan_kinds)
    unexpected_plan_kinds = sorted(set(plan_kinds) - {"weighted_mean", "internal_best"})
    if unexpected_plan_kinds:
        raise DatasetContractError(
            "only cost-selected SafeMPPI outputs may train Stage 3; got "
            f"{unexpected_plan_kinds}"
        )
    if tuple(grid.shape[1:]) != (3, 32, 32):
        raise DatasetContractError(f"grid must have shape [N,3,32,32], got {tuple(grid.shape)}")
    if tuple(low5.shape[1:]) != (5,):
        raise DatasetContractError(f"low5 must have shape [N,5], got {tuple(low5.shape)}")
    if hist.ndim != 3 or hist.shape[-1] != 2:
        raise DatasetContractError(f"hist must have shape [N,K,2], got {tuple(hist.shape)}")
    if tuple(plans.shape[1:]) != (10, 2):
        raise DatasetContractError(
            f"training targets must be generated planned [10,2] windows, got {tuple(plans.shape)}"
        )
    if tuple(verifier_state.shape[1:]) != (4,):
        raise DatasetContractError(
            "verifier_state must have shape [N,4], got "
            f"{tuple(verifier_state.shape)}"
        )
    if not bool(torch.isfinite(verifier_state).all()):
        raise DatasetContractError("verifier_state contains non-finite data")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in verifier_fingerprints
    ):
        raise DatasetContractError(
            "verifier_spec_fingerprint contains a non-SHA256 digest"
        )
    for name, tensor in (("grid", grid), ("low5", low5), ("hist", hist), ("U", plans)):
        if not bool(torch.isfinite(tensor).all()):
            raise DatasetContractError(f"`{name}` contains non-finite data")
    if bool((plans.abs() > 1.0 + 1.0e-6).any()):
        raise DatasetContractError("planned actions exceed the policy action bound")
    if not bool(torch.isin(direction, torch.tensor((0, 1))).all()):
        raise DatasetContractError("window_direction must encode only real R-first/U-first paths")
    if not torch.equal(source_trajectory_ids, trajectory_ids):
        raise DatasetContractError(
            "source_trajectory_ids must identify the exact unsplit source trajectory"
        )
    if not bool(torch.isfinite(trajectory_weight).all() and (trajectory_weight > 0.0).all()):
        raise DatasetContractError("trajectory_balanced_weight must be finite and positive")

    label_tensors = {
        key: _tensor(payload, key, length=count)
        for key in ("target_safe", "target_in_bounds", "target_socp_ok")
    }
    non_boolean = [key for key, value in label_tensors.items() if value.dtype != torch.bool]
    if non_boolean:
        raise DatasetContractError(
            "full-verifier target labels must be explicit boolean tensors: "
            + ", ".join(non_boolean)
        )
    safe = label_tensors["target_safe"]
    in_bounds = label_tensors["target_in_bounds"]
    socp_ok = label_tensors["target_socp_ok"]
    if not bool(torch.equal(safe, in_bounds & socp_ok)):
        raise DatasetContractError("per-target safety must equal strict-bounds AND full-SOCP")
    if not bool(safe.all() and in_bounds.all() and socp_ok.all()):
        bad = torch.where(~(safe & in_bounds & socp_ok))[0][:8].tolist()
        raise DatasetContractError(
            f"unverified or verifier-rejected training targets at rows {bad}"
        )

    hash_fields: dict[str, tuple[str, ...]] = {}
    for key in (
        "query_hashes",
        "generated_hashes",
        "verifier_input_hashes",
        "training_target_hashes",
    ):
        values = payload.get(key)
        if not isinstance(values, (tuple, list)) or len(values) != count:
            raise DatasetContractError(f"`{key}` must contain one SHA-256 identity per target")
        hash_fields[key] = tuple(str(item).lower() for item in values)
    baseline = hash_fields["query_hashes"]
    for key, values in hash_fields.items():
        if values != baseline:
            raise DatasetContractError(
                f"identity mismatch: generated/verifier/training hashes differ in `{key}`"
            )

    raw_rows = payload.get("trajectory_rows")
    if not isinstance(raw_rows, (tuple, list)) or not raw_rows:
        raise DatasetContractError("trajectory_rows provenance is required")
    trajectory_meta: dict[int, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise DatasetContractError("every trajectory row must be a mapping")
        tid = int(raw.get("trajectory_id", -1))
        if tid < 0 or tid in trajectory_meta:
            raise DatasetContractError(f"invalid/duplicate trajectory_id {tid}")
        route = str(raw.get("direction_class"))
        if route not in ("R-first", "U-first"):
            raise DatasetContractError(f"trajectory {tid} is not a real classified R/U path")
        raw_seed = raw.get("seed")
        if isinstance(raw_seed, (bool, np.bool_)) or not isinstance(
            raw_seed, (int, np.integer)
        ):
            raise DatasetContractError(
                f"trajectory {tid} requires an integer source-generation seed"
            )
        trajectory_meta[tid] = {
            **dict(raw),
            "trajectory_id": tid,
            "gamma": float(raw["gamma"]),
            "seed": int(raw_seed),
            "direction_class": route,
        }
    seen_ids = {int(value) for value in torch.unique(trajectory_ids)}
    if seen_ids != set(trajectory_meta):
        raise DatasetContractError("window trajectory IDs and trajectory_rows do not match exactly")

    canonical_gamma = torch.empty(count, dtype=torch.float64)
    for tid, meta in trajectory_meta.items():
        rows = torch.where(trajectory_ids == tid)[0]
        if len(rows) != int(meta.get("steps", -1)):
            raise DatasetContractError(
                f"trajectory {tid} row count differs from its real executed step count"
            )
        expected_steps = torch.arange(len(rows), dtype=torch.long)
        actual_steps = torch.sort(steps[rows]).values
        if not torch.equal(actual_steps, expected_steps):
            raise DatasetContractError(f"trajectory {tid} has padding, gaps, or duplicate windows")
        expected_weight = torch.full(
            (len(rows),), 1.0 / len(rows), dtype=torch.float64
        )
        if not torch.allclose(
            trajectory_weight[rows], expected_weight, atol=2e-7, rtol=2e-6
        ):
            raise DatasetContractError(
                f"trajectory {tid} weights do not sum to one real-trajectory unit"
            )
        route_code = 0 if meta["direction_class"] == "R-first" else 1
        if not bool((direction[rows] == route_code).all()):
            raise DatasetContractError(f"trajectory {tid} direction disagrees with its windows")
        if not bool((window_seeds[rows] == int(meta["seed"])).all()):
            raise DatasetContractError(
                f"trajectory {tid} source seed disagrees with its windows"
            )
        value = float(meta["gamma"])
        canonical_gamma[rows] = value
        scalar = torch.tensor(value, dtype=torch.float64)
        if not bool(torch.isclose(gamma_tensor[rows].double(), scalar, atol=1e-6).all()):
            raise DatasetContractError(f"trajectory {tid} gamma tensor disagrees with provenance")
        if not bool(torch.isclose(low5[rows, 4].double(), scalar, atol=1e-6).all()):
            raise DatasetContractError(f"trajectory {tid} endpoint-free gamma context is incorrect")
    if expected_gamma is not None and not bool(
        torch.isclose(
            canonical_gamma,
            torch.tensor(float(expected_gamma), dtype=torch.float64),
            atol=1e-12,
        ).all()
    ):
        raise DatasetContractError(
            f"per-gamma artifact {path} contains a row outside gamma={expected_gamma:g}"
        )

    # Reconstruct and hash every exact generated/verified/training object.
    for index in range(count):
        context = QueryContext(
            grid[index].numpy(),
            low5[index].numpy(),
            hist[index].numpy(),
            verifier_state[index].numpy(),
            verifier_fingerprints[index],
        )
        actual = query_content_hash(context, float(canonical_gamma[index]), plans[index].numpy())
        if actual != baseline[index]:
            raise DatasetContractError(
                f"row {index} content differs from its generated/verifier/training hash"
            )
    if len(set(baseline)) != len(baseline):
        raise DatasetContractError("duplicate exact query targets are not valid independent demos")
    return {
        "path": path,
        "grid": grid.contiguous(),
        "low5": low5.contiguous(),
        "hist": hist.contiguous(),
        "plans": plans.contiguous(),
        "gamma": canonical_gamma,
        "trajectory_ids": trajectory_ids,
        "steps": steps,
        "direction": direction,
        "trajectory_weight": trajectory_weight,
        "hashes": baseline,
        "trajectory_meta": trajectory_meta,
    }


def _balance_audit(
    trajectory_rows: Iterable[Mapping[str, Any]], expected_gammas: Sequence[float]
) -> dict[str, Any]:
    rows = list(trajectory_rows)
    per_gamma: dict[str, Any] = {}
    totals = []
    for gamma in expected_gammas:
        selected = [row for row in rows if math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-12)]
        r_count = sum(row["direction_class"] == "R-first" for row in selected)
        u_count = sum(row["direction_class"] == "U-first" for row in selected)
        if not selected:
            raise DatasetContractError(f"missing planned demonstrations for gamma={gamma:g}")
        if r_count != u_count:
            raise DatasetContractError(
                f"gamma={gamma:g} is not exact R/U trajectory balanced: {r_count}/{u_count}"
            )
        totals.append(len(selected))
        per_gamma[f"{float(gamma):g}"] = {
            "trajectories": len(selected), "R-first": r_count, "U-first": u_count
        }
    extra = sorted(
        {float(row["gamma"]) for row in rows}
        - {float(gamma) for gamma in expected_gammas}
    )
    if extra:
        raise DatasetContractError(f"dataset contains unsupported gamma strata: {extra}")
    if len(set(totals)) != 1:
        raise DatasetContractError(f"gamma trajectory mass is not exactly balanced: {totals}")
    return {
        "exact_gamma_trajectory_balance": True,
        "exact_R_U_trajectory_balance": True,
        "per_gamma": per_gamma,
    }


def load_planned_demo_manifest(
    manifest_path: str | Path,
    *,
    expected_gammas: Sequence[float] = GAMMAS,
) -> PlannedDemoPool:
    """Load only clean planned-demo artifacts and verify every target bitwise."""

    manifest_path = Path(manifest_path).resolve()
    _manifest, entries = _resolve_dataset_entries(manifest_path)
    pieces: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for entry in entries:
        path: Path = entry["path"]
        if not path.exists():
            raise FileNotFoundError(f"planned-demo dataset does not exist: {path}")
        actual_sha = sha256_file(path)
        expected_sha = str(entry["sha256"])
        if actual_sha != expected_sha:
            raise DatasetContractError(f"dataset checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, Mapping):
            raise DatasetContractError(f"dataset payload must be a mapping: {path}")
        piece = _validate_piece(payload, path=path, expected_gamma=entry.get("gamma"))
        pieces.append(piece)
        source_rows.append({"path": str(path), "sha256": actual_sha, "rows": len(piece["plans"])})

    tensors: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "grid", "low5", "hist", "plans", "gamma", "trajectory_ids",
            "steps", "direction", "trajectory_weight",
        )
    }
    hashes: list[str] = []
    trajectory_rows: list[dict[str, Any]] = []
    source_trajectories: dict[tuple[float, int], str] = {}
    next_tid = 0
    for piece in pieces:
        old_ids = sorted(piece["trajectory_meta"])
        remap = {old: next_tid + offset for offset, old in enumerate(old_ids)}
        next_tid += len(old_ids)
        remapped = torch.tensor(
            [remap[int(value)] for value in piece["trajectory_ids"].tolist()], dtype=torch.long
        )
        for key in (
            "grid", "low5", "hist", "plans", "gamma", "steps", "direction",
            "trajectory_weight",
        ):
            tensors[key].append(piece[key])
        tensors["trajectory_ids"].append(remapped)
        hashes.extend(piece["hashes"])
        for old in old_ids:
            row = dict(piece["trajectory_meta"][old])
            source_key = (float(row["gamma"]), int(row["seed"]))
            if source_key in source_trajectories:
                raise DatasetContractError(
                    "one source trajectory appears more than once across planned-demo artifacts: "
                    f"gamma={source_key[0]:g}, seed={source_key[1]}"
                )
            source_trajectories[source_key] = str(piece["path"])
            row["source_trajectory_id"] = old
            row["trajectory_id"] = remap[old]
            trajectory_rows.append(row)
    if len(set(hashes)) != len(hashes):
        raise DatasetContractError("duplicate exact query hashes occur across dataset shards")
    balance = _balance_audit(trajectory_rows, expected_gammas)
    source_rows.append({"balance": balance, "manifest": str(manifest_path)})
    return PlannedDemoPool(
        grid=torch.cat(tensors["grid"]),
        low5=torch.cat(tensors["low5"]),
        hist=torch.cat(tensors["hist"]),
        plans=torch.cat(tensors["plans"]),
        gamma=torch.cat(tensors["gamma"]).double(),
        trajectory_ids=torch.cat(tensors["trajectory_ids"]).long(),
        trajectory_steps=torch.cat(tensors["steps"]).long(),
        direction=torch.cat(tensors["direction"]).long(),
        trajectory_balanced_weight=torch.cat(tensors["trajectory_weight"]).double(),
        query_hashes=tuple(hashes),
        trajectory_rows=tuple(trajectory_rows),
        sources=tuple(source_rows),
    )


def validate_no_trajectory_leakage(
    pool: PlannedDemoPool,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
) -> None:
    train = {int(value) for value in pool.trajectory_ids[train_indices].tolist()}
    validation = {int(value) for value in pool.trajectory_ids[validation_indices].tolist()}
    overlap = sorted(train & validation)
    if overlap:
        raise DatasetContractError(f"source trajectory leakage across train/validation: {overlap}")


def make_group_split(
    pool: PlannedDemoPool,
    *,
    validation_trajectories_per_mode: int = 2,
    seed: int = 31_711,
) -> GroupSplit:
    """Split whole trajectories within every gamma/mode stratum."""

    if validation_trajectories_per_mode <= 0:
        raise ValueError("validation_trajectories_per_mode must be positive")
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    validation_ids: list[int] = []
    audit: dict[str, Any] = {"seed": int(seed), "per_gamma": {}}
    for gamma in GAMMAS:
        gamma_entry: dict[str, Any] = {}
        for mode, name in MODE_NAMES.items():
            ids = sorted(
                int(row["trajectory_id"])
                for row in pool.trajectory_rows
                if math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-12)
                and row["direction_class"] == name
            )
            if len(ids) <= validation_trajectories_per_mode:
                raise DatasetContractError(
                    f"gamma={gamma:g}/{name} has {len(ids)} trajectories; cannot hold out "
                    f"{validation_trajectories_per_mode} and retain training data"
                )
            shuffled = np.asarray(ids, dtype=np.int64)
            rng.shuffle(shuffled)
            held = sorted(int(value) for value in shuffled[:validation_trajectories_per_mode])
            kept = sorted(int(value) for value in shuffled[validation_trajectories_per_mode:])
            validation_ids.extend(held)
            train_ids.extend(kept)
            gamma_entry[name] = {"train": kept, "validation": held}
        audit["per_gamma"][f"{float(gamma):g}"] = gamma_entry
    train_mask = torch.isin(pool.trajectory_ids, torch.tensor(train_ids, dtype=torch.long))
    validation_mask = torch.isin(pool.trajectory_ids, torch.tensor(validation_ids, dtype=torch.long))
    train_indices = torch.where(train_mask)[0]
    validation_indices = torch.where(validation_mask)[0]
    if bool((train_mask & validation_mask).any()) or not bool((train_mask | validation_mask).all()):
        raise DatasetContractError("group split does not partition the complete planned-demo pool")
    validate_no_trajectory_leakage(pool, train_indices, validation_indices)
    audit.update(
        {
            "train_trajectories": len(set(train_ids)),
            "validation_trajectories": len(set(validation_ids)),
            "train_windows_available": len(train_indices),
            "validation_windows_available": len(validation_indices),
            "trajectory_leakage": 0,
        }
    )
    return GroupSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_trajectory_ids=tuple(sorted(train_ids)),
        validation_trajectory_ids=tuple(sorted(validation_ids)),
        audit=audit,
    )


def balanced_real_window_indices(
    pool: PlannedDemoPool,
    trajectory_ids: Sequence[int],
    *,
    windows_per_trajectory: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw equal real rows per trajectory, without replacement or synthesis."""

    if windows_per_trajectory <= 0:
        raise ValueError("windows_per_trajectory must be positive")
    selected: list[torch.Tensor] = []
    for tid in sorted(int(value) for value in trajectory_ids):
        rows = torch.where(pool.trajectory_ids == tid)[0]
        if len(rows) < windows_per_trajectory:
            raise DatasetContractError(
                f"trajectory {tid} has {len(rows)} real windows, below requested "
                f"{windows_per_trajectory}; padding/replacement is forbidden"
            )
        order = torch.randperm(len(rows), generator=generator)[:windows_per_trajectory]
        selected.append(rows[order])
    combined = torch.cat(selected)
    combined = combined[torch.randperm(len(combined), generator=generator)]
    # Exact trajectory balance plus exact trajectory counts from the validated
    # pool imply equal window mass for each gamma and each R/U mode.
    strata_counts: list[int] = []
    for gamma in GAMMAS:
        for mode in MODE_NAMES:
            mask = torch.isclose(pool.gamma[combined], torch.tensor(float(gamma), dtype=torch.float64))
            mask &= pool.direction[combined] == mode
            strata_counts.append(int(mask.sum()))
    if len(set(strata_counts)) != 1:
        raise AssertionError(f"balanced sampler produced unequal gamma/RU mass: {strata_counts}")
    return combined


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _batch_to_device(pool: PlannedDemoPool, rows: torch.Tensor, device: torch.device):
    return (
        pool.grid[rows].to(device, non_blocking=True),
        pool.low5[rows].to(device, non_blocking=True),
        pool.hist[rows].to(device, non_blocking=True),
        pool.plans[rows].to(device, non_blocking=True),
    )


def seeded_cfm_loss(
    policy: torch.nn.Module,
    grid: torch.Tensor,
    low5: torch.Tensor,
    hist: torch.Tensor,
    plans: torch.Tensor,
    *,
    generator: torch.Generator,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard CFM loss with explicit source/time RNG and no augmentation."""

    context = policy.ctx_from(grid, low5, hist)
    x1 = (plans / float(policy.u_max)).reshape(len(plans), int(policy.d))
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    tau = torch.rand(len(x1), device=x1.device, dtype=x1.dtype, generator=generator).clamp_(1e-4, 1.0)
    x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
    prediction = policy(x_tau, tau, context)
    per_sample = ((prediction - (x1 - x0)) ** 2).mean(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype)
    if weight.shape != per_sample.shape or bool((weight <= 0.0).any()):
        raise ValueError("sample_weight must be one positive scalar per CFM target")
    return (per_sample * weight).sum() / weight.sum()


@torch.no_grad()
def evaluate_cfm(
    policy: torch.nn.Module,
    pool: PlannedDemoPool,
    rows: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    amp: bool,
    trajectory_weighted: bool = True,
) -> float:
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    mass = 0.0
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        grid, low5, hist, plans = _batch_to_device(pool, batch_rows, device)
        weight = (
            pool.trajectory_balanced_weight[batch_rows].to(device)
            if trajectory_weighted
            else None
        )
        with _amp_context(device, amp):
            loss = seeded_cfm_loss(
                policy,
                grid,
                low5,
                hist,
                plans,
                generator=generator,
                sample_weight=weight,
            )
        batch_mass = float(weight.sum()) if weight is not None else float(len(batch_rows))
        total += float(loss) * batch_mass
        mass += batch_mass
    return total / max(mass, 1.0e-12)


def _crossing_time(values: np.ndarray, threshold: float = 1.0) -> float:
    for index in range(1, len(values)):
        left, right = float(values[index - 1]), float(values[index])
        if left < threshold <= right:
            width = right - left
            fraction = 1.0 if abs(width) <= 1e-12 else (threshold - left) / width
            return index - 1 + fraction
    return math.inf


def _rollout_mode(path: np.ndarray) -> str:
    right = _crossing_time(path[:, 0])
    up = _crossing_time(path[:, 1])
    if not np.isfinite(right) or not np.isfinite(up) or abs(right - up) <= 1e-5:
        return "unclassified"
    return "R-first" if right < up else "U-first"


def _segment_clearance(start: np.ndarray, end: np.ndarray, obstacles: np.ndarray, rr: float) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-16:
        closest = np.broadcast_to(start, (len(obstacles), 2))
    else:
        fraction = ((obstacles[:, :2] - start) @ delta) / denominator
        closest = start + np.clip(fraction, 0.0, 1.0)[:, None] * delta
    return float(
        (np.linalg.norm(closest - obstacles[:, :2], axis=1) - obstacles[:, 2] - rr).min()
    )


@torch.inference_mode()
def evaluate_id_rollouts(
    policy: torch.nn.Module,
    *,
    device: torch.device,
    repetitions: int,
    max_steps: int,
    reach_m: float,
    temperature: float = 1.0,
    nfe: int = 8,
    seed: int = 83_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Batched ID rollouts; scientific pretraining evaluation is fixed at T=1."""

    if repetitions <= 0 or max_steps <= 0 or nfe <= 0:
        raise ValueError("rollout counts, horizon, and nfe must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    # Keep scene/matplotlib dependencies outside the contract-only loader so
    # data validation and its tests remain lightweight and CPU-only.
    from .scene import context_from_state, make_id_scene

    env = make_id_scene()
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64)
    rr = float(env.r_robot)
    metadata = [
        (float(gamma), repetition)
        for gamma in GAMMAS
        for repetition in range(repetitions)
    ]
    count = len(metadata)
    initial = np.asarray(env.x0.detach().cpu().numpy(), dtype=np.float64)
    states = np.repeat(initial[None], count, axis=0)
    histories: list[list[np.ndarray]] = [[] for _ in range(count)]
    paths: list[list[np.ndarray]] = [[initial[:2].copy()] for _ in range(count)]
    active = np.ones(count, dtype=bool)
    reached = np.zeros(count, dtype=bool)
    collision = np.zeros(count, dtype=bool)
    in_bounds = np.ones(count, dtype=bool)
    min_clearance = np.full(count, np.inf, dtype=np.float64)
    generator = torch.Generator(device=device).manual_seed(seed)
    policy.eval()
    for _step in range(max_steps):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        contexts = [
            context_from_state(
                states[index], GOAL, metadata[index][0], histories[index], env
            )
            for index in indices
        ]
        grid = torch.as_tensor(
            np.stack([item.grid for item in contexts]), dtype=torch.float32, device=device
        )
        low5 = torch.as_tensor(
            np.stack([item.low5 for item in contexts]), dtype=torch.float32, device=device
        )
        hist = torch.as_tensor(
            np.stack([item.hist for item in contexts]), dtype=torch.float32, device=device
        )
        context = policy.ctx_from(grid, low5, hist)
        x = temperature * torch.randn(
            len(indices), int(policy.d), device=device, generator=generator
        )
        for flow_step in range(nfe):
            tau = torch.full(
                (len(indices),), flow_step / nfe, dtype=x.dtype, device=device
            )
            x = x + policy(x, tau, context) / nfe
        plans = (
            x.reshape(len(indices), int(policy.T), 2) * float(policy.u_max)
        ).clamp(-float(policy.u_max), float(policy.u_max))
        actions = plans[:, 0].float().cpu().numpy()
        for local, index in enumerate(indices):
            previous = states[index].copy()
            states[index] = step_state(previous, actions[local], dt=float(env.dt))
            histories[index].append(actions[local].copy())
            paths[index].append(states[index, :2].copy())
            segment_margin = _segment_clearance(
                previous[:2], states[index, :2], obstacles, rr
            )
            min_clearance[index] = min(min_clearance[index], segment_margin)
            collision[index] |= segment_margin < 0.0
            in_bounds[index] &= bool(
                ((states[index, :2] >= 0.0) & (states[index, :2] <= 5.0)).all()
            )
            reached[index] = float(np.linalg.norm(states[index, :2] - GOAL)) < reach_m
            if collision[index] or not in_bounds[index] or reached[index]:
                active[index] = False

    rows: list[dict[str, Any]] = []
    for index, (gamma, repetition) in enumerate(metadata):
        path = np.asarray(paths[index], dtype=np.float32)
        success = bool(reached[index] and not collision[index] and in_bounds[index])
        rows.append(
            {
                "gamma": gamma,
                "repetition": repetition,
                "success": success,
                "collision": bool(collision[index]),
                "in_bounds": bool(in_bounds[index]),
                "reached": bool(reached[index]),
                "steps": len(path) - 1,
                "endpoint_distance_m": float(np.linalg.norm(path[-1] - GOAL)),
                "min_clearance_m": float(min_clearance[index]),
                "mode": _rollout_mode(path) if success else "unclassified",
                "path": path,
            }
        )
    per_gamma: dict[str, Any] = {}
    for gamma in GAMMAS:
        selected = [row for row in rows if row["gamma"] == float(gamma)]
        successful = [row for row in selected if row["success"]]
        r_count = sum(row["mode"] == "R-first" for row in successful)
        u_count = sum(row["mode"] == "U-first" for row in successful)
        per_gamma[f"{float(gamma):g}"] = {
            "rollouts": len(selected),
            "successes": len(successful),
            "success_rate_sr": len(successful) / len(selected),
            "collisions": sum(row["collision"] for row in selected),
            "collision_rate_cr": sum(row["collision"] for row in selected) / len(selected),
            "out_of_bounds": sum(not row["in_bounds"] for row in selected),
            "R-first_successes": r_count,
            "U-first_successes": u_count,
            "unclassified_successes": sum(
                row["mode"] == "unclassified" for row in successful
            ),
            "both_R_U_modes_present": bool(r_count > 0 and u_count > 0),
        }
    summary = {
        "scene": "ordinary_symmetric_4x4_ID_stadium",
        "sampling_temperature": float(temperature),
        "temperature_role": "scientific ID evaluation" if temperature == 1.0 else "visualization diagnostic only",
        "nfe": int(nfe),
        "repetitions_per_gamma": int(repetitions),
        "all_gammas_have_both_R_U_modes": bool(
            all(entry["both_R_U_modes_present"] for entry in per_gamma.values())
        ),
        "global_success_rate_sr": sum(row["success"] for row in rows) / len(rows),
        "global_collision_rate_cr": sum(row["collision"] for row in rows) / len(rows),
        "per_gamma": per_gamma,
    }
    return summary, rows


def validate_temperature_one_mode_diversity(summary: Mapping[str, Any]) -> None:
    """Require observed successful R-first and U-first rollouts at every gamma."""

    if float(summary.get("sampling_temperature", math.nan)) != 1.0:
        raise RuntimeError("the scientific mode-diversity gate requires temperature T=1")
    per_gamma = summary.get("per_gamma")
    if not isinstance(per_gamma, Mapping):
        raise RuntimeError("temperature-1 evaluation is missing per-gamma mode counts")
    missing: list[str] = []
    for gamma in GAMMAS:
        key = f"{float(gamma):g}"
        entry = per_gamma.get(key)
        if not isinstance(entry, Mapping):
            missing.append(f"gamma={key} (missing evaluation)")
            continue
        if int(entry.get("R-first_successes", 0)) <= 0:
            missing.append(f"gamma={key} R-first")
        if int(entry.get("U-first_successes", 0)) <= 0:
            missing.append(f"gamma={key} U-first")
    if missing:
        raise RuntimeError(
            "temperature-1 pretrained-policy mode-diversity gate failed: "
            + ", ".join(missing)
        )


def _save_rollout_arrays(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    offsets = [0]
    positions: list[np.ndarray] = []
    for row in rows:
        path = np.asarray(row["path"], dtype=np.float32)
        positions.append(path)
        offsets.append(offsets[-1] + len(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions=np.concatenate(positions),
        offsets=np.asarray(offsets, dtype=np.int64),
        gamma=np.asarray([row["gamma"] for row in rows], dtype=np.float32),
        repetition=np.asarray([row["repetition"] for row in rows], dtype=np.int32),
        success=np.asarray([row["success"] for row in rows], dtype=bool),
        collision=np.asarray([row["collision"] for row in rows], dtype=bool),
        in_bounds=np.asarray([row["in_bounds"] for row in rows], dtype=bool),
        mode=np.asarray([row["mode"] for row in rows], dtype="U16"),
    )


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _configure_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.split(",")[0].strip() != "1":
            raise RuntimeError(
                "Stage 03 is assigned to physical GPU 1: launch with "
                "CUDA_VISIBLE_DEVICES=1 and use --device cuda:0"
            )
    torch.set_float32_matmul_precision("high")
    return device


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    started_wall = _utc_now()
    started = time.perf_counter()
    outdir = args.outdir.resolve()
    require_clean_fresh_outdir(outdir)
    device = _configure_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    for directory in (outdir / "data", outdir / "logs", outdir / "tables", outdir / "viz"):
        directory.mkdir(parents=True, exist_ok=True)
    dependencies = write_dependency_manifest(outdir / "logs/dependencies.json")
    assert_no_legacy_expansion_imports()

    pool = load_planned_demo_manifest(args.manifest)
    split = make_group_split(
        pool,
        validation_trajectories_per_mode=args.validation_trajectories_per_mode,
        seed=args.split_seed,
    )
    _atomic_json(outdir / "logs/split_audit.json", split.audit)
    sampler_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    minimum_train = min(
        int((pool.trajectory_ids == tid).sum()) for tid in split.train_trajectory_ids
    )
    minimum_validation = min(
        int((pool.trajectory_ids == tid).sum()) for tid in split.validation_trajectory_ids
    )
    train_per_trajectory = min(args.windows_per_trajectory, minimum_train)
    validation_per_trajectory = min(
        args.validation_windows_per_trajectory, minimum_validation
    )
    if args.smoke:
        validation_rows = balanced_real_window_indices(
            pool,
            split.validation_trajectory_ids,
            windows_per_trajectory=validation_per_trajectory,
            generator=torch.Generator(device="cpu").manual_seed(args.seed + 2),
        )
    else:
        # Full training and monitoring retain every exact verified target.  The
        # inverse-length weight makes each real source trajectory one unit;
        # exact trajectory counts then give exact gamma and R/U objective mass.
        validation_rows = split.validation_indices

    policy = HP.GridHPFlowPolicy(
        repr_dim=32,
        grid_hw=(32, 32),
        trunk_hidden=tuple(args.trunk_hidden),
        enc_depth=args.enc_depth,
    ).to(device)
    config = policy.config()
    if config.get("raw_start_goal") or policy.ctx_dim != 37 or policy.repr_dim != 32:
        raise RuntimeError("Stage 03 requires the fresh endpoint-free 37-D context and repr_dim=32")
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epochs = min(args.epochs, 2) if args.smoke else args.epochs

    def lr_factor(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        fraction = (epoch - args.warmup_epochs) / max(epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    cfm_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    fields = (
        "epoch", "train_cfm", "validation_cfm", "learning_rate",
        "encoder_gradient_norm", "epoch_seconds", "gpu_memory_mib",
    )
    history_path = outdir / "tables/training_history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        if args.smoke:
            train_rows = balanced_real_window_indices(
                pool,
                split.train_trajectory_ids,
                windows_per_trajectory=train_per_trajectory,
                generator=sampler_generator,
            )
        else:
            order = torch.randperm(len(split.train_indices), generator=sampler_generator)
            train_rows = split.train_indices[order]
        policy.train()
        weighted_loss = 0.0
        objective_mass = 0.0
        encoder_gradient = 0.0
        batches = 0
        epoch_mass = (
            float(len(train_rows))
            if args.smoke
            else float(pool.trajectory_balanced_weight[train_rows].sum())
        )
        batch_count = math.ceil(len(train_rows) / args.batch_size)
        optimizer_mass_per_step = epoch_mass / batch_count
        for offset in range(0, len(train_rows), args.batch_size):
            rows = train_rows[offset : offset + args.batch_size]
            grid, low5, hist, plans = _batch_to_device(pool, rows, device)
            sample_weight = (
                None
                if args.smoke
                else pool.trajectory_balanced_weight[rows].to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, args.amp):
                loss = seeded_cfm_loss(
                    policy,
                    grid,
                    low5,
                    hist,
                    plans,
                    generator=cfm_generator,
                    sample_weight=sample_weight,
                )
            batch_mass = (
                float(sample_weight.sum())
                if sample_weight is not None
                else float(len(rows))
            )
            # Account for a short last batch and small random differences in
            # inverse-trajectory mass.  Averaging these scaled minibatch
            # gradients over the epoch equals the declared weighted objective.
            backward_loss = loss * (batch_mass / optimizer_mass_per_step)
            backward_loss.backward()
            norm_squared = sum(
                float((parameter.grad.detach().float() ** 2).sum())
                for parameter in policy.enc_grid.parameters()
                if parameter.grad is not None
            )
            encoder_gradient += math.sqrt(norm_squared)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.gradient_clip)
            optimizer.step()
            weighted_loss += float(loss.detach()) * batch_mass
            objective_mass += batch_mass
            batches += 1
        scheduler.step()
        validation_loss = evaluate_cfm(
            policy,
            pool,
            validation_rows,
            batch_size=args.validation_batch_size,
            device=device,
            seed=args.seed + 10_000,
            amp=args.amp,
            trajectory_weighted=not args.smoke,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state_dict_cpu(policy)
        row = {
            "epoch": epoch,
            "train_cfm": weighted_loss / objective_mass,
            "validation_cfm": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_gradient_norm": encoder_gradient / max(batches, 1),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            ),
        }
        history.append(row)
        with history_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        if epoch % 10 == 0 or epoch == epochs - 1 or epoch == best_epoch:
            print(
                f"[pretrain {epoch:03d}/{epochs}] train={row['train_cfm']:.6f} "
                f"val={validation_loss:.6f} best={best_loss:.6f}@{best_epoch} "
                f"enc_grad={row['encoder_gradient_norm']:.3e}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    policy.load_state_dict(best_state)
    policy.eval()
    state_hash = model_state_hash(policy)
    checkpoint = outdir / "data/checkpoint_best.pt"
    phi0_path = outdir / "data/phi0_frozen.pt"
    candidate_checkpoint = outdir / "data/unpromoted_checkpoint_candidate.pt"
    candidate_phi0 = outdir / "data/unpromoted_phi0_candidate.pt"
    shared_extra = {
        "stage_schema": PRETRAIN_SCHEMA,
        "fresh_from_scratch": True,
        "endpoint_free": True,
        "best_epoch": best_epoch,
        "best_validation_cfm": best_loss,
        "model_state_sha256": state_hash,
        "source_manifest": str(Path(args.manifest).resolve()),
        "source_query_hash_digest": hashlib.sha256("".join(pool.query_hashes).encode()).hexdigest(),
        "expansion_promotion": False,
        "promotion_reason": "temperature-one all-gamma R/U gate not yet evaluated",
    }
    cpu_policy = policy.cpu()
    # Persist a diagnostic candidate so a failed gate is reproducible, but do
    # not create the production filenames consumed by Stage 4/5/6 yet.
    HP.save_hp(cpu_policy, candidate_checkpoint, extra=shared_extra)
    HP.save_hp(
        cpu_policy,
        candidate_phi0,
        extra={
            **shared_extra,
            "frozen_feature_snapshot": True,
            "feature_time": 0.9,
            "feature_dimension": 32,
        },
    )
    reloaded, phi_payload = HP.load_hp(candidate_phi0, device="cpu")
    if model_state_hash(reloaded) != state_hash or phi_payload["model_state_sha256"] != state_hash:
        raise RuntimeError("phi0 frozen snapshot failed its post-write hash check")
    _atomic_json(
        outdir / "data/unpromoted_phi0_manifest.json",
        {
            "path": str(candidate_phi0),
            "file_sha256": sha256_file(candidate_phi0),
            "model_state_sha256": state_hash,
            "feature_time": 0.9,
            "feature_dimension": 32,
            "frozen_during_expansion": True,
            "expansion_promotion": False,
        },
    )

    # Scientific ID evaluation is deliberately ordinary T=1 sampling.  The
    # smoother T=.5 setting is not used to claim SR/CR or model validity.
    policy = cpu_policy.to(device)
    eval_repetitions = min(args.eval_rollouts_per_gamma, 2) if args.smoke else args.eval_rollouts_per_gamma
    eval_steps = min(args.eval_max_steps, 20) if args.smoke else args.eval_max_steps
    id_metrics, id_rows = evaluate_id_rollouts(
        policy,
        device=device,
        repetitions=eval_repetitions,
        max_steps=eval_steps,
        reach_m=args.reach,
        temperature=1.0,
        nfe=args.nfe,
        seed=args.seed + 20_000,
    )
    _atomic_json(outdir / "tables/id_temperature1_metrics.json", id_metrics)
    _save_rollout_arrays(id_rows, outdir / "data/id_temperature1_rollouts.npz")
    mode_diversity_gate = {
        "required": not args.smoke,
        "temperature": 1.0,
        "criterion": "at least one successful R-first and U-first rollout at every gamma",
        "passed": bool(id_metrics["all_gammas_have_both_R_U_modes"]),
    }
    _atomic_json(outdir / "logs/mode_diversity_gate.json", mode_diversity_gate)
    if not args.smoke:
        validate_temperature_one_mode_diversity(id_metrics)

    id_metrics_sha256 = hashlib.sha256(
        json.dumps(
            _jsonable(id_metrics), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    promoted_extra = {
        **shared_extra,
        "expansion_promotion": not args.smoke,
        "promotion_reason": (
            "passed ordinary temperature-one all-gamma R/U gate"
            if not args.smoke
            else "smoke run is never eligible for expansion"
        ),
        "id_mode_diversity_gate_passed": bool(
            mode_diversity_gate["passed"] and not args.smoke
        ),
        "id_metrics_sha256": id_metrics_sha256,
        "id_evaluation_temperature": 1.0,
        "id_evaluation_uncertainty_tilting": False,
    }
    promoted_policy = policy.cpu()
    HP.save_hp(promoted_policy, checkpoint, extra=promoted_extra)
    HP.save_hp(
        promoted_policy,
        phi0_path,
        extra={
            **promoted_extra,
            "frozen_feature_snapshot": True,
            "feature_time": 0.9,
            "feature_dimension": 32,
        },
    )
    reloaded, phi_payload = HP.load_hp(phi0_path, device="cpu")
    if model_state_hash(reloaded) != state_hash or phi_payload["model_state_sha256"] != state_hash:
        raise RuntimeError("promoted phi0 snapshot failed its post-write hash check")
    _atomic_json(
        outdir / "data/phi0_manifest.json",
        {
            "path": str(phi0_path),
            "file_sha256": sha256_file(phi0_path),
            "model_state_sha256": state_hash,
            "feature_time": 0.9,
            "feature_dimension": 32,
            "frozen_during_expansion": True,
            "expansion_promotion": not args.smoke,
            "id_mode_diversity_gate_passed": bool(
                mode_diversity_gate["passed"] and not args.smoke
            ),
            "id_metrics_sha256": id_metrics_sha256,
        },
    )
    policy = promoted_policy.to(device)

    diagnostic: dict[str, Any] | None = None
    if args.visualization_rollouts_per_gamma > 0:
        diagnostic, diagnostic_rows = evaluate_id_rollouts(
            policy,
            device=device,
            repetitions=args.visualization_rollouts_per_gamma,
            max_steps=args.eval_max_steps,
            reach_m=args.reach,
            temperature=0.5,
            nfe=args.nfe,
            seed=args.seed + 30_000,
        )
        _atomic_json(outdir / "tables/id_temperature0.5_viz_diagnostic.json", diagnostic)
        _save_rollout_arrays(
            diagnostic_rows, outdir / "data/id_temperature0.5_viz_rollouts.npz"
        )

    summary = {
        "schema_version": PRETRAIN_SCHEMA,
        "status": "SMOKE_COMPLETE" if args.smoke else "PRETRAINED_AND_ID_EVALUATED",
        "started_at_utc": started_wall,
        "finished_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "contract": {
            "source": "fully verified generated SafeMPPI planned H=10 windows",
            "identity_rehashed_for_every_target": True,
            "executed_composites": 0,
            "unverified_targets": 0,
            "debug_training_targets": 0,
            "debug_target_share": 0.0,
            "synthetic_reflections": 0,
            "padding": 0,
            "raw_start_or_goal_inputs": False,
            "group_split_by_source_trajectory": True,
            "train_validation_trajectory_leakage": 0,
            "gamma_conditioning_is_not_a_curriculum": True,
        },
        "dataset": {
            "manifest": str(Path(args.manifest).resolve()),
            "windows": len(pool),
            "trajectories": len(pool.trajectory_rows),
            "sources": pool.sources,
            "split": split.audit,
            "all_exact_train_targets_used_each_epoch": not args.smoke,
            "trajectory_balancing": (
                "inverse-real-trajectory-length weighted objective"
                if not args.smoke
                else "equal real windows per trajectory without replacement"
            ),
            "smoke_train_windows_per_trajectory": train_per_trajectory if args.smoke else None,
            "smoke_validation_windows_per_trajectory": (
                validation_per_trajectory if args.smoke else None
            ),
        },
        "model": {
            "fresh_from_scratch": True,
            "endpoint_free": True,
            "repr_dim": 32,
            "context_dim": 37,
            "encoder_trainable_during_pretraining": True,
            "config": config,
            "state_sha256": state_hash,
        },
        "training": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "best_epoch": best_epoch,
            "best_validation_cfm": best_loss,
            "final": history[-1],
        },
        "id_evaluation_temperature1": id_metrics,
        "mode_diversity_gate": mode_diversity_gate,
        "temperature0.5_visualization_diagnostic": diagnostic,
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "phi0_frozen": str(phi0_path),
            "phi0_file_sha256": sha256_file(phi0_path),
            "history": str(history_path),
        },
        "dependencies": dependencies,
        "args": vars(args),
    }
    _atomic_json(outdir / "manifest.json", summary)
    _atomic_json(outdir / "logs/stage_summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "best_validation_cfm": best_loss,
                "phi0_state_sha256": state_hash,
                "temperature1_global_SR": id_metrics["global_success_rate_sr"],
                "temperature1_global_CR": id_metrics["global_collision_rate_cr"],
                "all_gammas_have_both_R_U_modes": id_metrics[
                    "all_gammas_have_both_R_U_modes"
                ],
                "manifest": str(outdir / "manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_STAGE2_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="logical CUDA device; caller must expose physical GPU 1 as cuda:0",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=1024)
    parser.add_argument("--windows-per-trajectory", type=int, default=64)
    parser.add_argument("--validation-windows-per-trajectory", type=int, default=32)
    parser.add_argument("--validation-trajectories-per-mode", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--trunk-hidden", nargs="+", type=int, default=(128, 64))
    parser.add_argument("--enc-depth", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--split-seed", type=int, default=31_711)
    parser.add_argument("--eval-rollouts-per-gamma", type=int, default=24)
    parser.add_argument("--visualization-rollouts-per-gamma", type=int, default=0)
    parser.add_argument("--eval-max-steps", type=int, default=260)
    parser.add_argument("--reach", type=float, default=0.20)
    parser.add_argument("--nfe", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    args.smoke = args.command == "smoke"
    positive = (
        "epochs", "batch_size", "validation_batch_size", "windows_per_trajectory",
        "validation_windows_per_trajectory", "eval_rollouts_per_gamma", "eval_max_steps", "nfe",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    run_stage(args)


if __name__ == "__main__":
    main()
