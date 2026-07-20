"""Hash-locked realized control-decision budgets for matched controls.

The Full arm can terminate an episode before its declared ``episode_max_steps``.
Giving a control the entire declared ceiling would then give it more FLOW
acquisition decisions than Full.  This module turns the exact Full episode
trace lengths into a portable, fingerprinted per-cell cap and validates the
realized usage embedded in control checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import clean_method_absence_manifest
from .scene import GAMMAS
from .stage5_expand import CHECKPOINT_SCHEMA, FULL_REPLAY_DESCRIPTION


REFERENCE_SCHEMA = "afe_full_realized_control_decision_budget_v1"
USAGE_SCHEMA = "afe_control_decision_budget_usage_v1"


class DecisionBudgetError(RuntimeError):
    """A Full reference or a control's realized usage is not auditable."""


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise DecisionBudgetError(
        f"expected mapping-like value, got {type(value).__name__}"
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise DecisionBudgetError("decision-budget metadata must be finite")
        return float(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            raise DecisionBudgetError("Full checkpoint state_dict has a non-tensor row")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _episode_gamma(value: Any) -> float:
    if hasattr(value, "gamma"):
        return float(value.gamma)
    row = _mapping(value)
    if "gamma" not in row:
        raise DecisionBudgetError("episode lacks gamma")
    return float(row["gamma"])


def _episode_seed(value: Any) -> int | None:
    if hasattr(value, "seed"):
        return int(value.seed)
    row = _mapping(value)
    return int(row["seed"]) if "seed" in row else None


def _episode_trace_count(value: Any) -> int:
    if hasattr(value, "traces"):
        traces = value.traces
    else:
        traces = _mapping(value).get("traces")
    if traces is None:
        raise DecisionBudgetError("episode lacks controller traces")
    return len(tuple(traces))


def _gamma_index(gamma: float) -> int:
    matches = [
        index for index, expected in enumerate(GAMMAS)
        if np.isclose(float(gamma), float(expected), rtol=0.0, atol=1e-9)
    ]
    if len(matches) != 1:
        raise DecisionBudgetError(f"unknown gamma {gamma!r}")
    return matches[0]


def episode_cells(
    episodes: Sequence[Any],
    *,
    round_index: int,
    episodes_per_gamma: int,
    expected_seed_base: int | None = None,
) -> list[dict[str, int | float]]:
    """Index exactly one episode collection by round/gamma/within-gamma index."""

    if round_index <= 0 or episodes_per_gamma <= 0:
        raise DecisionBudgetError("round and episodes_per_gamma must be positive")
    grouped: dict[int, list[Any]] = {index: [] for index in range(len(GAMMAS))}
    for episode in episodes:
        grouped[_gamma_index(_episode_gamma(episode))].append(episode)
    counts = {float(GAMMAS[index]): len(rows) for index, rows in grouped.items()}
    if any(count != episodes_per_gamma for count in counts.values()):
        raise DecisionBudgetError(
            f"round {round_index} does not have exactly {episodes_per_gamma} "
            f"episodes per gamma: {counts}"
        )

    result: list[dict[str, int | float]] = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for episode_index, episode in enumerate(grouped[gamma_index]):
            seed = _episode_seed(episode)
            if expected_seed_base is not None:
                expected_seed = (
                    int(expected_seed_base)
                    + int(round_index) * 1_000_000
                    + int(gamma_index) * 10_000
                    + int(episode_index)
                )
                if seed is None or seed != expected_seed:
                    raise DecisionBudgetError(
                        f"round {round_index} gamma {float(gamma)} episode "
                        f"{episode_index} seed {seed!r} != {expected_seed}"
                    )
            result.append({
                "round": int(round_index),
                "gamma": float(gamma),
                "episode_index": int(episode_index),
                "control_decisions": int(_episode_trace_count(episode)),
            })
    return result


def _recipe_without_extendable_rounds(recipe: Mapping[str, Any]) -> dict[str, Any]:
    result = _mapping(recipe)
    protocol = _mapping(result.get("matched_protocol", {}))
    protocol.pop("rounds", None)
    result["matched_protocol"] = protocol
    return result


def _validate_full_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(recipe)
    expected = {
        "arm": "full",
        "acquisition_mode": "afe",
        "progress_ranking": True,
        "eligibility_mode": "full",
        "replay_eligibility": "full_safe",
        "runtime_safety_claim": True,
        "uncertainty_tilting": True,
        "ordinary_audit_untilted": True,
    }
    drift = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if drift:
        raise DecisionBudgetError(f"reference is not the clean Full arm: {drift}")
    if row.get("legacy_mechanisms") != clean_method_absence_manifest():
        raise DecisionBudgetError("Full reference clean-method manifest is altered")
    if row.get("replay") != FULL_REPLAY_DESCRIPTION:
        raise DecisionBudgetError("Full reference is not uniform positive FLOW replay")
    protocol = row.get("matched_protocol")
    if not isinstance(protocol, Mapping) or not protocol:
        raise DecisionBudgetError("Full reference lacks a matched protocol")
    return row


def _reference_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(payload)
    return {key: value for key, value in row.items() if key != "fingerprint"}


def validate_reference_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and cryptographically validate a portable Full cap payload."""

    row = _mapping(payload)
    required = {
        "schema_version", "reference_dir", "reference_recipe_sha256",
        "source_checkpoint_sha256", "source_model_hash",
        "final_checkpoint_path", "final_checkpoint_sha256", "final_model_hash",
        "final_round", "matched_protocol", "caps", "total_control_decisions",
        "fingerprint",
    }
    if set(row) != required:
        raise DecisionBudgetError(
            "Full decision-budget payload fields differ: "
            f"missing={sorted(required - set(row))}, extra={sorted(set(row) - required)}"
        )
    if row["schema_version"] != REFERENCE_SCHEMA:
        raise DecisionBudgetError("unknown Full decision-budget schema")
    for key in (
        "reference_recipe_sha256", "source_checkpoint_sha256", "source_model_hash",
        "final_checkpoint_sha256", "final_model_hash", "fingerprint",
    ):
        if not _is_sha256(row[key]):
            raise DecisionBudgetError(f"Full decision-budget {key} is not SHA-256")
    if fingerprint(_reference_core(row)) != row["fingerprint"]:
        raise DecisionBudgetError("Full decision-budget fingerprint mismatch")

    protocol = _mapping(row["matched_protocol"])
    final_round = int(row["final_round"])
    episodes_per_gamma = int(protocol.get("episodes_per_gamma", 0))
    declared_rounds = int(protocol.get("rounds", 0))
    ceiling = int(protocol.get("episode_max_steps", -1))
    if final_round <= 0 or final_round != declared_rounds:
        raise DecisionBudgetError("Full cap rounds differ from matched protocol")
    if episodes_per_gamma <= 0 or ceiling < 0:
        raise DecisionBudgetError("Full cap protocol has invalid episode counts")

    caps = list(row["caps"])
    expected_cells = final_round * len(GAMMAS) * episodes_per_gamma
    if len(caps) != expected_cells:
        raise DecisionBudgetError(
            f"Full cap table has {len(caps)} cells, expected {expected_cells}"
        )
    normalized_caps: list[dict[str, int | float]] = []
    for cap in caps:
        item = _mapping(cap)
        if set(item) != {"round", "gamma", "episode_index", "max_control_decisions"}:
            raise DecisionBudgetError("malformed Full decision-cap cell")
        normalized_caps.append({
            "round": int(item["round"]),
            "gamma": float(GAMMAS[_gamma_index(float(item["gamma"]))]),
            "episode_index": int(item["episode_index"]),
            "max_control_decisions": int(item["max_control_decisions"]),
        })
    expected_keys = [
        (round_index, float(gamma), episode_index)
        for round_index in range(1, final_round + 1)
        for gamma in GAMMAS
        for episode_index in range(episodes_per_gamma)
    ]
    observed_keys = [
        (int(item["round"]), float(item["gamma"]), int(item["episode_index"]))
        for item in normalized_caps
    ]
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        raise DecisionBudgetError("Full decision-cap cells are not exact and canonical")
    if any(
        int(item["max_control_decisions"]) < 0
        or int(item["max_control_decisions"]) > ceiling
        for item in normalized_caps
    ):
        raise DecisionBudgetError("Full decision cap exceeds its declared ceiling")
    total = sum(int(item["max_control_decisions"]) for item in normalized_caps)
    if total != int(row["total_control_decisions"]):
        raise DecisionBudgetError("Full decision-cap total is inconsistent")
    return _canonical(row)


def load_full_reference(
    reference_dir: Path,
    *,
    final_checkpoint: Path,
    expected_protocol: Mapping[str, Any],
    expected_source_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Load all Full round bundles and derive exact realized decision caps."""

    root = Path(reference_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_protocol = _mapping(expected_protocol)
    requested_checkpoint = Path(final_checkpoint).resolve()
    if not requested_checkpoint.is_file():
        raise FileNotFoundError(requested_checkpoint)
    if requested_checkpoint.parent != root / "checkpoints":
        raise DecisionBudgetError(
            "explicit Full checkpoint must be inside reference_dir/checkpoints"
        )
    checkpoint = _mapping(torch.load(
        requested_checkpoint, map_location="cpu", weights_only=False,
    ))
    if checkpoint.get("afe_schema") != CHECKPOINT_SCHEMA:
        raise DecisionBudgetError("Full reference final checkpoint schema is invalid")
    final_round = int(checkpoint.get("round", -1))
    if final_round <= 0 or requested_checkpoint.name != f"round_{final_round:03d}.pt":
        raise DecisionBudgetError(
            "explicit Full checkpoint name/embedded round binding is invalid"
        )
    final_recipe = _validate_full_recipe(_mapping(checkpoint.get("recipe", {})))
    episodes_per_gamma = int(expected_protocol.get("episodes_per_gamma", 0))
    if final_round <= 0 or episodes_per_gamma <= 0:
        raise DecisionBudgetError("expected matched protocol is incomplete")
    expected_paths = [
        root / f"data/round_{round_index:03d}_bundle.pt"
        for round_index in range(0, final_round + 1)
    ]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise DecisionBudgetError(f"Full reference is missing round bundles: {missing}")
    if _mapping(final_recipe["matched_protocol"]) != expected_protocol:
        raise DecisionBudgetError("Full/control matched protocols differ")
    if final_recipe.get("source_checkpoint_sha256") != expected_source_checkpoint_sha256:
        raise DecisionBudgetError("Full/control source checkpoints differ")

    baseline = _mapping(torch.load(expected_paths[0], map_location="cpu", weights_only=False))
    if int(baseline.get("round", -1)) != 0 or baseline.get("episodes", ()):
        raise DecisionBudgetError("Full round-zero bundle is malformed")
    baseline_recipe = _validate_full_recipe(_mapping(baseline.get("recipe", {})))
    if _recipe_without_extendable_rounds(baseline_recipe) != _recipe_without_extendable_rounds(final_recipe):
        raise DecisionBudgetError("Full baseline recipe drifted beyond extendable rounds")
    del baseline

    cap_rows: list[dict[str, int | float]] = []
    full_seed = int(expected_protocol["seed"])
    ceiling = int(expected_protocol["episode_max_steps"])
    for round_index, path in enumerate(expected_paths[1:], start=1):
        bundle = _mapping(torch.load(path, map_location="cpu", weights_only=False))
        if int(bundle.get("round", -1)) != round_index:
            raise DecisionBudgetError(f"Full bundle {path.name} has the wrong round")
        embedded = _validate_full_recipe(_mapping(bundle.get("recipe", {})))
        embedded_protocol = _mapping(embedded["matched_protocol"])
        if int(embedded_protocol.get("rounds", 0)) > final_round:
            raise DecisionBudgetError("Full embedded recipe declares a later round")
        if _recipe_without_extendable_rounds(embedded) != _recipe_without_extendable_rounds(final_recipe):
            raise DecisionBudgetError(f"Full round {round_index} recipe drifted")
        cells = episode_cells(
            tuple(bundle.get("episodes", ())),
            round_index=round_index,
            episodes_per_gamma=episodes_per_gamma,
            expected_seed_base=full_seed,
        )
        for cell in cells:
            count = int(cell["control_decisions"])
            if count < 0 or count > ceiling:
                raise DecisionBudgetError(
                    f"Full round {round_index} realized {count} decisions beyond ceiling {ceiling}"
                )
            cap_rows.append({
                "round": int(cell["round"]),
                "gamma": float(cell["gamma"]),
                "episode_index": int(cell["episode_index"]),
                "max_control_decisions": count,
            })
        del bundle

    checkpoint_recipe = _validate_full_recipe(_mapping(checkpoint.get("recipe", {})))
    if checkpoint_recipe != final_recipe:
        raise DecisionBudgetError("Full final checkpoint recipe differs from logs recipe")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise DecisionBudgetError("Full reference final checkpoint lacks model state")
    final_model_hash = _state_dict_hash(state_dict)
    if checkpoint.get("current_model_hash") != final_model_hash:
        raise DecisionBudgetError("Full reference final model hash is invalid")
    del checkpoint

    core = {
        "schema_version": REFERENCE_SCHEMA,
        "reference_dir": str(root),
        "reference_recipe_sha256": fingerprint(final_recipe),
        "source_checkpoint_sha256": str(expected_source_checkpoint_sha256),
        "source_model_hash": str(final_recipe["source_model_hash"]),
        "final_checkpoint_path": str(requested_checkpoint),
        "final_checkpoint_sha256": sha256_file(requested_checkpoint),
        "final_model_hash": final_model_hash,
        "final_round": final_round,
        "matched_protocol": expected_protocol,
        "caps": cap_rows,
        "total_control_decisions": sum(
            int(item["max_control_decisions"]) for item in cap_rows
        ),
    }
    result = core | {"fingerprint": fingerprint(core)}
    return validate_reference_payload(result)


def cap_lookup(payload: Mapping[str, Any]) -> dict[tuple[int, float, int], int]:
    row = validate_reference_payload(payload)
    return {
        (int(item["round"]), float(item["gamma"]), int(item["episode_index"])):
        int(item["max_control_decisions"])
        for item in row["caps"]
    }


def build_usage(
    episodes: Sequence[Any],
    *,
    round_index: int,
    reference: Mapping[str, Any],
    expected_seed_base: int,
) -> dict[str, Any]:
    """Bind one control round's exact realized decisions to its Full caps."""

    ref = validate_reference_payload(reference)
    episodes_per_gamma = int(ref["matched_protocol"]["episodes_per_gamma"])
    lookup = cap_lookup(ref)
    cells = episode_cells(
        episodes,
        round_index=round_index,
        episodes_per_gamma=episodes_per_gamma,
        expected_seed_base=expected_seed_base,
    )
    usage_rows = []
    for cell in cells:
        key = (
            int(cell["round"]), float(cell["gamma"]), int(cell["episode_index"]),
        )
        cap = lookup[key]
        realized = int(cell["control_decisions"])
        if realized > cap:
            raise DecisionBudgetError(
                f"control cell {key} used {realized} decisions beyond Full cap {cap}"
            )
        usage_rows.append({
            "round": key[0],
            "gamma": key[1],
            "episode_index": key[2],
            "full_cap": cap,
            "realized_control_decisions": realized,
            "within_full_cap": True,
        })
    core = {
        "schema_version": USAGE_SCHEMA,
        "reference_fingerprint": ref["fingerprint"],
        "round": int(round_index),
        "cells": usage_rows,
        "realized_control_decisions": sum(
            int(item["realized_control_decisions"]) for item in usage_rows
        ),
        "full_control_decision_cap": sum(int(item["full_cap"]) for item in usage_rows),
        "all_cells_within_full_cap": True,
    }
    return core | {"fingerprint": fingerprint(core)}


def validate_usage(
    usage: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate checkpoint-only evidence of a control round's cellwise usage."""

    row = _mapping(usage)
    ref = validate_reference_payload(reference)
    required = {
        "schema_version", "reference_fingerprint", "round", "cells",
        "realized_control_decisions", "full_control_decision_cap",
        "all_cells_within_full_cap", "fingerprint",
    }
    if set(row) != required or row.get("schema_version") != USAGE_SCHEMA:
        raise DecisionBudgetError("malformed control decision-budget usage")
    if not _is_sha256(row.get("fingerprint")):
        raise DecisionBudgetError("control decision-budget usage lacks a fingerprint")
    core = {key: value for key, value in row.items() if key != "fingerprint"}
    if fingerprint(core) != row["fingerprint"]:
        raise DecisionBudgetError("control decision-budget usage fingerprint mismatch")
    if row["reference_fingerprint"] != ref["fingerprint"]:
        raise DecisionBudgetError("control usage points to another Full cap reference")
    round_index = int(row["round"])
    episodes_per_gamma = int(ref["matched_protocol"]["episodes_per_gamma"])
    lookup = cap_lookup(ref)
    expected_keys = [
        (round_index, float(gamma), episode_index)
        for gamma in GAMMAS
        for episode_index in range(episodes_per_gamma)
    ]
    cells = [_mapping(item) for item in row["cells"]]
    observed_keys = [
        (int(item["round"]), float(item["gamma"]), int(item["episode_index"]))
        for item in cells
    ]
    if observed_keys != expected_keys:
        raise DecisionBudgetError("control usage cells are not exact and canonical")
    realized_total = 0
    cap_total = 0
    for item, key in zip(cells, expected_keys):
        if set(item) != {
            "round", "gamma", "episode_index", "full_cap",
            "realized_control_decisions", "within_full_cap",
        }:
            raise DecisionBudgetError("malformed control usage cell")
        cap = lookup[key]
        realized = int(item["realized_control_decisions"])
        if int(item["full_cap"]) != cap or realized < 0 or realized > cap:
            raise DecisionBudgetError("control usage exceeds or alters its Full cell cap")
        if item["within_full_cap"] is not True:
            raise DecisionBudgetError("control usage cell is not marked within cap")
        realized_total += realized
        cap_total += cap
    if (
        realized_total != int(row["realized_control_decisions"])
        or cap_total != int(row["full_control_decision_cap"])
        or row["all_cells_within_full_cap"] is not True
    ):
        raise DecisionBudgetError("control decision-budget usage totals are inconsistent")
    return _canonical(row)
