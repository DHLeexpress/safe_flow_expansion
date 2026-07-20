#!/usr/bin/env python3
"""Stage 07: post-expansion galleries, active-expansion videos, and manifest.

This stage is a read-only consumer of completed Stage-05/06 runs.  It never
updates a model, verifier ledger, uncertainty matrix, or scientific result.
Ordinary temperature-0.5 rollouts are generated only for the rollout gallery;
all expansion candidates and held-out validity shown in the active videos are
the original temperature-one artifacts saved by the expansion stages.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

import grid_hp_expt as HP

from .ablations import AblationArm, ablation_spec, training_view
from .config import clean_method_absence_manifest
from .controller import ControlStepTrace, QueriedPlanTrace
from .decision_budget import (
    DecisionBudgetError,
    build_usage,
    episode_cells,
    validate_reference_payload,
    validate_usage,
)
from .policy import model_state_hash
from .schemas import QuerySource
from .scene import GAMMAS, make_ood_scene, verifier_spec_fingerprint
from .stage5_expand import evaluate_ordinary
from .store import VerificationStore
from .visualize_expansion import (
    ExpansionVizFrame,
    SceneSnapshot,
    build_expansion_frames,
    render_expansion_video,
    save_visualization_data,
)


STAGE = Path(__file__).resolve().parent / "stage_results/07_post_expansion_artifacts"
GALLERY_TEMPERATURE = 0.5
SCIENCE_TEMPERATURE = 1.0


class ArtifactStageError(RuntimeError):
    """A completed run cannot support a faithful post-expansion artifact."""


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    root: Path
    replay_eligibility: str
    runtime_safety_claim: bool
    acquisition_mode: str
    progress_ranking: bool
    eligibility_mode: str
    selected_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if self.replay_eligibility not in {"full_safe", "strict_bounds"}:
            raise ValueError("unknown replay eligibility")
        if self.replay_eligibility == "strict_bounds" and self.runtime_safety_claim:
            raise ValueError("strict-bounds offline run cannot claim runtime safety")
        if self.acquisition_mode not in {"afe", "uniform"}:
            raise ValueError("unknown acquisition mode")
        if self.eligibility_mode not in {"full", "bounds_only_offline"}:
            raise ValueError("unknown eligibility mode")
        object.__setattr__(self, "root", root)
        if self.selected_checkpoint is not None:
            checkpoint = Path(self.selected_checkpoint).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if checkpoint.parent != root / "checkpoints":
                raise ValueError("selected checkpoint must be inside run_root/checkpoints")
            object.__setattr__(self, "selected_checkpoint", checkpoint)


@dataclass(frozen=True)
class ExpansionSource:
    spec: RunSpec
    bundle_paths: tuple[Path, ...]
    bundles: tuple[dict[str, Any], ...]
    checkpoint_path: Path
    checkpoint_round: int
    store: VerificationStore
    traces: tuple[ControlStepTrace, ...]
    round_indices: tuple[int, ...]
    proximal_by_frame: dict[int, Any]
    audit_by_frame: dict[int, Any]
    replay_hashes_by_frame: dict[int, tuple[str, ...]]
    recipe: dict[str, Any]
    recipe_sha256: str
    checkpoint_sha256: str
    checkpoint_model_hash: str
    verifier_spec_fingerprint: str
    audit_bank_fingerprint: str
    matched_protocol: dict[str, Any]


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected mapping-like value, got {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value):
        return _stable_value(asdict(value))
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _stable_value(value.tolist())
    if isinstance(value, torch.Tensor):
        return _stable_value(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _stable_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if len(result) != 64:
        raise ArtifactStageError(f"{name} is not a SHA-256 digest")
    try:
        bytes.fromhex(result)
    except ValueError as exc:
        raise ArtifactStageError(f"{name} is not a SHA-256 digest") from exc
    return result


def _audit_mapping(value: Any, *, label: str) -> dict[str, Any]:
    row = _mapping(value)
    required = {
        "temperature", "uncertainty_tilting", "sampling_distribution",
        "context_bank_fingerprint", "context_bank_role", "per_gamma",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ArtifactStageError(
            f"{label}: audit lacks explicit provenance fields {missing}"
        )
    if float(row["temperature"]) != SCIENCE_TEMPERATURE:
        raise ArtifactStageError(f"{label}: expansion audit is not explicitly T=1")
    if row["uncertainty_tilting"] is not False:
        raise ArtifactStageError(f"{label}: expansion audit is not explicitly untilted")
    if row["sampling_distribution"] != "ordinary_conditional_flow_iid":
        raise ArtifactStageError(
            f"{label}: audit is not ordinary conditional-flow IID sampling"
        )
    _require_sha256(
        row["context_bank_fingerprint"], name=f"{label} audit-bank fingerprint",
    )
    if row["context_bank_role"] != "round_monitoring":
        raise ArtifactStageError(
            f"{label}: active-expansion video must use the round-monitoring audit bank"
        )
    per_gamma = row["per_gamma"]
    rows = per_gamma.values() if isinstance(per_gamma, Mapping) else per_gamma
    observed = tuple(sorted(float(_mapping(item)["gamma"]) for item in rows))
    expected = tuple(sorted(float(value) for value in GAMMAS))
    if observed != expected:
        raise ArtifactStageError(
            f"{label}: audit gamma allocation is not the fixed seven-level set"
        )
    return row


def _expected_arm_semantics(spec: RunSpec) -> dict[str, Any]:
    return {
        "full": {
            "acquisition_mode": "afe", "progress_ranking": True,
            "eligibility_mode": "full", "replay_eligibility": "full_safe",
            "runtime_safety_claim": True,
        },
        "minus_afe": {
            "acquisition_mode": "uniform", "progress_ranking": True,
            "eligibility_mode": "full", "replay_eligibility": "full_safe",
            "runtime_safety_claim": True,
        },
        "minus_progress": {
            "acquisition_mode": "afe", "progress_ranking": False,
            "eligibility_mode": "full", "replay_eligibility": "full_safe",
            "runtime_safety_claim": True,
        },
        "minus_socp": {
            "acquisition_mode": "afe", "progress_ranking": True,
            "eligibility_mode": "bounds_only_offline",
            "replay_eligibility": "strict_bounds",
            "runtime_safety_claim": False,
        },
    }.get(spec.key, {})


def _validate_recipe(spec: RunSpec, recipe_value: Any) -> dict[str, Any]:
    recipe = _mapping(recipe_value)
    required = {
        "arm", "acquisition_mode", "progress_ranking", "eligibility_mode",
        "replay_eligibility", "runtime_safety_claim", "uncertainty_tilting",
        "ordinary_audit_untilted", "source_checkpoint_sha256",
        "source_model_hash", "frozen_feature_hash", "audit_bank_fingerprint", "audit_bank_role",
        "verifier_spec_fingerprint", "legacy_mechanisms", "matched_protocol",
        "gamma_distribution", "beta",
    }
    missing = sorted(required - set(recipe))
    if missing:
        raise ArtifactStageError(
            f"{spec.label}: embedded recipe is incomplete; missing {missing}"
        )
    expected = _expected_arm_semantics(spec)
    if not expected:
        raise ArtifactStageError(f"{spec.label}: unsupported artifact arm key {spec.key!r}")
    if recipe["arm"] != spec.key:
        raise ArtifactStageError(
            f"{spec.label}: embedded recipe arm {recipe['arm']!r} is mislabeled"
        )
    declared = {
        "acquisition_mode": recipe["acquisition_mode"],
        "progress_ranking": bool(recipe["progress_ranking"]),
        "eligibility_mode": recipe["eligibility_mode"],
        "replay_eligibility": recipe["replay_eligibility"],
        "runtime_safety_claim": bool(recipe["runtime_safety_claim"]),
    }
    if declared != expected:
        raise ArtifactStageError(
            f"{spec.label}: embedded arm recipe is not the intended single switch: "
            f"{declared} != {expected}"
        )
    spec_declared = {
        "acquisition_mode": spec.acquisition_mode,
        "progress_ranking": spec.progress_ranking,
        "eligibility_mode": spec.eligibility_mode,
        "replay_eligibility": spec.replay_eligibility,
        "runtime_safety_claim": spec.runtime_safety_claim,
    }
    if spec_declared != expected:
        raise ArtifactStageError(f"{spec.label}: RunSpec mislabels its scientific arm")
    if bool(recipe["uncertainty_tilting"]) != (spec.acquisition_mode == "afe"):
        raise ArtifactStageError(f"{spec.label}: uncertainty-tilting recipe is inconsistent")
    if recipe["ordinary_audit_untilted"] is not True:
        raise ArtifactStageError(f"{spec.label}: recipe does not require an untilted audit")
    if _mapping(recipe["legacy_mechanisms"]) != clean_method_absence_manifest():
        raise ArtifactStageError(
            f"{spec.label}: clean-method absence manifest is missing or altered"
        )
    if recipe["audit_bank_role"] != "round_monitoring":
        raise ArtifactStageError(f"{spec.label}: wrong audit-bank role")
    for key in (
        "source_checkpoint_sha256", "source_model_hash", "frozen_feature_hash",
        "audit_bank_fingerprint", "verifier_spec_fingerprint",
    ):
        _require_sha256(recipe[key], name=f"{spec.label} recipe {key}")
    protocol = _mapping(recipe["matched_protocol"])
    if not protocol:
        raise ArtifactStageError(f"{spec.label}: matched protocol is empty")
    if float(protocol.get("expansion_temperature", float("nan"))) != 1.0:
        raise ArtifactStageError(f"{spec.label}: matched expansion temperature is not T=1")
    if float(protocol.get("beta", float("nan"))) != float(recipe["beta"]):
        raise ArtifactStageError(f"{spec.label}: beta differs within its recipe")
    if int(protocol.get("episodes_per_gamma", 0)) <= 0:
        raise ArtifactStageError(f"{spec.label}: fixed gamma allocation count is invalid")
    if spec.key != "full":
        reference_value = recipe.get("full_reference_decision_budget")
        if not isinstance(reference_value, Mapping):
            raise ArtifactStageError(
                f"{spec.label}: control lacks its selected-Full decision-budget reference"
            )
        try:
            reference = validate_reference_payload(reference_value)
        except DecisionBudgetError as exc:
            raise ArtifactStageError(
                f"{spec.label}: invalid selected-Full decision-budget reference: {exc}"
            ) from exc
        if _mapping(reference["matched_protocol"]) != protocol:
            raise ArtifactStageError(
                f"{spec.label}: Full decision caps use another matched protocol"
            )
        if recipe.get("control_decision_budget_rule") != (
            "for every (round,gamma,episode), max_steps equals the selected "
            "Full episode's realized len(traces); the control may terminate earlier"
        ):
            raise ArtifactStageError(
                f"{spec.label}: control does not declare the realized-Full cap rule"
            )
    gamma_text = str(recipe["gamma_distribution"]).lower()
    if "fixed" not in gamma_text or "seven" not in gamma_text or "no schedule" not in gamma_text:
        raise ArtifactStageError(f"{spec.label}: gamma allocation is not explicitly fixed")
    return recipe


def _episode_gamma(value: Any) -> float:
    if hasattr(value, "gamma"):
        return float(value.gamma)
    row = _mapping(value)
    if "gamma" not in row:
        raise ArtifactStageError("episode lacks its gamma allocation label")
    return float(row["gamma"])


def _validate_round_gamma_allocation(
    bundle: Mapping[str, Any], protocol: Mapping[str, Any], *, label: str,
) -> None:
    round_index = int(bundle.get("round", -1))
    if round_index == 0:
        if bundle.get("episodes", ()):
            raise ArtifactStageError(f"{label}: round-zero bundle must not have episodes")
        return
    episodes = tuple(bundle.get("episodes", ()))
    expected_count = int(protocol["episodes_per_gamma"])
    counts = {
        float(gamma): sum(np.isclose(_episode_gamma(item), gamma) for item in episodes)
        for gamma in GAMMAS
    }
    unknown = [
        _episode_gamma(item)
        for item in episodes
        if not any(np.isclose(_episode_gamma(item), gamma) for gamma in GAMMAS)
    ]
    if unknown or any(count != expected_count for count in counts.values()):
        raise ArtifactStageError(
            f"{label}: round {round_index} violates fixed per-gamma allocation: "
            f"counts={counts}, unknown={unknown[:3]}"
        )


def _store_identity(state: Mapping[str, Any]) -> tuple[tuple[str, ...], np.ndarray, int]:
    store = VerificationStore.from_state_dict(dict(state))
    return (
        tuple(record.query_hash for record in store.records),
        np.asarray(store.uncertainty.A, dtype=np.float64),
        int(store.uncertainty.count),
    )


def _round_from_name(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (ValueError, IndexError) as exc:
        raise ArtifactStageError(f"cannot parse round from {path}") from exc


def _query_trace(value: Any) -> QueriedPlanTrace:
    if isinstance(value, QueriedPlanTrace):
        return value
    row = _mapping(value)
    return QueriedPlanTrace(**{
        field: row[field]
        for field in QueriedPlanTrace.__dataclass_fields__
        if field in row
    })


def _control_trace(value: Any) -> ControlStepTrace:
    if isinstance(value, ControlStepTrace):
        return value
    row = _mapping(value)
    row["queried"] = tuple(_query_trace(item) for item in row.get("queried", ()))
    allowed = ControlStepTrace.__dataclass_fields__
    return ControlStepTrace(**{field: row[field] for field in allowed if field in row})


def _episode_traces(value: Any) -> tuple[ControlStepTrace, ...]:
    if hasattr(value, "traces"):
        rows = value.traces
    else:
        rows = _mapping(value).get("traces", ())
    return tuple(_control_trace(row) for row in rows)


def _checkpoint_for_round(root: Path, final_round: int) -> Path:
    exact = root / f"checkpoints/round_{final_round:03d}.pt"
    if exact.is_file():
        return exact
    candidates = sorted(root.glob("checkpoints/round_*.pt"))
    if not candidates:
        raise ArtifactStageError(f"no round checkpoint under {root}")
    result = candidates[-1]
    if _round_from_name(result) != final_round:
        raise ArtifactStageError(
            f"final bundle is round {final_round}, but final checkpoint is {result.name}"
        )
    return result


def _training_hashes(
    store: VerificationStore, spec: RunSpec,
) -> tuple[str, ...]:
    """Return identities from the immutable replay view actually optimized."""

    if spec.replay_eligibility == "strict_bounds":
        arm = ablation_spec(AblationArm.MINUS_SOCP)
        view = training_view(store.records, arm)
        hashes = []
        for item in view:
            item.validate_identity()
            # The wrapper retains actual SOCP rather than relabeling the
            # source record.  This assertion is deliberately identity-only.
            hashes.append(item.source_query_hash)
        return tuple(hashes)
    view = store.uniform_positive_view(source=QuerySource.FLOW)
    for item in view:
        item.validate_identity()
    return tuple(item.source_query_hash for item in view)


def load_expansion_source(spec: RunSpec) -> ExpansionSource:
    """Load every expansion round and tie traces to one cumulative store."""

    raw_checkpoint: dict[str, Any] | None = None
    selected_round: int | None = None
    if spec.selected_checkpoint is not None:
        checkpoint = spec.selected_checkpoint
        raw_checkpoint = _mapping(torch.load(
            checkpoint, map_location="cpu", weights_only=False,
        ))
        selected_round = int(raw_checkpoint.get("round", -1))
        if selected_round <= 0 or checkpoint.name != f"round_{selected_round:03d}.pt":
            raise ArtifactStageError(
                f"{spec.label}: explicit checkpoint name/embedded round is invalid"
            )
        bundle_paths = tuple(
            spec.root / f"data/round_{round_index:03d}_bundle.pt"
            for round_index in range(selected_round + 1)
        )
        missing = [str(path) for path in bundle_paths if not path.is_file()]
        if missing:
            raise ArtifactStageError(
                f"{spec.label}: selected checkpoint lacks prior bundles {missing}"
            )
    else:
        bundle_paths = tuple(sorted(spec.root.glob("data/round_*_bundle.pt")))
    if len(bundle_paths) < 2:
        raise ArtifactStageError(f"{spec.label}: expected baseline plus expansion bundles")
    bundles = tuple(
        _mapping(torch.load(path, map_location="cpu", weights_only=False))
        for path in bundle_paths
    )
    rounds = tuple(int(bundle.get("round", -1)) for bundle in bundles)
    if rounds != tuple(sorted(rounds)) or len(set(rounds)) != len(rounds):
        raise ArtifactStageError(f"{spec.label}: bundle rounds are not unique and ordered")
    if rounds[0] != 0:
        raise ArtifactStageError(f"{spec.label}: round-zero baseline bundle is missing")
    if any("recipe" not in bundle for bundle in bundles):
        raise ArtifactStageError(f"{spec.label}: a bundle has no embedded recipe")
    recipe = _validate_recipe(spec, bundles[-1]["recipe"])
    recipe_sha256 = _digest_json(recipe)
    protocol = _mapping(recipe["matched_protocol"])
    final_recipe_comparison = dict(recipe)
    final_protocol_comparison = _mapping(final_recipe_comparison.pop("matched_protocol"))
    final_declared_rounds = int(final_protocol_comparison.pop("rounds"))
    for bundle in bundles:
        embedded = _validate_recipe(spec, bundle["recipe"])
        embedded_comparison = dict(embedded)
        embedded_protocol = _mapping(embedded_comparison.pop("matched_protocol"))
        embedded_declared_rounds = int(embedded_protocol.pop("rounds"))
        if (
            embedded_comparison != final_recipe_comparison
            or embedded_protocol != final_protocol_comparison
            or embedded_declared_rounds > final_declared_rounds
        ):
            raise ArtifactStageError(
                f"{spec.label}: round {bundle.get('round')} recipe drifted"
            )
        bundle_arm = bundle.get("arm", "full" if spec.key == "full" else None)
        if bundle_arm != spec.key:
            raise ArtifactStageError(
                f"{spec.label}: bundle arm {bundle_arm!r} disagrees with RunSpec"
            )
        _validate_round_gamma_allocation(
            bundle, _mapping(embedded["matched_protocol"]), label=spec.label,
        )
        round_index = int(bundle.get("round", -1))
        if spec.key != "full" and round_index > 0:
            query_summary = bundle.get("query_summary")
            if not isinstance(query_summary, Mapping):
                raise ArtifactStageError(
                    f"{spec.label}: round {round_index} lacks decision-budget usage"
                )
            usage_value = query_summary.get(
                "full_reference_control_decision_budget"
            )
            try:
                reference = validate_reference_payload(
                    embedded["full_reference_decision_budget"]
                )
                usage = validate_usage(usage_value, reference)
                reconstructed = build_usage(
                    tuple(bundle.get("episodes", ())),
                    round_index=round_index,
                    reference=reference,
                    expected_seed_base=int(protocol["seed"]),
                )
            except (DecisionBudgetError, TypeError) as exc:
                raise ArtifactStageError(
                    f"{spec.label}: round {round_index} decision-budget evidence failed: {exc}"
                ) from exc
            if _stable_value(usage) != _stable_value(reconstructed):
                raise ArtifactStageError(
                    f"{spec.label}: round {round_index} cap usage differs from exact traces"
                )
        audit_value = bundle.get("audit", bundle.get("audit_actual_full_socp"))
        if audit_value is None:
            raise ArtifactStageError(
                f"{spec.label}: round {bundle.get('round')} has no audit"
            )
        audit_row = _audit_mapping(audit_value, label=spec.label)
        if audit_row["context_bank_fingerprint"] != recipe["audit_bank_fingerprint"]:
            raise ArtifactStageError(f"{spec.label}: audit bank differs from recipe")
        if audit_row["context_bank_role"] != recipe["audit_bank_role"]:
            raise ArtifactStageError(f"{spec.label}: audit-bank role differs from recipe")

    final_round = rounds[-1]
    if selected_round is not None and final_round != selected_round:
        raise ArtifactStageError(f"{spec.label}: explicit checkpoint/bundle round mismatch")
    checkpoint = (
        spec.selected_checkpoint
        if spec.selected_checkpoint is not None
        else _checkpoint_for_round(spec.root, final_round)
    )
    assert checkpoint is not None
    checkpoint_sha256 = _sha256(checkpoint)
    if raw_checkpoint is None:
        raw_checkpoint = _mapping(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        )
    if int(raw_checkpoint.get("round", -1)) != final_round:
        raise ArtifactStageError(f"{spec.label}: checkpoint round differs from bundle")
    if _mapping(raw_checkpoint.get("recipe", {})) != recipe:
        raise ArtifactStageError(f"{spec.label}: checkpoint recipe differs from bundle")
    checkpoint_model_hash = _require_sha256(
        raw_checkpoint.get("current_model_hash"),
        name=f"{spec.label} checkpoint current_model_hash",
    )
    history = raw_checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ArtifactStageError(f"{spec.label}: checkpoint has no round history")
    history_final = _mapping(history[-1])
    if (
        int(history_final.get("round", -1)) != final_round
        or history_final.get("model_hash") != checkpoint_model_hash
    ):
        raise ArtifactStageError(
            f"{spec.label}: checkpoint model hash is not bound to its final history round"
        )
    final_bundle = bundles[-1]
    bundle_history_fields = {
        "query": "query_summary",
        "solver": "solver",
        "audit": (
            "audit_actual_full_socp"
            if "audit_actual_full_socp" in final_bundle
            else "audit"
        ),
        "matrix": "matrix",
    }
    for history_key, bundle_key in bundle_history_fields.items():
        if history_key not in history_final or bundle_key not in final_bundle:
            raise ArtifactStageError(
                f"{spec.label}: final bundle/history lacks {history_key!r} binding"
            )
        if _stable_value(history_final[history_key]) != _stable_value(final_bundle[bundle_key]):
            raise ArtifactStageError(
                f"{spec.label}: checkpoint history {history_key} differs from final bundle"
            )
    final_state = bundles[-1].get("store_state")
    if not final_state:
        raise ArtifactStageError(f"{spec.label}: final bundle has no cumulative store")
    final_store = VerificationStore.from_state_dict(final_state)
    checkpoint_store_state = raw_checkpoint.get("verification_store_state")
    if not isinstance(checkpoint_store_state, Mapping):
        raise ArtifactStageError(f"{spec.label}: checkpoint has no cumulative store")
    bundle_identity = _store_identity(final_state)
    checkpoint_identity = _store_identity(checkpoint_store_state)
    if (
        bundle_identity[0] != checkpoint_identity[0]
        or bundle_identity[2] != checkpoint_identity[2]
        or not np.array_equal(bundle_identity[1], checkpoint_identity[1])
    ):
        raise ArtifactStageError(
            f"{spec.label}: final checkpoint store differs from final bundle"
        )
    verifier_fingerprint = str(recipe["verifier_spec_fingerprint"])
    for record in final_store.records:
        record.validate_identity()
        if record.context.verifier_spec_fingerprint != verifier_fingerprint:
            raise ArtifactStageError(
                f"{spec.label}: query ledger contains another verifier specification"
            )

    traces: list[ControlStepTrace] = []
    trace_rounds: list[int] = []
    proximal: dict[int, Any] = {}
    audits: dict[int, Any] = {}
    replay_hashes: dict[int, tuple[str, ...]] = {}
    baseline_store = VerificationStore.from_state_dict(bundles[0]["store_state"])
    previous_hashes = tuple(record.query_hash for record in baseline_store.records)
    if previous_hashes:
        raise ArtifactStageError(f"{spec.label}: round-zero acquisition ledger is not empty")
    for bundle in bundles[1:]:
        round_index = int(bundle["round"])
        round_traces = tuple(
            trace
            for episode in bundle.get("episodes", ())
            for trace in _episode_traces(episode)
        )
        if not round_traces:
            raise ArtifactStageError(
                f"{spec.label}: round {round_index} has no controller traces"
            )
        start_index = len(traces)
        traces.extend(round_traces)
        trace_rounds.extend([round_index] * len(round_traces))

        state = bundle.get("store_state")
        if not state:
            raise ArtifactStageError(
                f"{spec.label}: round {round_index} lacks its cumulative store snapshot"
            )
        round_store = VerificationStore.from_state_dict(state)
        current_hashes = tuple(record.query_hash for record in round_store.records)
        if current_hashes[: len(previous_hashes)] != previous_hashes:
            raise ArtifactStageError(
                f"{spec.label}: round {round_index} ledger is not append-only"
            )
        queried_this_round = tuple(
            query.query_hash for trace in round_traces for query in trace.queried
        )
        if queried_this_round != current_hashes[len(previous_hashes) :]:
            raise ArtifactStageError(
                f"{spec.label}: round {round_index} trace events do not match "
                "the exact ordered ledger append"
            )
        previous_hashes = current_hashes
        final_frame_index = start_index + len(round_traces) - 1
        solver = bundle.get("solver")
        if solver is None:
            raise ArtifactStageError(f"{spec.label}: round {round_index} has no proximal telemetry")
        proximal[final_frame_index] = solver
        audit = bundle.get("audit", bundle.get("audit_actual_full_socp"))
        _audit_mapping(audit, label=spec.label)
        audits[final_frame_index] = audit
        hashes = _training_hashes(round_store, spec)
        solver_count = int(_mapping(solver).get("positive_count", len(hashes)))
        if solver_count != len(hashes):
            raise ArtifactStageError(
                f"{spec.label}: solver trained {solver_count} rows but immutable replay has {len(hashes)}"
            )
        replay_hashes[final_frame_index] = hashes

    if previous_hashes != tuple(record.query_hash for record in final_store.records):
        raise ArtifactStageError(f"{spec.label}: final bundle/store mismatch")
    traced = tuple(query.query_hash for trace in traces for query in trace.queried)
    if traced != previous_hashes:
        raise ArtifactStageError(
            f"{spec.label}: all-round ordered trace/store identity mismatch"
        )
    return ExpansionSource(
        spec=spec,
        bundle_paths=bundle_paths,
        bundles=bundles,
        checkpoint_path=checkpoint,
        checkpoint_round=final_round,
        store=final_store,
        traces=tuple(traces),
        round_indices=tuple(trace_rounds),
        proximal_by_frame=proximal,
        audit_by_frame=audits,
        replay_hashes_by_frame=replay_hashes,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_model_hash=checkpoint_model_hash,
        verifier_spec_fingerprint=verifier_fingerprint,
        audit_bank_fingerprint=str(recipe["audit_bank_fingerprint"]),
        matched_protocol=protocol,
    )


def build_run_frames(source: ExpansionSource) -> tuple[ExpansionVizFrame, ...]:
    return build_expansion_frames(
        source.traces,
        source.store,
        source.proximal_by_frame,
        audit_results=source.audit_by_frame,
        replay_query_hashes=source.replay_hashes_by_frame,
        round_indices=source.round_indices,
        expansion_temperature=SCIENCE_TEMPERATURE,
        audit_temperature=SCIENCE_TEMPERATURE,
        replay_eligibility=source.spec.replay_eligibility,
        runtime_safety_claim=source.spec.runtime_safety_claim,
        method_label=source.spec.label,
        acquisition_mode=source.spec.acquisition_mode,
        progress_ranking=source.spec.progress_ranking,
    )


def validate_matched_sources(
    sources: Sequence[ExpansionSource],
) -> dict[str, Any]:
    """Reject any cross-arm drift beyond the three declared single switches."""

    rows = tuple(sources)
    expected_keys = {"full", "minus_afe", "minus_progress", "minus_socp"}
    keys = [source.spec.key for source in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ArtifactStageError(
            f"artifact comparison requires exactly {sorted(expected_keys)}, got {keys}"
        )

    def one(field: str) -> str:
        values = {str(source.recipe[field]) for source in rows}
        if len(values) != 1:
            raise ArtifactStageError(f"cross-arm {field} mismatch: {sorted(values)}")
        return next(iter(values))

    source_checkpoint_sha256 = one("source_checkpoint_sha256")
    source_model_hash = one("source_model_hash")
    frozen_feature_hash = one("frozen_feature_hash")
    audit_bank_fingerprint = one("audit_bank_fingerprint")
    verifier_fingerprint = one("verifier_spec_fingerprint")
    protocols = {_digest_json(source.matched_protocol) for source in rows}
    if len(protocols) != 1:
        detail = {
            source.spec.key: source.matched_protocol for source in rows
        }
        raise ArtifactStageError(f"cross-arm matched protocol drift: {detail}")
    protocol = rows[0].matched_protocol
    if int(protocol.get("rounds", -1)) != rows[0].checkpoint_round or any(
        source.checkpoint_round != rows[0].checkpoint_round for source in rows
    ):
        raise ArtifactStageError(
            "cross-arm completed rounds do not equal the declared matched protocol"
        )
    if any(float(source.recipe["beta"]) != float(protocol["beta"]) for source in rows):
        raise ArtifactStageError("cross-arm beta is not bound to the matched protocol")
    for source in rows:
        if source.audit_bank_fingerprint != audit_bank_fingerprint:
            raise ArtifactStageError("source audit-bank fingerprint cache is inconsistent")
        if source.verifier_spec_fingerprint != verifier_fingerprint:
            raise ArtifactStageError("source verifier fingerprint cache is inconsistent")
        _validate_recipe(source.spec, source.recipe)
    full = next(source for source in rows if source.spec.key == "full")
    control_references = [
        validate_reference_payload(
            source.recipe["full_reference_decision_budget"]
        )
        for source in rows
        if source.spec.key != "full"
    ]
    reference_fingerprints = {
        str(reference["fingerprint"]) for reference in control_references
    }
    if len(reference_fingerprints) != 1:
        raise ArtifactStageError("controls do not share one selected-Full cap reference")
    for reference in control_references:
        if (
            Path(reference["reference_dir"]).resolve() != full.spec.root
            or Path(reference["final_checkpoint_path"]).resolve()
            != full.checkpoint_path.resolve()
            or reference["reference_recipe_sha256"] != full.recipe_sha256
            or reference["final_checkpoint_sha256"] != full.checkpoint_sha256
            or reference["final_model_hash"] != full.checkpoint_model_hash
            or int(reference["final_round"]) != full.checkpoint_round
            or reference["source_checkpoint_sha256"] != source_checkpoint_sha256
            or reference["source_model_hash"] != source_model_hash
            or _mapping(reference["matched_protocol"]) != protocol
        ):
            raise ArtifactStageError(
                "a control's realized-decision caps are not bound to the displayed Full run"
            )
    expected_caps: list[dict[str, int | float]] = []
    episodes_per_gamma = int(protocol["episodes_per_gamma"])
    for bundle in full.bundles[1:]:
        round_index = int(bundle["round"])
        for cell in episode_cells(
            tuple(bundle.get("episodes", ())),
            round_index=round_index,
            episodes_per_gamma=episodes_per_gamma,
            expected_seed_base=int(protocol["seed"]),
        ):
            expected_caps.append({
                "round": int(cell["round"]),
                "gamma": float(cell["gamma"]),
                "episode_index": int(cell["episode_index"]),
                "max_control_decisions": int(cell["control_decisions"]),
            })
    if any(reference["caps"] != expected_caps for reference in control_references):
        raise ArtifactStageError(
            "control decision-cap table differs from the displayed Full traces"
        )
    return {
        "status": "PASS",
        "arms": keys,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_model_hash": source_model_hash,
        "frozen_feature_hash": frozen_feature_hash,
        "audit_bank_fingerprint": audit_bank_fingerprint,
        "verifier_spec_fingerprint": verifier_fingerprint,
        "matched_protocol_sha256": next(iter(protocols)),
        "matched_protocol": protocol,
        "full_reference_decision_budget_fingerprint": next(
            iter(reference_fingerprints)
        ),
        "control_decision_caps_equal_exact_full_trace_lengths": True,
        "every_control_cell_within_corresponding_full_cap": True,
        "only_intended_arm_switches": True,
    }


def _video_subset(
    frames: Sequence[ExpansionVizFrame], maximum: int,
) -> tuple[ExpansionVizFrame, ...]:
    rows = tuple(frames)
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    update_indices = {index for index, row in enumerate(rows) if row.proximal is not None}
    if len(update_indices) + 1 > maximum:
        raise ArtifactStageError(
            "video-max-events is too small to retain every proximal-update frame"
        )
    uniform = np.linspace(0, len(rows) - 1, maximum, dtype=int)
    chosen = set(int(index) for index in uniform) | update_indices | {len(rows) - 1}
    while len(chosen) > maximum:
        removable = sorted(chosen - update_indices - {len(rows) - 1})
        if not removable:
            break
        chosen.remove(removable[len(removable) // 2])
    return tuple(rows[index] for index in sorted(chosen))


def save_gallery_artifact(
    path: Path,
    *,
    label: str,
    checkpoint: Path,
    checkpoint_hash: str,
    checkpoint_model_hash: str,
    source_recipe_sha256: str | None,
    scene_verifier_spec_fingerprint: str,
    rollouts: Sequence[Any],
    summary: Mapping[str, Any],
    seed: int,
    nfe: int,
) -> Path:
    rows = [asdict(row) if is_dataclass(row) else _mapping(row) for row in rollouts]
    temperatures = {float(row.get("temperature", np.nan)) for row in rows}
    if temperatures != {GALLERY_TEMPERATURE}:
        raise ArtifactStageError(
            f"{label}: gallery artifact must be exclusively T=0.5, got {temperatures}"
        )
    checkpoint_hash = _require_sha256(checkpoint_hash, name=f"{label} checkpoint file")
    checkpoint_model_hash = _require_sha256(
        checkpoint_model_hash, name=f"{label} checkpoint model",
    )
    if source_recipe_sha256 is not None:
        source_recipe_sha256 = _require_sha256(
            source_recipe_sha256, name=f"{label} source recipe",
        )
    scene_verifier_spec_fingerprint = _require_sha256(
        scene_verifier_spec_fingerprint, name=f"{label} gallery scene verifier spec",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "afe_gallery_rollouts_v1",
        "label": label,
        "visualization_temperature": GALLERY_TEMPERATURE,
        "visualization_rollouts": rows,
        "gallery_diagnostics_not_scientific_metrics": dict(summary),
        "scientific_use_forbidden": True,
        "scientific_metrics_source_temperature": SCIENCE_TEMPERATURE,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_model_state_sha256": checkpoint_model_hash,
        "source_recipe_sha256": source_recipe_sha256,
        "scene_verifier_spec_fingerprint": scene_verifier_spec_fingerprint,
        "seed": int(seed),
        "nfe": int(nfe),
        "gamma_levels": [float(gamma) for gamma in GAMMAS],
        "gamma_allocation": {
            str(float(gamma)): sum(
                np.isclose(float(row["gamma"]), gamma) for row in rows
            )
            for gamma in GAMMAS
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _validate_gallery_allocation(
    rollouts: Sequence[Any], *, per_gamma: int, label: str,
) -> None:
    rows = [asdict(row) if is_dataclass(row) else _mapping(row) for row in rollouts]
    counts = {
        float(gamma): sum(np.isclose(float(row["gamma"]), gamma) for row in rows)
        for gamma in GAMMAS
    }
    unknown = [
        float(row["gamma"])
        for row in rows
        if not any(np.isclose(float(row["gamma"]), gamma) for gamma in GAMMAS)
    ]
    if unknown or any(count != int(per_gamma) for count in counts.values()):
        raise ArtifactStageError(
            f"{label}: T=0.5 gallery gamma allocation mismatch: "
            f"counts={counts}, unknown={unknown[:3]}"
        )


def _input_hashes(source: ExpansionSource) -> dict[str, str]:
    paths = source.bundle_paths + (source.checkpoint_path,)
    return {str(path.resolve()): _sha256(path) for path in paths}


def generate_run_artifacts(
    source: ExpansionSource,
    output_root: Path,
    *,
    device: torch.device,
    gallery_seed: int,
    gallery_rollouts_per_gamma: int,
    nfe: int,
    fps: int,
    seconds_per_event: float,
    dpi: int,
    video_max_events: int,
    model_loader: Callable[..., Any] = HP.load_hp,
    gallery_evaluator: Callable[..., Any] = evaluate_ordinary,
    video_renderer: Callable[..., Any] = render_expansion_video,
) -> dict[str, Any]:
    """Generate one run's artifacts and prove its inputs stayed unchanged."""

    before_hashes = _input_hashes(source)
    model, checkpoint_payload = model_loader(source.checkpoint_path, device=device)
    checkpoint_payload = _mapping(checkpoint_payload)
    before_model_hash = model_state_hash(model)
    if before_hashes[str(source.checkpoint_path.resolve())] != source.checkpoint_sha256:
        raise ArtifactStageError(f"{source.spec.label}: checkpoint file hash changed after load")
    if before_model_hash != source.checkpoint_model_hash:
        raise ArtifactStageError(
            f"{source.spec.label}: loaded model differs from embedded final model hash"
        )
    if checkpoint_payload.get("current_model_hash") != source.checkpoint_model_hash:
        raise ArtifactStageError(
            f"{source.spec.label}: loader payload lost final checkpoint model binding"
        )
    if int(checkpoint_payload.get("round", -1)) != source.checkpoint_round:
        raise ArtifactStageError(f"{source.spec.label}: loaded checkpoint round changed")
    if _mapping(checkpoint_payload.get("recipe", {})) != source.recipe:
        raise ArtifactStageError(f"{source.spec.label}: loaded checkpoint recipe changed")
    env = make_ood_scene(radius=1.2)
    current_verifier_fingerprint = verifier_spec_fingerprint(env, env.goal)
    if current_verifier_fingerprint != source.verifier_spec_fingerprint:
        raise ArtifactStageError(
            f"{source.spec.label}: gallery scene/verifier differs from the query ledger"
        )
    summary, rollouts = gallery_evaluator(
        model,
        env,
        seed=gallery_seed,
        per_gamma=gallery_rollouts_per_gamma,
        nfe=nfe,
        temperature=GALLERY_TEMPERATURE,
    )
    _validate_gallery_allocation(
        rollouts, per_gamma=gallery_rollouts_per_gamma, label=source.spec.label,
    )
    if model_state_hash(model) != before_model_hash:
        raise ArtifactStageError(f"{source.spec.label}: gallery inference mutated model weights")
    run_output = output_root / source.spec.key
    gallery_path = save_gallery_artifact(
        run_output / "data/visualization_rollouts.pt",
        label=source.spec.label,
        checkpoint=source.checkpoint_path,
        checkpoint_hash=before_hashes[str(source.checkpoint_path.resolve())],
        checkpoint_model_hash=source.checkpoint_model_hash,
        source_recipe_sha256=source.recipe_sha256,
        scene_verifier_spec_fingerprint=current_verifier_fingerprint,
        rollouts=rollouts,
        summary=summary,
        seed=gallery_seed,
        nfe=nfe,
    )

    frames = build_run_frames(source)
    viz_data = save_visualization_data(
        run_output / "data/active_expansion.json",
        SceneSnapshot.from_environment(env),
        frames,
        metadata={
            "run": source.spec.label,
            "source_run": str(source.spec.root),
            "source_trace_count": len(source.traces),
            "exact_training_identity_count": len(source.replay_hashes_by_frame),
            "no_curriculum_learning": True,
            "gamma_distribution": "fixed; no schedule",
            "gallery_temperature_is_separate": GALLERY_TEMPERATURE,
            "acquisition_mode": source.spec.acquisition_mode,
            "progress_ranking": source.spec.progress_ranking,
            "checkpoint_file_sha256": source.checkpoint_sha256,
            "checkpoint_model_state_sha256": source.checkpoint_model_hash,
            "source_recipe_sha256": source.recipe_sha256,
            "scene_verifier_spec_fingerprint": current_verifier_fingerprint,
            "audit_bank_fingerprint": source.audit_bank_fingerprint,
        },
    )
    video_frames = _video_subset(frames, video_max_events)
    video_path = run_output / "viz/active_expansion.mp4"
    video_manifest = video_renderer(
        SceneSnapshot.from_environment(env),
        video_frames,
        video_path,
        preview_png=run_output / "viz/active_expansion_preview.png",
        fps=fps,
        seconds_per_event=seconds_per_event,
        dpi=dpi,
    )
    after_hashes = _input_hashes(source)
    if after_hashes != before_hashes:
        raise ArtifactStageError(f"{source.spec.label}: a source run artifact was modified")
    final_replay = frames[-1].replay
    result = {
        "status": "PASS",
        "label": source.spec.label,
        "source_run": str(source.spec.root),
        "source_checkpoint_round": source.checkpoint_round,
        "source_checkpoint_file_sha256": source.checkpoint_sha256,
        "source_checkpoint_model_state_sha256": source.checkpoint_model_hash,
        "source_recipe_sha256": source.recipe_sha256,
        "clean_method_absence_manifest": source.recipe["legacy_mechanisms"],
        "scene_verifier_spec_fingerprint": current_verifier_fingerprint,
        "source_hashes_unchanged": True,
        "source_sha256": before_hashes,
        "gallery": {
            "path": str(gallery_path.resolve()),
            "temperature": GALLERY_TEMPERATURE,
            "scientific_use_forbidden": True,
            "rollouts": len(rollouts),
        },
        "active_expansion": {
            "data": str(viz_data.resolve()),
            "mp4": str(video_path.resolve()),
            "source_trace_count": len(frames),
            "rendered_event_count": len(video_frames),
            "candidate_temperature": SCIENCE_TEMPERATURE,
            "audit_temperature": SCIENCE_TEMPERATURE,
            "audit_provenance": asdict(frames[-1].audit_provenance),
            "acquisition_mode": source.spec.acquisition_mode,
            "progress_ranking": source.spec.progress_ranking,
            "query_acceptance_scope": "FLOW_only",
            "flow_query_count": frames[-1].query_count,
            "flow_positive_count": frames[-1].positive_count,
            "flow_query_acceptance": frames[-1].query_acceptance,
            "backup_query_count": frames[-1].backup_query_count,
            "backup_positive_count": frames[-1].backup_positive_count,
            "backup_acceptance": frames[-1].backup_acceptance,
            "replay_eligibility": source.spec.replay_eligibility,
            "exact_final_training_rows": len(final_replay),
            "actual_socp_failures_in_final_training_rows": sum(
                not row.safe for row in final_replay
            ),
            "runtime_safety_claim": source.spec.runtime_safety_claim,
            "no_runtime_safety_claim_banner": not source.spec.runtime_safety_claim,
            "video_manifest": video_manifest,
        },
        "no_curriculum_learning": True,
    }
    manifest_path = run_output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["manifest"] = str(manifest_path.resolve())
    return result


def generate_checkpoint_gallery(
    checkpoint: Path,
    output_root: Path,
    *,
    device: torch.device,
    gallery_seed: int,
    gallery_rollouts_per_gamma: int,
    nfe: int,
    expected_file_sha256: str | None = None,
    label: str = "Pretrained",
    key: str = "pretrained",
    model_loader: Callable[..., Any] = HP.load_hp,
    gallery_evaluator: Callable[..., Any] = evaluate_ordinary,
) -> dict[str, Any]:
    """Generate a gallery from an arbitrary internally hash-locked checkpoint."""

    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    before_file_hash = _sha256(source)
    if expected_file_sha256 is not None:
        expected = expected_file_sha256.strip().lower()
        if len(expected) != 64 or before_file_hash != expected:
            raise ArtifactStageError(
                f"{label}: checkpoint file SHA-256 mismatch "
                f"({before_file_hash} != {expected})"
            )
    model, payload = model_loader(source, device=device)
    embedded_model_hash = _mapping(payload).get("model_state_sha256")
    actual_model_hash = model_state_hash(model)
    if embedded_model_hash is None or str(embedded_model_hash) != actual_model_hash:
        raise ArtifactStageError(
            f"{label}: checkpoint lacks a valid embedded model_state_sha256"
        )
    env = make_ood_scene(radius=1.2)
    current_verifier_fingerprint = verifier_spec_fingerprint(env, env.goal)
    summary, rollouts = gallery_evaluator(
        model,
        env,
        seed=gallery_seed,
        per_gamma=gallery_rollouts_per_gamma,
        nfe=nfe,
        temperature=GALLERY_TEMPERATURE,
    )
    _validate_gallery_allocation(
        rollouts, per_gamma=gallery_rollouts_per_gamma, label=label,
    )
    if model_state_hash(model) != actual_model_hash:
        raise ArtifactStageError(f"{label}: gallery inference mutated model weights")
    if _sha256(source) != before_file_hash:
        raise ArtifactStageError(f"{label}: source checkpoint was modified")
    gallery = save_gallery_artifact(
        output_root / key / "data/visualization_rollouts.pt",
        label=label,
        checkpoint=source,
        checkpoint_hash=before_file_hash,
        checkpoint_model_hash=actual_model_hash,
        source_recipe_sha256=None,
        scene_verifier_spec_fingerprint=current_verifier_fingerprint,
        rollouts=rollouts,
        summary=summary,
        seed=gallery_seed,
        nfe=nfe,
    )
    result = {
        "status": "PASS",
        "label": label,
        "checkpoint": str(source),
        "checkpoint_file_sha256": before_file_hash,
        "model_state_sha256": actual_model_hash,
        "scene_verifier_spec_fingerprint": current_verifier_fingerprint,
        "embedded_model_hash_verified": True,
        "caller_file_hash_verified": expected_file_sha256 is not None,
        "source_hash_unchanged": True,
        "gallery": {
            "path": str(gallery.resolve()),
            "temperature": GALLERY_TEMPERATURE,
            "scientific_use_forbidden": True,
            "rollouts": len(rollouts),
        },
    }
    manifest = output_root / key / "MANIFEST.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["manifest"] = str(manifest.resolve())
    return result


def _resolve_specs(args: argparse.Namespace) -> tuple[RunSpec, ...]:
    paths: dict[str, Path | None] = {
        "minus_afe": args.minus_afe_run,
        "minus_progress": args.minus_progress_run,
        "minus_socp": args.minus_socp_run,
    }
    if args.ablations_root is not None:
        for key in paths:
            paths[key] = paths[key] or args.ablations_root / key
    missing = [key for key, path in paths.items() if path is None]
    if missing:
        raise ArtifactStageError(
            "all three arm runs are required; missing " + ", ".join(missing)
        )
    return (
        RunSpec(
            "full", "Full", args.full_run, "full_safe", True,
            "afe", True, "full", args.full_checkpoint,
        ),
        RunSpec(
            "minus_afe", "-AFE", paths["minus_afe"], "full_safe", True,
            "uniform", True, "full",
        ),
        RunSpec(
            "minus_progress", "-Progress", paths["minus_progress"], "full_safe", True,
            "afe", False, "full",
        ),
        RunSpec(
            "minus_socp", "-SOCP (offline only)", paths["minus_socp"],
            "strict_bounds", False, "afe", True, "bounds_only_offline",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument(
        "--full-checkpoint",
        type=Path,
        required=True,
        help=(
            "explicit promoted Full checkpoint; later bundles in --full-run are ignored"
        ),
    )
    parser.add_argument("--ablations-root", type=Path)
    parser.add_argument("--minus-afe-run", type=Path)
    parser.add_argument("--minus-progress-run", type=Path)
    parser.add_argument("--minus-socp-run", type=Path)
    parser.add_argument(
        "--pretrained-checkpoint", type=Path,
        help="optional arbitrary hash-locked checkpoint for the Pretrained T=0.5 panel",
    )
    parser.add_argument(
        "--pretrained-checkpoint-sha256",
        help="optional expected file SHA-256 in addition to mandatory embedded model-state verification",
    )
    parser.add_argument("--outdir", type=Path, default=STAGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=107000)
    parser.add_argument("--gallery-rollouts-per-gamma", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seconds-per-event", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=72)
    parser.add_argument(
        "--video-max-events", type=int, default=0,
        help="0 renders every trace; positive values retain all round-update frames plus deterministic keyframes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.gallery_rollouts_per_gamma <= 0 or args.nfe <= 0:
        raise ValueError("gallery rollout count and nfe must be positive")
    if args.fps <= 0 or args.seconds_per_event <= 0 or args.dpi <= 0:
        raise ValueError("video rendering parameters must be positive")
    if args.video_max_events < 0:
        raise ValueError("video-max-events cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    specs = _resolve_specs(args)
    sources = tuple(load_expansion_source(spec) for spec in specs)
    matched_validation = validate_matched_sources(sources)
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = {
        source.spec.key: generate_run_artifacts(
            source,
            args.outdir,
            device=device,
            gallery_seed=args.seed + index * 100_000,
            gallery_rollouts_per_gamma=args.gallery_rollouts_per_gamma,
            nfe=args.nfe,
            fps=args.fps,
            seconds_per_event=args.seconds_per_event,
            dpi=args.dpi,
            video_max_events=args.video_max_events,
        )
        for index, source in enumerate(sources)
    }
    pretrained = None
    if args.pretrained_checkpoint is not None:
        pretrained = generate_checkpoint_gallery(
            args.pretrained_checkpoint,
            args.outdir,
            device=device,
            gallery_seed=args.seed + 900_000,
            gallery_rollouts_per_gamma=args.gallery_rollouts_per_gamma,
            nfe=args.nfe,
            expected_file_sha256=args.pretrained_checkpoint_sha256,
        )
    elif args.pretrained_checkpoint_sha256 is not None:
        raise ArtifactStageError(
            "--pretrained-checkpoint-sha256 requires --pretrained-checkpoint"
        )
    manifest = {
        "status": "POST_EXPANSION_ARTIFACTS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gallery_temperature": GALLERY_TEMPERATURE,
        "gallery_scientific_use_forbidden": True,
        "expansion_and_audit_temperature": SCIENCE_TEMPERATURE,
        "no_curriculum_learning": True,
        "matched_cross_arm_validation": matched_validation,
        "runs": results,
        "pretrained": pretrained,
        "final_reports_inputs": {
            "full": str(specs[0].root),
            "full_checkpoint": str(sources[0].checkpoint_path),
            "full_viz": results["full"]["gallery"]["path"],
            "no_afe": str(specs[1].root),
            "no_afe_checkpoint": str(sources[1].checkpoint_path),
            "no_afe_viz": results["minus_afe"]["gallery"]["path"],
            "no_progress": str(specs[2].root),
            "no_progress_checkpoint": str(sources[2].checkpoint_path),
            "no_progress_viz": results["minus_progress"]["gallery"]["path"],
            "no_socp": str(specs[3].root),
            "no_socp_checkpoint": str(sources[3].checkpoint_path),
            "no_socp_viz": results["minus_socp"]["gallery"]["path"],
            "pretrained_viz": (
                pretrained["gallery"]["path"] if pretrained is not None else None
            ),
        },
    }
    destination = args.outdir / "MANIFEST.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
