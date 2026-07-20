#!/usr/bin/env python3
"""Stage 08: one-shot sealed validity audit and independent-model inference.

This stage is deliberately evaluation-only.  It accepts final expansion
checkpoints, samples the ordinary conditional flow at temperature one on the
untouched ``sealed_final_test`` context bank, and submits every sampled H=10
plan to the actual full verifier.  It has no uncertainty matrix, acquisition
store, replay view, or optimizer.

The run-manifest schema is ``afe_sealed_run_spec_v1``::

    {"runs": [{
      "label": "Full seed 105000",
      "method": "full",                  # full/minus_afe/minus_progress/minus_socp
      "checkpoint": "/absolute/or/relative/path.pt",
      "expansion_training_seed": 105000,
      "independent_full_replica": true,
      "selected_main": true
    }, ...]}

At least two independently pretrained+expanded Full checkpoints are required.
Exactly one selected-main checkpoint is required for each displayed method.
The selected Full checkpoint is also one of the independent Full replicas.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import grid_hp_expt as HP

from .deps import sha256_file
from .config import clean_method_absence_manifest
from .decision_budget import (
    DecisionBudgetError,
    validate_reference_payload,
    validate_usage,
)
from .policy import model_state_hash
from .schemas import QuerySource
from .scene import GAMMAS, make_ood_scene, verifier_spec_fingerprint
from .stage4_baseline import audit_model, load_audit_bank_artifact
from .stage5_expand import (
    CHECKPOINT_SCHEMA,
    FULL_REPLAY_DESCRIPTION,
    _proximal_unusable_reason,
)
from .store import VerificationStore
from .validity import (
    VALID_AUDIT_MODES,
    aggregate_sealed_full_runs,
    validate_sealed_audit_protocol,
)


STAGE = Path(__file__).resolve().parent / "stage_results/08_sealed_validity"
RUN_SPEC_SCHEMA = "afe_sealed_run_spec_v1"
OUTPUT_SCHEMA = "afe_sealed_validity_v1"
PROTOCOL_SCHEMA = "afe_sealed_audit_protocol_v1"
METHODS = ("full", "minus_afe", "minus_progress", "minus_socp")
DISPLAY_BY_METHOD = {
    "full": "Full",
    "minus_afe": "-AFE",
    "minus_progress": "-Progress",
    "minus_socp": "-SOCP",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a raw checkpoint state exactly as :func:`model_state_hash` does."""

    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint state_dict contains a non-tensor value")
        digest.update(str(name).encode("utf-8"))
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@dataclass(frozen=True)
class RunSpec:
    label: str
    method: str
    checkpoint: Path
    expansion_training_seed: int
    independent_full_replica: bool
    selected_main: bool

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("sealed run labels cannot be empty")
        if self.method not in METHODS:
            raise ValueError(f"unsupported sealed method {self.method!r}")
        if self.independent_full_replica and self.method != "full":
            raise ValueError("only Full can be an independent Full replica")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)


def load_run_specs(path: Path) -> list[RunSpec]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != RUN_SPEC_SCHEMA:
        raise ValueError(f"run manifest must use schema_version={RUN_SPEC_SCHEMA!r}")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("run manifest must contain a runs list")
    allowed = {
        "label", "method", "checkpoint", "expansion_training_seed",
        "independent_full_replica", "selected_main",
    }
    specs = []
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict) or set(raw) != allowed:
            raise ValueError(
                f"run {index} must contain exactly fields {sorted(allowed)}"
            )
        if (
            not isinstance(raw["label"], str)
            or not isinstance(raw["method"], str)
            or not isinstance(raw["checkpoint"], str)
            or not isinstance(raw["expansion_training_seed"], int)
            or isinstance(raw["expansion_training_seed"], bool)
            or not isinstance(raw["independent_full_replica"], bool)
            or not isinstance(raw["selected_main"], bool)
        ):
            raise ValueError(f"run {index} has invalid run-spec field types")
        checkpoint = Path(raw["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        specs.append(RunSpec(
            label=str(raw["label"]),
            method=str(raw["method"]),
            checkpoint=checkpoint.resolve(),
            expansion_training_seed=int(raw["expansion_training_seed"]),
            independent_full_replica=bool(raw["independent_full_replica"]),
            selected_main=bool(raw["selected_main"]),
        ))
    _validate_run_spec_roles(specs)
    return specs


def _validate_run_spec_roles(specs: Sequence[RunSpec]) -> None:
    if len(specs) < 5:
        raise ValueError("sealed evaluation needs >=2 Full replicas and three controls")
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("sealed run labels must be unique")
    paths = [str(spec.checkpoint) for spec in specs]
    if len(set(paths)) != len(paths):
        raise ValueError("a final checkpoint may appear only once in the sealed run manifest")
    replicas = [spec for spec in specs if spec.independent_full_replica]
    if len(replicas) < 2:
        raise ValueError("need at least two independent Full replicas")
    replica_seeds = [spec.expansion_training_seed for spec in replicas]
    if len(set(replica_seeds)) != len(replica_seeds):
        raise ValueError("independent Full replicas have duplicate expansion seeds")
    selected = [spec for spec in specs if spec.selected_main]
    selected_counts = {
        method: sum(spec.method == method for spec in selected) for method in METHODS
    }
    if selected_counts != {method: 1 for method in METHODS}:
        raise ValueError(
            "selected_main must identify exactly one Full/-AFE/-Progress/-SOCP run"
        )
    selected_full = next(spec for spec in selected if spec.method == "full")
    if not selected_full.independent_full_replica:
        raise ValueError("the selected main Full must also be an independent Full replica")


def _checkpoint_method(recipe: Mapping[str, Any]) -> str:
    arm = recipe.get("arm")
    if arm is not None:
        method = str(arm)
        if method not in METHODS:
            raise ValueError(f"checkpoint declares an unsupported final arm {method!r}")
    else:
        acquisition = recipe.get("acquisition", recipe.get("acquisition_mode"))
        progress = recipe.get("progress_ranking")
        if acquisition == "uniform" and progress is True:
            method = "minus_afe"
        elif acquisition == "afe" and progress is False:
            method = "minus_progress"
        elif acquisition == "afe" and progress is True:
            method = "full"
        else:
            raise ValueError("checkpoint recipe does not identify a registered final method")

    expected = {
        "full": ("afe", True, "full", "full_safe", True),
        "minus_afe": ("uniform", True, "full", "full_safe", True),
        "minus_progress": ("afe", False, "full", "full_safe", True),
        "minus_socp": ("afe", True, "bounds_only_offline", "strict_bounds", False),
    }[method]
    observed = (
        recipe.get("acquisition", recipe.get("acquisition_mode")),
        recipe.get("progress_ranking"),
        recipe.get("eligibility_mode"),
        recipe.get("replay_eligibility"),
        recipe.get("runtime_safety_claim"),
    )
    if observed != expected:
        raise ValueError(
            f"checkpoint arm {method!r} has inconsistent acquisition/progress/"
            "eligibility/runtime semantics"
        )
    expected_tilting = expected[0] == "afe"
    if (
        "uncertainty_tilting" in recipe
        and bool(recipe["uncertainty_tilting"]) != expected_tilting
    ):
        raise ValueError(f"checkpoint arm {method!r} has inconsistent tilting metadata")
    return method


def _validate_clean_uniform_flow_replay(
    payload: Mapping[str, Any],
    recipe: Mapping[str, Any],
    *,
    method: str,
    label: str,
) -> dict[str, int]:
    """Reconstruct the final ledger and prove clean FLOW-only replay.

    Recipe claims alone are insufficient for a sealed audit.  This gate checks
    every expansion round's solver telemetry against cumulative fresh FLOW
    queries, then independently checks those totals against the exact final
    verification store.  SafeMPPI rows may update cumulative ``A`` but can
    never account for a CFM training row.
    """

    if recipe.get("legacy_mechanisms") != clean_method_absence_manifest():
        raise ValueError(
            f"{label}: clean-method absence manifest is missing or altered"
        )
    if method == "full" and recipe.get("replay") != FULL_REPLAY_DESCRIPTION:
        raise ValueError(
            f"{label}: Full recipe does not declare uniform positive FLOW-only replay"
        )
    if method != "full" and recipe.get(
        "uniform_replay_no_frontier_weighting"
    ) is not True:
        raise ValueError(
            f"{label}: control recipe lacks uniform replay/no-frontier evidence"
        )

    raw_store = payload.get("verification_store_state")
    if not isinstance(raw_store, Mapping):
        raise ValueError(f"{label}: checkpoint lacks its exact verification store")
    try:
        store = VerificationStore.from_state_dict(dict(raw_store))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label}: invalid final verification store: {exc}") from exc
    flow_records = tuple(
        record for record in store.records if record.source is QuerySource.FLOW
    )
    backup_records = tuple(
        record
        for record in store.records
        if record.source is QuerySource.SAFEMPPI_BACKUP
    )
    eligible_records = tuple(
        record
        for record in flow_records
        if (
            record.safety.strict_bounds
            if method == "minus_socp"
            else record.safe
        )
    )

    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{label}: checkpoint is missing expansion history")
    rounds = [int(row.get("round", -1)) for row in history if isinstance(row, Mapping)]
    if len(rounds) != len(history) or rounds != list(range(rounds[-1] + 1)):
        raise ValueError(f"{label}: expansion history is not contiguous from round zero")

    cumulative_flow = 0
    cumulative_eligible = 0
    cumulative_backup = 0
    for row in history:
        round_index = int(row["round"])
        if round_index == 0:
            if row.get("solver") is not None or row.get("query") is not None:
                raise ValueError(f"{label}: round-zero history must not train or query")
            continue
        query = row.get("query")
        solver = row.get("solver")
        if not isinstance(query, Mapping) or not isinstance(solver, Mapping):
            raise ValueError(
                f"{label}: round {round_index} lacks query/solver evidence"
            )
        flow_calls = int(query.get("new_verifier_calls", -1))
        flow_positive = int(query.get("new_positive_queries", -1))
        flow_negative = int(query.get("new_negative_queries", -1))
        backup_calls = int(query.get("backup_verifier_calls", -1))
        backup_positive = int(query.get("backup_positive_queries", -1))
        backup_negative = int(query.get("backup_negative_queries", -1))
        total_calls = int(query.get("new_total_full_verifier_calls", -1))
        if min(
            flow_calls, flow_positive, flow_negative,
            backup_calls, backup_positive, backup_negative, total_calls,
        ) < 0:
            raise ValueError(f"{label}: round {round_index} has incomplete query counts")
        if (
            flow_positive + flow_negative != flow_calls
            or backup_positive + backup_negative != backup_calls
            or flow_calls + backup_calls != total_calls
        ):
            raise ValueError(f"{label}: round {round_index} query counts are inconsistent")
        if method == "minus_socp":
            eligibility = query.get("actual_vs_training_eligibility")
            if not isinstance(eligibility, Mapping):
                raise ValueError(
                    f"{label}: -SOCP round {round_index} lacks bounds eligibility counts"
                )
            eligible_delta = int(eligibility.get("training_eligible", -1))
            if not 0 <= eligible_delta <= flow_calls:
                raise ValueError(
                    f"{label}: -SOCP round {round_index} eligibility is inconsistent"
                )
        else:
            eligible_delta = flow_positive
        cumulative_flow += flow_calls
        cumulative_eligible += eligible_delta
        cumulative_backup += backup_calls

        if solver.get("sampling") != "uniform_full_positive_pass_seeded_reshuffle":
            raise ValueError(
                f"{label}: round {round_index} solver is not uniform positive replay"
            )
        if int(solver.get("positive_count", -1)) != cumulative_eligible:
            raise ValueError(
                f"{label}: round {round_index} solver positive count is not the "
                "cumulative eligible FLOW ledger"
            )
        expected_input_count = (
            cumulative_flow
            if method in {"minus_afe", "minus_progress"}
            else cumulative_eligible
        )
        if int(solver.get("total_record_count", -1)) != expected_input_count:
            raise ValueError(
                f"{label}: round {round_index} solver input count is not its exact "
                "FLOW-only replay view"
            )
        unusable = _proximal_unusable_reason(solver)
        if unusable is not None:
            raise ValueError(
                f"{label}: round {round_index} has unusable proximal telemetry: {unusable}"
            )
        trace = solver.get("trace")
        if not isinstance(trace, (tuple, list)):
            raise ValueError(f"{label}: round {round_index} solver trace is malformed")
        if cumulative_eligible:
            for step in trace:
                if not isinstance(step, Mapping):
                    raise ValueError(
                        f"{label}: round {round_index} solver step is malformed"
                    )
                sizes = step.get("microbatch_sizes")
                order_digest = step.get("record_order_sha256")
                if (
                    float(step.get("positive_coverage", -1.0)) != 1.0
                    or int(step.get("unique_record_count", -1))
                    != cumulative_eligible
                    or not isinstance(sizes, (tuple, list))
                    or len(sizes) != int(step.get("microbatch_count", -1))
                    or sum(int(size) for size in sizes) != cumulative_eligible
                    or any(int(size) <= 0 for size in sizes)
                    or not isinstance(order_digest, str)
                    or len(order_digest) != 64
                ):
                    raise ValueError(
                        f"{label}: round {round_index} did not retain exact "
                        "compact evidence for a full uniform eligible-FLOW pass"
                    )

    if cumulative_flow != len(flow_records):
        raise ValueError(f"{label}: history FLOW counts differ from the exact ledger")
    if cumulative_backup != len(backup_records):
        raise ValueError(f"{label}: history backup counts differ from the exact ledger")
    if cumulative_eligible != len(eligible_records):
        raise ValueError(
            f"{label}: history training eligibility differs from exact FLOW records"
        )
    if store.query_count != cumulative_flow + cumulative_backup:
        raise ValueError(f"{label}: verifier store source accounting is incomplete")
    return {
        "flow_queries": cumulative_flow,
        "eligible_flow_replay_rows": cumulative_eligible,
        "backup_queries_excluded_from_replay": cumulative_backup,
    }


def _runtime_from_history(history: object) -> dict[str, Any]:
    unavailable = {
        "available": False,
        "control_decisions": 0,
        "episode_count": 0,
        "fallback_steps": 0,
        "fallback_frequency": None,
        "failclosed_episodes": 0,
        "failclosed_frequency": None,
        "source": "expansion_checkpoint_history",
    }
    if not isinstance(history, list):
        return unavailable
    queries = [row.get("query") for row in history if isinstance(row, dict)]
    queries = [dict(query) for query in queries if isinstance(query, Mapping)]
    if not queries:
        return unavailable
    control = sum(int(row.get("control_decisions", 0)) for row in queries)
    episodes = sum(int(row.get("episodes", 0)) for row in queries)
    fallback = sum(int(row.get("fallback_steps", 0)) for row in queries)
    failclosed = sum(int(row.get("fail_closed_episodes", 0)) for row in queries)
    if control <= 0 or episodes <= 0:
        raise ValueError("checkpoint runtime history has non-positive denominators")
    if not 0 <= fallback <= control or not 0 <= failclosed <= episodes:
        raise ValueError("checkpoint runtime history has invalid fallback/failclosed counts")
    return {
        "available": True,
        "control_decisions": control,
        "episode_count": episodes,
        "fallback_steps": fallback,
        "fallback_frequency": fallback / control,
        "failclosed_episodes": failclosed,
        "failclosed_frequency": failclosed / episodes,
        "source": "expansion_checkpoint_history",
    }


def _preflight_checkpoint(spec: RunSpec, verifier_spec: str) -> dict[str, Any]:
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("afe_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"{spec.label}: final checkpoint is not schema {CHECKPOINT_SCHEMA!r}"
        )
    round_index = int(payload.get("round", -1))
    if round_index <= 0:
        raise ValueError(f"{spec.label}: checkpoint is not an expanded final round")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{spec.label}: checkpoint has no model state")
    state_hash = _state_dict_hash(state_dict)
    if payload.get("current_model_hash") != state_hash:
        raise ValueError(f"{spec.label}: checkpoint current-model hash mismatch")
    config = payload.get("config")
    if not isinstance(config, Mapping) or int(config.get("repr_dim", -1)) != 32:
        raise ValueError(f"{spec.label}: final model does not use repr_dim=32")
    if bool(config.get("raw_start_goal", False)):
        raise ValueError(f"{spec.label}: rejected raw start/goal architecture")
    recipe = payload.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError(f"{spec.label}: checkpoint is missing its expansion recipe")
    declared_method = _checkpoint_method(recipe)
    if declared_method != spec.method:
        raise ValueError(
            f"{spec.label}: manifest method {spec.method!r} != checkpoint {declared_method!r}"
        )
    source_pretrain_hash = str(recipe.get("source_model_hash", ""))
    source_pretrain_checkpoint_sha = str(recipe.get("source_checkpoint_sha256", ""))
    if not _is_sha256(source_pretrain_hash) or not _is_sha256(
        source_pretrain_checkpoint_sha
    ):
        raise ValueError(f"{spec.label}: source-pretrain hash provenance is incomplete")
    if str(payload.get("frozen_feature_hash", "")) != source_pretrain_hash:
        raise ValueError(f"{spec.label}: frozen phi0 differs from source pretrain")
    seed_evidence: dict[str, int] = {}
    run_config = recipe.get("run_config")
    if isinstance(run_config, Mapping) and "seed" in run_config:
        seed_evidence["run_config.seed"] = int(run_config["seed"])
    matched_protocol = recipe.get("matched_protocol")
    if isinstance(matched_protocol, Mapping) and "seed" in matched_protocol:
        seed_evidence["matched_protocol.seed"] = int(matched_protocol["seed"])
    if "expansion_training_seed" in recipe:
        seed_evidence["expansion_training_seed"] = int(
            recipe["expansion_training_seed"]
        )
    if not seed_evidence:
        raise ValueError(f"{spec.label}: checkpoint recipe does not bind its expansion seed")
    if set(seed_evidence.values()) != {spec.expansion_training_seed}:
        raise ValueError(
            f"{spec.label}: expansion training seed evidence disagrees: {seed_evidence}"
        )
    expansion_verifier_spec = recipe.get("verifier_spec_fingerprint")
    if expansion_verifier_spec is not None and str(expansion_verifier_spec) != verifier_spec:
        raise ValueError(f"{spec.label}: expansion/audit verifier specifications differ")
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{spec.label}: checkpoint is missing expansion history")
    final_history = history[-1]
    if (
        not isinstance(final_history, Mapping)
        or int(final_history.get("round", -1)) != round_index
        or final_history.get("model_hash") != state_hash
    ):
        raise ValueError(f"{spec.label}: final history is inconsistent with checkpoint")
    decision_budget_evidence: dict[str, Any] | None = None
    if declared_method != "full":
        reference_value = recipe.get("full_reference_decision_budget")
        if not isinstance(reference_value, Mapping):
            raise ValueError(
                f"{spec.label}: control lacks selected-Full decision-budget evidence"
            )
        try:
            reference = validate_reference_payload(reference_value)
        except DecisionBudgetError as exc:
            raise ValueError(
                f"{spec.label}: invalid selected-Full decision-budget evidence: {exc}"
            ) from exc
        if not isinstance(matched_protocol, Mapping) or dict(
            reference["matched_protocol"]
        ) != dict(matched_protocol):
            raise ValueError(
                f"{spec.label}: selected-Full decision caps use another protocol"
            )
        usages = []
        for history_row in history:
            if not isinstance(history_row, Mapping):
                raise ValueError(f"{spec.label}: malformed expansion history")
            history_round = int(history_row.get("round", -1))
            if history_round == 0:
                continue
            query = history_row.get("query")
            if not isinstance(query, Mapping):
                raise ValueError(
                    f"{spec.label}: round {history_round} lacks query evidence"
                )
            try:
                usage = validate_usage(
                    query.get("full_reference_control_decision_budget"), reference,
                )
            except (DecisionBudgetError, TypeError) as exc:
                raise ValueError(
                    f"{spec.label}: round {history_round} violates its Full cap: {exc}"
                ) from exc
            if int(usage["round"]) != history_round:
                raise ValueError(
                    f"{spec.label}: decision usage is attached to the wrong round"
                )
            if int(query.get("control_decisions", -1)) != int(
                usage["realized_control_decisions"]
            ):
                raise ValueError(
                    f"{spec.label}: query decisions differ from cellwise cap usage"
                )
            usages.append(usage)
        if [int(row["round"]) for row in usages] != list(range(1, round_index + 1)):
            raise ValueError(
                f"{spec.label}: decision-budget history is not contiguous"
            )
        decision_budget_evidence = {
            "reference": reference,
            "reference_fingerprint": reference["fingerprint"],
            "round_usage_fingerprints": [row["fingerprint"] for row in usages],
            "all_control_cells_within_corresponding_full_cap": True,
        }
    replay_evidence = _validate_clean_uniform_flow_replay(
        payload,
        recipe,
        method=declared_method,
        label=spec.label,
    )
    runtime_safety_claim = spec.method != "minus_socp"
    declared_claim = recipe.get("runtime_safety_claim", final_history.get("runtime_safety_claim"))
    if declared_claim is not None and bool(declared_claim) != runtime_safety_claim:
        raise ValueError(f"{spec.label}: runtime safety-claim metadata is inconsistent")
    return {
        "label": spec.label,
        "method": spec.method,
        "checkpoint_path": str(spec.checkpoint),
        "checkpoint_file_sha256": sha256_file(spec.checkpoint),
        "model_state_sha256": state_hash,
        "expansion_training_seed": spec.expansion_training_seed,
        "expansion_training_seed_binding": "checkpoint_recipe_and_run_manifest",
        "expansion_training_seed_checkpoint_fields": sorted(seed_evidence),
        "source_pretrain_hash": source_pretrain_hash,
        "source_pretrain_checkpoint_sha256": source_pretrain_checkpoint_sha,
        "checkpoint_round": round_index,
        "runtime_safety_claim": runtime_safety_claim,
        "clean_uniform_flow_replay_evidence": replay_evidence,
        "independent_full_replica": spec.independent_full_replica,
        "selected_main": spec.selected_main,
        "expansion_verifier_spec_fingerprint": (
            str(expansion_verifier_spec) if expansion_verifier_spec is not None else None
        ),
        "runtime": _runtime_from_history(history),
        "full_reference_decision_budget_evidence": decision_budget_evidence,
    }


def _validate_preflight_identities(rows: Sequence[Mapping[str, Any]]) -> None:
    for field in ("checkpoint_file_sha256", "model_state_sha256"):
        values = [str(row[field]) for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"sealed inputs contain duplicate {field}")
    replicas = [row for row in rows if bool(row["independent_full_replica"])]
    pretrain_hashes = [str(row["source_pretrain_hash"]) for row in replicas]
    if len(set(pretrain_hashes)) != len(pretrain_hashes):
        raise ValueError("Full replicas are not independently pretrained")
    seeds = [int(row["expansion_training_seed"]) for row in replicas]
    if len(set(seeds)) != len(seeds):
        raise ValueError("independent Full replicas have duplicate expansion seeds")
    selected_full_rows = [
        row for row in rows
        if row["method"] == "full" and bool(row["selected_main"])
    ]
    if len(selected_full_rows) != 1:
        raise ValueError("sealed preflight needs exactly one selected Full")
    selected_full = selected_full_rows[0]
    control_references = []
    for row in rows:
        if row["method"] == "full":
            continue
        evidence = row.get("full_reference_decision_budget_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{row['label']}: missing Full cap evidence")
        reference = evidence.get("reference")
        if not isinstance(reference, Mapping):
            raise ValueError(f"{row['label']}: malformed Full cap evidence")
        reference = validate_reference_payload(reference)
        control_references.append(reference)
        if (
            reference["final_checkpoint_sha256"]
            != selected_full["checkpoint_file_sha256"]
            or reference["final_model_hash"] != selected_full["model_state_sha256"]
            or int(reference["final_round"]) != int(selected_full["checkpoint_round"])
            or reference["source_checkpoint_sha256"]
            != selected_full["source_pretrain_checkpoint_sha256"]
            or reference["source_model_hash"] != selected_full["source_pretrain_hash"]
            or Path(reference["final_checkpoint_path"]).resolve()
            != Path(selected_full["checkpoint_path"]).resolve()
        ):
            raise ValueError(
                f"{row['label']}: control caps are not bound to selected Full"
            )
    if len({row["fingerprint"] for row in control_references}) != 1:
        raise ValueError("sealed controls do not share one selected-Full cap reference")


def _run_aggregate(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in audit["per_gamma"]]
    sample_count = sum(int(row["sample_count"]) for row in rows)
    safe_count = sum(int(row["safe_count"]) for row in rows)
    progress_count = sum(int(row["safe_progress_count"]) for row in rows)
    modes = sorted({
        str(mode)
        for row in rows
        for mode, count in row["mode_counts"].items()
        if int(count) > 0
    })
    coverage = float(np.mean([
        int(row["safe_mode_coverage"]) / len(VALID_AUDIT_MODES) for row in rows
    ]))
    return {
        "sample_count": sample_count,
        "safe_count": safe_count,
        "safe_progress_count": progress_count,
        "V": safe_count / sample_count,
        "Vprog": progress_count / sample_count,
        "valid_mode_coverage_count": len(modes),
        "valid_mode_coverage_fraction": coverage,
        "valid_mode_coverage_definition": (
            "mean across gamma of observed safe local modes / 3 preregistered modes"
        ),
        "valid_modes": modes,
    }


def _build_protocol(
    *,
    sealed_bank: Path,
    bank: Any,
    bank_artifact: Mapping[str, Any],
    plans_per_context: int,
    progress_threshold: float,
    nfe: int,
    audit_seed: int,
    verifier_spec: str,
    independent_model_confidence: float,
) -> dict[str, Any]:
    if plans_per_context <= 0 or nfe <= 0:
        raise ValueError("sealed plan count and NFE must be positive")
    if not math.isfinite(progress_threshold):
        raise ValueError("sealed progress threshold must be finite")
    if not 0.0 < independent_model_confidence < 1.0:
        raise ValueError("independent-model confidence must lie in (0,1)")
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "context_bank_path": str(sealed_bank.resolve()),
        "context_bank_file_sha256": sha256_file(sealed_bank),
        "context_bank_fingerprint": str(bank.fingerprint),
        "context_bank_role": str(bank.role),
        "context_bank_artifact_fingerprint": str(
            bank_artifact["artifact_fingerprint"]
        ),
        "context_bank_source_provenance_fingerprint": str(
            bank_artifact["source_provenance_fingerprint"]
        ),
        "context_bank_source_provenance": dict(bank_artifact["source_provenance"]),
        "context_count": len(bank),
        "plans_per_context": int(plans_per_context),
        "progress_threshold": float(progress_threshold),
        "nfe": int(nfe),
        "temperature": 1.0,
        "uncertainty_tilting": False,
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "gammas": [float(gamma) for gamma in GAMMAS],
        "verifier_spec_fingerprint": verifier_spec,
        "audit_seed": int(audit_seed),
        "conditional_plan_sampling_confidence": 0.95,
        "independent_training_seed_confidence": float(independent_model_confidence),
        "full_verifier_label": "strict task-space bounds AND full SOCP certificate",
        "audit_samples_added_to_training_or_acquisition": False,
        "audit_invocations_per_model": 1,
    }
    if protocol["context_bank_role"] != "sealed_final_test":
        raise ValueError("Stage 08 accepts only a sealed_final_test context bank")
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    return protocol


def run_sealed_validity(
    *,
    run_manifest: Path,
    sealed_bank: Path,
    outdir: Path,
    device: torch.device,
    plans_per_context: int,
    progress_threshold: float,
    nfe: int,
    audit_seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Run every registered checkpoint exactly once and write JSON/PT twins."""

    output_json = outdir / "final_validity_report.json"
    output_pt = outdir / "final_validity_report.pt"
    if output_json.exists() or output_pt.exists():
        raise FileExistsError(
            "sealed final validity outputs already exist; the one-shot stage will not overwrite"
        )
    specs = load_run_specs(run_manifest)
    bank, bank_artifact = load_audit_bank_artifact(
        sealed_bank, require_locked_provenance=True,
    )
    if bank.role != "sealed_final_test":
        raise ValueError("Stage 08 cannot audit a round-monitoring bank")
    env = make_ood_scene(radius=1.2)
    verifier_spec = verifier_spec_fingerprint(env, env.goal)
    bank_provenance = bank_artifact["source_provenance"]
    if (
        bank_provenance.get("purpose") != "sealed_final_test"
        or bank_provenance.get("scene_verifier_spec_fingerprint") != verifier_spec
    ):
        raise ValueError(
            "sealed bank provenance does not match its role/current full verifier"
        )
    protocol = _build_protocol(
        sealed_bank=sealed_bank,
        bank=bank,
        bank_artifact=bank_artifact,
        plans_per_context=plans_per_context,
        progress_threshold=progress_threshold,
        nfe=nfe,
        audit_seed=audit_seed,
        verifier_spec=verifier_spec,
        independent_model_confidence=confidence,
    )
    preflight = [_preflight_checkpoint(spec, verifier_spec) for spec in specs]
    _validate_preflight_identities(preflight)

    per_run: list[dict[str, Any]] = []
    for spec, identity in zip(specs, preflight):
        model, payload = HP.load_hp(spec.checkpoint, device=device)
        loaded_hash = model_state_hash(model)
        if loaded_hash != identity["model_state_sha256"]:
            raise RuntimeError(f"{spec.label}: loaded model changed after checkpoint preflight")
        if payload.get("current_model_hash") != loaded_hash:
            raise RuntimeError(f"{spec.label}: loaded checkpoint hash mismatch")
        audit = audit_model(
            model,
            env,
            bank,
            plans_per_context=plans_per_context,
            seed=audit_seed,
            nfe=nfe,
            progress_threshold=progress_threshold,
        ).to_dict()
        audit = validate_sealed_audit_protocol(audit, protocol)
        run_id = f"{spec.method}:{_fingerprint({
            'checkpoint_file_sha256': identity['checkpoint_file_sha256'],
            'model_state_sha256': identity['model_state_sha256'],
            'expansion_training_seed': identity['expansion_training_seed'],
            'protocol_fingerprint': protocol['protocol_fingerprint'],
        })[:20]}"
        per_run.append(dict(identity) | {
            "run_id": run_id,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "audit_verifier_spec_fingerprint": verifier_spec,
            "audit": audit,
            "aggregate": _run_aggregate(audit),
        })
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len({row["run_id"] for row in per_run}) != len(per_run):
        raise RuntimeError("sealed run IDs are not unique")
    replicas = [row for row in per_run if row["independent_full_replica"]]
    independent = aggregate_sealed_full_runs(replicas, confidence=confidence)
    selected_main_run_ids = {
        DISPLAY_BY_METHOD[method]: next(
            row["run_id"]
            for row in per_run
            if row["method"] == method and row["selected_main"]
        )
        for method in METHODS
    }
    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "SEALED_FINAL_VALIDITY_COMPLETE",
        "generated_at_utc": generated,
        "bank_fingerprint": protocol["context_bank_fingerprint"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "protocol": protocol,
        "per_run": per_run,
        "independent_full_aggregate": independent,
        "selected_main_run_ids": selected_main_run_ids,
        "provenance": {
            "run_manifest_path": str(run_manifest.resolve()),
            "run_manifest_file_sha256": sha256_file(run_manifest),
            "sealed_bank_path": str(sealed_bank.resolve()),
            "sealed_bank_file_sha256": protocol["context_bank_file_sha256"],
            "stage8_source_sha256": sha256_file(Path(__file__)),
            "validity_source_sha256": sha256_file(Path(__file__).with_name("validity.py")),
            "one_shot_evaluation": True,
            "audit_invocations_per_model": 1,
            "audit_results_used_for_training_or_checkpoint_selection": False,
            "independent_replication_unit": "independently_pretrained_and_expanded_model",
            "plan_samples_pooled_across_models": False,
            "temperature_point_five_role": "none; visualization only outside this stage",
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    _atomic_torch(output_pt, report)
    _atomic_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--sealed-bank", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=STAGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plans-per-context", type=int, default=32)
    parser.add_argument("--progress-threshold", type=float, default=0.10)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--audit-seed", type=int, default=108000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if not args.run_manifest.is_file() or not args.sealed_bank.is_file():
        raise FileNotFoundError("run manifest and sealed audit bank must exist")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    report = run_sealed_validity(
        run_manifest=args.run_manifest,
        sealed_bank=args.sealed_bank,
        outdir=args.outdir,
        device=device,
        plans_per_context=args.plans_per_context,
        progress_threshold=args.progress_threshold,
        nfe=args.nfe,
        audit_seed=args.audit_seed,
        confidence=args.confidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
