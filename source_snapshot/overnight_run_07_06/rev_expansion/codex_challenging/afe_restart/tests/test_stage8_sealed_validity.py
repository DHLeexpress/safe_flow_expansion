from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from afe_restart.config import clean_method_absence_manifest
from afe_restart.decision_budget import (
    REFERENCE_SCHEMA,
    build_usage,
    fingerprint,
)
from afe_restart.policy import model_state_hash
from afe_restart.scene import GAMMAS
from afe_restart.schemas import (
    ProgressResult,
    QueryContext,
    QuerySource,
    SafetyResult,
    VerificationRecord,
)
from afe_restart.stage5_expand import CHECKPOINT_SCHEMA, FULL_REPLAY_DESCRIPTION
from afe_restart.stage8_sealed_validity import (
    OUTPUT_SCHEMA,
    RUN_SPEC_SCHEMA,
    RunSpec,
    _preflight_checkpoint,
    run_sealed_validity,
)
from afe_restart.store import VerificationStore
from afe_restart.validity import (
    aggregate_sealed_full_runs,
    validate_sealed_audit_protocol,
)


class TinyModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))


class FakeBank:
    role = "sealed_final_test"
    fingerprint = "c" * 64

    def __len__(self) -> int:
        return 2


def _protocol(*, plans: int = 2) -> dict:
    return {
        "context_bank_fingerprint": "c" * 64,
        "context_bank_role": "sealed_final_test",
        "context_count": 2,
        "plans_per_context": plans,
        "progress_threshold": 0.1,
        "nfe": 8,
        "temperature": 1.0,
        "uncertainty_tilting": False,
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "gammas": list(GAMMAS),
        "audit_seed": 91,
    }


def _audit(*, safe_count: int = 2, plans: int = 2) -> dict:
    n = 2 * plans
    progress = max(0, safe_count - 1)
    modes = {}
    if safe_count:
        modes["left-of-goal-ray"] = safe_count
    rows = [{
        "gamma": gamma,
        "sample_count": n,
        "safe_count": safe_count,
        "safe_progress_count": progress,
        "validity_mass": safe_count / n,
        "progress_validity": progress / n,
        "mode_counts": modes,
        "safe_mode_coverage": int(bool(safe_count)),
    } for gamma in GAMMAS]
    return {
        "context_count": 2,
        "plans_per_context": plans,
        "total_verifier_calls": 2 * plans * len(GAMMAS),
        "seed": 91,
        "temperature": 1.0,
        "progress_threshold": 0.1,
        "context_bank_fingerprint": "c" * 64,
        "context_bank_role": "sealed_final_test",
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "uncertainty_tilting": False,
        "confidence_interval_scope": (
            "conditional_plan_sampling_wilson_on_fixed_context_bank_single_model"
        ),
        "independent_training_seed_count": 1,
        "independent_training_seed_ci": False,
        "per_gamma": rows,
    }


def test_strict_sealed_protocol_rejects_duplicate_or_missing_gamma_and_count_drift() -> None:
    protocol = _protocol()
    assert validate_sealed_audit_protocol(_audit(), protocol)["temperature"] == 1.0

    duplicate = _audit()
    duplicate["per_gamma"][-1]["gamma"] = duplicate["per_gamma"][0]["gamma"]
    with pytest.raises(ValueError, match="duplicate gamma"):
        validate_sealed_audit_protocol(duplicate, protocol)

    drift = _audit()
    drift["per_gamma"][0]["mode_counts"] = {}
    with pytest.raises(ValueError, match="mode"):
        validate_sealed_audit_protocol(drift, protocol)


def _aggregate_row(seed: int, model: str, pretrain: str, safe: int) -> dict:
    audit = _audit(safe_count=safe)
    audit["context_bank_fingerprint"] = "c" * 64
    return {
        "run_id": f"full:{seed}",
        "method": "full",
        "expansion_training_seed": seed,
        "model_state_sha256": model * 64,
        "source_pretrain_hash": pretrain * 64,
        "protocol_fingerprint": "d" * 64,
        "audit": audit,
        "runtime": {
            "available": True,
            "fallback_frequency": seed / 1000,
            "failclosed_frequency": seed / 2000,
        },
    }


def test_independent_full_aggregate_has_student_t_coverage_and_runtime() -> None:
    result = aggregate_sealed_full_runs([
        _aggregate_row(11, "1", "a", 1),
        _aggregate_row(22, "2", "b", 2),
        _aggregate_row(33, "3", "e", 3),
    ])
    assert result["schema_version"] == "afe_sealed_independent_full_aggregate_v1"
    assert result["per_gamma"][0]["valid_mode_coverage"]["method"].startswith("student_t")
    overall = result["aggregate_over_gammas"]
    assert overall["valid_mode_coverage"]["independent_training_seed_count"] == 3
    assert overall["fallback_frequency"]["mean"] == pytest.approx(0.022)
    assert overall["failclosed_frequency"]["mean"] == pytest.approx(0.011)

    duplicate = [
        _aggregate_row(11, "1", "a", 1),
        _aggregate_row(11, "2", "b", 2),
    ]
    with pytest.raises(ValueError, match="duplicate expansion"):
        aggregate_sealed_full_runs(duplicate)


def _write_checkpoint(
    path: Path,
    *,
    value: float,
    method: str,
    seed: int,
    pretrain_hash: str,
) -> None:
    model = TinyModel(value)
    state_hash = model_state_hash(model)
    semantics = {
        "full": ("afe", True, "full", "full_safe", True),
        "minus_afe": ("uniform", True, "full", "full_safe", True),
        "minus_progress": ("afe", False, "full", "full_safe", True),
        "minus_socp": ("afe", True, "bounds_only_offline", "strict_bounds", False),
    }[method]
    recipe = {
        "arm": method,
        "acquisition": semantics[0],
        "acquisition_mode": semantics[0],
        "progress_ranking": semantics[1],
        "eligibility_mode": semantics[2],
        "replay_eligibility": semantics[3],
        "runtime_safety_claim": semantics[4],
        "uncertainty_tilting": semantics[0] == "afe",
        "legacy_mechanisms": clean_method_absence_manifest(),
        "uniform_replay_no_frontier_weighting": True,
    }
    matched_protocol = {
        "seed": seed,
        "rounds": 1,
        "episodes_per_gamma": 1,
        "episode_max_steps": 1,
    }
    recipe.update({
        "source_model_hash": pretrain_hash,
        "source_checkpoint_sha256": "f" * 64,
        "run_config": {"seed": seed},
        "matched_protocol": matched_protocol,
        "expansion_training_seed": seed,
        "verifier_spec_fingerprint": "v" * 64,
        "replay": FULL_REPLAY_DESCRIPTION if method == "full" else None,
    })
    query = {
        "control_decisions": len(GAMMAS),
        "episodes": 7,
        "fallback_steps": 2,
        "fail_closed_episodes": 1,
        "new_total_full_verifier_calls": 0,
        "new_verifier_calls": 0,
        "new_positive_queries": 0,
        "new_negative_queries": 0,
        "backup_verifier_calls": 0,
        "backup_positive_queries": 0,
        "backup_negative_queries": 0,
    }
    if method == "minus_socp":
        query["actual_vs_training_eligibility"] = {"training_eligible": 0}
    if method != "full":
        reference_core = {
            "schema_version": REFERENCE_SCHEMA,
            "reference_dir": str(path.parent),
            "reference_recipe_sha256": "c" * 64,
            "source_checkpoint_sha256": "f" * 64,
            "source_model_hash": pretrain_hash,
            "final_checkpoint_path": str(path),
            "final_checkpoint_sha256": "d" * 64,
            "final_model_hash": state_hash,
            "final_round": 1,
            "matched_protocol": matched_protocol,
            "caps": [{
                "round": 1,
                "gamma": float(gamma),
                "episode_index": 0,
                "max_control_decisions": 1,
            } for gamma in GAMMAS],
            "total_control_decisions": len(GAMMAS),
        }
        reference = reference_core | {"fingerprint": fingerprint(reference_core)}
        recipe["full_reference_decision_budget"] = reference
        episodes = [{
            "gamma": float(gamma),
            "seed": seed + 1_000_000 + gamma_index * 10_000,
            "traces": [object()],
        } for gamma_index, gamma in enumerate(GAMMAS)]
        query["full_reference_control_decision_budget"] = build_usage(
            episodes,
            round_index=1,
            reference=reference,
            expected_seed_base=seed,
        )
    solver = {
        "positive_count": 0,
        "total_record_count": 0,
        "optimizer_steps": 0,
        "stopping_reason": "no_positive_records",
        "converged": False,
        "sampling": "uniform_full_positive_pass_seeded_reshuffle",
        "trace": [],
    }
    store = VerificationStore()
    payload = {
        "state_dict": model.state_dict(),
        "config": {"repr_dim": 32, "raw_start_goal": False},
        "afe_schema": CHECKPOINT_SCHEMA,
        "round": 1,
        "recipe": recipe,
        "history": [
            {
                "round": 0,
                "model_hash": "unused",
                "query": None,
                "solver": None,
            },
            {
                "round": 1,
                "model_hash": state_hash,
                "query": query,
                "solver": solver,
                "runtime_safety_claim": method != "minus_socp",
            },
        ],
        "frozen_feature_hash": pretrain_hash,
        "current_model_hash": state_hash,
        "verification_store_state": store.state_dict(),
    }
    torch.save(payload, path)


def _run_spec(path: Path, *, seed: int = 101) -> RunSpec:
    return RunSpec(
        label="Full test",
        method="full",
        checkpoint=path,
        expansion_training_seed=seed,
        independent_full_replica=True,
        selected_main=True,
    )


def test_stage8_preflight_rejects_ad_hoc_manifest_and_max_step_solver(
    tmp_path: Path,
) -> None:
    path = tmp_path / "full.pt"
    _write_checkpoint(
        path,
        value=1.0,
        method="full",
        seed=101,
        pretrain_hash="a" * 64,
    )
    clean = torch.load(path, map_location="cpu", weights_only=False)
    evidence = _preflight_checkpoint(_run_spec(path), "v" * 64)
    assert evidence["clean_uniform_flow_replay_evidence"] == {
        "flow_queries": 0,
        "eligible_flow_replay_rows": 0,
        "backup_queries_excluded_from_replay": 0,
    }

    ad_hoc = copy.deepcopy(clean)
    ad_hoc["recipe"]["legacy_mechanisms"]["demo_frac"] = "present"
    torch.save(ad_hoc, path)
    with pytest.raises(ValueError, match="absence manifest"):
        _preflight_checkpoint(_run_spec(path), "v" * 64)

    capped = copy.deepcopy(clean)
    capped_solver = capped["history"][-1]["solver"]
    capped_solver.update({
        "positive_count": 1,
        "total_record_count": 1,
        "optimizer_steps": 12,
        "stopping_reason": "max_steps",
        "trace": [{
            "positive_coverage": 1.0,
            "projected_to_update_bound": False,
        }],
    })
    capped["history"][-1]["query"].update({
        "new_total_full_verifier_calls": 1,
        "new_verifier_calls": 1,
        "new_positive_queries": 1,
    })
    torch.save(capped, path)
    with pytest.raises(ValueError, match="unusable proximal telemetry"):
        _preflight_checkpoint(_run_spec(path), "v" * 64)


def test_stage8_preflight_proves_backup_is_not_a_training_row(tmp_path: Path) -> None:
    path = tmp_path / "full_backup.pt"
    _write_checkpoint(
        path,
        value=1.0,
        method="full",
        seed=101,
        pretrain_hash="a" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    store = VerificationStore()
    z = np.zeros(32, dtype=np.float64)
    z[0] = 1.0
    context = QueryContext(
        grid=np.zeros((2, 2), dtype=np.float32),
        low5=np.zeros(5, dtype=np.float32),
        hist=np.zeros((2, 2), dtype=np.float32),
        verifier_state=np.zeros(4, dtype=np.float64),
        verifier_spec_fingerprint="d" * 64,
    )
    backup = VerificationRecord(
        context=context,
        gamma=0.5,
        plan=np.zeros((10, 2), dtype=np.float32),
        source=QuerySource.SAFEMPPI_BACKUP,
        feature_z=z,
        acquisition_sigma=store.uncertainty.sigma(z),
        safety=SafetyResult(True, True, 0.2, 0.1, 0.1),
        progress=ProgressResult(5.0, 4.5),
    )
    store.append(backup)
    payload["verification_store_state"] = store.state_dict()
    query = payload["history"][-1]["query"]
    query.update({
        "new_total_full_verifier_calls": 1,
        "backup_verifier_calls": 1,
        "backup_positive_queries": 1,
    })
    # This is the forbidden implicit-distillation claim: the only verified row
    # is a backup, yet the solver says it trained one positive target.
    payload["history"][-1]["solver"] = {
        "positive_count": 1,
        "total_record_count": 1,
        "optimizer_steps": 1,
        "stopping_reason": "gradient_tolerance",
        "converged": True,
        "sampling": "uniform_full_positive_pass_seeded_reshuffle",
        "trace": [{
            "positive_coverage": 1.0,
            "projected_to_update_bound": False,
        }],
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="cumulative eligible FLOW ledger"):
        _preflight_checkpoint(_run_spec(path), "v" * 64)


def test_stage8_writes_bound_json_pt_twins_and_refuses_overwrite(
    tmp_path: Path, monkeypatch,
) -> None:
    methods = ["full", "full", "minus_afe", "minus_progress", "minus_socp"]
    seeds = [101, 202, 101, 101, 101]
    paths = []
    for index, (method, seed) in enumerate(zip(methods, seeds), start=1):
        path = tmp_path / f"run_{index}.pt"
        pretrain = ("a" if index != 2 else "b") * 64
        _write_checkpoint(
            path,
            value=float(index),
            method=method,
            seed=seed,
            pretrain_hash=pretrain,
        )
        paths.append(path)
    selected_full_payload = torch.load(paths[0], weights_only=False)
    selected_full_file_hash = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    for path in paths[2:]:
        payload = torch.load(path, weights_only=False)
        reference_core = copy.deepcopy(
            payload["recipe"]["full_reference_decision_budget"]
        )
        reference_core.pop("fingerprint")
        reference_core.update({
            "reference_dir": str(paths[0].parent),
            "final_checkpoint_path": str(paths[0]),
            "final_checkpoint_sha256": selected_full_file_hash,
            "final_model_hash": selected_full_payload["current_model_hash"],
            "source_checkpoint_sha256": selected_full_payload["recipe"][
                "source_checkpoint_sha256"
            ],
            "source_model_hash": selected_full_payload["recipe"]["source_model_hash"],
        })
        reference = reference_core | {"fingerprint": fingerprint(reference_core)}
        payload["recipe"]["full_reference_decision_budget"] = reference
        episodes = [{
            "gamma": float(gamma),
            "seed": 101 + 1_000_000 + gamma_index * 10_000,
            "traces": [object()],
        } for gamma_index, gamma in enumerate(GAMMAS)]
        payload["history"][-1]["query"][
            "full_reference_control_decision_budget"
        ] = build_usage(
            episodes,
            round_index=1,
            reference=reference,
            expected_seed_base=101,
        )
        torch.save(payload, path)
    run_manifest = tmp_path / "runs.json"
    run_manifest.write_text(json.dumps({
        "schema_version": RUN_SPEC_SCHEMA,
        "runs": [{
            "label": f"run {index}",
            "method": method,
            "checkpoint": path.name,
            "expansion_training_seed": seed,
            "independent_full_replica": method == "full",
            "selected_main": index != 2,
        } for index, (path, method, seed) in enumerate(
            zip(paths, methods, seeds), start=1
        )],
    }))
    bank_path = tmp_path / "sealed.pt"
    bank_path.write_bytes(b"sealed-bank-placeholder")

    fake_bank_artifact = {
        "artifact_fingerprint": "8" * 64,
        "source_provenance_fingerprint": "9" * 64,
        "source_provenance": {
            "purpose": "sealed_final_test",
            "scene_verifier_spec_fingerprint": "v" * 64,
            "expert_planner": {
                "smooth_weight": 8.0,
                "noise_var_mult": 3.0,
                "retreat_weight": 1.0,
            }
        },
    }
    monkeypatch.setattr(
        "afe_restart.stage8_sealed_validity.load_audit_bank_artifact",
        lambda _path, require_locked_provenance: (FakeBank(), fake_bank_artifact),
    )
    monkeypatch.setattr(
        "afe_restart.stage8_sealed_validity.make_ood_scene",
        lambda radius: SimpleNamespace(goal=torch.tensor([4.5, 4.5])),
    )
    monkeypatch.setattr(
        "afe_restart.stage8_sealed_validity.verifier_spec_fingerprint",
        lambda _env, _goal: "v" * 64,
    )

    load_calls = []

    def fake_load(path, device):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = float(payload["state_dict"]["weight"].item())
        model = TinyModel(value).to(device)
        model.load_state_dict(payload["state_dict"])
        load_calls.append(str(path))
        return model.eval(), payload

    monkeypatch.setattr("afe_restart.stage8_sealed_validity.HP.load_hp", fake_load)

    class FakeResult:
        def to_dict(self):
            return copy.deepcopy(_audit())

    audit_calls = []

    def fake_audit(model, _env, _bank, **kwargs):
        assert kwargs == {
            "plans_per_context": 2,
            "seed": 91,
            "nfe": 8,
            "progress_threshold": 0.1,
        }
        audit_calls.append(model_state_hash(model))
        return FakeResult()

    monkeypatch.setattr("afe_restart.stage8_sealed_validity.audit_model", fake_audit)
    outdir = tmp_path / "out"
    report = run_sealed_validity(
        run_manifest=run_manifest,
        sealed_bank=bank_path,
        outdir=outdir,
        device=torch.device("cpu"),
        plans_per_context=2,
        progress_threshold=0.1,
        nfe=8,
        audit_seed=91,
    )
    assert report["schema_version"] == OUTPUT_SCHEMA
    assert report["bank_fingerprint"] == report["protocol"]["context_bank_fingerprint"]
    assert report["protocol_fingerprint"] == report["protocol"]["protocol_fingerprint"]
    assert set(report["selected_main_run_ids"]) == {"Full", "-AFE", "-Progress", "-SOCP"}
    assert len(load_calls) == len(audit_calls) == 5
    assert len(set(audit_calls)) == 5
    assert report["independent_full_aggregate"]["independent_training_seed_count"] == 2
    json_report = json.loads((outdir / "final_validity_report.json").read_text())
    pt_report = torch.load(
        outdir / "final_validity_report.pt", map_location="cpu", weights_only=False
    )
    assert json_report == pt_report == report

    with pytest.raises(FileExistsError, match="one-shot"):
        run_sealed_validity(
            run_manifest=run_manifest,
            sealed_bank=bank_path,
            outdir=outdir,
            device=torch.device("cpu"),
            plans_per_context=2,
            progress_threshold=0.1,
            nfe=8,
            audit_seed=91,
        )
