from __future__ import annotations

import json
import hashlib
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from afe_restart.controller import ControlStepTrace, QueriedPlanTrace
from afe_restart.config import clean_method_absence_manifest
from afe_restart.decision_budget import (
    REFERENCE_SCHEMA,
    build_usage,
    fingerprint,
)
from afe_restart.policy import model_state_hash
from afe_restart.scene import GAMMAS, make_ood_scene, verifier_spec_fingerprint
from afe_restart.schemas import (
    ProgressResult,
    QueryContext,
    QuerySource,
    SafetyResult,
    VerificationRecord,
)
from afe_restart.stage7_artifacts import (
    ArtifactStageError,
    RunSpec,
    build_run_frames,
    generate_checkpoint_gallery,
    generate_run_artifacts,
    load_expansion_source,
    save_gallery_artifact,
    validate_matched_sources,
)
from afe_restart.store import VerificationStore
from afe_restart.uncertainty import CumulativeLinearUncertainty
from afe_restart.visualize_expansion import load_visualization_data


class _Tiny(torch.nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


def _scene_fingerprint() -> str:
    env = make_ood_scene(radius=1.2)
    return verifier_spec_fingerprint(env, env.goal)


def _context(gamma_index: int) -> QueryContext:
    value = 0.50 + 0.01 * gamma_index
    return QueryContext(
        grid=np.full((2, 2), value, dtype=np.float32),
        low5=np.asarray([0.5, 0.5, 0.0, 0.0, 0.5], dtype=np.float32),
        hist=np.zeros((2, 2), dtype=np.float32),
        verifier_state=np.asarray([value, value, 0.0, 0.0], dtype=np.float64),
        verifier_spec_fingerprint=_scene_fingerprint(),
    )


def _unsafe_bounds_record(
    plan: np.ndarray, gamma: float, gamma_index: int,
) -> VerificationRecord:
    feature = np.zeros(32, dtype=np.float64)
    feature[gamma_index] = 1.0
    return VerificationRecord(
        context=_context(gamma_index),
        gamma=gamma,
        plan=plan,
        source=QuerySource.FLOW,
        feature_z=feature,
        acquisition_sigma=1.0,
        safety=SafetyResult(
            strict_bounds=True,
            socp_certified=False,
            min_clearance=-0.1,
            certificate_slack=-0.2,
            feasible_face_margin=-np.inf,
        ),
        progress=ProgressResult(
            initial_goal_distance=5.0,
            terminal_goal_distance=4.8,
        ),
        executed=False,
    )


def _audit() -> dict:
    return {
        "temperature": 1.0,
        "uncertainty_tilting": False,
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "context_bank_fingerprint": "a" * 64,
        "context_bank_role": "round_monitoring",
        "per_gamma": [
            {
                "gamma": float(gamma),
                "sample_count": 10,
                "validity_mass": 0.2,
                "progress_validity": 0.1,
                "validity_interval": {"low": 0.05, "high": 0.45},
                "progress_validity_interval": {"low": 0.01, "high": 0.30},
            }
            for gamma in GAMMAS
        ],
    }


def _solver(count: int = len(GAMMAS)) -> dict:
    return {
        "positive_count": count,
        "total_record_count": count,
        "optimizer_steps": 1,
        "final_update_norm": 0.01,
        "stopping_reason": "max_steps",
        "trace": [{
            "original_record_indices": list(range(count)),
            "positive_coverage": 1.0,
            "objective": 0.8,
            "cfm_loss": 0.79,
            "proximal_penalty": 0.01,
            "gradient_norm": 0.2,
            "update_norm": 0.01,
        }],
    }


def _protocol() -> dict:
    return {
        "seed": 7,
        "candidate_count": 2,
        "verifier_budget": 1,
        "fallback_verifier_budget": 1,
        "beta": 0.2,
        "backup_smooth_weight": 8.0,
        "backup_noise_var_mult": 3.0,
        "backup_retreat_weight": 1.0,
        "rounds": 1,
        "episodes_per_gamma": 1,
        "episode_max_steps": 1,
        "expansion_temperature": 1.0,
        "nfe": 1,
        "ridge_lambda": 1.0,
        "prox_eta": 0.05,
        "learning_rate": 2e-5,
        "microbatch": 8,
        "solver_max_steps": 1,
        "solver_min_steps": 1,
        "update_norm_limit": 0.1,
        "relative_loss_tolerance": 0.01,
        "gradient_tolerance": 1e-5,
        "audit_plans_per_context": 1,
        "audit_progress_threshold": 0.1,
        "eval_rollouts": 1,
    }


def _recipe(model_hash: str) -> dict:
    protocol = _protocol()
    reference_core = {
        "schema_version": REFERENCE_SCHEMA,
        "reference_dir": "/fake/full",
        "reference_recipe_sha256": "4" * 64,
        "source_checkpoint_sha256": "1" * 64,
        "source_model_hash": "2" * 64,
        "final_checkpoint_path": "/fake/full/checkpoints/round_001.pt",
        "final_checkpoint_sha256": "3" * 64,
        "final_model_hash": model_hash,
        "final_round": 1,
        "matched_protocol": protocol,
        "caps": [
            {
                "round": 1,
                "gamma": float(gamma),
                "episode_index": 0,
                "max_control_decisions": 1,
            }
            for gamma in GAMMAS
        ],
        "total_control_decisions": len(GAMMAS),
    }
    reference = reference_core | {"fingerprint": fingerprint(reference_core)}
    return {
        "arm": "minus_socp",
        "acquisition_mode": "afe",
        "progress_ranking": True,
        "eligibility_mode": "bounds_only_offline",
        "replay_eligibility": "strict_bounds",
        "runtime_safety_claim": False,
        "uncertainty_tilting": True,
        "ordinary_audit_untilted": True,
        "source_checkpoint_sha256": "1" * 64,
        "source_model_hash": "2" * 64,
        "frozen_feature_hash": "2" * 64,
        "audit_bank_fingerprint": "a" * 64,
        "audit_bank_role": "round_monitoring",
        "verifier_spec_fingerprint": _scene_fingerprint(),
        "legacy_mechanisms": clean_method_absence_manifest(),
        "matched_protocol": protocol,
        "gamma_distribution": "fixed uniform over all seven gammas; no schedule",
        "beta": 0.2,
        "current_model_hash_for_test": model_hash,
        "full_reference_decision_budget": reference,
        "control_decision_budget_rule": (
            "for every (round,gamma,episode), max_steps equals the selected "
            "Full episode's realized len(traces); the control may terminate earlier"
        ),
    }


def _write_offline_run(root: Path) -> RunSpec:
    (root / "data").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)
    empty = VerificationStore(CumulativeLinearUncertainty(lambda_=1.0))
    checkpoint_model_hash = model_state_hash(_Tiny())
    recipe = _recipe(checkpoint_model_hash)
    torch.save({
        "round": 0,
        "arm": "minus_socp",
        "recipe": recipe,
        "episodes": [],
        "store_state": empty.state_dict(),
        "audit": _audit(),
    }, root / "data/round_000_bundle.pt")

    store = VerificationStore(CumulativeLinearUncertainty(lambda_=1.0))
    episodes = []
    for gamma_index, gamma in enumerate(GAMMAS):
        plan = np.full((10, 2), 0.01 * (gamma_index + 1), dtype=np.float32)
        record = _unsafe_bounds_record(plan, float(gamma), gamma_index)
        store.append(record)
        query = QueriedPlanTrace(
            candidate_index=0,
            query_hash=record.query_hash,
            source="flow",
            plan_kind="flow",
            acquisition_sigma=1.0,
            safe=False,
            in_bounds=True,
            socp_ok=False,
            progress_m=record.progress_value,
            clearance_m=-0.1,
            cache_hit=False,
            executed=True,
        )
        state = record.context.verifier_state.copy()
        trace = ControlStepTrace(
            step=0,
            gamma=float(gamma),
            state_before=state,
            candidate_plans=np.stack((plan, -plan)),
            candidate_sigmas=np.asarray([1.0, 0.8]),
            acquisition_probabilities=np.asarray([0.55, 0.45]),
            acquisition_order=np.asarray([0, 1]),
            queried=(query,),
            verifier_calls=1,
            cache_hits=0,
            selected_query_hash=record.query_hash,
            selected_source="flow",
            action=plan[0],
            state_after=state,
            fallback_used=False,
            fail_closed=False,
            acquisition_entropy=0.69,
            acquisition_ess=1.98,
            eligibility_mode="bounds_only_offline",
            runtime_safety_claim=False,
            selected_actual_full_safe=False,
        )
        episodes.append({
            "gamma": float(gamma),
            "seed": 7 + 1_000_000 + gamma_index * 10_000,
            "traces": [trace],
        })
    solver = _solver()
    audit = _audit()
    query_summary = {
        "new_verifier_calls": len(GAMMAS),
        "full_reference_control_decision_budget": build_usage(
            episodes,
            round_index=1,
            reference=recipe["full_reference_decision_budget"],
            expected_seed_base=7,
        ),
    }
    matrix = {"observations": len(GAMMAS)}
    torch.save({
        "round": 1,
        "arm": "minus_socp",
        "recipe": recipe,
        "episodes": episodes,
        "store_state": store.state_dict(),
        "query_summary": query_summary,
        "solver": solver,
        "audit": audit,
        "matrix": matrix,
    }, root / "data/round_001_bundle.pt")
    torch.save({
        "round": 1,
        "recipe": recipe,
        "current_model_hash": checkpoint_model_hash,
        "history": [{
            "round": 1,
            "model_hash": checkpoint_model_hash,
            "query": query_summary,
            "solver": solver,
            "audit": audit,
            "matrix": matrix,
        }],
        "verification_store_state": store.state_dict(),
    }, root / "checkpoints/round_001.pt")
    return RunSpec(
        key="minus_socp",
        label="-SOCP (offline only)",
        root=root,
        replay_eligibility="strict_bounds",
        runtime_safety_claim=False,
        acquisition_mode="afe",
        progress_ranking=True,
        eligibility_mode="bounds_only_offline",
    )


def test_offline_source_uses_exact_bounds_view_without_relabeling(tmp_path: Path) -> None:
    source = load_expansion_source(_write_offline_run(tmp_path / "run"))
    frames = build_run_frames(source)
    assert len(frames) == len(GAMMAS)
    frame = frames[-1]
    assert frame.replay_eligibility == "strict_bounds"
    assert not frame.runtime_safety_claim
    assert frame.executed is not None and not frame.executed.safe
    assert len(frame.replay) == len(GAMMAS)
    assert all(not item.safe and item.strict_bounds for item in frame.replay)
    source_record = source.store.records[0]
    assert not source_record.safety.socp_certified
    assert not source_record.executed
    assert frame.replay[0].query_hash == source_record.query_hash


def test_gallery_writer_rejects_scientific_temperature(tmp_path: Path) -> None:
    with pytest.raises(ArtifactStageError, match="exclusively T=0.5"):
        save_gallery_artifact(
            tmp_path / "gallery.pt",
            label="Full",
            checkpoint=tmp_path / "checkpoint.pt",
            checkpoint_hash="0" * 64,
            checkpoint_model_hash="1" * 64,
            source_recipe_sha256="2" * 64,
            scene_verifier_spec_fingerprint="3" * 64,
            rollouts=[{"temperature": 1.0}],
            summary={},
            seed=1,
            nfe=1,
        )


def test_generate_artifacts_is_read_only_and_marks_offline_video(tmp_path: Path) -> None:
    source = load_expansion_source(_write_offline_run(tmp_path / "run"))

    def loader(path, *, device):
        assert path == source.checkpoint_path and str(device) == "cpu"
        return _Tiny(), {
            "current_model_hash": source.checkpoint_model_hash,
            "round": source.checkpoint_round,
            "recipe": source.recipe,
        }

    def evaluator(model, env, *, seed, per_gamma, nfe, temperature):
        del model, env, seed, nfe
        assert temperature == 0.5
        rollouts = [
            {
                "gamma": float(gamma),
                "seed": index,
                "temperature": 0.5,
                "states": np.asarray([[0.5, 0.5, 0, 0], [0.51, 0.51, 0, 0]]),
                "actions": np.zeros((1, 2)),
                "reached": False,
                "collision": False,
                "out_of_bounds": False,
                "timeout": True,
                "min_clearance_m": 0.1,
                "path_length_m": 0.02,
                "time_to_goal_s": None,
                "detour_mode": "unresolved",
            }
            for index, gamma in enumerate(GAMMAS)
            for _ in range(per_gamma)
        ]
        return {str(gamma): {"n": per_gamma} for gamma in GAMMAS}, rollouts

    def renderer(scene, frames, output, **kwargs):
        del scene
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"offline video stub")
        preview = Path(kwargs["preview_png"])
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(b"preview")
        assert not frames[-1].runtime_safety_claim
        assert not frames[-1].replay[0].safe
        return {
            "status": "PASS",
            "runtime_safety_claim": False,
            "no_runtime_safety_claim_banner": True,
        }

    result = generate_run_artifacts(
        source,
        tmp_path / "out",
        device=torch.device("cpu"),
        gallery_seed=1,
        gallery_rollouts_per_gamma=1,
        nfe=1,
        fps=1,
        seconds_per_event=1.0,
        dpi=24,
        video_max_events=0,
        model_loader=loader,
        gallery_evaluator=evaluator,
        video_renderer=renderer,
    )
    assert result["source_hashes_unchanged"]
    assert result["gallery"]["temperature"] == 0.5
    active = result["active_expansion"]
    assert not active["runtime_safety_claim"]
    assert active["no_runtime_safety_claim_banner"]
    assert active["actual_socp_failures_in_final_training_rows"] == len(GAMMAS)
    assert active["query_acceptance_scope"] == "FLOW_only"
    payload = torch.load(result["gallery"]["path"], weights_only=False)
    assert payload["scientific_use_forbidden"]
    assert payload["visualization_temperature"] == 0.5
    assert payload["checkpoint_model_state_sha256"] == source.checkpoint_model_hash
    assert payload["source_recipe_sha256"] == source.recipe_sha256
    scene, frames, metadata = load_visualization_data(active["data"])
    del scene
    assert metadata["no_curriculum_learning"]
    assert not frames[-1].runtime_safety_claim
    per_run_manifest = json.loads(Path(result["manifest"]).read_text())
    assert per_run_manifest["no_curriculum_learning"]


def test_arbitrary_pretrained_checkpoint_requires_hash_lock_and_stays_gallery_only(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"arbitrary checkpoint bytes")
    file_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

    model = Tiny()
    state_hash = model_state_hash(model)

    def loader(path, *, device):
        assert path == checkpoint.resolve() and str(device) == "cpu"
        return model, {"model_state_sha256": state_hash}

    def evaluator(model, env, *, seed, per_gamma, nfe, temperature):
        del model, env, seed, nfe
        return {}, [
            {
                "gamma": float(gamma),
                "temperature": temperature,
                "states": np.asarray([[0.5, 0.5, 0, 0]]),
                "actions": np.empty((0, 2)),
                "reached": False,
                "collision": False,
                "out_of_bounds": False,
                "timeout": True,
                "min_clearance_m": 0.2,
                "time_to_goal_s": None,
                "detour_mode": "unresolved",
            }
            for gamma in GAMMAS
            for _ in range(per_gamma)
        ]

    result = generate_checkpoint_gallery(
        checkpoint,
        tmp_path / "out",
        device=torch.device("cpu"),
        gallery_seed=1,
        gallery_rollouts_per_gamma=1,
        nfe=1,
        expected_file_sha256=file_hash,
        model_loader=loader,
        gallery_evaluator=evaluator,
    )
    assert result["embedded_model_hash_verified"]
    assert result["caller_file_hash_verified"]
    payload = torch.load(result["gallery"]["path"], weights_only=False)
    assert payload["visualization_temperature"] == 0.5
    assert payload["scientific_use_forbidden"]

    with pytest.raises(ArtifactStageError, match="file SHA-256 mismatch"):
        generate_checkpoint_gallery(
            checkpoint,
            tmp_path / "bad",
            device=torch.device("cpu"),
            gallery_seed=1,
            gallery_rollouts_per_gamma=1,
            nfe=1,
            expected_file_sha256="0" * 64,
            model_loader=loader,
            gallery_evaluator=evaluator,
        )


def test_cross_arm_validator_allows_only_the_declared_single_switches(
    tmp_path: Path,
) -> None:
    base = load_expansion_source(_write_offline_run(tmp_path / "run"))
    semantics = {
        "full": ("afe", True, "full", "full_safe", True),
        "minus_afe": ("uniform", True, "full", "full_safe", True),
        "minus_progress": ("afe", False, "full", "full_safe", True),
        "minus_socp": (
            "afe", True, "bounds_only_offline", "strict_bounds", False,
        ),
    }
    labels = {
        "full": "Full", "minus_afe": "-AFE",
        "minus_progress": "-Progress", "minus_socp": "-SOCP (offline only)",
    }
    sources = []
    for key, (acquisition, progress, eligibility, replay, safety_claim) in semantics.items():
        recipe = copy.deepcopy(base.recipe)
        recipe.update({
            "arm": key,
            "acquisition_mode": acquisition,
            "progress_ranking": progress,
            "eligibility_mode": eligibility,
            "replay_eligibility": replay,
            "runtime_safety_claim": safety_claim,
            "uncertainty_tilting": acquisition == "afe",
        })
        if key != "full":
            reference_core = copy.deepcopy(
                recipe["full_reference_decision_budget"]
            )
            reference_core.pop("fingerprint")
            reference_core.update({
                "reference_dir": str(base.spec.root),
                "reference_recipe_sha256": base.recipe_sha256,
                "final_checkpoint_path": str(base.checkpoint_path),
                "final_checkpoint_sha256": base.checkpoint_sha256,
                "final_model_hash": base.checkpoint_model_hash,
                "final_round": base.checkpoint_round,
            })
            recipe["full_reference_decision_budget"] = reference_core | {
                "fingerprint": fingerprint(reference_core)
            }
        else:
            recipe.pop("full_reference_decision_budget", None)
            recipe.pop("control_decision_budget_rule", None)
        spec = RunSpec(
            key=key,
            label=labels[key],
            root=base.spec.root,
            replay_eligibility=replay,
            runtime_safety_claim=safety_claim,
            acquisition_mode=acquisition,
            progress_ranking=progress,
            eligibility_mode=eligibility,
        )
        sources.append(replace(base, spec=spec, recipe=recipe))
    result = validate_matched_sources(sources)
    assert result["only_intended_arm_switches"]

    bad_recipe = copy.deepcopy(sources[1].recipe)
    bad_protocol = copy.deepcopy(sources[1].matched_protocol)
    bad_protocol["beta"] = 0.5
    bad_recipe["beta"] = 0.5
    bad_recipe["matched_protocol"] = bad_protocol
    bad = replace(
        sources[1], recipe=bad_recipe, matched_protocol=bad_protocol,
    )
    with pytest.raises(ArtifactStageError, match="matched protocol drift"):
        validate_matched_sources((sources[0], bad, sources[2], sources[3]))


def test_source_rejects_missing_clean_absence_manifest(tmp_path: Path) -> None:
    spec = _write_offline_run(tmp_path / "run")
    baseline = spec.root / "data/round_000_bundle.pt"
    payload = torch.load(baseline, weights_only=False)
    payload["recipe"]["legacy_mechanisms"]["demo_frac"] = "present"
    torch.save(payload, baseline)
    with pytest.raises(ArtifactStageError, match="absence manifest"):
        load_expansion_source(spec)


def test_explicit_checkpoint_selects_earlier_round_despite_later_bundle(tmp_path: Path) -> None:
    spec = _write_offline_run(tmp_path / "run")
    torch.save(
        {"round": 2, "recipe": {"intentionally": "unselected"}},
        spec.root / "data/round_002_bundle.pt",
    )
    selected = replace(
        spec,
        selected_checkpoint=spec.root / "checkpoints/round_001.pt",
    )
    source = load_expansion_source(selected)
    assert source.checkpoint_round == 1
    assert [path.name for path in source.bundle_paths] == [
        "round_000_bundle.pt", "round_001_bundle.pt",
    ]
