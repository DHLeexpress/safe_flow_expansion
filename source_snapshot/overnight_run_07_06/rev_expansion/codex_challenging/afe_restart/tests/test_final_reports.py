from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from afe_restart.config import clean_method_absence_manifest
from afe_restart.final_reports import (
    ArtifactError,
    _table_rows,
    generate_reports,
    load_gallery,
    load_id_demo_paths,
    load_run,
    load_sealed_validity,
)
from afe_restart.scene import GAMMAS


SOURCE_CHECKPOINT_SHA = "1" * 64
SOURCE_MODEL_SHA = "2" * 64
MONITORING_BANK_SHA = "3" * 64
SEALED_BANK_SHA = "4" * 64
VERIFIER_SHA = "5" * 64
PROTOCOL_SHA = "6" * 64


MATCHED_PROTOCOL = {
    "seed": 105000,
    "candidate_count": 64,
    "verifier_budget": 8,
    "fallback_verifier_budget": 8,
    "beta": 0.2,
    "backup_smooth_weight": 8.0,
    "backup_noise_var_mult": 3.0,
    "backup_retreat_weight": 1.0,
    "rounds": 1,
    "episodes_per_gamma": 1,
    "episode_max_steps": 240,
    "expansion_temperature": 1.0,
    "nfe": 8,
    "ridge_lambda": 0.01,
    "prox_eta": 0.05,
    "learning_rate": 2e-5,
    "microbatch": 256,
    "solver_max_steps": 12,
    "solver_min_steps": 2,
    "update_norm_limit": 0.12,
    "relative_loss_tolerance": 2e-3,
    "gradient_tolerance": 1e-5,
    "audit_plans_per_context": 4,
    "audit_progress_threshold": 0.1,
    "eval_rollouts": 6,
}


ARM = {
    "Full": {
        "arm": "full", "method": "planned-window AFE",
        "acquisition": "afe", "acquisition_mode": "afe",
        "progress_ranking": True, "eligibility_mode": "full",
        "replay_eligibility": "full_safe", "runtime_safety_claim": True,
        "uncertainty_tilting": True, "ordinary_audit_untilted": True,
    },
    "-AFE": {
        "arm": "minus_afe", "acquisition_mode": "uniform",
        "progress_ranking": True, "eligibility_mode": "full",
        "replay_eligibility": "full_safe", "runtime_safety_claim": True,
    },
    "-Progress": {
        "arm": "minus_progress", "acquisition_mode": "afe",
        "progress_ranking": False, "eligibility_mode": "full",
        "replay_eligibility": "full_safe", "runtime_safety_claim": True,
    },
    "-SOCP": {
        "arm": "minus_socp", "acquisition_mode": "afe",
        "progress_ranking": True, "eligibility_mode": "bounds_only_offline",
        "replay_eligibility": "strict_bounds", "runtime_safety_claim": False,
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(
    round_index: int, *, role: str, fingerprint: str, seed: int | None = None,
) -> dict:
    rows = []
    for index, gamma in enumerate(GAMMAS):
        safe = min(9, 3 + round_index + index // 3)
        progress = max(0, safe - 1)
        left = safe // 2
        right = safe - left
        rows.append({
            "gamma": gamma,
            "sample_count": 10,
            "safe_count": safe,
            "safe_progress_count": progress,
            "validity_mass": safe / 10,
            "validity_interval": {"low": 0.0, "high": 1.0},
            "progress_validity": progress / 10,
            "progress_validity_interval": {"low": 0.0, "high": 1.0},
            "mean_progress": 0.1,
            "mean_safe_progress": 0.2,
            "mode_counts": {
                "left-of-goal-ray": left,
                "right-of-goal-ray": right,
                "goal-ray": 0,
            },
            "safe_mode_coverage": int(left > 0) + int(right > 0),
        })
    return {
        "context_count": 5,
        "plans_per_context": 2,
        "total_verifier_calls": 70,
        "seed": 123 + round_index if seed is None else seed,
        "temperature": 1.0,
        "progress_threshold": 0.1,
        "context_bank_fingerprint": fingerprint,
        "context_bank_role": role,
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "uncertainty_tilting": False,
        "confidence_interval_scope": "conditional_plan_sampling",
        "independent_training_seed_count": 1,
        "independent_training_seed_ci": False,
        "per_gamma": rows,
    }


def _rollouts(temperature: float, count: int = 2) -> list[dict]:
    rows = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for repetition in range(count):
            t = np.linspace(0, 1, 20)
            bend = (.12 if repetition == 0 else -.12) * np.sin(np.pi * t)
            path = np.stack((.5 + 4 * t + bend, .5 + 4 * t - bend), axis=1)
            rows.append({
                "gamma": gamma,
                "seed": gamma_index * 100 + repetition,
                "temperature": temperature,
                "states": np.column_stack((path, np.zeros((len(path), 2)))),
                "actions": np.zeros((len(path) - 1, 2), dtype=np.float32),
                "reached": repetition == 0,
                "collision": repetition == 1,
                "out_of_bounds": False,
                "timeout": False,
                "min_clearance_m": .2 + .01 * gamma_index,
                "time_to_goal_s": 1.9 if repetition == 0 else None,
                "detour_mode": "upper-left" if repetition == 0 else "lower-right",
            })
    return rows


def _recipe(label: str) -> dict:
    return {
        **ARM[label],
        "source_checkpoint": "/immutable/pretrained.pt",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
        "source_model_hash": SOURCE_MODEL_SHA,
        "audit_bank": "/immutable/round_monitoring.pt",
        "audit_bank_fingerprint": MONITORING_BANK_SHA,
        "audit_bank_role": "round_monitoring",
        "verifier_spec_fingerprint": VERIFIER_SHA,
        "matched_protocol": copy.deepcopy(MATCHED_PROTOCOL),
        "legacy_mechanisms": clean_method_absence_manifest(),
        "gamma_distribution": "fixed uniform over all seven gammas; no schedule",
        "backup_planner": {
            "smooth_weight": 8.0, "noise_var_mult": 3.0, "retreat_weight": 1.0,
        },
        "sampling_temperature": 1.0,
        "expansion_temperature": 1.0,
        "audit_temperature": 1.0,
        "visualization_temperature": 0.5,
        "ordinary_audit_untilted": True,
    }


def _write_run(root: Path, label: str, *, valid_audit: bool = True) -> Path:
    (root / "data").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)
    recipe = _recipe(label)
    ordinary = _rollouts(1.0)
    audit0 = _audit(0, role="round_monitoring", fingerprint=MONITORING_BANK_SHA)
    audit1 = _audit(1, role="round_monitoring", fingerprint=MONITORING_BANK_SHA)
    if not valid_audit:
        audit1["per_gamma"][0].pop("safe_count")
    schema = "afe_expansion_round_v1" if label == "Full" else "afe_matched_ablation_round_v1"
    arm = ARM[label]["arm"]
    common = {"schema_version": schema, "arm": arm, "recipe": recipe}
    torch.save({
        **common,
        "round": 0,
        "ordinary_rollouts": ordinary,
        "audit": audit0,
        "episodes": [],
        "matrix": {"logdet": 0.0, "eigenvalue_min": 1.0, "eigenvalue_max": 1.0},
        "store_state": {},
    }, root / "data/round_000_bundle.pt")
    episodes = [
        {"gamma": gamma, "actions": np.zeros((4, 2)), "fallback_steps": index % 2,
         "fail_closed": index == 0}
        for index, gamma in enumerate(GAMMAS)
    ]
    query = {
        "episodes": len(GAMMAS), "fail_closed_episodes": 1,
        "new_verifier_calls": 70, "new_positive_queries": 35,
        "new_negative_queries": 35, "query_acceptance": .5,
        "fallback_frequency": .1,
    }
    solver = {
        "sampling": "uniform_full_positive_pass_seeded_reshuffle",
        "final_update_norm": .012, "positive_count": 35,
        "trace": [{"cfm_loss": .8, "update_norm": .012, "positive_coverage": 1.0}],
    }
    matrix = {"logdet": 4.2, "eigenvalue_min": 1.0, "eigenvalue_max": 72.0}
    torch.save({
        **common,
        "round": 1,
        "ordinary_rollouts": ordinary,
        "audit": audit1,
        "episodes": episodes,
        "query_summary": query,
        "solver": solver,
        "matrix": matrix,
        "store_state": {},
    }, root / "data/round_001_bundle.pt")
    model_hash = hashlib.sha256(label.encode()).hexdigest()
    checkpoint_history = [
        {"round": 0, "model_hash": SOURCE_MODEL_SHA},
        {"round": 1, "model_hash": model_hash},
    ]
    torch.save({
        "round": 1,
        "recipe": recipe,
        "current_model_hash": model_hash,
        "history": checkpoint_history,
    }, root / "checkpoints/round_001.pt")
    # These mutable logs are intentionally not authoritative.
    (root / "logs/history.json").write_text(json.dumps([{"round": 999}]))
    (root / "logs/recipe.json").write_text(json.dumps({"arm": "tampered"}))
    return root


def _write_id_data(path: Path) -> Path:
    plans = []
    trajectory_ids = []
    steps = []
    gamma_rows = []
    seeds = []
    metadata = []
    trajectory = 0
    for gamma in GAMMAS:
        for mode_index, mode in enumerate(("R-first", "U-first")):
            plan = np.zeros((10, 2), dtype=np.float32)
            plan[0] = [1.0, .4] if mode_index == 0 else [.4, 1.0]
            plans.append(plan); trajectory_ids.append(trajectory); steps.append(0)
            gamma_rows.append(gamma); seeds.append(1000 + trajectory)
            metadata.append({
                "trajectory_id": trajectory, "gamma": gamma,
                "seed": 1000 + trajectory, "direction_class": mode,
                "steps": 1, "min_clearance_m": .2,
            })
            trajectory += 1
    torch.save({
        "schema_version": "planned_window_demo_v1",
        "U": torch.tensor(np.asarray(plans)),
        "window_trajectory_ids": torch.tensor(trajectory_ids),
        "window_steps": torch.tensor(steps), "gamma": torch.tensor(gamma_rows),
        "window_seeds": torch.tensor(seeds), "trajectory_rows": metadata,
        "start": torch.tensor([.5, .5]), "goal": torch.tensor([4.5, 4.5]),
    }, path)
    return path


def _write_gallery(path: Path, run_root: Path, label: str, temperature: float = .5) -> Path:
    checkpoint = run_root / "checkpoints/round_001.pt"
    torch.save({
        "schema_version": "afe_gallery_rollouts_v1",
        "label": "-SOCP (offline only)" if label == "-SOCP" else label,
        "visualization_temperature": temperature,
        "visualization_rollouts": _rollouts(temperature),
        "gallery_diagnostics_not_scientific_metrics": {},
        "scientific_use_forbidden": True,
        "scientific_metrics_source_temperature": 1.0,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "seed": 1,
        "nfe": 8,
    }, path)
    return path


def _aggregate(audit: dict) -> dict:
    n = sum(row["sample_count"] for row in audit["per_gamma"])
    safe = sum(row["safe_count"] for row in audit["per_gamma"])
    progress = sum(row["safe_progress_count"] for row in audit["per_gamma"])
    valid_modes = sorted({
        mode for row in audit["per_gamma"]
        for mode, count in row["mode_counts"].items() if count > 0
    })
    return {
        "sample_count": n,
        "safe_count": safe,
        "safe_progress_count": progress,
        "V": safe / n,
        "Vprog": progress / n,
        "valid_mode_coverage_count": len(valid_modes),
        "valid_mode_coverage_fraction": len(valid_modes) / 3,
        "valid_modes": valid_modes,
    }


def _interval(mean: float) -> dict:
    return {
        "mean": mean, "low": max(0.0, mean - .1), "high": min(1.0, mean + .1),
        "standard_deviation_across_training_seeds": .05,
        "confidence": .95,
        "independent_training_seed_count": 2,
        "method": "student_t_across_independent_training_seed_estimates",
    }


def _write_sealed(path: Path, run_roots: dict[str, Path]) -> Path:
    bank_path = path.parent / "sealed_bank.pt"
    bank_path.write_bytes(b"sealed-bank-for-final-report-test")
    protocol = {
        "schema_version": "afe_sealed_audit_protocol_v1",
        "context_bank_path": str(bank_path.resolve()),
        "context_bank_file_sha256": _sha(bank_path),
        "context_bank_fingerprint": SEALED_BANK_SHA,
        "context_bank_role": "sealed_final_test",
        "context_bank_artifact_fingerprint": "b" * 64,
        "context_bank_source_provenance_fingerprint": "c" * 64,
        "context_bank_source_provenance": {"source": "locked-unit-test-bank"},
        "context_count": 5,
        "plans_per_context": 2,
        "progress_threshold": .1,
        "nfe": 8,
        "temperature": 1.0,
        "uncertainty_tilting": False,
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "gammas": list(GAMMAS),
        "verifier_spec_fingerprint": VERIFIER_SHA,
        "audit_seed": 91,
        "conditional_plan_sampling_confidence": .95,
        "independent_training_seed_confidence": .95,
        "full_verifier_label": "strict task-space bounds AND full SOCP certificate",
        "audit_samples_added_to_training_or_acquisition": False,
        "audit_invocations_per_model": 1,
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    protocol["protocol_fingerprint"] = protocol_sha
    selected: dict[str, str] = {}
    per_run = []
    methods = {
        "Full": "full", "-AFE": "minus_afe",
        "-Progress": "minus_progress", "-SOCP": "minus_socp",
    }
    for index, label in enumerate(("Full", "-AFE", "-Progress", "-SOCP")):
        run_id = f"main-{methods[label]}"
        selected[label] = run_id
        root = run_roots[label]
        checkpoint = root / "checkpoints/round_001.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        audit = _audit(
            10 + index, role="sealed_final_test", fingerprint=SEALED_BANK_SHA, seed=91
        )
        available = label != "-SOCP"
        per_run.append({
            "run_id": run_id,
            "label": label,
            "method": methods[label],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_file_sha256": _sha(checkpoint),
            "model_state_sha256": checkpoint_payload["current_model_hash"],
            "expansion_training_seed": MATCHED_PROTOCOL["seed"],
            "source_pretrain_hash": SOURCE_MODEL_SHA,
            "source_pretrain_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
            "checkpoint_round": 1,
            "runtime_safety_claim": available,
            "independent_full_replica": label == "Full",
            "selected_main": True,
            "expansion_verifier_spec_fingerprint": VERIFIER_SHA,
            "audit_verifier_spec_fingerprint": VERIFIER_SHA,
            "protocol_fingerprint": protocol_sha,
            "audit": audit,
            "aggregate": _aggregate(audit),
            "runtime": {
                "available": available,
                "control_decisions": 100 if available else 0,
                "episode_count": 20 if available else 0,
                "fallback_steps": 10 if available else 0,
                "fallback_frequency": .1 if available else None,
                "failclosed_episodes": 2 if available else 0,
                "failclosed_frequency": .1 if available else None,
                "source": "embedded expansion bundles" if available else "offline arm",
            },
        })
    # A second independently trained Full replica supplies the scientific
    # replication unit but is not selected for the gallery/table row.
    replica_checkpoint = path.parent / "independent_full_2.pt"
    replica_model_hash = "8" * 64
    replica_recipe = _recipe("Full")
    replica_recipe["source_model_hash"] = "9" * 64
    replica_recipe["source_checkpoint_sha256"] = "a" * 64
    torch.save({
        "round": 1,
        "recipe": replica_recipe,
        "current_model_hash": replica_model_hash,
        "history": [{"round": 1, "model_hash": replica_model_hash}],
    }, replica_checkpoint)
    replica = copy.deepcopy(per_run[0])
    replica.update({
        "run_id": "independent-full-2",
        "label": "Full independent seed 205000",
        "checkpoint_path": str(replica_checkpoint.resolve()),
        "checkpoint_file_sha256": _sha(replica_checkpoint),
        "model_state_sha256": replica_model_hash,
        "expansion_training_seed": 205000,
        "source_pretrain_hash": "9" * 64,
        "source_pretrain_checkpoint_sha256": "a" * 64,
        "independent_full_replica": True,
        "selected_main": False,
    })
    per_run.append(replica)
    independent = {
        "schema_version": "afe_sealed_independent_full_aggregate_v1",
        "context_bank_fingerprint": SEALED_BANK_SHA,
        "context_bank_role": "sealed_final_test",
        "training_seeds": [105000, 205000],
        "independent_training_seed_count": 2,
        "replication_unit": "independently_trained_model",
        "plan_samples_pooled_across_training_seeds": False,
        "confidence_interval_scope": "across_independent_training_seed_estimates",
        "protocol_fingerprint": protocol_sha,
        "run_ids": [per_run[0]["run_id"], replica["run_id"]],
        "model_state_sha256s": [per_run[0]["model_state_sha256"], replica_model_hash],
        "source_pretrain_hashes": [SOURCE_MODEL_SHA, "9" * 64],
        "aggregate_over_gammas": {
            "validity": _interval(.6),
            "progress_validity": _interval(.5),
            "V": _interval(.6),
            "Vprog": _interval(.5),
            "valid_mode_coverage": _interval(2 / 3),
            "fallback_frequency": _interval(.1),
            "failclosed_frequency": _interval(.1),
        },
        "per_gamma": [{
            "gamma": gamma,
            "validity": _interval(.6),
            "progress_validity": _interval(.5),
            "V": _interval(.6),
            "Vprog": _interval(.5),
            "valid_mode_coverage": _interval(2 / 3),
        } for gamma in GAMMAS],
        "runtime": {
            "source": "expansion_checkpoint_history_not_sealed_audit_execution",
            "available_run_count": 2,
            "missing_run_count": 0,
            "fallback_frequency": _interval(.1),
            "failclosed_frequency": _interval(.1),
        },
    }
    payload = {
        "schema_version": "afe_sealed_validity_v1",
        "bank_fingerprint": SEALED_BANK_SHA,
        "protocol_fingerprint": protocol_sha,
        "protocol": protocol,
        "per_run": per_run,
        "independent_full_aggregate": independent,
        "selected_main_run_ids": selected,
        "provenance": {
            "generator": "unit-test",
            "sealed_bank_path": str(bank_path.resolve()),
            "sealed_bank_file_sha256": _sha(bank_path),
            "one_shot_evaluation": True,
            "audit_invocations_per_model": 1,
            "audit_results_used_for_training_or_checkpoint_selection": False,
            "plan_samples_pooled_across_models": False,
        },
    }
    path.write_text(json.dumps(payload))
    return path


def _all_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path], Path]:
    roots = {
        label: _write_run(tmp_path / ARM[label]["arm"], label)
        for label in ("Full", "-AFE", "-Progress", "-SOCP")
    }
    galleries = {
        label: _write_gallery(tmp_path / f"{ARM[label]['arm']}_gallery.pt", root, label)
        for label, root in roots.items()
    }
    sealed = _write_sealed(tmp_path / "sealed.json", roots)
    return roots, galleries, sealed


def _generate_kwargs(
    tmp_path: Path, roots: dict[str, Path], galleries: dict[str, Path], sealed: Path,
) -> dict:
    return {
        "full_path": roots["Full"], "full_viz_path": galleries["Full"],
        "no_afe_path": roots["-AFE"], "no_afe_viz_path": galleries["-AFE"],
        "no_progress_path": roots["-Progress"],
        "no_progress_viz_path": galleries["-Progress"],
        "no_socp_path": roots["-SOCP"], "no_socp_viz_path": galleries["-SOCP"],
        "sealed_validity_path": sealed,
        "id_demos_path": _write_id_data(tmp_path / "id.pt"),
        "output_dir": tmp_path / "reports",
    }


def test_id_demo_paths_are_reconstructed_and_balanced(tmp_path: Path) -> None:
    rows = load_id_demo_paths(_write_id_data(tmp_path / "id.pt"))
    assert len(rows) == 2 * len(GAMMAS)
    assert {row.mode for row in rows} == {"R-first", "U-first"}
    assert all(row.path.shape == (2, 2) for row in rows)


def test_run_uses_embedded_recipe_and_history_not_mutable_logs(tmp_path: Path) -> None:
    root = _write_run(tmp_path / "full", "Full")
    run = load_run(root, "Full", require_full=True)
    assert run.recipe["arm"] == "full"
    assert [row["round"] for row in run.history] == [0, 1]


def test_explicit_checkpoint_ignores_only_strictly_later_rounds(tmp_path: Path) -> None:
    root = _write_run(tmp_path / "full", "Full")
    selected = root / "checkpoints/round_001.pt"
    later_bundle = torch.load(
        root / "data/round_001_bundle.pt", map_location="cpu", weights_only=False,
    )
    later_recipe = copy.deepcopy(later_bundle["recipe"])
    later_recipe["matched_protocol"]["rounds"] = 2
    later_bundle.update({
        "round": 2,
        "recipe": later_recipe,
        "audit": _audit(2, role="round_monitoring", fingerprint=MONITORING_BANK_SHA),
    })
    torch.save(later_bundle, root / "data/round_002_bundle.pt")
    later_model_hash = "a" * 64
    torch.save({
        "round": 2,
        "recipe": later_recipe,
        "current_model_hash": later_model_hash,
        "history": [
            {"round": 0, "model_hash": SOURCE_MODEL_SHA},
            {"round": 1, "model_hash": hashlib.sha256(b"Full").hexdigest()},
            {"round": 2, "model_hash": later_model_hash},
        ],
    }, root / "checkpoints/round_002.pt")

    run = load_run(
        root, "Full", require_full=True, selected_checkpoint=selected,
    )
    assert run.final_round == 1
    assert run.checkpoint_path.resolve() == selected.resolve()
    assert [row["round"] for row in run.history] == [0, 1]

    with pytest.raises(ArtifactError, match="embedded recipes changed"):
        load_run(root, "Full", require_full=True)


def test_explicit_checkpoint_rejects_gap_and_cross_run_path(tmp_path: Path) -> None:
    root = _write_run(tmp_path / "full", "Full")
    other = _write_run(tmp_path / "other", "Full")
    with pytest.raises(ArtifactError, match="not inside this run"):
        load_run(
            root, "Full", require_full=True,
            selected_checkpoint=other / "checkpoints/round_001.pt",
        )

    checkpoint = torch.load(
        root / "checkpoints/round_001.pt", map_location="cpu", weights_only=False,
    )
    checkpoint["round"] = 2
    checkpoint["history"].append({
        "round": 2, "model_hash": checkpoint["current_model_hash"],
    })
    selected = root / "checkpoints/round_002.pt"
    torch.save(checkpoint, selected)
    (root / "data/round_001_bundle.pt").unlink()
    with pytest.raises(ArtifactError, match="contiguous bundle prefix"):
        load_run(root, "Full", require_full=True, selected_checkpoint=selected)


def test_missing_integer_audit_count_is_rejected_without_rounding(tmp_path: Path) -> None:
    root = _write_run(tmp_path / "broken", "Full", valid_audit=False)
    with pytest.raises(ArtifactError, match="safe_count must be an explicit integer"):
        load_run(root, "Full", require_full=True)


def test_gallery_is_bound_to_exact_run_checkpoint_and_temperature(tmp_path: Path) -> None:
    root = _write_run(tmp_path / "full", "Full")
    run = load_run(root, "Full", require_full=True)
    wrong_temperature = _write_gallery(tmp_path / "wrong_t.pt", root, "Full", 1.0)
    with pytest.raises(ArtifactError, match="T=0.5"):
        load_gallery(wrong_temperature, "Full", run)
    gallery = _write_gallery(tmp_path / "gallery.pt", root, "Full")
    payload = torch.load(gallery, map_location="cpu", weights_only=False)
    payload["checkpoint_sha256"] = "f" * 64
    torch.save(payload, gallery)
    with pytest.raises(ArtifactError, match="checkpoint file provenance mismatch"):
        load_gallery(gallery, "Full", run)


def test_matched_protocol_or_arm_mismatch_fails_report(tmp_path: Path) -> None:
    roots, galleries, sealed = _all_inputs(tmp_path)
    bundle_path = roots["-AFE"] / "data/round_001_bundle.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    bundle["arm"] = "minus_progress"
    torch.save(bundle, bundle_path)
    with pytest.raises(ArtifactError, match="arm label"):
        generate_reports(**_generate_kwargs(tmp_path, roots, galleries, sealed))


def test_sealed_checkpoint_provenance_is_required(tmp_path: Path) -> None:
    roots, galleries, sealed = _all_inputs(tmp_path)
    payload = json.loads(sealed.read_text())
    selected = payload["selected_main_run_ids"]["Full"]
    row = next(row for row in payload["per_run"] if row["run_id"] == selected)
    row["checkpoint_file_sha256"] = "f" * 64
    sealed.write_text(json.dumps(payload))
    runs = {
        label: load_run(roots[label], label, require_full=label == "Full")
        for label in ("Full", "-AFE", "-Progress", "-SOCP")
    }
    with pytest.raises(ArtifactError, match="checkpoint file hash mismatch"):
        load_sealed_validity(sealed, runs)


def test_generate_all_outputs_uses_sealed_validity(tmp_path: Path) -> None:
    roots, galleries, sealed = _all_inputs(tmp_path)
    manifest = generate_reports(**_generate_kwargs(tmp_path, roots, galleries, sealed))
    expected = {
        "rollouts.png", "internals.png", "scatter.png", "table.md",
        "table.tex", "final_validity_report.md", "manifest.json",
    }
    assert expected == {path.name for path in (tmp_path / "reports").iterdir()}
    for key in ("rollouts", "internals", "scatter", "table_md", "table_tex", "validity"):
        assert Path(manifest[key]).is_file() and Path(manifest[key]).stat().st_size > 0
    report = (tmp_path / "reports/final_validity_report.md").read_text()
    assert "afe_sealed_validity_v1" in report
    assert "Round-monitoring audits" in report
    assert "Independent-training-seed Full aggregate" in report
    assert "valid-mode coverage" in report
    assert "offline/no claim" in report
    table = (tmp_path / "reports/table.md").read_text()
    assert "sealed-final bank" in table and "-SOCP" in table


def test_offline_no_socp_suppresses_runtime_but_keeps_sealed_validity(tmp_path: Path) -> None:
    roots, _galleries, sealed_path = _all_inputs(tmp_path)
    runs = {
        label: load_run(roots[label], label, require_full=label == "Full")
        for label in ("Full", "-AFE", "-Progress", "-SOCP")
    }
    sealed = load_sealed_validity(sealed_path, runs)
    rows = _table_rows({"-SOCP": runs["-SOCP"]}, sealed)
    assert rows
    assert all(row["fallback"] == "offline/no claim" for row in rows)
    assert all(row["failclosed"] == "offline/no claim" for row in rows)
    assert all(row["V"] != "—" and row["Vprog"] != "—" for row in rows)
