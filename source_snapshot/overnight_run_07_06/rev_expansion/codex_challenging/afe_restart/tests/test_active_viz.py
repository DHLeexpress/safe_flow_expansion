from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from afe_restart.controller import ControlStepTrace, QueriedPlanTrace
from afe_restart.schemas import (
    ProgressResult,
    QueryContext,
    QuerySource,
    SafetyResult,
    VerificationRecord,
)
from afe_restart.store import VerificationStore
from afe_restart.uncertainty import CumulativeLinearUncertainty
from afe_restart.visualize_expansion import (
    RENDER_LABELS,
    SceneSnapshot,
    build_expansion_frames,
    load_visualization_data,
    render_expansion_video,
    save_visualization_data,
)


def _context(value: float) -> QueryContext:
    velocity = 0.0 if np.isclose(value, 0.5) else 0.1
    return QueryContext(
        grid=np.full((2, 2), value, dtype=np.float32),
        low5=np.asarray([value, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
        hist=np.zeros((2, 2), dtype=np.float32),
        verifier_state=np.asarray([value, value, velocity, velocity], dtype=np.float64),
        verifier_spec_fingerprint="a" * 64,
    )


def _record(
    context: QueryContext,
    plan: np.ndarray,
    feature_index: int,
    *,
    safe: bool,
    source: QuerySource,
    gamma: float,
) -> VerificationRecord:
    feature = np.zeros(32, dtype=np.float64)
    feature[feature_index] = 1.0
    return VerificationRecord(
        context=context,
        gamma=gamma,
        plan=plan,
        source=source,
        feature_z=feature,
        acquisition_sigma=1.0,
        safety=SafetyResult(
            strict_bounds=True,
            socp_certified=safe,
            min_clearance=0.2 if safe else -0.05,
            certificate_slack=0.1 if safe else -0.1,
            feasible_face_margin=0.1 if safe else -np.inf,
        ),
        progress=ProgressResult(
            initial_goal_distance=5.0,
            terminal_goal_distance=4.8 if safe else 5.1,
        ),
    )


def _query(record: VerificationRecord, candidate_index: int, *, executed: bool) -> QueriedPlanTrace:
    return QueriedPlanTrace(
        candidate_index=candidate_index,
        query_hash=record.query_hash,
        source=record.source.value,
        plan_kind="flow" if record.source is QuerySource.FLOW else "mean",
        acquisition_sigma=record.acquisition_sigma,
        safe=record.safe,
        in_bounds=record.safety.strict_bounds,
        socp_ok=record.safety.socp_certified,
        progress_m=record.progress_value,
        clearance_m=record.safety.min_clearance,
        cache_hit=False,
        executed=executed,
    )


def _trace(
    step: int,
    state: np.ndarray,
    candidates: np.ndarray,
    queried: tuple[QueriedPlanTrace, ...],
    selected: VerificationRecord | None,
    *,
    fallback: bool,
    gamma: float,
) -> ControlStepTrace:
    sigmas = np.linspace(1.0, 0.8, len(candidates), dtype=np.float64)
    return ControlStepTrace(
        step=step,
        gamma=gamma,
        state_before=state,
        candidate_plans=candidates,
        candidate_sigmas=sigmas,
        acquisition_probabilities=np.full(len(candidates), 1.0 / len(candidates)),
        acquisition_order=np.arange(len(candidates)),
        queried=queried,
        verifier_calls=len(queried),
        cache_hits=0,
        selected_query_hash=selected.query_hash if selected else None,
        selected_source=selected.source.value if selected else None,
        action=np.asarray(selected.plan[0]) if selected else None,
        state_after=state,
        fallback_used=fallback,
        fail_closed=selected is None,
        acquisition_entropy=1.0,
        acquisition_ess=float(len(candidates)),
    )


def _synthetic_controller_data():
    safe_plan = np.full((10, 2), (0.10, 0.08), dtype=np.float32)
    rejected_plan = np.full((10, 2), (0.04, -0.10), dtype=np.float32)
    backup_plan = np.full((10, 2), (0.07, 0.11), dtype=np.float32)
    filler_plan = np.full((10, 2), (-0.03, 0.06), dtype=np.float32)
    first = _record(
        _context(0.5), safe_plan, 0, safe=True, source=QuerySource.FLOW, gamma=0.1,
    )
    rejected = _record(
        _context(0.51), rejected_plan, 1, safe=False, source=QuerySource.FLOW, gamma=0.5,
    )
    backup = _record(
        _context(0.51), backup_plan, 2, safe=True,
        source=QuerySource.SAFEMPPI_BACKUP, gamma=0.5,
    )
    store = VerificationStore(CumulativeLinearUncertainty(lambda_=1.0))
    store.append(first)
    store.append(rejected)
    store.append(backup)
    traces = (
        _trace(
            0,
            np.asarray([0.5, 0.5, 0.0, 0.0]),
            np.stack((safe_plan, filler_plan)),
            (_query(first, 0, executed=True),),
            first,
            fallback=False,
            gamma=0.1,
        ),
        _trace(
            1,
            np.asarray([0.51, 0.51, 0.1, 0.1]),
            np.stack((rejected_plan, filler_plan)),
            (
                _query(rejected, 0, executed=False),
                _query(backup, -1, executed=True),
            ),
            backup,
            fallback=True,
            gamma=0.5,
        ),
    )
    proximal = {
        "positive_count": 1,
        "total_record_count": 1,
        "optimizer_steps": 3,
        "final_update_norm": 0.012,
        "stopping_reason": "relative_loss_tolerance",
        "trace": [
            {
                "original_record_indices": [0],
                "positive_coverage": 1.0,
                "objective": 0.82,
                "cfm_loss": 0.80,
                "proximal_penalty": 0.02,
                "gradient_norm": 0.14,
                "update_norm": 0.012,
            }
        ],
    }
    audit = {
        "temperature": 1.0,
        "uncertainty_tilting": False,
        "sampling_distribution": "ordinary_conditional_flow_iid",
        "context_bank_fingerprint": "c" * 64,
        "context_bank_role": "round_monitoring",
        "per_gamma": [
            {
                "gamma": gamma,
                "sample_count": 20,
                "validity_mass": validity,
                "progress_validity": validity * 0.75,
                "validity_interval": {"low": max(0.0, validity - 0.1), "high": min(1.0, validity + 0.1)},
                "progress_validity_interval": {
                    "low": max(0.0, validity * 0.75 - 0.1),
                    "high": min(1.0, validity * 0.75 + 0.1),
                },
            }
            for gamma, validity in ((0.1, 0.3), (0.5, 0.55), (1.0, 0.8))
        ],
    }
    return traces, store, proximal, audit


def test_build_frames_ties_A_queries_execution_and_uniform_replay_exactly() -> None:
    traces, store, proximal, audit = _synthetic_controller_data()
    frames = build_expansion_frames(
        traces,
        store,
        proximal,
        audit_results=audit,
        round_indices=(0, 1),
    )

    assert len(frames) == 2
    assert frames[0].A_observation_count == frames[0].query_count == 1
    assert frames[1].A_observation_count == 3
    assert frames[1].query_count == 2
    assert frames[1].positive_count == 1
    assert frames[1].backup_query_count == 1
    assert frames[1].backup_positive_count == 1
    assert frames[1].query_acceptance == 0.5
    assert frames[1].backup_acceptance == 1.0
    assert frames[1].fallback_count == 1
    assert frames[1].executed is not None
    assert frames[1].executed.source == "safemppi_backup"
    assert {plan.query_hash for plan in frames[1].replay} == {
        store.records[0].query_hash,
    }
    assert store.records[2].query_hash not in {
        plan.query_hash for plan in frames[1].replay
    }
    assert all(plan.safe for plan in frames[1].replay)
    assert {metric.gamma for metric in frames[1].audit} == {0.1, 0.5, 1.0}
    assert frames[1].proximal is not None
    assert frames[1].proximal.positive_coverage == 1.0
    np.testing.assert_allclose(
        np.sort(frames[-1].A_eigenvalues),
        np.sort(np.linalg.eigvalsh(store.uncertainty.A)),
    )


def test_saved_cli_format_and_ffmpeg_smoke_emit_required_outputs_and_labels(tmp_path: Path) -> None:
    traces, store, proximal, audit = _synthetic_controller_data()
    frames = build_expansion_frames(traces, store, proximal, audit_results=audit)
    scene = SceneSnapshot(
        obstacles=np.asarray([[2.5, 2.5, 1.2], [1.0, 4.0, 0.2]]),
        robot_radius=0.15,
        start=np.asarray([0.5, 0.5]),
        goal=np.asarray([4.5, 4.5]),
    )
    data_path = save_visualization_data(tmp_path / "data.json", scene, frames)
    loaded_scene, loaded_frames, metadata = load_visualization_data(data_path)
    assert metadata == {}
    np.testing.assert_array_equal(loaded_scene.obstacles, scene.obstacles)
    assert [frame.to_dict() for frame in loaded_frames] == [frame.to_dict() for frame in frames]

    output = tmp_path / "viz" / "active_expansion.mp4"
    preview = tmp_path / "viz" / "active_expansion.png"
    manifest = render_expansion_video(
        loaded_scene,
        loaded_frames,
        output,
        preview_png=preview,
        fps=2,
        seconds_per_event=0.5,
        dpi=32,
    )
    manifest_path = output.with_name("active_expansion_manifest.json")
    frame_log = output.with_name("active_expansion_frames.jsonl")
    assert output.stat().st_size > 1_000
    assert preview.stat().st_size > 1_000
    assert manifest_path.exists() and frame_log.exists()
    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["status"] == "PASS"
    assert saved_manifest["temperatures"] == {"expansion": 1.0, "independent_audit": 1.0}
    assert saved_manifest["colormaps"] == {"sigma": "viridis", "gamma": "plasma_trunc"}
    assert saved_manifest["replay_distribution"] == "uniform_positive_flow_query_ledger"
    assert saved_manifest["query_acceptance_scope"] == "FLOW_only"
    assert saved_manifest["final_frame"]["backup_query_count"] == 1
    assert saved_manifest["labels"] == RENDER_LABELS
    assert "ordinary T=1 flow candidates" in RENDER_LABELS["acquisition"]
    assert "Uniform positive flow-query replay" in RENDER_LABELS["replay"]
    assert manifest["final_frame"]["audit"]
    rows = [json.loads(line) for line in frame_log.read_text().splitlines()]
    assert len(rows) == 2
    # The certified backup is visible as the executed runtime action but is
    # excluded from CFM replay; it cannot become implicit expert distillation.
    assert len(rows[-1]["replay_hashes"]) == 1
    assert rows[-1]["executed_source"] == "safemppi_backup"


def test_frames_reject_implicit_audit_provenance_and_set_only_trace_matching() -> None:
    traces, store, proximal, audit = _synthetic_controller_data()
    missing = dict(audit)
    del missing["uncertainty_tilting"]
    with pytest.raises(ValueError, match="missing explicit 'uncertainty_tilting'"):
        build_expansion_frames(traces, store, proximal, audit_results=missing)
    with pytest.raises(ValueError, match="event-order identity mismatch"):
        build_expansion_frames(traces[::-1], store, proximal, audit_results=audit)


def test_minus_afe_and_minus_progress_semantics_survive_json(tmp_path: Path) -> None:
    traces, store, proximal, audit = _synthetic_controller_data()
    frames = build_expansion_frames(
        traces,
        store,
        proximal,
        audit_results=audit,
        acquisition_mode="uniform",
        progress_ranking=False,
        method_label="synthetic controls",
    )
    assert frames[-1].acquisition_mode == "uniform"
    assert not frames[-1].progress_ranking
    scene = SceneSnapshot(
        obstacles=np.asarray([[2.5, 2.5, 1.2]]),
        robot_radius=0.15,
        start=np.asarray([0.5, 0.5]),
        goal=np.asarray([4.5, 4.5]),
    )
    path = save_visualization_data(tmp_path / "controls.json", scene, frames)
    payload = json.loads(path.read_text())
    assert payload["semantics"]["acquisition_mode"] == "uniform"
    assert payload["semantics"]["progress_ranking"] is False
    assert payload["semantics"]["query_acceptance_denominator"].startswith("FLOW")
