#!/usr/bin/env python3
"""Stage 06: run the three matched, no-curriculum expansion controls.

Every arm starts from the same checkpoint and fixed audit bank.  Each
round/gamma/episode is capped by the corresponding selected Full episode's
realized control-decision count, bound through an explicit Full checkpoint.
All arms use temperature one for acquisition and independent validity audits;
temperature 0.5 remains a later visualization-only choice.  The
``minus_socp`` arm is explicitly offline and all reported validity still uses
the actual full SOCP.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import grid_hp_expt as HP

from .ablations import (
    AblationArm,
    AblationSpec,
    MatchedProtocol,
    ablation_spec,
    arm_manifest,
    assert_matched_protocols,
    eligibility_counts,
    training_view,
)
from .config import (
    AFEConfig,
    FeatureConfig,
    SamplingConfig,
    clean_method_absence_manifest,
)
from .controller import EpisodeResult, PlannedWindowAFEController
from .decision_budget import (
    build_usage,
    cap_lookup,
    load_full_reference,
    validate_reference_payload,
)
from .fallback import SafeMPPIBackup
from .policy import (
    FrozenFeatureModel,
    ledger_cfm_loss,
    model_state_hash,
    require_promoted_fresh_pretrain,
)
from .proximal_update import ProximalConfig, solve_proximal_update
from .scene import GAMMAS, make_ood_scene, verifier_spec_fingerprint
from .stage4_baseline import audit_model, load_audit_bank_artifact
from .stage5_expand import (
    _matrix_summary,
    _query_round_summary,
    _require_usable_proximal_solve,
    _round_checkpoint,
    _save_torch,
    _write_json,
    behavioral_gate,
    evaluate_ordinary,
)
from .store import VerificationStore
from .uncertainty import CumulativeLinearUncertainty


STAGE = Path(__file__).resolve().parent / "stage_results/06_matched_ablations"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _selected_without_socp(episodes: list[EpisodeResult]) -> int:
    return sum(
        trace.selected_query_hash is not None
        and trace.selected_actual_full_safe is False
        for episode in episodes
        for trace in episode.traces
    )


def _assert_offline_trace_integrity(
    episodes: list[EpisodeResult], store: VerificationStore, spec: AblationSpec,
) -> None:
    if spec.runtime_safety_claim:
        if any(not episode.runtime_safety_claim for episode in episodes):
            raise RuntimeError("certified control unexpectedly dropped its safety claim")
        return
    if any(episode.runtime_safety_claim for episode in episodes):
        raise RuntimeError("offline -SOCP episode carried a runtime-safety claim")
    by_hash = {record.query_hash: record for record in store.records}
    for episode in episodes:
        for trace in episode.traces:
            if trace.selected_query_hash is None or trace.selected_actual_full_safe is not False:
                continue
            record = by_hash[trace.selected_query_hash]
            if record.safety.socp_certified or record.executed:
                raise RuntimeError(
                    "offline SOCP-failing selection corrupted actual certificate telemetry"
                )


def _matched_protocol(args: argparse.Namespace) -> MatchedProtocol:
    return MatchedProtocol(
        seed=args.seed,
        candidate_count=args.candidate_count,
        verifier_budget=args.verifier_budget,
        fallback_verifier_budget=args.fallback_verifier_budget,
        beta=args.beta,
        backup_smooth_weight=args.backup_smooth_weight,
        backup_noise_var_mult=args.backup_noise_var_mult,
        backup_retreat_weight=args.backup_retreat_weight,
        rounds=args.rounds,
        episodes_per_gamma=args.episodes_per_gamma,
        episode_max_steps=args.episode_max_steps,
        expansion_temperature=1.0,
        nfe=args.nfe,
        ridge_lambda=args.ridge_lambda,
        prox_eta=args.prox_eta,
        learning_rate=args.learning_rate,
        microbatch=args.microbatch,
        solver_max_steps=args.solver_max_steps,
        solver_min_steps=args.solver_min_steps,
        update_norm_limit=args.update_norm_limit,
        relative_loss_tolerance=args.relative_loss_tolerance,
        gradient_tolerance=args.gradient_tolerance,
        audit_plans_per_context=args.audit_plans_per_context,
        audit_progress_threshold=args.audit_progress_threshold,
        eval_rollouts=args.eval_rollouts,
    )


def _run_arm(
    args: argparse.Namespace,
    spec: AblationSpec,
    protocol: MatchedProtocol,
    *,
    checkpoint_sha256: str,
    full_reference: dict[str, Any],
) -> dict[str, object]:
    arm_dir = args.outdir / spec.arm.value
    arm_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint_payload = HP.load_hp(args.checkpoint, device=device)
    source_model_hash = require_promoted_fresh_pretrain(model, checkpoint_payload)
    full_reference = validate_reference_payload(full_reference)
    if source_model_hash != full_reference["source_model_hash"]:
        raise RuntimeError(
            "control source model differs from the selected Full reference source"
        )
    frozen = FrozenFeatureModel.from_pretrained(model, s=0.9, expected_dim=32)
    frozen_hash = frozen.state_hash
    store = VerificationStore(CumulativeLinearUncertainty(lambda_=args.ridge_lambda))
    bank, bank_artifact = load_audit_bank_artifact(
        args.audit_bank, require_locked_provenance=True,
    )
    if bank.role != "round_monitoring":
        raise RuntimeError(
            "Stage 6 round-by-round ablations require an audit bank with "
            f"role='round_monitoring', got {bank.role!r}; sealed_final_test is "
            "reserved for the one-shot final evaluation"
        )
    env = make_ood_scene(radius=1.2)
    current_verifier_spec = verifier_spec_fingerprint(env, env.goal)
    bank_provenance = bank_artifact["source_provenance"]
    if (
        bank_provenance.get("purpose") != "round_monitoring"
        or bank_provenance.get("scene_verifier_spec_fingerprint")
        != current_verifier_spec
    ):
        raise RuntimeError(
            "round-monitoring bank provenance does not match its role/current verifier"
        )
    afe_config = AFEConfig(
        sampling=SamplingConfig(
            candidate_count=args.candidate_count,
            verifier_budget=args.verifier_budget,
            beta=args.beta,
            expansion_temperature=1.0,
            audit_temperature=1.0,
            visualization_temperature=0.5,
            nfe=args.nfe,
        ),
        features=FeatureConfig(
            representation_dim=32,
            feature_time=0.9,
            ridge_lambda=args.ridge_lambda,
        ),
    )
    controller = PlannedWindowAFEController(
        model,
        frozen,
        store,
        config=afe_config,
        backup=SafeMPPIBackup(
            smooth_weight=args.backup_smooth_weight,
            noise_var_mult=args.backup_noise_var_mult,
            retreat_weight=args.backup_retreat_weight,
        ),
        device=device,
        fallback_verifier_budget=args.fallback_verifier_budget,
        acquisition_mode=spec.acquisition_mode,
        progress_ranking=spec.progress_ranking,
        eligibility_mode=spec.eligibility_mode,
    )
    recipe = arm_manifest(spec, protocol) | {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_model_hash": source_model_hash,
        "audit_bank": str(args.audit_bank.resolve()),
        "audit_bank_file_sha256": _sha256(args.audit_bank),
        "audit_bank_fingerprint": bank.fingerprint,
        "audit_bank_role": bank.role,
        "audit_bank_artifact_fingerprint": bank_artifact["artifact_fingerprint"],
        "audit_bank_source_provenance_fingerprint": bank_artifact[
            "source_provenance_fingerprint"
        ],
        "verifier_spec_fingerprint": current_verifier_spec,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "beta": args.beta,
        "expansion_training_seed": args.seed,
        "backup_planner": {
            "smooth_weight": args.backup_smooth_weight,
            "noise_var_mult": args.backup_noise_var_mult,
            "retreat_weight": args.backup_retreat_weight,
        },
        "feature_time": 0.9,
        "frozen_feature_hash": frozen_hash,
        "gamma_distribution": "fixed uniform over all seven gammas; no schedule",
        "expansion_temperature": 1.0,
        "audit_temperature": 1.0,
        "visualization_temperature": 0.5,
        "progress_is_never_part_of_actual_safety_label": True,
        "uncertainty_tilting": spec.acquisition_mode == "afe",
        "ordinary_audit_untilted": True,
        "full_reference_decision_budget": full_reference,
        "control_decision_budget_rule": (
            "for every (round,gamma,episode), max_steps equals the selected "
            "Full episode's realized len(traces); the control may terminate earlier"
        ),
    }
    _write_json(arm_dir / "logs/recipe.json", recipe)

    baseline_audit = audit_model(
        model,
        env,
        bank,
        plans_per_context=args.audit_plans_per_context,
        seed=args.seed - 1,
        nfe=args.nfe,
        progress_threshold=args.audit_progress_threshold,
    )
    baseline_eval, baseline_rollouts = evaluate_ordinary(
        model,
        env,
        seed=args.seed - 10_000,
        per_gamma=args.eval_rollouts,
        nfe=args.nfe,
        temperature=1.0,
    )
    history: list[dict[str, object]] = [{
        "round": 0,
        # ``audit`` is the common artifact schema consumed by final_reports;
        # the explicit alias prevents anyone from mistaking the bounds-only
        # training rule for the validity label in the offline -SOCP arm.
        "audit": baseline_audit.to_dict(),
        "audit_actual_full_socp": baseline_audit.to_dict(),
        "ordinary_per_gamma": baseline_eval,
        "matrix": _matrix_summary(store),
        "query": None,
        "solver": None,
        "model_hash": source_model_hash,
        "runtime_safety_claim": spec.runtime_safety_claim,
    }]
    _save_torch(arm_dir / "data/round_000_bundle.pt", {
        "schema_version": "afe_matched_ablation_round_v1",
        "round": 0,
        "arm": spec.arm.value,
        "recipe": recipe,
        "episodes": [],
        "ordinary_rollouts": [asdict(row) for row in baseline_rollouts],
        "audit": baseline_audit.to_dict(),
        "audit_actual_full_socp": baseline_audit.to_dict(),
        "store_state": store.state_dict(),
    })
    _round_checkpoint(
        arm_dir / "checkpoints/round_000.pt",
        model, frozen, store, 0, recipe, history,
    )

    started = time.perf_counter()
    all_episodes: list[EpisodeResult] = []
    decision_caps = cap_lookup(full_reference)
    decision_usage: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        before_queries = store.query_count
        episodes: list[EpisodeResult] = []
        for gamma_index, gamma in enumerate(GAMMAS):
            for episode_index in range(args.episodes_per_gamma):
                episode_seed = (
                    args.seed
                    + round_index * 1_000_000
                    + gamma_index * 10_000
                    + episode_index
                )
                decision_cap = decision_caps[
                    (round_index, float(gamma), episode_index)
                ]
                episodes.append(controller.run_episode(
                    env,
                    gamma,
                    seed=episode_seed,
                    max_steps=decision_cap,
                    reach=0.20,
                ))
        all_episodes.extend(episodes)
        round_usage = build_usage(
            episodes,
            round_index=round_index,
            reference=full_reference,
            expected_seed_base=args.seed,
        )
        decision_usage.append(round_usage)
        _assert_offline_trace_integrity(episodes, store, spec)
        new_records = store.records[before_queries:]
        query_summary = _query_round_summary(episodes, store, before_queries)
        query_summary["actual_vs_training_eligibility"] = eligibility_counts(new_records, spec)
        query_summary["offline_selected_without_actual_socp"] = _selected_without_socp(episodes)
        query_summary["runtime_safety_claim"] = spec.runtime_safety_claim
        query_summary["full_reference_control_decision_budget"] = round_usage

        replay = training_view(store.records, spec)
        solver = solve_proximal_update(
            model,
            replay,
            ledger_cfm_loss,
            ProximalConfig(
                eta=args.prox_eta,
                learning_rate=args.learning_rate,
                batch_size=args.microbatch,
                max_steps=args.solver_max_steps,
                min_steps=args.solver_min_steps,
                update_norm_limit=args.update_norm_limit,
                relative_loss_tolerance=args.relative_loss_tolerance,
                gradient_tolerance=args.gradient_tolerance,
                tolerance_patience=2,
                seed=args.seed + round_index,
            ),
        )
        _require_usable_proximal_solve(
            solver,
            label=f"Stage 06 {spec.arm.value}",
            round_index=round_index,
            output_dir=arm_dir,
        )
        expected_training_count = int(
            eligibility_counts(store.records, spec)["training_eligible"]
        )
        if solver.positive_count != expected_training_count:
            raise RuntimeError(
                "uniform replay did not contain exactly the declared eligible rows: "
                f"solver={solver.positive_count}, expected={expected_training_count}"
            )
        if frozen.state_hash != frozen_hash:
            raise RuntimeError("frozen uncertainty representation changed")

        # This audit is identical in all arms, including offline -SOCP: it
        # always uses ordinary temperature-one samples and the actual full
        # strict-bounds AND SOCP label.
        audit = audit_model(
            model,
            env,
            bank,
            plans_per_context=args.audit_plans_per_context,
            seed=args.seed + round_index * 100,
            nfe=args.nfe,
            progress_threshold=args.audit_progress_threshold,
        )
        ordinary, ordinary_rollouts = evaluate_ordinary(
            model,
            env,
            seed=args.seed + round_index * 100_000,
            per_gamma=args.eval_rollouts,
            nfe=args.nfe,
            temperature=1.0,
        )
        gate_passed, gate_checks = behavioral_gate(ordinary)
        record = {
            "round": round_index,
            "query": query_summary,
            "solver": solver.to_dict(),
            "audit": audit.to_dict(),
            "audit_actual_full_socp": audit.to_dict(),
            "ordinary_per_gamma": ordinary,
            "tuning_gate": {
                "passed": gate_passed,
                "target_nonzero_sr": gate_checks,
                "role": "checkpoint-selection heuristic only; not final evidence",
                "rollouts_per_gamma": args.eval_rollouts,
            },
            "matrix": _matrix_summary(store),
            "model_hash": model_state_hash(model),
            "runtime_safety_claim": spec.runtime_safety_claim,
            "wall_seconds_total": time.perf_counter() - started,
        }
        history.append(record)
        _save_torch(arm_dir / f"data/round_{round_index:03d}_bundle.pt", {
            "schema_version": "afe_matched_ablation_round_v1",
            "round": round_index,
            "arm": spec.arm.value,
            "recipe": recipe,
            "episodes": episodes,
            "ordinary_rollouts": [asdict(row) for row in ordinary_rollouts],
            "audit": audit.to_dict(),
            "audit_actual_full_socp": audit.to_dict(),
            "query_summary": query_summary,
            "solver": solver.to_dict(),
            "matrix": record["matrix"],
            "store_state": store.state_dict(),
        })
        _round_checkpoint(
            arm_dir / f"checkpoints/round_{round_index:03d}.pt",
            model, frozen, store, round_index, recipe, history,
        )
        _write_json(arm_dir / "logs/history.json", history)
        print(
            f"[{spec.arm.value} round {round_index}/{args.rounds}] "
            f"q={query_summary['new_verifier_calls']} "
            f"actual-acc={query_summary['query_acceptance']:.3f} "
            f"train={solver.positive_count} "
            f"update={solver.final_update_norm:.4g}",
            flush=True,
        )

    final_counts = eligibility_counts(store.records, spec)
    report = {
        "status": "MATCHED_ABLATION_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": spec.arm.value,
        "recipe": recipe,
        "source_checkpoint_config": checkpoint_payload.get("config", {}),
        "rounds_completed": len(history) - 1,
        "query_accounting": final_counts,
        "uncertainty_observations": store.uncertainty.count,
        "ledger_queries": store.query_count,
        "exact_uncertainty_accounting": store.uncertainty.count == store.query_count,
        "offline_selected_without_actual_socp": _selected_without_socp(all_episodes),
        "runtime_safety_claim": spec.runtime_safety_claim,
        "full_reference_decision_budget": full_reference,
        "realized_control_decision_usage": decision_usage,
        "all_control_cells_within_full_realized_bound": all(
            bool(row["all_cells_within_full_cap"]) for row in decision_usage
        ),
        "final": history[-1],
        "validity_reporting": {
            "query_actual_full_acceptance": (
                "actual strict-bounds AND SOCP among all queried plans"
            ),
            "training_eligibility": spec.replay_eligibility,
            "model_validity": (
                "independent, uncertainty-untilted, temperature-one, fixed-context "
                "actual full-SOCP audit with per-gamma confidence intervals"
            ),
            "performance_validity": (
                "same independent audit with progress threshold reported separately"
            ),
            "runtime": (
                "certified same-verifier execution/fallback"
                if spec.runtime_safety_claim
                else "OFFLINE ONLY; no runtime certificate claim"
            ),
        },
    }
    if store.query_count != store.uncertainty.count:
        raise RuntimeError("query ledger and uncertainty observations diverged")
    _write_json(arm_dir / "REPORT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--full-reference-dir",
        type=Path,
        required=True,
        help=(
            "selected completed Stage-05 Full run; its exact per-episode "
            "len(traces) values cap the corresponding control episodes"
        ),
    )
    parser.add_argument(
        "--full-reference-checkpoint",
        type=Path,
        required=True,
        help=(
            "explicit promoted Full round checkpoint; only bundles through its "
            "embedded round are authoritative, even if the directory has later artifacts"
        ),
    )
    parser.add_argument("--audit-bank", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=STAGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=105000)
    parser.add_argument(
        "--arm",
        choices=("all",) + tuple(arm.value for arm in AblationArm),
        default="all",
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--episodes-per-gamma", type=int, default=1)
    parser.add_argument("--episode-max-steps", type=int, default=240)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--verifier-budget", type=int, default=8)
    parser.add_argument("--fallback-verifier-budget", type=int, default=8)
    parser.add_argument("--backup-smooth-weight", type=float, default=8.0)
    parser.add_argument("--backup-noise-var-mult", type=float, default=3.0)
    parser.add_argument("--backup-retreat-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--prox-eta", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--solver-max-steps", type=int, default=12)
    parser.add_argument("--solver-min-steps", type=int, default=2)
    parser.add_argument("--update-norm-limit", type=float, default=0.12)
    parser.add_argument("--relative-loss-tolerance", type=float, default=2e-3)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-5)
    parser.add_argument("--audit-plans-per-context", type=int, default=4)
    parser.add_argument("--audit-progress-threshold", type=float, default=0.10)
    parser.add_argument("--eval-rollouts", type=int, default=6)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    if not args.checkpoint.is_file() or not args.audit_bank.is_file():
        raise FileNotFoundError("checkpoint and fixed audit bank must exist")
    args.outdir.mkdir(parents=True, exist_ok=True)
    protocol = _matched_protocol(args)
    selected = (
        tuple(AblationArm)
        if args.arm == "all"
        else (AblationArm(args.arm),)
    )
    # Construct one immutable value per arm and assert equality rather than
    # merely trusting that a shared argparse namespace stayed unchanged.
    assert_matched_protocols([protocol for _arm in selected])
    checkpoint_hash = _sha256(args.checkpoint)
    full_reference = load_full_reference(
        args.full_reference_dir,
        final_checkpoint=args.full_reference_checkpoint,
        expected_protocol=protocol.__dict__,
        expected_source_checkpoint_sha256=checkpoint_hash,
    )
    reports = {
        arm.value: _run_arm(
            args,
            ablation_spec(arm),
            protocol,
            checkpoint_sha256=checkpoint_hash,
            full_reference=full_reference,
        )
        for arm in selected
    }
    source_model_hashes = {
        str(report["recipe"]["source_model_hash"])
        for report in reports.values()
    }
    audit_fingerprints = {
        str(report["recipe"]["audit_bank_fingerprint"])
        for report in reports.values()
    }
    if len(source_model_hashes) != 1 or len(audit_fingerprints) != 1:
        raise RuntimeError("ablation arms did not share one checkpoint and audit bank")
    actual_counts = {
        arm: int(report["ledger_queries"])
        for arm, report in reports.items()
    }
    combined = {
        "status": "MATCHED_ABLATIONS_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_hash,
        "audit_bank": str(args.audit_bank.resolve()),
        "source_model_hash": next(iter(source_model_hashes)),
        "audit_bank_fingerprint": next(iter(audit_fingerprints)),
        "full_reference_decision_budget": full_reference,
        "full_reference_decision_budget_fingerprint": full_reference["fingerprint"],
        "declared_protocol_identical": True,
        "declared_protocol": protocol.__dict__,
        "actual_query_counts": actual_counts,
        "actual_query_counts_equal": len(set(actual_counts.values())) <= 1,
        "actual_count_note": (
            "Every control cell is capped by the corresponding selected Full "
            "episode's realized control-decision count. A control may terminate "
            "earlier, so aggregate query counts remain reported rather than forced equal."
        ),
        "arms": reports,
    }
    _write_json(args.outdir / "REPORT.json", combined)
    print(json.dumps(combined, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
