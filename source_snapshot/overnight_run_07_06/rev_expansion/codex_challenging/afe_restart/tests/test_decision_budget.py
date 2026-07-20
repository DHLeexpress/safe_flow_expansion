from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from afe_restart.ablations import MatchedProtocol
from afe_restart.config import clean_method_absence_manifest
from afe_restart.decision_budget import (
    DecisionBudgetError,
    build_usage,
    load_full_reference,
    validate_reference_payload,
    validate_usage,
)
from afe_restart.scene import GAMMAS
from afe_restart.stage5_expand import CHECKPOINT_SCHEMA, FULL_REPLAY_DESCRIPTION


def _protocol() -> dict:
    return MatchedProtocol(
        seed=5,
        candidate_count=4,
        verifier_budget=2,
        fallback_verifier_budget=1,
        beta=0.2,
        backup_smooth_weight=8.0,
        backup_noise_var_mult=3.0,
        backup_retreat_weight=1.0,
        rounds=2,
        episodes_per_gamma=2,
        episode_max_steps=12,
        expansion_temperature=1.0,
        nfe=1,
        ridge_lambda=0.01,
        prox_eta=0.1,
        learning_rate=1e-3,
        microbatch=2,
        solver_max_steps=2,
        solver_min_steps=1,
        update_norm_limit=0.2,
        relative_loss_tolerance=0.01,
        gradient_tolerance=1e-5,
        audit_plans_per_context=2,
        audit_progress_threshold=0.1,
        eval_rollouts=2,
    ).__dict__


def _recipe(protocol: dict) -> dict:
    return {
        "method": "planned-window AFE",
        "arm": "full",
        "acquisition_mode": "afe",
        "progress_ranking": True,
        "eligibility_mode": "full",
        "replay_eligibility": "full_safe",
        "runtime_safety_claim": True,
        "uncertainty_tilting": True,
        "ordinary_audit_untilted": True,
        "legacy_mechanisms": clean_method_absence_manifest(),
        "replay": FULL_REPLAY_DESCRIPTION,
        "matched_protocol": protocol,
        "source_checkpoint_sha256": "a" * 64,
        "source_model_hash": "b" * 64,
    }


def _episodes(protocol: dict, round_index: int, *, trim: int = 0) -> list[dict]:
    rows = []
    for gamma_index, gamma in enumerate(GAMMAS):
        for episode_index in range(protocol["episodes_per_gamma"]):
            count = round_index + gamma_index + episode_index + 1 - trim
            rows.append({
                "gamma": float(gamma),
                "seed": (
                    protocol["seed"] + round_index * 1_000_000
                    + gamma_index * 10_000 + episode_index
                ),
                "traces": [object()] * count,
            })
    return rows


def _write_full(root: Path) -> tuple[dict, dict]:
    protocol = _protocol()
    recipe = _recipe(protocol)
    (root / "data").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "checkpoints").mkdir()
    (root / "logs/recipe.json").write_text(json.dumps(recipe))
    torch.save(
        {"round": 0, "recipe": recipe, "episodes": []},
        root / "data/round_000_bundle.pt",
    )
    for round_index in range(1, 3):
        torch.save(
            {
                "round": round_index,
                "recipe": recipe,
                "episodes": _episodes(protocol, round_index),
            },
            root / f"data/round_{round_index:03d}_bundle.pt",
        )
    state = {"weight": torch.tensor([1.0])}
    # Match the production state hashing algorithm.
    import hashlib
    digest = hashlib.sha256()
    tensor = state["weight"].contiguous()
    digest.update(b"weight")
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    model_hash = digest.hexdigest()
    torch.save(
        {
            "afe_schema": CHECKPOINT_SCHEMA,
            "round": 2,
            "recipe": recipe,
            "state_dict": state,
            "current_model_hash": model_hash,
        },
        root / "checkpoints/round_002.pt",
    )
    return protocol, recipe


def test_full_reference_derives_exact_cell_caps_and_hash_locks_them(tmp_path: Path) -> None:
    protocol, _recipe_value = _write_full(tmp_path / "full")
    reference = load_full_reference(
        tmp_path / "full",
        final_checkpoint=tmp_path / "full/checkpoints/round_002.pt",
        expected_protocol=protocol,
        expected_source_checkpoint_sha256="a" * 64,
    )
    assert len(reference["caps"]) == 2 * len(GAMMAS) * 2
    assert reference["caps"][0] == {
        "round": 1,
        "gamma": 0.1,
        "episode_index": 0,
        "max_control_decisions": 2,
    }
    assert validate_reference_payload(reference)["fingerprint"] == reference["fingerprint"]

    tampered = copy.deepcopy(reference)
    tampered["caps"][0]["max_control_decisions"] += 1
    with pytest.raises(DecisionBudgetError, match="fingerprint mismatch"):
        validate_reference_payload(tampered)


def test_control_usage_is_cellwise_and_may_stop_earlier_but_not_later(tmp_path: Path) -> None:
    protocol, _recipe_value = _write_full(tmp_path / "full")
    reference = load_full_reference(
        tmp_path / "full",
        final_checkpoint=tmp_path / "full/checkpoints/round_002.pt",
        expected_protocol=protocol,
        expected_source_checkpoint_sha256="a" * 64,
    )
    shorter = _episodes(protocol, 1, trim=1)
    usage = build_usage(
        shorter,
        round_index=1,
        reference=reference,
        expected_seed_base=protocol["seed"],
    )
    assert usage["all_cells_within_full_cap"]
    assert usage["realized_control_decisions"] < usage["full_control_decision_cap"]
    assert validate_usage(usage, reference)["fingerprint"] == usage["fingerprint"]

    longer = _episodes(protocol, 1)
    longer[0]["traces"].append(object())
    with pytest.raises(DecisionBudgetError, match="beyond Full cap"):
        build_usage(
            longer,
            round_index=1,
            reference=reference,
            expected_seed_base=protocol["seed"],
        )


def test_full_reference_rejects_gamma_allocation_or_protocol_drift(tmp_path: Path) -> None:
    protocol, _recipe_value = _write_full(tmp_path / "full")
    round_one = tmp_path / "full/data/round_001_bundle.pt"
    payload = torch.load(round_one, weights_only=False)
    payload["episodes"].pop()
    torch.save(payload, round_one)
    with pytest.raises(DecisionBudgetError, match="exactly 2 episodes per gamma"):
        load_full_reference(
            tmp_path / "full",
            final_checkpoint=tmp_path / "full/checkpoints/round_002.pt",
            expected_protocol=protocol,
            expected_source_checkpoint_sha256="a" * 64,
        )


def test_explicit_full_checkpoint_ignores_only_strictly_later_artifacts(tmp_path: Path) -> None:
    protocol, recipe = _write_full(tmp_path / "full")
    later_recipe = copy.deepcopy(recipe)
    later_recipe["matched_protocol"]["rounds"] = 10
    (tmp_path / "full/logs/recipe.json").write_text(json.dumps(later_recipe))
    torch.save(
        {"round": 3, "recipe": later_recipe, "episodes": []},
        tmp_path / "full/data/round_003_bundle.pt",
    )
    reference = load_full_reference(
        tmp_path / "full",
        final_checkpoint=tmp_path / "full/checkpoints/round_002.pt",
        expected_protocol=protocol,
        expected_source_checkpoint_sha256="a" * 64,
    )
    assert reference["final_round"] == 2
    assert all(int(row["round"]) <= 2 for row in reference["caps"])

    protocol_drift = copy.deepcopy(protocol)
    protocol_drift["beta"] = 0.5
    with pytest.raises(DecisionBudgetError, match="matched protocols differ"):
        load_full_reference(
            tmp_path / "full",
            final_checkpoint=tmp_path / "full/checkpoints/round_002.pt",
            expected_protocol=protocol_drift,
            expected_source_checkpoint_sha256="a" * 64,
        )
