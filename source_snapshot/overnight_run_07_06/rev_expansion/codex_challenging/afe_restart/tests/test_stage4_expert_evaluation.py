from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from afe_restart.audit import ImmutableContextBank
from afe_restart.evaluation import RolloutResult
from afe_restart.stage4_baseline import (
    AUDIT_BANK_SCHEMA,
    BASELINE_SCHEMA,
    EXPERT_SCHEMA,
    AuditStateContext,
    SafeMPPIExpertSettings,
    evaluate_fresh_ood_expert,
    load_audit_bank_artifact,
    make_baseline_rollout_bundle,
    save_audit_bank,
    save_expert_evaluation,
)
import afe_restart.stage4_baseline as stage4


def _fake_expert_rollout(*, env, gamma, seed, device, config):
    del env, device, config
    attempt = int(seed) % 4
    if attempt % 2 == 0:
        path = np.asarray(
            ((0.5, 0.5), (1.0, 2.2), (2.3, 4.0), (4.5, 4.5)),
            dtype=np.float32,
        )
    else:
        path = np.asarray(
            ((0.5, 0.5), (2.2, 1.0), (4.0, 2.3), (4.5, 4.5)),
            dtype=np.float32,
        )
    actions = np.zeros((len(path) - 1, 2), dtype=np.float32)
    success = attempt == 0
    collision = attempt == 1
    in_bounds = attempt != 2
    dead_reason = (
        None
        if success
        else "collision_after_verified_action"
        if collision
        else "out_of_bounds_after_verified_action"
        if not in_bounds
        else "timeout"
    )
    return {
        "gamma": float(gamma),
        "seed": int(seed),
        "path": path,
        "executed_actions": actions,
        "success": success,
        "reached": success,
        "collision": collision,
        "in_bounds": in_bounds,
        "dead_reason": dead_reason,
        "min_clearance_m": -0.1 if collision else 0.2,
        "path_length_m": 6.2,
        "wall_seconds": 0.25 + 0.01 * attempt,
    }


def test_fresh_expert_uses_fixed_budget_per_gamma_and_retains_failures() -> None:
    env = SimpleNamespace(dt=0.1)
    observed_settings = []

    def capture(**kwargs):
        config = kwargs["config"]
        observed_settings.append((
            config.smooth_weight, config.noise_var_mult, config.retreat_weight,
        ))
        return _fake_expert_rollout(**kwargs)

    rows, per_gamma = evaluate_fresh_ood_expert(
        env,
        device=torch.device("cpu"),
        gammas=(0.1, 0.5),
        attempts_per_gamma=4,
        seed0=100,
        planner_settings=SafeMPPIExpertSettings(17.0, 1.75, 0.4),
        rollout_fn=capture,
    )
    assert len(rows) == 8
    assert len({row["seed"] for row in rows}) == 8
    assert all(row["path"].dtype == np.float32 for row in rows)
    assert all(row["actions"].dtype == np.float32 for row in rows)
    assert all(len(row["path"]) == len(row["actions"]) + 1 for row in rows)
    assert {row["attempt_index"] for row in rows if row["gamma"] == 0.1} == set(range(4))
    for gamma in ("0.1", "0.5"):
        metrics = per_gamma[gamma]
        assert metrics["n"] == 4
        assert metrics["successes"] == 1
        assert metrics["success_rate"] == 0.25
        assert metrics["collisions"] == 1
        assert metrics["out_of_bounds_count"] == 1
        assert metrics["timeouts"] == 1
        interval = metrics["success_rate_wilson_95"]
        assert 0.0 <= interval["low"] <= 0.25 <= interval["high"] <= 1.0
    assert observed_settings == [(17.0, 1.75, 0.4)] * 8


def test_expert_artifact_saves_normalized_rows_and_per_gamma_metrics(
    tmp_path: Path,
) -> None:
    rows, per_gamma = evaluate_fresh_ood_expert(
        SimpleNamespace(dt=0.1),
        device=torch.device("cpu"),
        gammas=(0.2,),
        attempts_per_gamma=4,
        seed0=200,
        rollout_fn=_fake_expert_rollout,
    )
    output = tmp_path / "expert.pt"
    save_expert_evaluation(
        output,
        rows=rows,
        per_gamma=per_gamma,
        attempts_per_gamma=4,
        seed0=200,
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == EXPERT_SCHEMA
    assert payload["attempt_policy"].startswith("fixed count")
    assert payload["expert_planner"] == {
        "smooth_weight": 8.0,
        "noise_var_mult": 3.0,
        "retreat_weight": 1.0,
    }
    assert len(payload["rollouts"]) == 4
    assert payload["per_gamma"]["0.2"]["n"] == 4
    required = {
        "path",
        "actions",
        "collision",
        "out_of_bounds",
        "timeout",
        "min_clearance_m",
        "time_to_goal_s",
        "wall_time_s",
        "detour_mode",
        "seed",
    }
    assert required <= set(payload["rollouts"][0])


def test_stage4_baseline_bundle_declares_ordinary_temperature_one() -> None:
    rollout = RolloutResult(
        gamma=0.1,
        seed=1,
        temperature=1.0,
        states=np.zeros((2, 4), dtype=np.float32),
        actions=np.zeros((1, 2), dtype=np.float32),
        reached=False,
        collision=False,
        out_of_bounds=False,
        timeout=True,
        min_clearance_m=0.2,
        path_length_m=0.0,
        time_to_goal_s=None,
        detour_mode="unresolved",
    )
    bundle = make_baseline_rollout_bundle(
        rollouts=(rollout,),
        expert_rows=(),
        expert_per_gamma={},
        expert_artifact=Path("expert.pt"),
    )
    assert bundle["schema_version"] == BASELINE_SCHEMA
    assert bundle["temperature"] == 1.0
    assert bundle["ordinary_flow_temperature"] == 1.0
    assert bundle["ordinary_flow_evaluation"] == {
        "temperature": 1.0,
        "sampling_distribution": "ordinary conditional flow",
        "uncertainty_tilting": False,
        "safety_filter": False,
    }

    wrong_temperature = RolloutResult(
        **{**rollout.__dict__, "temperature": 0.5}
    )
    with pytest.raises(ValueError, match="only ordinary T=1"):
        make_baseline_rollout_bundle(
            rollouts=(wrong_temperature,),
            expert_rows=(),
            expert_per_gamma={},
            expert_artifact=Path("expert.pt"),
        )


def test_audit_bank_locks_and_validates_expert_source_provenance(tmp_path: Path) -> None:
    bank = ImmutableContextBank([
        AuditStateContext(
            state=np.zeros(4, dtype=np.float32),
            executed_history=np.zeros((0, 2), dtype=np.float32),
            expert_seed=12,
            expert_mode="upper-left",
            source_step=3,
        )
    ], role="sealed_final_test")
    provenance = {
        "schema_version": "afe_audit_bank_source_provenance_v1",
        "expert_planner": SafeMPPIExpertSettings(19.0, 2.25, 0.3).to_dict(),
        "seed0": 12,
    }
    path = tmp_path / "sealed.pt"
    metadata = save_audit_bank(bank, path, provenance=provenance)
    assert metadata["schema_version"] == AUDIT_BANK_SCHEMA
    restored, restored_metadata = load_audit_bank_artifact(
        path, require_locked_provenance=True,
    )
    assert restored.fingerprint == bank.fingerprint
    assert restored_metadata["source_provenance"] == provenance
    assert restored_metadata["artifact_fingerprint"] == metadata["artifact_fingerprint"]

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["source_provenance"]["expert_planner"]["smooth_weight"] = 20.0
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="provenance fingerprint"):
        load_audit_bank_artifact(path, require_locked_provenance=True)


def test_stage4_main_requires_promoted_stage3_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        sys, "argv", ["stage4", "--checkpoint", str(checkpoint), "--device", "cpu"]
    )
    monkeypatch.setattr(stage4, "make_ood_scene", lambda radius: object())
    monkeypatch.setattr(stage4.HP, "load_hp", lambda *_args, **_kwargs: (object(), {}))

    def reject(_model, _payload):
        raise RuntimeError("promotion gate sentinel")

    monkeypatch.setattr(stage4, "require_promoted_fresh_pretrain", reject)
    with pytest.raises(RuntimeError, match="promotion gate sentinel"):
        stage4.main()
