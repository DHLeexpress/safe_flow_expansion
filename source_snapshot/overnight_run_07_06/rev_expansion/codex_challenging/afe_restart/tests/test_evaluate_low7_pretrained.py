from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from afe_restart import evaluate_low7_pretrained as evaluation
from afe_restart.policy import model_state_hash


def _candidate_checkpoint(path: Path, *, fixed_goal_grid: bool = False) -> str:
    policy = evaluation.HP.GridHPFlowPolicy(
        repr_dim=32,
        grid_hw=(32, 32),
        trunk_hidden=(160, 96),
        enc_depth=3,
        raw_condition_dim=7,
        conditioning_schema="low7_closest_boundary",
    )
    evaluation.HP.save_hp(
        policy,
        path,
        extra={
            "stage_schema": evaluation.CHECKPOINT_STAGE_SCHEMA,
            "fresh_from_scratch": True,
            "endpoint_free": True,
            "domain_randomized_start_goal": not fixed_goal_grid,
            "domain_randomized_start": True,
            "fixed_goal": ([4.7, 4.7] if fixed_goal_grid else None),
            "zero_initial_velocity": fixed_goal_grid,
            "diagonal_start_exclusion": False,
            "encoder_trainable_during_pretraining": True,
            "expansion_promotion": False,
            "source_manifest": "/sealed/low7/manifest.json",
            "source_query_hash_digest": "a" * 64,
            "model_state_sha256": model_state_hash(policy),
            "best_epoch": 12,
            "best_validation_cfm": 0.25,
        },
    )
    return evaluation.sha256_file(path)


def test_strict_candidate_loader_accepts_only_low7_unpromoted_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checksum = _candidate_checkpoint(checkpoint)

    policy, contract = evaluation.load_low7_candidate(checkpoint, checksum, "cpu")

    assert policy.ctx_dim == 39
    assert policy.trunk[0].in_features == 91
    assert contract["file_sha256"] == checksum
    assert contract["parameter_count"] == evaluation.EXPECTED_PARAMETER_COUNT
    assert contract["expansion_promotion"] is False

    with pytest.raises(evaluation.CheckpointContractError, match="caller-declared"):
        evaluation.load_low7_candidate(checkpoint, "0" * 64, "cpu")


def test_candidate_loader_accepts_fixed_goal_full_grid_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "fixed_grid_candidate.pt"
    checksum = _candidate_checkpoint(checkpoint, fixed_goal_grid=True)

    _policy, contract = evaluation.load_low7_candidate(checkpoint, checksum, "cpu")

    assert contract["fixed_goal_grid"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("expansion_promotion", True, "expansion_promotion"),
        ("config.schema_version", "w8sg-hp-v2-low5-only", "schema_version"),
    ),
)
def test_strict_candidate_loader_rejects_contract_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    _candidate_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if field.startswith("config."):
        payload["config"][field.split(".", 1)[1]] = value
    else:
        payload[field] = value
    torch.save(payload, checkpoint)

    with pytest.raises(evaluation.CheckpointContractError, match=message):
        evaluation.load_low7_candidate(
            checkpoint, evaluation.sha256_file(checkpoint), "cpu"
        )


def test_declared_scenes_have_canonical_endpoints_and_exact_ood_geometry() -> None:
    obstacle_counts = {}
    for name in evaluation.SCENE_NAMES:
        env = evaluation.build_scene(evaluation.get_scene_profile(name))
        snapshot = evaluation.validate_scene_contract(name, env)
        obstacle_counts[name] = len(snapshot["obstacles"])
        np.testing.assert_allclose(snapshot["start_state"], (0.3, 0.3, 0.0, 0.0))
        np.testing.assert_allclose(snapshot["goal"], (4.7, 4.7))

    assert obstacle_counts["low7_id_canonical_v1"] == 72
    assert obstacle_counts["low7_radius1_canonical_v1"] == 69
    assert obstacle_counts["low7_radius03_canonical_v1"] == 72


class _ZeroPolicy:
    d = 20

    def __init__(self) -> None:
        self.conditions: list[torch.Tensor] = []

    def ctx_from(
        self, grid: torch.Tensor, condition: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        assert grid.shape[1:] == (3, 32, 32)
        assert history.shape[1:] == (16, 2)
        assert condition.shape[1] == 7
        self.conditions.append(condition.detach().cpu().clone())
        return torch.cat(
            (condition, torch.zeros(len(condition), 32, device=condition.device)), dim=1
        )

    def sample(
        self,
        n: int,
        context: torch.Tensor,
        *,
        nfe: int,
        temp: float,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        assert context.shape == (n, 39)
        assert initial_noise.shape == (n, 20)
        return torch.zeros(n, 10, 2, device=context.device)


def test_raw_rollout_uses_low7_gamma_last_and_never_calls_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_verifier(*_args, **_kwargs):
        raise AssertionError("verifier must not participate in raw rollout generation")

    monkeypatch.setattr(evaluation, "verify_full_plan", forbidden_verifier)
    env = evaluation.build_scene(
        evaluation.get_scene_profile("low7_id_canonical_v1")
    )
    policy = _ZeroPolicy()

    episodes, plans = evaluation.run_raw_rollouts(
        policy,
        env,
        "low7_id_canonical_v1",
        m=2,
        gammas=(0.1, 0.7),
        horizon=2,
        device="cpu",
    )

    assert len(episodes) == 4
    assert len(plans) == 8
    assert {episode["status"] for episode in episodes} == {"timeout"}
    assert len(policy.conditions) == 2
    for conditions in policy.conditions:
        torch.testing.assert_close(
            conditions[:, -1], torch.tensor((0.1, 0.1, 0.7, 0.7))
        )
    assert plans[0]["seed"] == evaluation.raw_noise_seed(0.1, 0, 0)


def test_seed_stream_is_fixed_and_plan_validity_matches_existing_progress_bar() -> None:
    assert evaluation.raw_noise_seed(0.5, 3, 7) == evaluation.raw_noise_seed(0.5, 3, 7)
    assert evaluation.raw_noise_seed(0.5, 3, 7) != evaluation.raw_noise_seed(0.5, 3, 8)

    def result(*, safe: bool, progress: float, socp_ok: bool = True):
        return SimpleNamespace(
            safe=safe,
            in_bounds=True,
            socp_ok=socp_ok,
            bounds_margin_m=0.1,
            physical_clearance_m=0.2,
            face_margin_m=0.2 if socp_ok else -float("inf"),
            certificate_residual=0.0 if socp_ok else -0.1,
            progress_m=progress,
            start_goal_distance_m=1.0,
        )

    safe_and_progressing = evaluation.plan_validity_from_verification(
        result(safe=True, progress=0.10)
    )
    safe_but_stalled = evaluation.plan_validity_from_verification(
        result(safe=True, progress=0.099)
    )
    unsafe = evaluation.plan_validity_from_verification(
        result(safe=False, progress=1.0, socp_ok=False)
    )

    assert safe_and_progressing["v_safe"] is True
    assert safe_and_progressing["v_full"] is True
    assert safe_but_stalled["v_safe"] is True
    assert safe_but_stalled["v_full"] is False
    assert unsafe["v_safe"] is False
    assert unsafe["v_full"] is False


def test_posthoc_worker_matches_training_verifier_and_uses_strict_bounds() -> None:
    scene_name = "low7_id_canonical_v1"
    env = evaluation.build_scene(evaluation.get_scene_profile(scene_name))
    state = env.x0.detach().cpu().numpy().astype(np.float64)
    plan = np.zeros((10, 2), dtype=np.float32)
    direct = evaluation.verify_full_plan(
        state, plan, env, 0.5, goal=env.goal.detach().cpu().numpy()
    )
    replay = evaluation._verify_plan_chunk((evaluation.N_THETA, [(scene_name, state, plan, 0.5)]))[0]
    assert replay["v_safe"] is direct.safe
    assert replay["bounds_margin"] == direct.bounds_margin_m
    assert replay["residual"] == direct.certificate_residual

    episode = {
        "path": np.asarray([[5.01, 2.5]], dtype=np.float64),
        "status": "timeout",
    }
    assert evaluation.trajectory_metrics(episode, env)["oob"] is True
