from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import grid_hp_expt as HP

from afe_restart.dynamics import step_state
from afe_restart.policy import model_state_hash
from afe_restart.scene import GIANT_CENTER, context_from_state, make_ood_scene
from afe_restart.stage4b_mizuta import (
    GALLERY_TEMPERATURE,
    LOW_GUIDANCE_SWEEP,
    ROLLOUT_SCHEMA,
    SCIENTIFIC_TEMPERATURE,
    MizutaConfig,
    _require_clean_output_dir,
    algorithm_contract,
    build_rollout_artifact,
    cbf_reward,
    collision_radii,
    describe_ood_scene,
    double_integrator_rollout,
    guided_generate,
    load_clean_stage3_checkpoint,
    make_parser,
    stage_cost_batch,
    transition_clearance,
)
import afe_restart.stage4b_mizuta as mizuta_module


def test_restart_ood_scene_has_exact_giant_and_per_obstacle_radii() -> None:
    env = make_ood_scene(radius=1.2)
    scene = describe_ood_scene(env)
    obstacles = env.obstacles.detach().cpu().numpy()
    giant = np.flatnonzero(
        np.linalg.norm(obstacles[:, :2] - GIANT_CENTER[None], axis=1) < 1.0e-7
    )
    assert len(giant) == 1
    assert scene["giant_index"] == int(giant[0])
    assert scene["giant_radius"] == pytest.approx(1.2)
    assert scene["per_obstacle_radius_model"] is True
    assert scene["start"] == pytest.approx([0.5, 0.5])
    assert scene["goal"] == pytest.approx([4.5, 4.5])

    radii = collision_radii(env, margin=0.05)
    assert radii.shape == (len(obstacles),)
    assert radii[int(giant[0])] == pytest.approx(1.2 + float(env.r_robot) + 0.05)
    small = np.flatnonzero(np.isclose(obstacles[:, 2], 0.2))[0]
    assert radii[int(giant[0])] - radii[int(small)] == pytest.approx(1.0)


def test_cost_and_cbf_use_the_giant_radius_without_mean_collapse() -> None:
    config = MizutaConfig(
        "unit", safe_coef=0.02, collision_weight=2.0,
        n_samples=4, n_elite=2, n_copies=2,
    )
    positions = torch.tensor([[[2.5, 3.6]]], dtype=torch.float32)
    velocities = torch.zeros_like(positions)
    controls = torch.zeros_like(positions)
    goal = torch.tensor([4.5, 4.5], dtype=torch.float32)
    obstacle_xy = torch.tensor([[0.0, 0.0], [2.5, 2.5]], dtype=torch.float32)
    per_obstacle = torch.tensor([0.25, 1.25], dtype=torch.float32)
    collapsed = torch.tensor([0.25, 0.25], dtype=torch.float32)

    true_cost = stage_cost_batch(
        positions, controls, goal, obstacle_xy, per_obstacle, config
    )
    collapsed_cost = stage_cost_batch(
        positions, controls, goal, obstacle_xy, collapsed, config
    )
    assert float(true_cost) > float(collapsed_cost) + 1.0

    true_reward = cbf_reward(
        positions, velocities, obstacle_xy, per_obstacle, config
    )
    collapsed_reward = cbf_reward(
        positions, velocities, obstacle_xy, collapsed, config
    )
    assert float(true_reward) < float(collapsed_reward)
    with pytest.raises(ValueError, match="one collision radius"):
        stage_cost_batch(
            positions, controls, goal, obstacle_xy, torch.tensor([0.25]), config
        )


def test_differentiable_rollout_matches_restart_double_integrator() -> None:
    state = np.asarray((0.5, 0.6, 0.2, -0.1), dtype=np.float64)
    controls = torch.tensor(
        [[[0.4, -0.2], [-0.1, 0.3], [0.0, 0.2]]], dtype=torch.float64
    )
    positions, velocities = double_integrator_rollout(state, controls, dt=0.1)
    expected = state.copy()
    for index, action in enumerate(controls[0].numpy()):
        expected = step_state(expected, action, dt=0.1)
        assert positions[0, index].numpy() == pytest.approx(expected[:2])
        assert velocities[0, index].numpy() == pytest.approx(expected[2:])


def test_guided_generate_applies_safety_gradient_exactly_once(monkeypatch) -> None:
    class ConstantField(torch.nn.Module):
        T = 1
        d = 2
        u_max = 1.0

        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def _expand_ctx(self, context: torch.Tensor, count: int) -> torch.Tensor:
            return context.expand(count, -1)

        def forward(
            self, value: torch.Tensor, tau: torch.Tensor, context: torch.Tensor
        ) -> torch.Tensor:
            del tau, context
            return value.new_tensor((3.0, 4.0)).expand_as(value)

    # With dt=1, p_x has derivative 0.5 with respect to endpoint u_x.
    # Global norm matching rescales that gradient to ||v||=5, so a single
    # safe_coef=0.2 application contributes [1,0].  One half Euler step then
    # produces [2,2]; a duplicated safety term would instead produce [2.5,2].
    def controlled_cbf(positions, velocities, obstacle_xy, obstacle_radii, config):
        del velocities, obstacle_xy, obstacle_radii, config
        return positions[:, -1, 0]

    def zero_goal(positions, goal):
        del goal
        return positions[:, -1, 1] * 0.0

    monkeypatch.setattr(mizuta_module, "cbf_reward", controlled_cbf)
    monkeypatch.setattr(mizuta_module, "goal_reward", zero_goal)
    config = MizutaConfig(
        "one-safe-term",
        safe_coef=0.2,
        collision_weight=1.0,
        goal_guidance_coef=0.0,
        n_samples=1,
        n_elite=1,
        n_copies=1,
    )
    result = guided_generate(
        ConstantField(),
        torch.zeros(1),
        np.zeros(4, dtype=np.float32),
        torch.zeros(2),
        torch.zeros((1, 2)),
        torch.ones(1),
        1.0,
        torch.zeros((1, 2)),
        (0.0, 0.5),
        config,
    )
    np.testing.assert_allclose(result.detach().numpy(), [[2.0, 2.0]], atol=1.0e-6)


def test_restart_context_uses_actual_state_gamma_and_transition_geometry() -> None:
    env = make_ood_scene(radius=1.2)
    state = np.asarray((1.1, 0.7, 0.2, -0.1), dtype=np.float64)
    context = context_from_state(
        state,
        env.goal.detach().cpu().numpy(),
        0.7,
        [np.asarray((0.1, -0.2), dtype=np.float32)],
        env,
    )
    assert context.verifier_state == pytest.approx(state)
    assert float(context.low5[-1]) == pytest.approx(0.7)

    # Both segment endpoints are clear of the giant, but the continuous
    # segment crosses its interior and must therefore report a collision.
    crossing = np.asarray((1.0, 2.5, 30.0, 0.0), dtype=np.float64)
    clearance, in_bounds = transition_clearance(crossing, np.zeros(2), env)
    assert clearance < 0.0
    assert in_bounds is True

    # A quadratic transition can leave and re-enter the workspace between
    # two in-bounds endpoints; the exact coordinate extremum catches it.
    excursion = np.asarray((0.1, 0.5, -5.0, 0.0), dtype=np.float64)
    _, in_bounds = transition_clearance(
        excursion, np.asarray((100.0, 0.0), dtype=np.float64), env
    )
    assert in_bounds is False


def _row(gamma: float, repetition: int, temperature: float) -> dict:
    success = repetition == 0
    path = np.asarray(((0.5, 0.5), (1.0, 1.1)), dtype=np.float32)
    return {
        "gamma": gamma,
        "seed": 100_000 + int(round(gamma * 1_000)) * 10 + repetition,
        "repetition": repetition,
        "temperature": temperature,
        "config_tag": "lg020",
        "path": path,
        "actions": np.zeros((1, 2), dtype=np.float32),
        "success": success,
        "reached": success,
        "collision": not success,
        "out_of_bounds": False,
        "timeout": False,
        "failure_reason": None if success else "collision",
        "min_clearance_m": 0.1 if success else -0.1,
        "path_length_m": 0.8,
        "endpoint_distance_m": 4.8,
        "goal_progress_m": 0.8,
        "rollout_duration_s": 0.1,
        "time_to_goal_s": 0.1 if success else None,
        "wall_time_s": 0.01,
        "detour_mode": "upper-left" if success else "unresolved",
        "verifier_calls": 0,
        "safety_filter_used": False,
        "obstacle_radius_model": "per-obstacle",
    }


def test_scientific_and_gallery_artifacts_have_disjoint_temperature_roles() -> None:
    config = MizutaConfig(
        "lg020", safe_coef=0.02, collision_weight=2.0,
        n_samples=4, n_elite=2, n_copies=2,
    )
    gammas = (0.1, 0.5)
    scientific_rows = [
        _row(gamma, repetition, SCIENTIFIC_TEMPERATURE)
        for gamma in gammas
        for repetition in range(2)
    ]
    scientific = build_rollout_artifact(
        role="scientific",
        temperature=SCIENTIFIC_TEMPERATURE,
        config=config,
        rows=scientific_rows,
        gammas=gammas,
        repetitions=2,
        checkpoint_file_sha256="a" * 64,
        checkpoint_state_sha256="b" * 64,
        checkpoint_config_sha256="9" * 64,
        scene_fingerprint_sha256="c" * 64,
        reference_source_sha256="d" * 64,
        implementation_sha256="e" * 64,
    )
    assert scientific["schema_version"] == ROLLOUT_SCHEMA
    assert scientific["temperature"] == 1.0
    assert scientific["metrics_included"] is True
    assert scientific["per_gamma"]["0.1"]["n"] == 2
    assert scientific["per_gamma"]["0.1"]["success_rate"] == 0.5
    assert scientific["verifier_calls"] == scientific["socp_calls"] == 0

    gallery_rows = [
        _row(gamma, repetition, GALLERY_TEMPERATURE)
        for gamma in gammas
        for repetition in range(2)
    ]
    gallery = build_rollout_artifact(
        role="gallery_only",
        temperature=GALLERY_TEMPERATURE,
        config=config,
        rows=gallery_rows,
        gammas=gammas,
        repetitions=2,
        checkpoint_file_sha256="a" * 64,
        checkpoint_state_sha256="b" * 64,
        checkpoint_config_sha256="9" * 64,
        scene_fingerprint_sha256="c" * 64,
        reference_source_sha256="d" * 64,
        implementation_sha256="e" * 64,
    )
    assert gallery["temperature"] == 0.5
    assert gallery["metrics_included"] is False
    assert gallery["per_gamma"] is None and gallery["overall"] is None

    with pytest.raises(ValueError, match="requires temperature 1"):
        build_rollout_artifact(
            role="scientific",
            temperature=0.5,
            config=config,
            rows=gallery_rows,
            gammas=gammas,
            repetitions=2,
            checkpoint_file_sha256="a" * 64,
            checkpoint_state_sha256="b" * 64,
            checkpoint_config_sha256="9" * 64,
            scene_fingerprint_sha256="c" * 64,
            reference_source_sha256="d" * 64,
            implementation_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="fixed count"):
        build_rollout_artifact(
            role="scientific",
            temperature=1.0,
            config=config,
            rows=scientific_rows[:-1],
            gammas=gammas,
            repetitions=2,
            checkpoint_file_sha256="a" * 64,
            checkpoint_state_sha256="b" * 64,
            checkpoint_config_sha256="9" * 64,
            scene_fingerprint_sha256="c" * 64,
            reference_source_sha256="d" * 64,
            implementation_sha256="e" * 64,
        )


def test_sweep_is_bounded_low_guidance_and_module_has_no_verifier_adapter() -> None:
    assert len(LOW_GUIDANCE_SWEEP) == 4
    assert all(config.low_guidance_admissible for config in LOW_GUIDANCE_SWEEP)
    assert max(config.safe_coef for config in LOW_GUIDANCE_SWEEP) == pytest.approx(0.04)
    assert max(config.collision_weight for config in LOW_GUIDANCE_SWEEP) == pytest.approx(4.0)
    source = inspect.getsource(mizuta_module)
    assert "from .verifier" not in source
    assert "verify_plan(" not in source


def test_algorithm_and_cli_contract_exclude_ad_hoc_or_legacy_modes(tmp_path) -> None:
    contract = algorithm_contract()
    assert contract["goal_guidance_term_count"] == 1
    assert contract["safety_guidance_term_count"] == 1
    assert contract["generation_verifier_free"] is True
    assert contract["generation_socp_free"] is True
    assert contract["scientific_temperature"] == 1.0
    assert contract["gallery_temperature"] == 0.5
    assert contract["legacy_artifact_reuse"] is False
    assert contract["resume_supported"] is False
    assert len(contract["preregistered_low_guidance_configs"]) == 4

    parser = make_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--resume" not in option_strings
    assert "--legacy" not in option_strings
    assert "--nfe" not in option_strings
    assert "--n-samples" not in option_strings
    assert "--n-elite" not in option_strings
    assert "--n-copies" not in option_strings

    clean = tmp_path / "clean"
    clean.mkdir()
    _require_clean_output_dir(clean)
    (clean / "legacy.pt").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="never resumes or retains legacy"):
        _require_clean_output_dir(clean)


def test_checkpoint_loader_accepts_only_fresh_endpoint_free_stage3(tmp_path) -> None:
    policy = HP.GridHPFlowPolicy(repr_dim=32, grid_hw=(32, 32))
    state_hash = model_state_hash(policy)
    checkpoint = tmp_path / "checkpoint_best.pt"
    HP.save_hp(
        policy,
        checkpoint,
        extra={
            "stage_schema": "afe_fresh_pretrain_v1",
            "fresh_from_scratch": True,
            "endpoint_free": True,
            "expansion_promotion": True,
            "id_mode_diversity_gate_passed": True,
            "id_evaluation_temperature": 1.0,
            "id_evaluation_uncertainty_tilting": False,
            "model_state_sha256": state_hash,
            "source_manifest": "/sealed/stage2/manifest.json",
            "source_query_hash_digest": "f" * 64,
            "id_metrics_sha256": "e" * 64,
        },
    )
    loaded, payload, file_hash, loaded_hash = load_clean_stage3_checkpoint(
        checkpoint, torch.device("cpu")
    )
    assert loaded.training is False
    assert payload["endpoint_free"] is True
    assert len(file_hash) == 64
    assert loaded_hash == state_hash

    legacy = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy["fresh_from_scratch"] = False
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(RuntimeError, match="fresh endpoint-free"):
        load_clean_stage3_checkpoint(legacy_path, torch.device("cpu"))
