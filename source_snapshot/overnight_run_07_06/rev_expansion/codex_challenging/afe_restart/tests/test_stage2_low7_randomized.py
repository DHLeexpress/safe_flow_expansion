from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from afe_restart.scene import context_from_state_low7, make_id_scene
from afe_restart.schemas import QueryContext, query_content_hash
from afe_restart.stage2_low7_randomized import (
    DATA_SCHEMA,
    NominalScheduleError,
    _atomic_json,
    _build_dataset,
    _planner_seed,
    _replay_selected_polytope,
    _validate_dataset_payload,
    collect_gamma_shard,
    generate_endpoint_payload,
    generate_fixed_goal_grid_payload,
    grid_features,
    load_endpoint_bank,
    make_parser,
    planner_scene,
)
from afe_restart.verifier import VP, verify_plan


def _endpoint_bank(tmp_path: Path, *, count: int = 8):
    path = tmp_path / "endpoints.json"
    _atomic_json(path, generate_endpoint_payload(pair_count=count, seed=20260717))
    return load_endpoint_bank(path)


def test_endpoint_bank_is_deterministic_iid_free_space_without_pair_filter(
    tmp_path: Path,
) -> None:
    first = generate_endpoint_payload(pair_count=32, seed=71)
    second = generate_endpoint_payload(pair_count=32, seed=71)
    assert first["pairs"] == second["pairs"]
    assert make_parser().parse_args(["endpoints"]).pairs == 100
    sampling = first["sampling"]
    assert sampling["start_goal_independent"] is True
    assert sampling["endpoint_relation_constraints"] == []
    assert sampling["diagonal_constraint"] is False
    assert sampling["minimum_start_goal_distance_m"] is None

    path = tmp_path / "endpoints.json"
    _atomic_json(path, first)
    bank = load_endpoint_bank(path)
    assert bank.starts.shape == bank.goals.shape == (32, 2)
    # This confirms the stored bank is not restricted to either fixed diagonal
    # direction used by the legacy corpus.
    displacement = bank.goals - bank.starts
    assert bool((displacement[:, 0] < 0).any())
    assert bool((displacement[:, 0] > 0).any())
    assert bool((displacement[:, 1] < 0).any())
    assert bool((displacement[:, 1] > 0).any())


def test_fixed_goal_grid_adds_diagonal_starts_without_weakening_clearance(
    tmp_path: Path,
) -> None:
    first = generate_fixed_goal_grid_payload(seed=0)
    second = generate_fixed_goal_grid_payload(seed=0)
    assert first["pairs"] == second["pairs"]
    sampling = first["sampling"]
    assert first["pair_count"] == 881
    assert sampling["legacy_off_diagonal_count"] == 566
    assert sampling["new_diagonal_region_count"] == 315
    assert sampling["strict_free_clearance_m"] == 0.05
    assert sampling["minimum_obstacle_clearance_m"] > 0.05
    assert sampling["diagonal_constraint"] is False
    assert sampling["fixed_goal"] == [4.7, 4.7]

    path = tmp_path / "fixed_grid_endpoints.json"
    _atomic_json(path, first)
    bank = load_endpoint_bank(path)
    assert bank.starts.shape == bank.goals.shape == (881, 2)
    assert np.all(bank.goals == np.asarray((4.7, 4.7), dtype=np.float32))
    assert bool((np.abs(bank.starts[:, 1] - bank.starts[:, 0]) < 1.0).any())


def test_planner_retry_seed_is_paired_across_gamma() -> None:
    # Gamma is not an input, so every shard uses this identical pair/retry bank.
    assert _planner_seed(1000, 7, 2, 3) == 1023
    assert len({_planner_seed(1000, pair, retry, 3) for pair in range(4) for retry in range(3)}) == 12


def test_cli_parser_registers_each_additive_command_once(tmp_path: Path) -> None:
    parser = make_parser()
    assert parser.parse_args(["endpoints"]).pairs == 100
    collect = parser.parse_args(
        [
            "collect",
            "--endpoint-manifest",
            str(tmp_path / "endpoints.json"),
            "--gamma",
            "0.5",
            "--outdir",
            str(tmp_path / "shard"),
        ]
    )
    assert collect.command == "collect"
    assert (
        collect.max_steps,
        collect.reach,
        collect.smooth_weight,
        collect.retreat_weight,
        collect.noise_var_mult,
        collect.max_debug_candidates,
        collect.max_proposals,
    ) == (800, 0.15, 0.12, 0.0, 3.0, 0, 2)
    assert parser.parse_args(
        [
            "combine",
            "--shard-manifests",
            str(tmp_path / "shard.json"),
            "--outdir",
            str(tmp_path / "combined"),
        ]
    ).command == "combine"
    assert parser.parse_args(
        ["render", "--manifest", str(tmp_path / "manifest.json")]
    ).command == "render"
    assert parser.parse_args(
        [
            "starts",
            "--endpoint-manifest",
            str(tmp_path / "endpoints.json"),
            "--output",
            str(tmp_path / "starts.png"),
        ]
    ).command == "starts"
    video = parser.parse_args(
        ["video", "--manifest", str(tmp_path / "manifest.json")]
    )
    assert video.command == "video" and video.frame_stride == 1


def test_collector_rejects_teacher_recipe_override(tmp_path: Path) -> None:
    bank = _endpoint_bank(tmp_path, count=1)
    args = make_parser().parse_args(
        [
            "collect",
            "--endpoint-manifest",
            str(bank.path),
            "--gamma",
            "0.5",
            "--outdir",
            str(tmp_path / "shard"),
            "--smooth-weight",
            "0.2",
        ]
    )
    with pytest.raises(ValueError, match="pins the canonical SafeMPPI teacher recipe"):
        collect_gamma_shard(args)


def test_v3_dataset_uses_low7_and_preserves_exact_selected_query_identity(
    tmp_path: Path,
) -> None:
    bank = _endpoint_bank(tmp_path, count=2)
    gamma = 0.1
    start = bank.starts[0]
    goal = bank.goals[0]
    states = np.asarray(
        [[start[0], start[1], 0.0, 0.0]] * 3, dtype=np.float32
    )
    contexts: list[QueryContext] = []
    plans: list[np.ndarray] = []
    hashes: list[str] = []
    for step in range(2):
        low7 = np.asarray(
            [0.4, -0.2, 0.0, 0.0, 0.1, -0.1, np.float32(gamma)],
            dtype=np.float32,
        )
        context = QueryContext(
            np.zeros((3, 32, 32), dtype=np.float32),
            low7,
            np.zeros((16, 2), dtype=np.float32),
            states[step].astype(np.float64),
            "a" * 64,
        )
        plan = np.full((10, 2), step / 10.0, dtype=np.float32)
        contexts.append(context)
        plans.append(plan)
        hashes.append(query_content_hash(context, gamma, plan))
    episode = {
        "pair_id": 0,
        "retry_index": 0,
        "success": True,
        "steps": 2,
        "gamma": gamma,
        "seed": 123,
        "states": states,
        "goal": goal,
        "selected_query_indices": np.asarray([0, 1], dtype=np.int64),
        "query_steps": np.asarray([0, 1], dtype=np.int32),
        "query_plans": np.asarray(plans, dtype=np.float32),
        "query_hashes": hashes,
        "query_kinds": ["weighted_mean", "internal_best"],
        "contexts": contexts,
        "query_safe": np.ones(2, dtype=bool),
        "query_in_bounds": np.ones(2, dtype=bool),
        "query_socp_ok": np.ones(2, dtype=bool),
        "query_progress_m": np.asarray([0.1, 0.2]),
        "query_physical_clearance_m": np.asarray([0.3, 0.4]),
        "min_clearance_m": 0.3,
        "path_length_m": 0.1,
        "query_acceptance": 1.0,
        "candidate_meta": "candidates/example.json",
    }
    output = tmp_path / "shard.pt"
    payload = _build_dataset(
        [episode], output, endpoint_manifest=bank, gamma=gamma
    )
    _validate_dataset_payload(
        payload, endpoint_manifest=bank, expected_gamma=gamma
    )

    assert payload["schema_version"] == DATA_SCHEMA
    assert "low7" in payload and "low5" not in payload
    assert "window_direction" not in payload
    assert payload["low7"].shape == (2, 7)
    assert payload["window_pair_ids"].tolist() == [0, 0]
    assert payload["window_trajectory_ids"].tolist() == [0, 0]
    assert payload["window_steps"].tolist() == [0, 1]
    assert torch.equal(payload["window_start"], torch.from_numpy(np.stack([start, start])))
    assert torch.equal(payload["window_goal"], torch.from_numpy(np.stack([goal, goal])))
    assert payload["trajectory_balanced_weight"].tolist() == [0.5, 0.5]
    assert payload["query_hashes"] == hashes
    assert payload["target_query_hash"] == hashes
    assert output.exists()


def test_polytope_replay_authenticates_exact_safe_cost_selected_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = np.asarray((0.5, 0.5), dtype=np.float32)
    goal = np.asarray((4.5, 4.5), dtype=np.float32)
    gamma = 0.5
    env = make_id_scene(start=start, goal=goal)
    state = env.x0.detach().cpu().numpy().astype(np.float64)
    plan = np.zeros((10, 2), dtype=np.float32)
    context = context_from_state_low7(state, goal, gamma, [], env)
    result = verify_plan(state, plan, env, gamma, goal=goal)
    assert result.safe and result.in_bounds and result.socp_ok
    episode = {
        "gamma": gamma,
        "selected_query_indices": np.asarray([0], dtype=np.int64),
        "query_steps": np.asarray([0], dtype=np.int32),
        "query_kinds": ["weighted_mean"],
        "query_safe": np.asarray([result.safe]),
        "query_in_bounds": np.asarray([result.in_bounds]),
        "query_socp_ok": np.asarray([result.socp_ok]),
        "contexts": [context],
        "query_plans": np.asarray([plan]),
        "executed_actions": np.asarray([plan[0]]),
        "query_hashes": [query_content_hash(context, gamma, plan)],
        "goal": goal,
        "query_certificate_worst_step": np.asarray(
            [result.certificate_worst_step], dtype=np.int16
        ),
    }
    for field in (
        "bounds_margin_m",
        "physical_clearance_m",
        "face_margin_m",
        "certificate_residual",
        "progress_m",
        "start_goal_distance_m",
        "terminal_goal_distance_m",
    ):
        episode[f"query_{field}"] = np.asarray(
            [getattr(result, field)], dtype=np.float64
        )

    replay = _replay_selected_polytope(episode, 0, env)
    assert replay["query_hash"] == episode["query_hashes"][0]
    assert replay["certified_cost_selected_count"] == 1
    assert len(replay["fitted_face_sha256"]) == 64
    assert len(replay["nominal_face_sha256"]) == 64
    assert replay["fitted_face_signatures"]
    assert replay["nominal_face_signatures"]
    assert replay["nominal_sensing_radius_m"] == 2.0
    assert replay["nominal_polytope_nbase"] == 16
    assert replay["nominal_worst_residual"] >= -1.0e-8
    assert 1 <= replay["nominal_worst_horizon_step"] <= 10
    sample_points = np.asarray(
        [[0.5, 0.5], [0.7, 0.6], [1.2, 0.8]], dtype=np.float64
    )
    nominal_hp, _arrays = grid_features.polytope_HP(
        state[:2],
        planner_scene.planner_obstacles(env),
        sensing=2.0,
        n_base=16,
        predict_gain=0.0,
    )
    replayed_hp = VP.H_grid(
        replay["nominal_faces"],
        (sample_points[:, 0] - state[0]).reshape(-1, 1),
        (sample_points[:, 1] - state[1]).reshape(-1, 1),
    ).reshape(-1)
    np.testing.assert_allclose(
        replayed_hp, nominal_hp(sample_points), atol=2.0e-15, rtol=2.0e-15
    )

    original_polytope_hp = grid_features.polytope_HP

    def failing_nominal(*args, **kwargs):
        _hp, arrays = original_polytope_hp(*args, **kwargs)

        def values(points):
            result = np.zeros(len(np.asarray(points)), dtype=np.float64)
            result[0] = 1.0
            return result

        return values, arrays

    monkeypatch.setattr(grid_features, "polytope_HP", failing_nominal)
    with pytest.raises(NominalScheduleError, match="nominal schedule failed"):
        _replay_selected_polytope(episode, 0, env)
    monkeypatch.setattr(grid_features, "polytope_HP", original_polytope_hp)

    episode["query_kinds"] = ["debug_candidate"]
    with pytest.raises(RuntimeError, match="certified cost-selected"):
        _replay_selected_polytope(episode, 0, env)
