from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from afe_restart.config import GAMMAS
from afe_restart.schemas import QueryContext, query_content_hash
from afe_restart.stage3_pretrain import (
    DatasetContractError,
    balanced_real_window_indices,
    load_planned_demo_manifest,
    make_group_split,
    validate_temperature_one_mode_diversity,
    validate_no_trajectory_leakage,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_payload() -> dict:
    grids = []
    low5s = []
    histories = []
    verifier_states = []
    verifier_fingerprints = []
    plans = []
    gammas = []
    trajectory_ids = []
    steps = []
    directions = []
    hashes = []
    trajectory_rows = []
    tid = 0
    # Two real source trajectories per gamma/mode permits a one-trajectory
    # validation holdout without ever splitting one trajectory's windows.
    for gamma_index, gamma in enumerate(GAMMAS):
        for direction, route in ((0, "R-first"), (1, "U-first")):
            for copy_index in range(2):
                grid = torch.zeros(3, 32, 32, dtype=torch.float32)
                grid[2].fill_(0.01 * (1 + tid))
                low5 = torch.tensor((0.2, 0.3, 0.0, 0.0, gamma), dtype=torch.float32)
                hist = torch.zeros(10, 2, dtype=torch.float32)
                plan = torch.zeros(10, 2, dtype=torch.float32)
                plan[:, direction] = 0.1 + 0.001 * (gamma_index * 4 + direction * 2 + copy_index)
                verifier_state = np.asarray(
                    [0.2, 0.3, 0.001 * tid, -0.001 * tid], dtype=np.float64
                )
                verifier_fingerprint = "e" * 64
                context = QueryContext(
                    grid.numpy(),
                    low5.numpy(),
                    hist.numpy(),
                    verifier_state,
                    verifier_fingerprint,
                )
                identity = query_content_hash(context, gamma, plan.numpy())
                grids.append(grid)
                low5s.append(low5)
                histories.append(hist)
                verifier_states.append(torch.from_numpy(verifier_state))
                verifier_fingerprints.append(verifier_fingerprint)
                plans.append(plan)
                gammas.append(gamma)
                trajectory_ids.append(tid)
                steps.append(0)
                directions.append(direction)
                hashes.append(identity)
                trajectory_rows.append(
                    {
                        "trajectory_id": tid,
                        "gamma": gamma,
                        "seed": 1000 + tid,
                        "direction_class": route,
                        "steps": 1,
                    }
                )
                tid += 1
    count = len(plans)
    return {
        "schema_version": "afe_planned_demo_v2_exact_verifier_identity",
        "grid": torch.stack(grids),
        "low5": torch.stack(low5s),
        "hist": torch.stack(histories),
        "verifier_state": torch.stack(verifier_states),
        "verifier_spec_fingerprint": list(verifier_fingerprints),
        "U": torch.stack(plans),
        "window_plan_kind": ["weighted_mean"] * count,
        "gamma": torch.tensor(gammas, dtype=torch.float32),
        "window_seeds": 1000 + torch.arange(count),
        "window_trajectory_ids": torch.tensor(trajectory_ids),
        "source_trajectory_ids": torch.tensor(trajectory_ids),
        "window_steps": torch.tensor(steps),
        "window_direction": torch.tensor(directions, dtype=torch.int8),
        "trajectory_balanced_weight": torch.ones(count, dtype=torch.float32),
        "target_safe": torch.ones(count, dtype=torch.bool),
        "target_in_bounds": torch.ones(count, dtype=torch.bool),
        "target_socp_ok": torch.ones(count, dtype=torch.bool),
        "query_hashes": list(hashes),
        "generated_hashes": list(hashes),
        "verifier_input_hashes": list(hashes),
        "training_target_hashes": list(hashes),
        "trajectory_rows": trajectory_rows,
        "contract": {
            "generated_equals_verified_equals_training": True,
            "planned_horizon": 10,
            "only_first_action_executed": True,
            "all_targets_pre_execution_fully_verified": True,
            "progress_not_in_safety_label": True,
            "synthetic_reflections": 0,
            "padding": 0,
            "debug_training_targets": 0,
            "debug_target_share": 0.0,
        },
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    data = tmp_path / "planned_id_balanced.pt"
    torch.save(payload, data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "afe_planned_demo_v2_exact_verifier_identity",
                "dataset": data.name,
                "dataset_sha256": _file_sha256(data),
            }
        )
    )
    return manifest


def test_clean_planned_targets_group_split_and_balanced_sampler(tmp_path: Path) -> None:
    pool = load_planned_demo_manifest(_write(tmp_path, _fixture_payload()))
    split = make_group_split(pool, validation_trajectories_per_mode=1, seed=9)
    validate_no_trajectory_leakage(pool, split.train_indices, split.validation_indices)
    assert not (set(split.train_trajectory_ids) & set(split.validation_trajectory_ids))
    rows = balanced_real_window_indices(
        pool,
        split.train_trajectory_ids,
        windows_per_trajectory=1,
        generator=torch.Generator().manual_seed(4),
    )
    counts = {}
    for gamma in GAMMAS:
        for mode in (0, 1):
            mask = torch.isclose(pool.gamma[rows], torch.tensor(gamma, dtype=torch.float64))
            mask &= pool.direction[rows] == mode
            counts[(gamma, mode)] = int(mask.sum())
    assert len(set(counts.values())) == 1


@pytest.mark.parametrize(
    "mutation, message",
    (
        (
            lambda payload: payload["contract"].__setitem__(
                "only_first_action_executed", False
            ),
            "executed-composite",
        ),
        (
            lambda payload: payload["target_socp_ok"].__setitem__(0, False),
            "safety must equal",
        ),
        (
            lambda payload: payload["training_target_hashes"].__setitem__(0, "0" * 64),
            "identity mismatch",
        ),
        (
            lambda payload: payload["window_plan_kind"].__setitem__(
                0, "debug_candidate"
            ),
            "cost-selected SafeMPPI",
        ),
    ),
)
def test_rejects_executed_composite_unverified_and_hash_mismatch(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = copy.deepcopy(_fixture_payload())
    mutation(payload)
    with pytest.raises(DatasetContractError, match=message):
        load_planned_demo_manifest(_write(tmp_path, payload))


def test_rejects_source_trajectory_leakage(tmp_path: Path) -> None:
    pool = load_planned_demo_manifest(_write(tmp_path, _fixture_payload()))
    same = torch.tensor((0,), dtype=torch.long)
    with pytest.raises(DatasetContractError, match="trajectory leakage"):
        validate_no_trajectory_leakage(pool, same, same)


def test_manifest_requires_dataset_checksum(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _fixture_payload())
    contents = json.loads(manifest.read_text())
    del contents["dataset_sha256"]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(DatasetContractError, match="requires a SHA-256 checksum"):
        load_planned_demo_manifest(manifest)


def test_rejects_duplicate_source_trajectory_provenance(tmp_path: Path) -> None:
    payload = _fixture_payload()
    payload["trajectory_rows"][1]["seed"] = payload["trajectory_rows"][0]["seed"]
    payload["window_seeds"][1] = payload["window_seeds"][0]
    with pytest.raises(DatasetContractError, match="source trajectory appears more than once"):
        load_planned_demo_manifest(_write(tmp_path, payload))


def test_rejects_window_level_source_provenance_mismatch(tmp_path: Path) -> None:
    payload = _fixture_payload()
    payload["window_seeds"][0] += 1
    with pytest.raises(DatasetContractError, match="source seed disagrees"):
        load_planned_demo_manifest(_write(tmp_path, payload))


def test_rejects_truthy_non_boolean_verifier_labels(tmp_path: Path) -> None:
    payload = _fixture_payload()
    payload["target_safe"] = torch.ones(len(payload["U"]), dtype=torch.uint8)
    with pytest.raises(DatasetContractError, match="explicit boolean tensors"):
        load_planned_demo_manifest(_write(tmp_path, payload))


def test_rejects_non_integer_direction_labels(tmp_path: Path) -> None:
    payload = _fixture_payload()
    payload["window_direction"] = payload["window_direction"].float() + 0.25
    with pytest.raises(DatasetContractError, match="must use an integer dtype"):
        load_planned_demo_manifest(_write(tmp_path, payload))


def test_rejects_gamma_or_direction_trajectory_imbalance(tmp_path: Path) -> None:
    payload = _fixture_payload()
    # Relabel one U trajectory as R in both trajectory and row tensors, while
    # preserving the otherwise valid planned-window identity.
    target_tid = next(
        row["trajectory_id"]
        for row in payload["trajectory_rows"]
        if row["gamma"] == GAMMAS[0] and row["direction_class"] == "U-first"
    )
    payload["trajectory_rows"][target_tid]["direction_class"] = "R-first"
    payload["window_direction"][target_tid] = 0
    with pytest.raises(DatasetContractError, match="not exact R/U trajectory balanced"):
        load_planned_demo_manifest(_write(tmp_path, payload))


def test_temperature_one_mode_diversity_is_a_required_per_gamma_gate() -> None:
    summary = {
        "sampling_temperature": 1.0,
        "per_gamma": {
            f"{float(gamma):g}": {"R-first_successes": 1, "U-first_successes": 1}
            for gamma in GAMMAS
        },
    }
    validate_temperature_one_mode_diversity(summary)
    summary["per_gamma"][f"{float(GAMMAS[-1]):g}"]["U-first_successes"] = 0
    with pytest.raises(RuntimeError, match="mode-diversity gate failed"):
        validate_temperature_one_mode_diversity(summary)


def test_mode_diversity_gate_rejects_non_scientific_temperature() -> None:
    with pytest.raises(RuntimeError, match="requires temperature T=1"):
        validate_temperature_one_mode_diversity(
            {"sampling_temperature": 0.5, "per_gamma": {}}
        )
