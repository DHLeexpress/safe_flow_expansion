from __future__ import annotations

from types import SimpleNamespace

import pytest

import grid_hp_expt as HP

from afe_restart.policy import model_state_hash, require_promoted_fresh_pretrain
from afe_restart.stage3_pretrain import require_clean_fresh_outdir, run_stage


def _valid_checkpoint(model) -> dict[str, object]:
    return {
        "config": model.config(),
        "stage_schema": "afe_fresh_pretrain_v1",
        "fresh_from_scratch": True,
        "endpoint_free": True,
        "expansion_promotion": True,
        "id_mode_diversity_gate_passed": True,
        "id_evaluation_temperature": 1.0,
        "id_evaluation_uncertainty_tilting": False,
        "model_state_sha256": model_state_hash(model),
        "source_query_hash_digest": "a" * 64,
        "source_manifest": "/sealed/stage2/manifest.json",
        "id_metrics_sha256": "b" * 64,
    }


def test_fresh_stage3_rejects_nonempty_outdir_before_touching_old_artifacts(
    tmp_path,
) -> None:
    outdir = tmp_path / "stage3"
    production = outdir / "data/checkpoint_best.pt"
    production.parent.mkdir(parents=True)
    production.write_bytes(b"previous-promoted-checkpoint")

    # run_stage reaches the clean-output guard before it needs any remaining
    # arguments, dataset access, dependency logging, or training setup.
    args = SimpleNamespace(device="cpu", outdir=outdir)
    with pytest.raises(RuntimeError, match="refuses a nonempty output directory"):
        run_stage(args)

    assert production.read_bytes() == b"previous-promoted-checkpoint"
    assert sorted(path.relative_to(outdir) for path in outdir.rglob("*")) == [
        production.parent.relative_to(outdir),
        production.relative_to(outdir),
    ]


def test_fresh_stage3_allows_missing_or_empty_outdir(tmp_path) -> None:
    missing = tmp_path / "missing"
    require_clean_fresh_outdir(missing)
    empty = tmp_path / "empty"
    empty.mkdir()
    require_clean_fresh_outdir(empty)


def test_complete_promoted_checkpoint_contract_is_accepted() -> None:
    model = HP.GridHPFlowPolicy(repr_dim=32, grid_hw=(32, 32))
    checkpoint = _valid_checkpoint(model)
    assert require_promoted_fresh_pretrain(model, checkpoint) == model_state_hash(model)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("source_query_hash_digest", None),
        ("source_query_hash_digest", "z" * 64),
        ("source_manifest", ""),
        ("id_metrics_sha256", None),
        ("id_metrics_sha256", "A" * 64),
        ("frozen_feature_snapshot", True),
    ),
)
def test_promotion_guard_rejects_missing_or_invalid_provenance(
    field: str, invalid: object
) -> None:
    model = HP.GridHPFlowPolicy(repr_dim=32, grid_hw=(32, 32))
    checkpoint = _valid_checkpoint(model)
    checkpoint[field] = invalid
    with pytest.raises(RuntimeError, match="requires a fresh endpoint-free Stage-03"):
        require_promoted_fresh_pretrain(model, checkpoint)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("arch", "other"),
        ("schema_version", "legacy"),
        ("raw_start_goal", True),
        ("H_pred", 9),
        ("grid_shape", (3, 32, 32)),
        ("K_hist", 8),
        ("u_max", 2.0),
        ("use_gru", True),
        ("boundary_adapter", True),
        ("repr_dim", 31),
        ("ctx_dim", 39),
    ),
)
def test_promotion_guard_rejects_architecture_or_endpoint_contract_drift(
    field: str, invalid: object
) -> None:
    model = HP.GridHPFlowPolicy(repr_dim=32, grid_hw=(32, 32))
    checkpoint = _valid_checkpoint(model)
    checkpoint["config"] = {**checkpoint["config"], field: invalid}
    with pytest.raises(RuntimeError, match="requires a fresh endpoint-free Stage-03"):
        require_promoted_fresh_pretrain(model, checkpoint)
