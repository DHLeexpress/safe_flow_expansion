from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import low7_rbf_v3_support_sweep_driver as DRIVER


def args():
    return SimpleNamespace(
        python="/env/python",
        checkpoint=Path("/authenticated/checkpoint.pt"),
        checkpoint_sha256="a" * 64,
        verifier_workers=128,
    )


def value_after(command, option):
    return command[command.index(option) + 1]


def test_six_arm_matrix_and_gpu_serial_queues_are_exact():
    assert {
        index: [arm.arm_id for arm in queue]
        for index, queue in DRIVER.GPU_QUEUES.items()
    } == {
        1: ["opt016_demo000", "opt016_demo0250", "opt032_demo0125"],
        3: ["opt016_demo0125", "opt032_demo000", "opt032_demo0250"],
    }
    arms = [arm for queue in DRIVER.GPU_QUEUES.values() for arm in queue]
    assert len(arms) == len({arm.arm_id for arm in arms}) == 6


@pytest.mark.parametrize("steps,demo", ((16, 0.0), (16, 0.125), (32, 0.25)))
def test_trainer_command_does_not_confuse_optimizer_dose_with_replicas(steps, demo):
    arm = DRIVER.Arm(steps, demo)
    command = DRIVER.trainer_command(args(), arm, Path("/new/run"))
    assert value_after(command, "--optimizer-steps-per-round") == str(steps)
    assert value_after(command, "--rollout-replicas") == "8"
    assert value_after(command, "--demo-frac") == f"{demo:g}"
    assert value_after(command, "--rounds") == "100"
    assert value_after(command, "--protocol-profile") == "v3_support_sweep"
    assert "--nvp-audit-all-k" not in command
    assert value_after(command, "--verifier-workers") == "128"


def test_timing_preflight_changes_only_profile_and_round_count():
    arm = DRIVER.Arm(32, 0.0)
    full = DRIVER.trainer_command(args(), arm, Path("/full"))
    timing = DRIVER.trainer_command(args(), arm, Path("/timing"), preflight=True)
    assert value_after(timing, "--protocol-profile") == "v3_support_preflight"
    assert value_after(timing, "--rounds") == "1"
    ignored = {"--protocol-profile", "--rounds", "--outdir"}
    def options(command):
        return {
            token: command[index + 1]
            for index, token in enumerate(command[:-1])
            if token.startswith("--") and command[index + 1].startswith("--") is False
            and token not in ignored
        }
    assert options(full) == options(timing)


def test_global_selection_applies_J_first_then_declared_ties():
    base = {
        "CR": 0.1, "timeout": 0.1, "minimum_clearance": 0.2,
        "round": 20,
    }
    rows = [
        {**base, "arm_id": "a", "J": 0.3, "SR": 0.9},
        {**base, "arm_id": "b", "J": 0.4, "SR": 0.7},
        {**base, "arm_id": "c", "J": 0.4, "SR": 0.8},
    ]
    assert min(rows, key=DRIVER.global_key)["arm_id"] == "c"


def test_recipe_contract_rejects_rollout_or_audit_drift():
    arm = DRIVER.Arm(16, 0.125)
    recipe = {
        "algorithm": "afe_rbf_low7_v3_optimizer_demo_support_v1",
        "protocol_profile": "v3_support_sweep", "rounds": 100,
        "rollout_replicas": 8, "K": 16, "B": 4, "T": 300,
        "batch": 128, "afe_lr": 1.0e-5, "negative_alpha": 0.0,
        "optimizer_steps_per_round": 16, "demo_frac": 0.125,
        "replay_window": 2, "gp_replay_window": 2,
        "replay_update_mode": "fixed_macro_steps_exact_epoch",
        "replay_loss_weighting": "gamma_episode_context_query_equal_mass",
        "execution_rule": "nominal_hp_max_step_margin",
        "nvp_all_k_audit": False, "verifier_workers": 128,
        "no_curriculum": True, "no_anchor": True, "no_prox": True,
        "no_fallback": True,
        "demo_reference": {"pair_leakage": 0},
        "scene": {"profile": {"name": DRIVER.SCENE}},
    }
    DRIVER._support_recipe_contract(recipe, arm, 100, 128)
    recipe["rollout_replicas"] = 16
    with pytest.raises(RuntimeError, match="recipe mismatch"):
        DRIVER._support_recipe_contract(recipe, arm, 100, 128)


def test_launcher_allocates_exactly_half_the_online_cpu_count():
    launcher = (ROOT / "run_low7_rbf_v3_support_sweep.sh").read_text()
    assert "CPU_COUNT=$(getconf _NPROCESSORS_ONLN)" in launcher
    assert "WORKERS=$((CPU_COUNT / 2))" in launcher
    assert "check_gpu 1" in launcher and "check_gpu 3" in launcher
    assert "grep -q libx264" not in launcher
    assert "grep libx264 >/dev/null" in launcher
