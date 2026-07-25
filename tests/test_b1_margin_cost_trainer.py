from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts import b1_margin_cost_trainer as launcher


def _fake_trainer():
    seen = []

    def original_validate(args):
        seen.append((args.protocol_profile, args.execution_rule))
        assert args.protocol_profile == "b1_balanced_r0_margin50"
        assert args.execution_rule == "nominal_hp_max_step_margin"

    module = SimpleNamespace(
        EX=SimpleNamespace(
            SAFEMPPI_COST="nominal_hp_safemppi_cost",
            MAX_STEP_MARGIN="nominal_hp_max_step_margin",
        ),
        validate_protocol_args=original_validate,
    )
    return module, seen


def test_override_is_scoped_to_validation_and_restores_cost_rule():
    module, seen = _fake_trainer()
    launcher._install_cost_validation_override(module)
    args = Namespace(
        protocol_profile="b1_balanced_r0_margin50",
        execution_rule="nominal_hp_safemppi_cost",
    )

    module.validate_protocol_args(args)

    assert seen == [
        ("b1_balanced_r0_margin50", "nominal_hp_max_step_margin")
    ]
    assert args.execution_rule == "nominal_hp_safemppi_cost"


@pytest.mark.parametrize(
    ("profile", "rule"),
    [
        ("b1_balanced_r0_sweep", "nominal_hp_safemppi_cost"),
        ("b1_balanced_r0_margin50", "nominal_hp_max_step_margin"),
    ],
)
def test_override_rejects_any_other_protocol_or_rule(profile, rule):
    module, seen = _fake_trainer()
    launcher._install_cost_validation_override(module)
    args = Namespace(protocol_profile=profile, execution_rule=rule)

    with pytest.raises(ValueError):
        module.validate_protocol_args(args)

    assert seen == []
    assert args.execution_rule == rule
