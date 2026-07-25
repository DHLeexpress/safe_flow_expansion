from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_metrics_common import GAMMAS, load_jsonl_round, load_path_archive


def test_local_baseline_archives_cover_declared_gamma_grid() -> None:
    base = ROOT / "provenance/paper_baselines/local_native_cost_m10"
    archives = [base / "safemppi_ood_m10.npz"]
    archives.extend(sorted(base.glob("kazuki_wg0_ws*_ood_m10.npz")))
    assert len(archives) == 7
    for archive in archives:
        cells = load_path_archive(archive)
        assert tuple(cells) == GAMMAS
        for cell in cells.values():
            assert len(cell["paths"]) == 10
            assert len(cell["outcomes"]) == 10
            assert all(np.asarray(path).shape[1] == 2 for path in cell["paths"])


def test_max_safety_input_resolves_round_15() -> None:
    path = (
        ROOT
        / "provenance/b1_margin_goal/fixedtemp_m200_revised_r0_r15.jsonl"
    )
    selected, cells = load_jsonl_round(path, 15)
    assert selected == 15
    assert tuple(cells) == GAMMAS
    assert all(cell["m"] == 200 for cell in cells.values())


def test_kazuki_spec_has_no_implicit_coefficient_selection() -> None:
    spec = json.loads(
        (ROOT / "configs/kazuki_native_cost_sweep.json").read_text()
    )
    observed = {
        (float(pair["goal_coef"]), float(pair["safe_coef"]))
        for pair in spec["pairs"]
    }
    assert observed == {
        (0.0, 0.0),
        (0.0, 0.1),
        (0.0, 0.3),
        (0.0, 0.5),
        (0.0, 0.7),
        (0.0, 0.9),
    }
    assert "selected" not in spec
