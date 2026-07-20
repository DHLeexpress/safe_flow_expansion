#!/usr/bin/env python3
"""Final, artifact-driven reports for the planned-window AFE restart.

Scientific quantities in this module come from ordinary, untilted temperature-one
audits and rollouts.  The rollout gallery is a separate temperature-0.5 visual
diagnostic.  No value is copied between those two roles.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import torch

from .config import clean_method_absence_manifest
from .dynamics import step_state
from .scene import GAMMAS, GIANT_CENTER, GOAL, START, make_id_scene, make_ood_scene


PLASMA = LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(np.linspace(0.04, 0.88, 256))
)
GAMMA_NORM = Normalize(vmin=min(GAMMAS), vmax=max(GAMMAS))
GAMMA_COLOR = {float(g): PLASMA(GAMMA_NORM(float(g))) for g in GAMMAS}
METHOD_STYLE = {
    "Expert": ("o", 90),
    "Pretrained": ("s", 68),
    r"CFM-MPPI$^{*}$": ("v", 90),
    "Full": ("*", 225),
    "-SOCP": ("X", 80),
    "-Progress": ("P", 80),
    "-AFE": ("D", 67),
}


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot support the requested scientific claim."""


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"expected a mapping-like record, got {type(value).__name__}")


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool(value: Any, default: bool = False) -> bool:
    return default if value is None else bool(value)


def _gamma_key(gamma: float) -> str:
    return f"{float(gamma):g}"


def _gamma_color(gamma: float) -> Any:
    """Use the continuous map so float32 gamma serialization is harmless."""
    return PLASMA(GAMMA_NORM(float(gamma)))


def _lookup_gamma(mapping: Mapping[str, Any], gamma: float) -> Any:
    for key in (_gamma_key(gamma), str(float(gamma)), str(gamma)):
        if key in mapping:
            return mapping[key]
    for key, value in mapping.items():
        try:
            if math.isclose(float(key), float(gamma), abs_tol=1e-8):
                return value
        except (TypeError, ValueError):
            pass
    return None


def _wilson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class RolloutRow:
    gamma: float
    path: np.ndarray
    success: bool
    collision: bool
    out_of_bounds: bool
    timeout: bool
    min_clearance_m: float
    time_to_goal_s: float | None
    mode: str
    seed: int | None
    temperature: float | None


@dataclass
class RunArtifacts:
    label: str
    root: Path
    history: list[dict[str, Any]]
    bundles: list[dict[str, Any]]
    final_rollouts: list[RolloutRow]
    final_audit: dict[str, Any]
    recipe: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_sha256: str
    final_round: int

    @property
    def final_bundle(self) -> dict[str, Any]:
        return self.bundles[-1]


@dataclass(frozen=True)
class GalleryArtifacts:
    label: str
    rows: tuple[RolloutRow, ...]
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_sha256: str


@dataclass(frozen=True)
class SealedValidity:
    payload: dict[str, Any]
    protocol: dict[str, Any]
    per_method: dict[str, dict[str, Any]]
    independent_full_aggregate: dict[str, Any]


RUN_LABELS = ("Full", "-AFE", "-Progress", "-SOCP")
CONTROL_ARMS = {
    "-AFE": {
        "arm": "minus_afe",
        "acquisition_mode": "uniform",
        "progress_ranking": True,
        "eligibility_mode": "full",
        "replay_eligibility": "full_safe",
        "runtime_safety_claim": True,
    },
    "-Progress": {
        "arm": "minus_progress",
        "acquisition_mode": "afe",
        "progress_ranking": False,
        "eligibility_mode": "full",
        "replay_eligibility": "full_safe",
        "runtime_safety_claim": True,
    },
    "-SOCP": {
        "arm": "minus_socp",
        "acquisition_mode": "afe",
        "progress_ranking": True,
        "eligibility_mode": "bounds_only_offline",
        "replay_eligibility": "strict_bounds",
        "runtime_safety_claim": False,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _strict_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactError(f"{field} must be an explicit integer count")
    result = int(value)
    if result < minimum:
        raise ArtifactError(f"{field} must be >= {minimum}")
    return result


def _exact_probability(value: Any, numerator: int, denominator: int, field: str) -> float:
    result = _finite(value)
    expected = numerator / denominator if denominator else math.nan
    if not math.isfinite(result) or denominator <= 0 or not math.isclose(
        result, expected, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ArtifactError(
            f"{field}={value!r} is inconsistent with integer counts "
            f"{numerator}/{denominator}"
        )
    return result


def _normalise_rollout(raw: Any) -> RolloutRow:
    row = _mapping(raw)
    states = row.get("states")
    path = row.get("path")
    if path is None and states is not None:
        path = np.asarray(states)[:, :2]
    if path is None:
        raise ArtifactError("rollout record has neither `path` nor `states`")
    path_array = np.asarray(path, dtype=np.float64)
    if path_array.ndim != 2 or path_array.shape[1] < 2 or len(path_array) == 0:
        raise ArtifactError(f"rollout path has invalid shape {path_array.shape}")
    path_array = path_array[:, :2].copy()
    in_bounds = row.get("in_bounds")
    out_of_bounds = row.get("out_of_bounds")
    if out_of_bounds is None and in_bounds is not None:
        out_of_bounds = not bool(in_bounds)
    reached = _bool(row.get("reached", row.get("success")))
    collision = _bool(row.get("collision"))
    out_of_bounds = _bool(out_of_bounds)
    fail_closed = _bool(row.get("fail_closed"))
    success = _bool(row.get("success"), reached and not collision and not out_of_bounds)
    success = success and not fail_closed
    timeout = _bool(row.get("timeout"), not success and not collision and not out_of_bounds)
    clearance = _finite(row.get("min_clearance_m", row.get("min_clearance")))
    time_goal = row.get("time_to_goal_s")
    if time_goal is None and success:
        actions = row.get("actions", row.get("executed_actions", ()))
        time_goal = len(actions) * 0.1
    return RolloutRow(
        gamma=float(row["gamma"]),
        path=path_array,
        success=success,
        collision=collision,
        out_of_bounds=out_of_bounds,
        timeout=timeout,
        min_clearance_m=clearance,
        time_to_goal_s=None if time_goal is None else _finite(time_goal),
        mode=str(row.get("detour_mode", row.get("mode", row.get("direction_class", "unresolved")))),
        seed=None if row.get("seed") is None else int(row["seed"]),
        temperature=None if row.get("temperature") is None else float(row["temperature"]),
    )


def _npz_rollouts(path: Path) -> list[RolloutRow]:
    with np.load(path, allow_pickle=True) as data:
        if "positions" in data and "offsets" in data:
            positions = np.asarray(data["positions"])
            offsets = np.asarray(data["offsets"], dtype=int)
            rows = []
            for index in range(len(offsets) - 1):
                raw = {
                    "path": positions[offsets[index] : offsets[index + 1]],
                    "gamma": data["gamma"][index],
                    "success": data["success"][index] if "success" in data else False,
                    "collision": data["collision"][index] if "collision" in data else False,
                    "in_bounds": data["in_bounds"][index] if "in_bounds" in data else True,
                    "mode": data["mode"][index] if "mode" in data else "unresolved",
                    "temperature": data["temperature"][index] if "temperature" in data else None,
                }
                rows.append(_normalise_rollout(raw))
            return rows
        for key in ("rollouts", "records", "rows"):
            if key in data:
                return [_normalise_rollout(row) for row in data[key].tolist()]
    raise ArtifactError(f"unsupported rollout NPZ schema: {path}")


def load_rollouts(path: str | Path, *, section: str | None = None) -> list[RolloutRow]:
    """Load rollout rows from a stage directory, PT/JSON/NPZ bundle, or list."""
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        preferred = (
            "data/rollouts_temp0.5.pt", "data/rollouts_t0.5.pt",
            "data/visualization_rollouts.pt", "data/baseline_rollouts.pt",
        )
        for name in preferred:
            candidate = source / name
            if candidate.exists():
                source = candidate
                break
        else:
            candidates = sorted(source.glob("data/round_*_bundle.pt"))
            if not candidates:
                candidates = sorted(source.rglob("*temp0.5*.npz"))
            if not candidates:
                raise ArtifactError(f"no rollout artifact found under {source}")
            source = candidates[-1]
    if source.suffix == ".npz":
        return _npz_rollouts(source)
    payload = _json(source) if source.suffix == ".json" else _torch(source)
    if isinstance(payload, list):
        raw_rows = payload
    else:
        body = _mapping(payload)
        if section is not None:
            raw_rows = body.get(section)
            if raw_rows is None:
                raise ArtifactError(f"{source} has no `{section}` rollout section")
        else:
            raw_rows = next(
                (body[key] for key in ("visualization_rollouts", "ordinary_rollouts", "rollouts", "records") if key in body),
                None,
            )
        if raw_rows is None:
            raise ArtifactError(f"{source} contains no rollout rows")
    rows = [_normalise_rollout(row) for row in raw_rows]
    if not rows:
        raise ArtifactError(f"rollout artifact is empty: {source}")
    return rows


def _run_root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.parent.name in {"logs", "data", "checkpoints"}:
        return path.parent.parent
    return path.parent


def _history_from_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    per_gamma: dict[str, Any] = {}
    for gamma in GAMMAS:
        rows = [
            _normalise_rollout(row)
            for row in bundle.get("ordinary_rollouts", ())
            if math.isclose(float(_mapping(row).get("gamma", math.nan)), gamma, abs_tol=1e-8)
        ]
        successes = [row for row in rows if row.success]
        mode_counts = Counter(row.mode for row in successes)
        per_gamma[_gamma_key(gamma)] = {
            "n": len(rows),
            "success_rate": sum(row.success for row in rows) / len(rows) if rows else math.nan,
            "collision_rate": sum(row.collision for row in rows) / len(rows) if rows else math.nan,
            "mode_counts_successes": dict(mode_counts),
        }
    return {
        "round": int(bundle.get("round", 0)),
        "audit": bundle.get("audit"),
        "query": bundle.get("query_summary"),
        "solver": bundle.get("solver"),
        "matrix": bundle.get("matrix"),
        # This summary is derived from the immutable embedded rollout rows.
        # Mutable ``logs/history.json`` is deliberately never consulted.
        "ordinary_per_gamma": per_gamma,
    }


def _validate_audit_counts(
    audit: Mapping[str, Any],
    label: str,
    *,
    expected_bank_role: str,
    expected_bank_fingerprint: str | None = None,
) -> None:
    prefix = f"{label} audit"
    if "temperature" not in audit or float(audit["temperature"]) != 1.0:
        raise ArtifactError(f"{prefix} must explicitly record temperature=1.0")
    if audit.get("uncertainty_tilting") is not False:
        raise ArtifactError(f"{prefix} must explicitly record uncertainty_tilting=false")
    if audit.get("sampling_distribution") != "ordinary_conditional_flow_iid":
        raise ArtifactError(f"{prefix} is not explicit ordinary conditional-flow IID sampling")
    if audit.get("context_bank_role") != expected_bank_role:
        raise ArtifactError(
            f"{prefix} bank role is {audit.get('context_bank_role')!r}, "
            f"expected {expected_bank_role!r}"
        )
    fingerprint = _sha256(audit.get("context_bank_fingerprint"), f"{prefix} bank fingerprint")
    if expected_bank_fingerprint is not None and fingerprint != expected_bank_fingerprint:
        raise ArtifactError(f"{prefix} bank fingerprint disagrees with run provenance")
    context_count = _strict_int(audit.get("context_count"), f"{prefix}.context_count", minimum=1)
    plans_per_context = _strict_int(
        audit.get("plans_per_context"), f"{prefix}.plans_per_context", minimum=1
    )
    total_calls = _strict_int(
        audit.get("total_verifier_calls"), f"{prefix}.total_verifier_calls", minimum=1
    )
    rows = _audit_rows(audit)
    if len(rows) != len(GAMMAS):
        raise ArtifactError(f"{prefix} must contain exactly all seven gamma rows")
    observed_gammas = tuple(float(row.get("gamma", math.nan)) for row in rows)
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-8)
        for observed, expected in zip(observed_gammas, GAMMAS)
    ):
        raise ArtifactError(f"{prefix} gamma rows do not match the fixed seven levels")
    expected_per_gamma = context_count * plans_per_context
    summed = 0
    for row in rows:
        gamma = float(row["gamma"])
        n = _strict_int(row.get("sample_count"), f"{prefix}[{gamma:g}].sample_count", minimum=1)
        safe = _strict_int(row.get("safe_count"), f"{prefix}[{gamma:g}].safe_count")
        progress = _strict_int(
            row.get("safe_progress_count"), f"{prefix}[{gamma:g}].safe_progress_count"
        )
        if n != expected_per_gamma:
            raise ArtifactError(
                f"{prefix}[{gamma:g}] sample_count={n}, expected "
                f"context_count*plans_per_context={expected_per_gamma}"
            )
        if not 0 <= progress <= safe <= n:
            raise ArtifactError(f"{prefix}[{gamma:g}] integer counts are not nested")
        _exact_probability(row.get("validity_mass"), safe, n, f"{prefix}[{gamma:g}].validity_mass")
        _exact_probability(
            row.get("progress_validity"), progress, n,
            f"{prefix}[{gamma:g}].progress_validity",
        )
        mode_counts = row.get("mode_counts")
        if not isinstance(mode_counts, Mapping):
            raise ArtifactError(f"{prefix}[{gamma:g}].mode_counts is missing")
        counted_modes = sum(
            _strict_int(value, f"{prefix}[{gamma:g}].mode_counts[{key!r}]")
            for key, value in mode_counts.items()
        )
        if counted_modes != safe:
            raise ArtifactError(
                f"{prefix}[{gamma:g}] mode counts {counted_modes} != safe_count {safe}"
            )
        coverage = _strict_int(
            row.get("safe_mode_coverage"), f"{prefix}[{gamma:g}].safe_mode_coverage"
        )
        if coverage != sum(value > 0 for value in mode_counts.values()):
            raise ArtifactError(f"{prefix}[{gamma:g}] safe_mode_coverage is inconsistent")
        summed += n
    if total_calls != summed:
        raise ArtifactError(f"{prefix} total_verifier_calls != sum of per-gamma sample_count")


def _fixed_gamma_claim(recipe: Mapping[str, Any], label: str) -> None:
    claim = str(recipe.get("gamma_distribution", "")).lower()
    if not all(token in claim for token in ("fixed", "uniform", "seven")) or "no schedule" not in claim:
        raise ArtifactError(f"{label}: recipe lacks an explicit fixed-uniform seven-gamma/no-schedule claim")
    if recipe.get("legacy_mechanisms") != clean_method_absence_manifest():
        raise ArtifactError(f"{label}: recipe clean-method absence claims are missing or inconsistent")


def _matched_protocol(run: RunArtifacts) -> dict[str, Any]:
    protocol = run.recipe.get("matched_protocol")
    if not isinstance(protocol, Mapping):
        raise ArtifactError(f"{run.label}: embedded recipe lacks matched_protocol")
    return dict(protocol)


def load_run(
    path: str | Path,
    label: str,
    *,
    require_full: bool = False,
    selected_checkpoint: str | Path | None = None,
) -> RunArtifacts:
    """Load one exact expansion horizon from a run directory.

    ``selected_checkpoint`` is an explicit scientific cutoff.  Its embedded
    round selects the contiguous bundle prefix ``round_000..round_N``; only
    strictly later exploratory bundles are ignored.  The checkpoint must live
    in this run's own ``checkpoints`` directory, and its filename, embedded
    round, final bundle, history, and recipe are all bound below.
    """
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    root = _run_root(source)
    if selected_checkpoint is None and source.is_file() and source.parent.name == "checkpoints":
        selected_checkpoint = source

    selected_checkpoint_path: Path | None = None
    selected_checkpoint_payload: dict[str, Any] | None = None
    selected_round: int | None = None
    if selected_checkpoint is not None:
        selected_checkpoint_path = Path(selected_checkpoint).resolve()
        if not selected_checkpoint_path.is_file():
            raise FileNotFoundError(selected_checkpoint_path)
        if selected_checkpoint_path.parent != (root / "checkpoints").resolve():
            raise ArtifactError(
                f"{label}: selected checkpoint is not inside this run's checkpoints directory"
            )
        selected_checkpoint_payload = _mapping(_torch(selected_checkpoint_path))
        selected_round = _strict_int(
            selected_checkpoint_payload.get("round"),
            f"{label} selected checkpoint round",
            minimum=1,
        )
        expected_name = f"round_{selected_round:03d}.pt"
        if selected_checkpoint_path.name != expected_name:
            raise ArtifactError(
                f"{label}: selected checkpoint filename does not bind embedded "
                f"round {selected_round}: expected {expected_name}"
            )

    bundle_paths: list[Path]
    if selected_round is not None:
        bundle_paths = [
            root / f"data/round_{round_index:03d}_bundle.pt"
            for round_index in range(selected_round + 1)
        ]
        missing = [str(item) for item in bundle_paths if not item.is_file()]
        if missing:
            raise ArtifactError(
                f"{label}: selected checkpoint round {selected_round} lacks "
                f"contiguous bundle prefix 0..N; missing {missing}"
            )
        expected = set(bundle_paths)
        for candidate in root.glob("data/round_*_bundle.pt"):
            stem = candidate.name
            prefix, suffix = "round_", "_bundle.pt"
            middle = stem[len(prefix) : -len(suffix)] if (
                stem.startswith(prefix) and stem.endswith(suffix)
            ) else ""
            if not middle.isdigit():
                raise ArtifactError(
                    f"{label}: noncanonical round bundle cannot be classified "
                    f"as a strictly later artifact: {candidate.name}"
                )
            candidate_round = int(middle)
            if candidate.name != f"round_{candidate_round:03d}_bundle.pt":
                raise ArtifactError(
                    f"{label}: noncanonical round bundle cannot be classified "
                    f"as a strictly later artifact: {candidate.name}"
                )
            if candidate_round <= selected_round and candidate not in expected:
                raise ArtifactError(
                    f"{label}: unexpected bundle at or before selected round: {candidate}"
                )
            # Canonically named rounds strictly greater than selected_round are
            # exploratory artifacts and intentionally do not affect this run.
    elif source.is_file() and source.suffix in {".pt", ".pth"} and "bundle" in source.name:
        # Preserve cumulative runtime accounting when a caller points at the
        # latest bundle rather than its run directory.
        siblings = sorted(root.glob("data/round_*_bundle.pt"))
        bundle_paths = siblings if source in siblings else [source]
    else:
        bundle_paths = sorted(root.glob("data/round_*_bundle.pt"))
    if not bundle_paths:
        raise ArtifactError(f"{label}: no AFE round bundles under {root}")
    bundles = [_mapping(_torch(item)) for item in bundle_paths]
    bundles.sort(key=lambda row: int(row.get("round", 0)))
    rounds = tuple(_strict_int(row.get("round"), f"{label} bundle round") for row in bundles)
    if rounds != tuple(range(rounds[-1] + 1)):
        raise ArtifactError(f"{label}: embedded round bundles are not contiguous from round zero")
    expected_schema = "afe_expansion_round_v1" if label == "Full" else "afe_matched_ablation_round_v1"
    for bundle in bundles:
        if bundle.get("schema_version") != expected_schema:
            raise ArtifactError(f"{label}: unexpected embedded bundle schema")
    embedded_recipes = [_mapping(bundle.get("recipe")) for bundle in bundles]
    if any(not recipe for recipe in embedded_recipes):
        raise ArtifactError(f"{label}: every bundle must embed its immutable recipe")
    recipe = embedded_recipes[0]
    if any(candidate != recipe for candidate in embedded_recipes[1:]):
        raise ArtifactError(f"{label}: embedded recipes changed across rounds")
    _fixed_gamma_claim(recipe, label)
    if recipe.get("ordinary_audit_untilted") is not True:
        raise ArtifactError(f"{label}: recipe lacks the explicit ordinary-untilted audit claim")
    if recipe.get("visualization_temperature") != 0.5:
        raise ArtifactError(f"{label}: recipe does not separate the T=0.5 gallery")
    scientific_temperature = (
        recipe.get("sampling_temperature")
        if label == "Full"
        else recipe.get("expansion_temperature")
    )
    if scientific_temperature != 1.0:
        raise ArtifactError(f"{label}: recipe scientific sampling temperature is not T=1")
    if label != "Full" and recipe.get("audit_temperature") != 1.0:
        raise ArtifactError(f"{label}: recipe audit temperature is not T=1")
    expected_arm = CONTROL_ARMS.get(label)
    if label == "Full":
        if recipe.get("method") != "planned-window AFE":
            raise ArtifactError("Full: embedded method label is not planned-window AFE")
        expected_full = {
            "arm": "full",
            "acquisition_mode": "afe",
            "progress_ranking": True,
            "eligibility_mode": "full",
            "replay_eligibility": "full_safe",
            "runtime_safety_claim": True,
            "uncertainty_tilting": True,
            "ordinary_audit_untilted": True,
        }
        if any(recipe.get(key) != value for key, value in expected_full.items()):
            raise ArtifactError("Full: embedded arm switches do not identify the Full method")
        if any(bundle.get("arm") not in (None, "full") for bundle in bundles):
            raise ArtifactError("Full: embedded bundle arm label is inconsistent")
    elif expected_arm is None:
        raise ArtifactError(f"unknown scientific run label {label!r}")
    else:
        for key, expected in expected_arm.items():
            if recipe.get(key) != expected:
                raise ArtifactError(
                    f"{label}: recipe field {key}={recipe.get(key)!r}, expected {expected!r}"
                )
        if any(bundle.get("arm") != expected_arm["arm"] for bundle in bundles):
            raise ArtifactError(f"{label}: embedded bundle arm label is inconsistent")
    source_checkpoint_sha256 = _sha256(
        recipe.get("source_checkpoint_sha256"), f"{label} source checkpoint"
    )
    del source_checkpoint_sha256
    _sha256(recipe.get("source_model_hash"), f"{label} source model")
    bank_fingerprint = _sha256(
        recipe.get("audit_bank_fingerprint"), f"{label} round-monitoring bank"
    )
    if recipe.get("audit_bank_role") != "round_monitoring":
        raise ArtifactError(f"{label}: embedded recipe did not use round_monitoring")
    _sha256(recipe.get("verifier_spec_fingerprint"), f"{label} verifier specification")

    history = [_history_from_bundle(bundle) for bundle in bundles]
    final = bundles[-1]
    raw_rollouts = final.get("ordinary_rollouts")
    audit = _mapping(final.get("audit"))
    if not raw_rollouts:
        raise ArtifactError(f"{label}: final bundle has no ordinary T=1 rollouts")
    if not audit or not audit.get("per_gamma"):
        raise ArtifactError(
            f"{label}: final bundle has no independent held-out audit artifact "
            "(its interval scope must still be labeled explicitly)"
        )
    for round_index, bundle in zip(rounds, bundles):
        embedded_audit = _mapping(bundle.get("audit"))
        _validate_audit_counts(
            embedded_audit,
            f"{label} round {round_index}",
            expected_bank_role="round_monitoring",
            expected_bank_fingerprint=bank_fingerprint,
        )
    rollouts = [_normalise_rollout(row) for row in raw_rollouts]
    _require_scientific_temperature(rollouts, label)
    if {round(row.gamma, 8) for row in rollouts} != {round(float(gamma), 8) for gamma in GAMMAS}:
        raise ArtifactError(f"{label}: final ordinary rollouts do not cover all seven gammas")

    final_round = rounds[-1]
    checkpoint_path = (
        selected_checkpoint_path
        if selected_checkpoint_path is not None
        else root / f"checkpoints/round_{final_round:03d}.pt"
    )
    if not checkpoint_path.is_file():
        raise ArtifactError(f"{label}: final round checkpoint is missing")
    checkpoint = (
        selected_checkpoint_payload
        if selected_checkpoint_payload is not None
        else _mapping(_torch(checkpoint_path))
    )
    if _strict_int(checkpoint.get("round"), f"{label} checkpoint round") != final_round:
        raise ArtifactError(f"{label}: final checkpoint round disagrees with final bundle")
    if _mapping(checkpoint.get("recipe")) != recipe:
        raise ArtifactError(f"{label}: final checkpoint recipe differs from embedded bundles")
    model_hash = _sha256(checkpoint.get("current_model_hash"), f"{label} final model")
    checkpoint_history = checkpoint.get("history")
    if not isinstance(checkpoint_history, list) or not checkpoint_history:
        raise ArtifactError(f"{label}: final checkpoint lacks embedded round history")
    checkpoint_final = _mapping(checkpoint_history[-1])
    if (
        _strict_int(checkpoint_final.get("round"), f"{label} checkpoint history round") != final_round
        or checkpoint_final.get("model_hash") != model_hash
    ):
        raise ArtifactError(f"{label}: final checkpoint model/round provenance is inconsistent")
    if require_full:
        expanded = [row for row in history if row.get("query")]
        if not expanded:
            raise ArtifactError("Full: no expansion round with verifier-query accounting")
        if recipe.get("acquisition") != "afe":
            raise ArtifactError("Full: recipe is not AFE uncertainty acquisition")
        store = _mapping(final.get("store_state"))
        uncertainty = _mapping(store.get("uncertainty"))
        records = list(store.get("records", ()))
        if store and int(uncertainty.get("count", len(records))) != len(records):
            raise ArtifactError("Full: cumulative A count differs from unique verifier records")
        for row in expanded:
            solver = _mapping(row.get("solver"))
            sampling = str(solver.get("sampling", ""))
            if solver and sampling and "uniform" not in sampling:
                raise ArtifactError("Full: replay solver is not a uniform positive full pass")
    return RunArtifacts(
        label, root, history, bundles, rollouts, audit, recipe,
        checkpoint_path, _sha256_file(checkpoint_path), model_hash, final_round,
    )


def _require_visual_temperature(rows: Sequence[RolloutRow], label: str) -> None:
    temperatures = {row.temperature for row in rows if row.temperature is not None}
    if not temperatures:
        raise ArtifactError(f"{label}: visualization rollouts do not record temperature")
    if temperatures != {0.5}:
        raise ArtifactError(f"{label}: gallery must use only T=0.5, found {temperatures}")


def _require_scientific_temperature(
    rows: Sequence[RolloutRow], label: str, *, allow_unrecorded: bool = False
) -> None:
    temperatures = {row.temperature for row in rows if row.temperature is not None}
    missing = sum(row.temperature is None for row in rows)
    if missing and not allow_unrecorded:
        raise ArtifactError(f"{label}: scientific rollouts do not record temperature")
    if temperatures and temperatures != {1.0}:
        raise ArtifactError(f"{label}: primary metrics must use only T=1, found {temperatures}")


def _validate_matched_runs(runs: Mapping[str, RunArtifacts]) -> None:
    if tuple(runs) != RUN_LABELS:
        raise ArtifactError(f"final reports require run order/pairs {RUN_LABELS}, got {tuple(runs)}")
    provenance_fields = (
        "source_checkpoint",
        "source_checkpoint_sha256",
        "source_model_hash",
        "audit_bank",
        "audit_bank_fingerprint",
        "audit_bank_role",
        "verifier_spec_fingerprint",
    )
    reference = runs["Full"]
    for field in provenance_fields:
        expected = reference.recipe.get(field)
        mismatched = [label for label, run in runs.items() if run.recipe.get(field) != expected]
        if mismatched:
            raise ArtifactError(
                f"matched runs disagree on {field}; mismatched arms: {', '.join(mismatched)}"
            )
    protocol = _matched_protocol(reference)
    for label, run in runs.items():
        if _matched_protocol(run) != protocol:
            raise ArtifactError(f"{label}: matched_protocol differs from Full")
    backup = reference.recipe.get("backup_planner")
    if not isinstance(backup, Mapping):
        raise ArtifactError("Full: recipe lacks backup_planner provenance")
    for label, run in runs.items():
        if run.recipe.get("backup_planner") != backup:
            raise ArtifactError(f"{label}: backup planner differs from Full")


def load_gallery(
    path: str | Path,
    label: str,
    run: RunArtifacts,
) -> GalleryArtifacts:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _mapping(_torch(source))
    if payload.get("schema_version") != "afe_gallery_rollouts_v1":
        raise ArtifactError(f"{label}: gallery lacks afe_gallery_rollouts_v1 provenance")
    expected_gallery_label = "-SOCP (offline only)" if label == "-SOCP" else label
    if payload.get("label") != expected_gallery_label:
        raise ArtifactError(
            f"{label}: gallery label {payload.get('label')!r} != {expected_gallery_label!r}"
        )
    if payload.get("visualization_temperature") != 0.5:
        raise ArtifactError(f"{label}: gallery does not explicitly declare T=0.5")
    if payload.get("scientific_use_forbidden") is not True:
        raise ArtifactError(f"{label}: gallery is not marked visualization-only")
    if payload.get("scientific_metrics_source_temperature") != 1.0:
        raise ArtifactError(f"{label}: gallery lacks the distinct T=1 scientific-source claim")
    rows = tuple(_normalise_rollout(row) for row in payload.get("visualization_rollouts", ()))
    if not rows:
        raise ArtifactError(f"{label}: gallery contains no rollouts")
    _require_visual_temperature(rows, label)
    if {round(row.gamma, 8) for row in rows} != {round(float(gamma), 8) for gamma in GAMMAS}:
        raise ArtifactError(f"{label}: gallery does not cover all seven gammas")
    checkpoint_path = Path(str(payload.get("checkpoint", ""))).resolve()
    if checkpoint_path != run.checkpoint_path.resolve():
        raise ArtifactError(f"{label}: gallery was not generated from the displayed run checkpoint")
    declared_file_hash = _sha256(payload.get("checkpoint_sha256"), f"{label} gallery checkpoint")
    if declared_file_hash != run.checkpoint_sha256 or _sha256_file(checkpoint_path) != declared_file_hash:
        raise ArtifactError(f"{label}: gallery checkpoint file provenance mismatch")
    # A checkpoint file hash binds the gallery to the exact checkpoint bytes;
    # the model hash is then read from those same immutable bytes and matched
    # to the run's embedded final-model provenance.
    checkpoint = _mapping(_torch(checkpoint_path))
    model_hash = _sha256(checkpoint.get("current_model_hash"), f"{label} gallery model")
    if model_hash != run.model_state_sha256:
        raise ArtifactError(f"{label}: gallery checkpoint model differs from final run model")
    if payload.get("model_state_sha256") not in (None, model_hash):
        raise ArtifactError(f"{label}: gallery's optional model hash is inconsistent")
    return GalleryArtifacts(
        label=label,
        rows=rows,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=declared_file_hash,
        model_state_sha256=model_hash,
    )


def load_id_demo_paths(path: str | Path) -> list[RolloutRow]:
    """Reconstruct exact ID paths from selected planned targets and U[0]."""
    source = Path(path).resolve()
    if source.is_dir():
        candidates = list(source.glob("data/planned_id_balanced.pt")) + list(source.glob("planned_id_balanced.pt"))
        if not candidates:
            raise ArtifactError(f"no planned ID demo dataset under {source}")
        source = candidates[0]
    payload = _mapping(_torch(source))
    required = {"U", "window_trajectory_ids", "window_steps", "gamma", "trajectory_rows"}
    missing = required - payload.keys()
    if missing:
        raise ArtifactError(f"ID planned-demo artifact is missing {sorted(missing)}")
    plans = np.asarray(payload["U"])
    trajectory_ids = np.asarray(payload["window_trajectory_ids"], dtype=int)
    steps = np.asarray(payload["window_steps"], dtype=int)
    gammas = np.asarray(payload["gamma"], dtype=float)
    seeds = np.asarray(payload.get("window_seeds", np.full(len(plans), -1)), dtype=int)
    metadata = {int(row["trajectory_id"]): _mapping(row) for row in payload["trajectory_rows"]}
    rows: list[RolloutRow] = []
    start = np.asarray(payload.get("start", START), dtype=float).reshape(2)
    for trajectory_id in sorted(set(trajectory_ids.tolist())):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        indices = indices[np.argsort(steps[indices])]
        if not np.array_equal(steps[indices], np.arange(len(indices))):
            raise ArtifactError(f"ID trajectory {trajectory_id} has noncontiguous real windows")
        state = np.asarray([start[0], start[1], 0.0, 0.0], dtype=float)
        path_rows = [state[:2].copy()]
        for index in indices:
            state = step_state(state, plans[index, 0], 0.1)
            path_rows.append(state[:2].copy())
        meta = metadata.get(trajectory_id, {})
        rows.append(RolloutRow(
            gamma=float(gammas[indices[0]]), path=np.asarray(path_rows), success=True,
            collision=False, out_of_bounds=False, timeout=False,
            min_clearance_m=_finite(meta.get("min_clearance_m")), time_to_goal_s=len(indices) * 0.1,
            mode=str(meta.get("direction_class", "unresolved")),
            seed=int(meta.get("seed", seeds[indices[0]])), temperature=None,
        ))
    for gamma in GAMMAS:
        selected = [row for row in rows if math.isclose(row.gamma, gamma, abs_tol=1e-6)]
        counts = Counter(row.mode for row in selected)
        if not selected or counts["R-first"] != counts["U-first"]:
            raise ArtifactError(f"ID demonstrations are not R/U balanced for gamma={gamma:g}")
    return rows


def _scene(axis: Any, title: str, *, giant: bool, bold: bool = False) -> None:
    env = make_ood_scene(radius=1.2) if giant else make_id_scene()
    axis.set_facecolor("#f8f7f4")
    rr = float(env.r_robot)
    for obstacle in env.obstacles.detach().cpu().numpy():
        is_giant = giant and np.linalg.norm(obstacle[:2] - GIANT_CENTER) < 1e-6
        axis.add_patch(Circle(
            obstacle[:2], float(obstacle[2]) + rr,
            color="#686868" if is_giant else "#cccccc",
            ec="#b2182b" if is_giant else "none", lw=1.4 if is_giant else 0, zorder=1,
        ))
    axis.plot(*START, "ks", ms=5.5, zorder=8)
    axis.plot(*GOAL, "*", color="gold", mec="k", ms=13, zorder=8)
    axis.set(xlim=(-0.42, 5.42), ylim=(-0.42, 5.42), aspect="equal")
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_title(title, pad=6, fontsize=17, fontweight="bold" if bold else "normal")


def _pick(rows: Sequence[RolloutRow], gamma: float, count: int = 3) -> list[RolloutRow]:
    group = [row for row in rows if math.isclose(row.gamma, gamma, abs_tol=1e-7)]
    good = [row for row in group if row.success]
    bad = [row for row in group if not row.success]
    return (good[:2] + bad[:count])[:count] or group[:count]


def _draw_rows(axis: Any, title: str, rows: Sequence[RolloutRow] | None, *, bold: bool = False) -> None:
    _scene(axis, title, giant=True, bold=bold)
    if not rows:
        axis.text(.5, .48, "artifact not provided", transform=axis.transAxes,
                  ha="center", va="center", color="0.45", fontsize=11)
        return
    shown = (0.1, 0.5, 1.0)
    for gamma in shown:
        for row in _pick(rows, gamma):
            axis.plot(row.path[:, 0], row.path[:, 1], color=GAMMA_COLOR[gamma],
                      lw=1.45, alpha=.92, zorder=3)
            axis.plot(row.path[::3, 0], row.path[::3, 1], ".", color="k", ms=1.5, alpha=.45, zorder=4)
            if not row.success:
                axis.plot(*row.path[-1], "x", color="#cc3311", ms=8, mew=2.1, zorder=7)


def rollout_figure(
    output: Path,
    *,
    id_rows: Sequence[RolloutRow],
    expert: Sequence[RolloutRow] | None,
    pretrained: Sequence[RolloutRow] | None,
    mizuta: Sequence[RolloutRow] | None,
    galleries: Mapping[str, Sequence[RolloutRow] | None],
) -> None:
    matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 12.5})
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 10.0))
    _scene(axes[0, 0], "ID balanced planned demos", giant=False)
    for row in id_rows:
        axes[0, 0].plot(row.path[:, 0], row.path[:, 1], color=_gamma_color(row.gamma),
                        lw=.82, alpha=.24, zorder=3)
    axes[0, 0].plot(*START, "o", color="k", ms=4, label="start seed", zorder=9)
    axes[0, 0].plot(*GOAL, "*", color="gold", mec="k", ms=8, label="goal seed", zorder=9)
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    _draw_rows(axes[0, 1], "OOD Expert", expert)
    _draw_rows(axes[0, 2], "Pretrained", pretrained)
    _draw_rows(axes[0, 3], r"CFM-MPPI$^{*}$ / Mizuta", mizuta)
    _draw_rows(axes[1, 0], "-SOCP", galleries.get("-SOCP"))
    _draw_rows(axes[1, 1], "-Progress", galleries.get("-Progress"))
    _draw_rows(axes[1, 2], "-AFE", galleries.get("-AFE"))
    _draw_rows(axes[1, 3], "Full", galleries.get("Full"), bold=True)
    scalar = plt.cm.ScalarMappable(cmap=PLASMA, norm=GAMMA_NORM); scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=axes, location="right", fraction=.022, pad=.02, ticks=GAMMAS)
    cbar.set_label(r"safety level $\gamma$", fontsize=13)
    fig.text(.5, .02, "Rollout gallery: T=0.5 visualization only; all reported metrics use ordinary T=1.",
             ha="center", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _audit_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = audit.get("per_gamma", ())
    if isinstance(rows, Mapping):
        values = []
        for key, raw in rows.items():
            row = _mapping(raw)
            row.setdefault("gamma", float(key))
            values.append(row)
        rows = values
    return sorted((_mapping(row) for row in rows), key=lambda row: float(row["gamma"]))


def _last_solver_trace(solver: Mapping[str, Any]) -> dict[str, Any]:
    trace = list(solver.get("trace", ()))
    return _mapping(trace[-1]) if trace else {}


def internals_figure(output: Path, run: RunArtifacts) -> None:
    history = run.history
    expanded = [row for row in history if row.get("query")]
    rounds = np.asarray([int(row.get("round", index)) for index, row in enumerate(expanded)], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2))

    a = axes[0, 0]
    acceptance = [_finite(_mapping(row["query"]).get("query_acceptance")) for row in expanded]
    a.plot(rounds, acceptance, "-o", color="#009988", lw=2, ms=4)
    a.set(title="(A) uncertainty-tilted query acceptance", xlabel="expansion round", ylabel="accepted / queried")
    a.set_ylim(-.02, 1.02); a.grid(alpha=.3)
    a.text(.03, .05, "query efficiency — not model validity", transform=a.transAxes, fontsize=9)

    b = axes[0, 1]
    for gamma in GAMMAS:
        validity, performance = [], []
        for row in expanded:
            match = next((item for item in _audit_rows(_mapping(row.get("audit")))
                          if math.isclose(float(item["gamma"]), gamma, abs_tol=1e-8)), None)
            validity.append(_finite(match.get("validity_mass")) if match else math.nan)
            performance.append(_finite(match.get("progress_validity")) if match else math.nan)
        b.plot(rounds, validity, color=GAMMA_COLOR[gamma], lw=1.8, label=rf"$\gamma={gamma:g}$")
        b.plot(rounds, performance, color=GAMMA_COLOR[gamma], lw=1.2, ls="--")
    b.set(title="(B) round-monitoring validity (ordinary T=1)", xlabel="expansion round", ylabel="mass")
    b.set_ylim(-.02, 1.02); b.grid(alpha=.3); b.legend(fontsize=7.5, ncol=2)
    b.text(.03, .04, "solid V   dashed Vprog", transform=b.transAxes, fontsize=9)

    c = axes[0, 2]
    positives = np.cumsum([int(_mapping(row["query"]).get("new_positive_queries", 0)) for row in expanded])
    negatives = np.cumsum([int(_mapping(row["query"]).get("new_negative_queries", 0)) for row in expanded])
    c.plot(rounds, positives, "-o", color="#009944", ms=3, label="verified +")
    c.plot(rounds, negatives, "-o", color="#cc3311", ms=3, label="verified −")
    c.set(title="(C) cumulative full-verifier queries", xlabel="expansion round", ylabel="unique planned windows")
    c.grid(alpha=.3); c.legend(fontsize=9)

    d = axes[1, 0]
    logdet = [_finite(_mapping(row.get("matrix")).get("logdet")) for row in expanded]
    eig_min = [_finite(_mapping(row.get("matrix")).get("eigenvalue_min")) for row in expanded]
    eig_max = [_finite(_mapping(row.get("matrix")).get("eigenvalue_max")) for row in expanded]
    d.plot(rounds, logdet, color="#0072B2", lw=2, label=r"$\log\det A_n$")
    d2 = d.twinx()
    d2.plot(rounds, eig_min, "--", color="#6a3d9a", label=r"$\lambda_{min}$")
    d2.plot(rounds, eig_max, "-", color="#e31a1c", label=r"$\lambda_{max}$")
    d.set(title=r"(D) cumulative fixed-feature $A_n$", xlabel="expansion round", ylabel="log determinant")
    d2.set_ylabel("eigenvalue"); d.grid(alpha=.3)
    d.legend(loc="upper left", fontsize=8); d2.legend(loc="lower right", fontsize=8)

    e = axes[1, 1]
    solver_rows = [_mapping(row.get("solver")) for row in expanded]
    traces = [_last_solver_trace(row) for row in solver_rows]
    losses = [_finite(row.get("cfm_loss")) for row in traces]
    update = [_finite(solver.get("final_update_norm", trace.get("update_norm"))) for solver, trace in zip(solver_rows, traces)]
    coverage = [_finite(trace.get("positive_coverage"), 0.0) for trace in traces]
    e.plot(rounds, losses, color="#D55E00", lw=1.7, label="CFM loss")
    e.plot(rounds, update, color="#0072B2", lw=1.7, label="update norm")
    positive = [value for value in losses + update if math.isfinite(value) and value > 0]
    if positive:
        e.set_yscale("log")
    e2 = e.twinx(); e2.plot(rounds, coverage, ":s", color="#009944", ms=3, label="uniform replay coverage")
    e2.set_ylim(-.03, 1.03); e2.set_ylabel("positive-ledger coverage")
    e.set(title="(E) proximal solve / uniform replay", xlabel="expansion round")
    e.grid(alpha=.3); e.legend(loc="upper left", fontsize=8); e2.legend(loc="lower right", fontsize=8)

    f = axes[1, 2]
    fallback = [_finite(_mapping(row["query"]).get("fallback_frequency"), 0.0) for row in expanded]
    failclosed = [int(_mapping(row["query"]).get("fail_closed_episodes", 0)) /
                  max(1, int(_mapping(row["query"]).get("episodes", 0))) for row in expanded]
    modes = []
    for row in expanded:
        per_gamma = _mapping(row.get("ordinary_per_gamma"))
        union: set[str] = set()
        for summary in per_gamma.values():
            union.update(key for key, value in _mapping(summary).get("mode_counts_successes", {}).items() if value)
        modes.append(len(union))
    f.plot(rounds, fallback, color="#f0ad00", lw=1.8, label="certified fallback / step")
    f.plot(rounds, failclosed, color="#cc3311", lw=1.8, label="fail-closed / episode")
    f2 = f.twinx(); f2.plot(rounds, modes, "--o", color="#6a3d9a", ms=3, label="successful modes")
    f.set_ylim(-.03, 1.03); f2.set_ylim(-.1, max(2.1, max(modes, default=0) + .2))
    f.set(title="(F) runtime backup / behavioral diversity", xlabel="expansion round", ylabel="frequency")
    f2.set_ylabel("mode coverage"); f.grid(alpha=.3)
    f.legend(loc="upper left", fontsize=8); f2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Planned-window AFE — training internals (T=1 acquisition and audit)", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _rollout_summary(rows: Sequence[RolloutRow]) -> dict[str, Any]:
    if not rows:
        return {}
    successes = [row for row in rows if row.success]
    return {
        "n": len(rows),
        "successes": sum(row.success for row in rows),
        "collisions": sum(row.collision for row in rows),
        "success_rate": sum(row.success for row in rows) / len(rows),
        "collision_rate": sum(row.collision for row in rows) / len(rows),
        "clearance": float(np.nanmean([row.min_clearance_m for row in successes])) if successes else math.nan,
        "time": float(np.nanmean([row.time_to_goal_s for row in successes if row.time_to_goal_s is not None])) if successes else math.nan,
        "modes": len({row.mode for row in successes if row.mode not in {"unresolved", "unclassified", "diagonal"}}),
    }


def scatter_figure(output: Path, methods: Mapping[str, RunArtifacts | Sequence[RolloutRow] | None]) -> None:
    fig, (reliability, quality) = plt.subplots(1, 2, figsize=(16.2, 5.4))
    handles = []
    for zorder, (name, artifact) in enumerate(methods.items(), start=3):
        if artifact is None:
            continue
        rows = artifact.final_rollouts if isinstance(artifact, RunArtifacts) else list(artifact)
        marker, size = METHOD_STYLE[name]
        for gamma in GAMMAS:
            summary = _rollout_summary([row for row in rows if math.isclose(row.gamma, gamma, abs_tol=1e-8)])
            if not summary:
                continue
            kwargs = {"edgecolors": "k", "linewidths": .8} if marker == "*" else {}
            reliability.scatter(100 * summary["success_rate"], 100 * summary["collision_rate"],
                                c=[GAMMA_COLOR[gamma]], marker=marker, s=size, zorder=zorder, **kwargs)
            if math.isfinite(summary["clearance"]) and math.isfinite(summary["time"]):
                quality.scatter(summary["time"], summary["clearance"], c=[GAMMA_COLOR[gamma]],
                                marker=marker, s=size, zorder=zorder, **kwargs)
        handles.append(Line2D([], [], color="#666", marker=marker, ls="", ms=11 if marker == "*" else 8, label=name))
    reliability.set(xlabel="success rate SR [%]", ylabel="collision rate CR [%]", xlim=(-5, 105), ylim=(-3, 105))
    quality.set(xlabel="time to goal [s]", ylabel="min clearance (successes) [m]")
    reliability.grid(alpha=.3); quality.grid(alpha=.3)
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)), frameon=False,
               bbox_to_anchor=(.5, 1.03), fontsize=10.5)
    scalar = plt.cm.ScalarMappable(cmap=PLASMA, norm=GAMMA_NORM); scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=[reliability, quality], location="right", fraction=.025, pad=.015, ticks=GAMMAS)
    cbar.set_label(r"safety level $\gamma$")
    fig.text(.5, .01, "Ordinary T=1 metrics.  Uncertainty is acquisition-only and is not encoded here.", ha="center", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)


def _runtime_by_gamma(run: RunArtifacts, gamma: float | None) -> dict[str, float]:
    explicit_claim = run.recipe.get("runtime_safety_claim")
    if explicit_claim is None:
        explicit_claim = run.final_bundle.get("runtime_safety_claim")
    if explicit_claim is None and run.history:
        explicit_claim = run.history[-1].get("runtime_safety_claim")
    runtime_claim = (
        bool(explicit_claim)
        if explicit_claim is not None
        else run.label != "-SOCP"
    )
    if not runtime_claim:
        return {"fallback": math.nan, "failclosed": math.nan, "claim": False}
    episodes: list[Any] = []
    for bundle in run.bundles[1:]:
        episodes.extend(bundle.get("episodes", ()))
    selected = []
    for raw in episodes:
        row = _mapping(raw)
        if gamma is None or math.isclose(float(row.get("gamma", math.nan)), gamma, abs_tol=1e-8):
            selected.append(row)
    actions = sum(len(np.asarray(row.get("actions", ())).reshape(-1, 2)) for row in selected)
    decisions = sum(
        len(row.get("traces", ()))
        if row.get("traces") is not None
        else len(np.asarray(row.get("actions", ())).reshape(-1, 2))
        for row in selected
    )
    fallback = sum(int(row.get("fallback_steps", 0)) for row in selected)
    return {
        "fallback": fallback / decisions if decisions else math.nan,
        "control_decisions": decisions,
        "executed_actions": actions,
        "failclosed": sum(_bool(row.get("fail_closed")) for row in selected) / len(selected) if selected else math.nan,
        "claim": True,
    }


def _audit_summary(audit: Mapping[str, Any], gamma: float | None) -> dict[str, Any]:
    rows = _audit_rows(audit)
    if gamma is not None:
        rows = [row for row in rows if math.isclose(float(row["gamma"]), gamma, abs_tol=1e-8)]
    n = sum(_strict_int(row.get("sample_count"), "audit.sample_count", minimum=1) for row in rows)
    safe = sum(_strict_int(row.get("safe_count"), "audit.safe_count") for row in rows)
    progress = sum(
        _strict_int(row.get("safe_progress_count"), "audit.safe_progress_count")
        for row in rows
    )
    if not 0 <= progress <= safe <= n:
        raise ArtifactError("audit aggregate integer counts are not nested")
    return {
        "n": n, "safe": safe, "progress": progress,
        "V": safe / n if n else math.nan, "V_ci": _wilson(safe, n),
        "Vprog": progress / n if n else math.nan, "Vprog_ci": _wilson(progress, n),
    }


def _validate_runtime_record(raw: Any, label: str, *, expected_available: bool) -> dict[str, Any]:
    runtime = _mapping(raw)
    if runtime.get("available") is not expected_available:
        raise ArtifactError(f"{label}: sealed runtime availability is inconsistent")
    decisions = _strict_int(runtime.get("control_decisions"), f"{label}.control_decisions")
    episodes = _strict_int(runtime.get("episode_count"), f"{label}.episode_count")
    fallback = _strict_int(runtime.get("fallback_steps"), f"{label}.fallback_steps")
    failclosed = _strict_int(
        runtime.get("failclosed_episodes"), f"{label}.failclosed_episodes"
    )
    if fallback > decisions or failclosed > episodes:
        raise ArtifactError(f"{label}: sealed runtime counts exceed their denominators")
    if expected_available:
        if decisions <= 0 or episodes <= 0:
            raise ArtifactError(f"{label}: available runtime accounting has zero exposure")
        _exact_probability(
            runtime.get("fallback_frequency"), fallback, decisions,
            f"{label}.fallback_frequency",
        )
        _exact_probability(
            runtime.get("failclosed_frequency"), failclosed, episodes,
            f"{label}.failclosed_frequency",
        )
    else:
        if any((decisions, episodes, fallback, failclosed)):
            raise ArtifactError(f"{label}: unavailable runtime accounting must use explicit zero counts")
        if runtime.get("fallback_frequency") is not None or runtime.get("failclosed_frequency") is not None:
            raise ArtifactError(f"{label}: unavailable runtime frequencies must be null")
    if not isinstance(runtime.get("source"), str) or not runtime["source"]:
        raise ArtifactError(f"{label}: sealed runtime source is missing")
    return runtime


def _validate_sealed_aggregate(
    raw: Any,
    audit: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    aggregate = _mapping(raw)
    audit_counts = _audit_summary(audit, None)
    n = _strict_int(aggregate.get("sample_count"), f"{label}.aggregate.sample_count", minimum=1)
    safe = _strict_int(aggregate.get("safe_count"), f"{label}.aggregate.safe_count")
    progress = _strict_int(
        aggregate.get("safe_progress_count"), f"{label}.aggregate.safe_progress_count"
    )
    if (n, safe, progress) != (
        audit_counts["n"], audit_counts["safe"], audit_counts["progress"]
    ):
        raise ArtifactError(f"{label}: sealed aggregate counts disagree with sealed audit rows")
    _exact_probability(aggregate.get("V"), safe, n, f"{label}.aggregate.V")
    _exact_probability(aggregate.get("Vprog"), progress, n, f"{label}.aggregate.Vprog")
    modes = aggregate.get("valid_modes")
    if not isinstance(modes, list) or any(not isinstance(mode, str) or not mode for mode in modes):
        raise ArtifactError(f"{label}: aggregate.valid_modes must be an explicit string list")
    if len(set(modes)) != len(modes):
        raise ArtifactError(f"{label}: aggregate.valid_modes contains duplicates")
    coverage = _strict_int(
        aggregate.get("valid_mode_coverage_count"),
        f"{label}.aggregate.valid_mode_coverage_count",
    )
    if coverage != len(modes):
        raise ArtifactError(f"{label}: valid-mode coverage count disagrees with valid_modes")
    fraction = _finite(aggregate.get("valid_mode_coverage_fraction"))
    if not 0.0 <= fraction <= 1.0:
        raise ArtifactError(f"{label}: valid-mode coverage fraction is not a probability")
    return aggregate


def _validate_interval(raw: Any, field: str) -> dict[str, Any]:
    interval = _mapping(raw)
    values = {key: _finite(interval.get(key)) for key in ("low", "mean", "high")}
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()):
        raise ArtifactError(f"{field} must contain finite probability low/mean/high")
    if not values["low"] <= values["mean"] <= values["high"]:
        raise ArtifactError(f"{field} low/mean/high are not ordered")
    return interval


def load_sealed_validity(
    path: str | Path,
    runs: Mapping[str, RunArtifacts],
) -> SealedValidity:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _mapping(_json(source) if source.suffix == ".json" else _torch(source))
    if payload.get("schema_version") != "afe_sealed_validity_v1":
        raise ArtifactError("sealed validity artifact has the wrong schema")
    bank_fingerprint = _sha256(payload.get("bank_fingerprint"), "sealed bank_fingerprint")
    protocol_fingerprint = _sha256(
        payload.get("protocol_fingerprint"), "sealed protocol_fingerprint"
    )
    protocol = _mapping(payload.get("protocol"))
    if _sha256(protocol.get("protocol_fingerprint"), "sealed protocol fingerprint") != protocol_fingerprint:
        raise ArtifactError("sealed protocol fingerprint alias is inconsistent")
    fingerprint_payload = dict(protocol)
    fingerprint_payload.pop("protocol_fingerprint", None)
    recomputed_protocol = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recomputed_protocol != protocol_fingerprint:
        raise ArtifactError("sealed protocol fingerprint does not hash its protocol")
    nested_bank = protocol.get("bank")
    if isinstance(nested_bank, Mapping):
        bank = dict(nested_bank)
        nested_fingerprint = bank.get("fingerprint")
        bank_role = bank.get("role")
        context_count = bank.get("context_count")
        bank_path = bank.get("path")
        bank_file_sha256 = bank.get("file_sha256")
    else:
        nested_fingerprint = protocol.get("context_bank_fingerprint")
        bank_role = protocol.get("context_bank_role")
        context_count = protocol.get("context_count")
        bank_path = protocol.get("context_bank_path")
        bank_file_sha256 = protocol.get("context_bank_file_sha256")
    if _sha256(nested_fingerprint, "sealed protocol bank fingerprint") != bank_fingerprint:
        raise ArtifactError("sealed top-level and protocol bank fingerprints disagree")
    if bank_role != "sealed_final_test":
        raise ArtifactError("sealed protocol bank role is not sealed_final_test")
    sealed_bank_path = Path(str(bank_path or "")).resolve()
    if not sealed_bank_path.is_file():
        raise ArtifactError("sealed protocol context-bank file is unavailable")
    if _sha256(bank_file_sha256, "sealed context-bank file") != _sha256_file(sealed_bank_path):
        raise ArtifactError("sealed protocol context-bank file hash mismatch")
    _sha256(
        protocol.get("context_bank_artifact_fingerprint"),
        "sealed context-bank artifact fingerprint",
    )
    _sha256(
        protocol.get("context_bank_source_provenance_fingerprint"),
        "sealed context-bank source-provenance fingerprint",
    )
    if not isinstance(protocol.get("context_bank_source_provenance"), Mapping):
        raise ArtifactError("sealed protocol lacks locked context-bank source provenance")
    _strict_int(context_count, "sealed protocol context_count", minimum=1)
    _strict_int(protocol.get("plans_per_context"), "sealed protocol plans_per_context", minimum=1)
    _strict_int(protocol.get("nfe"), "sealed protocol nfe", minimum=1)
    _strict_int(protocol.get("audit_seed"), "sealed protocol audit_seed")
    if protocol.get("temperature") != 1.0 or protocol.get("uncertainty_tilting") is not False:
        raise ArtifactError("sealed protocol is not explicitly untilted temperature one")
    if protocol.get("sampling_distribution") != "ordinary_conditional_flow_iid":
        raise ArtifactError("sealed protocol is not ordinary conditional-flow IID")
    if tuple(float(value) for value in protocol.get("gammas", ())) != tuple(float(g) for g in GAMMAS):
        raise ArtifactError("sealed protocol does not use the fixed seven gamma levels")
    verifier = _sha256(
        protocol.get("verifier_spec_fingerprint"), "sealed verifier specification"
    )
    if verifier != runs["Full"].recipe.get("verifier_spec_fingerprint"):
        raise ArtifactError("sealed verifier specification differs from expansion runs")
    if not isinstance(payload.get("provenance"), Mapping) or not payload["provenance"]:
        raise ArtifactError("sealed validity provenance is missing")
    provenance = _mapping(payload["provenance"])
    if (
        Path(str(provenance.get("sealed_bank_path", ""))).resolve() != sealed_bank_path
        or provenance.get("sealed_bank_file_sha256") != bank_file_sha256
        or provenance.get("one_shot_evaluation") is not True
        or provenance.get("audit_invocations_per_model") != 1
        or provenance.get("audit_results_used_for_training_or_checkpoint_selection") is not False
        or provenance.get("plan_samples_pooled_across_models") is not False
    ):
        raise ArtifactError("sealed validity provenance claims are incomplete or inconsistent")

    raw_runs = payload.get("per_run")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ArtifactError("sealed validity per_run must be a nonempty list")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_runs:
        row = _mapping(raw)
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in by_id:
            raise ArtifactError("sealed per_run identifiers must be nonempty and unique")
        if not isinstance(row.get("label"), str) or not str(row["label"]).strip():
            raise ArtifactError(f"sealed run {run_id} label is missing")
        if row.get("method") not in {"full", "minus_afe", "minus_progress", "minus_socp"}:
            raise ArtifactError(f"sealed run {run_id} method is unsupported")
        if row.get("protocol_fingerprint") != protocol_fingerprint:
            raise ArtifactError(f"sealed run {run_id} used a different protocol")
        if not isinstance(row.get("runtime_safety_claim"), bool):
            raise ArtifactError(f"sealed run {run_id} lacks an explicit runtime-safety claim")
        if not isinstance(row.get("independent_full_replica"), bool):
            raise ArtifactError(f"sealed run {run_id} lacks its independent-replica role")
        if not isinstance(row.get("selected_main"), bool):
            raise ArtifactError(f"sealed run {run_id} lacks its selected-main role")
        if row.get("expansion_verifier_spec_fingerprint") != verifier:
            raise ArtifactError(f"sealed run {run_id} expansion/audit verifier mismatch")
        if row.get("audit_verifier_spec_fingerprint") != verifier:
            raise ArtifactError(f"sealed run {run_id} sealed-audit verifier mismatch")
        checkpoint_path = Path(str(row.get("checkpoint_path", ""))).resolve()
        if not checkpoint_path.is_file():
            raise ArtifactError(f"sealed run {run_id} checkpoint file is unavailable")
        checkpoint_file_hash = _sha256(
            row.get("checkpoint_file_sha256"), f"sealed run {run_id} checkpoint"
        )
        if _sha256_file(checkpoint_path) != checkpoint_file_hash:
            raise ArtifactError(f"sealed run {run_id} checkpoint file hash mismatch")
        checkpoint = _mapping(_torch(checkpoint_path))
        if _sha256(
            row.get("model_state_sha256"), f"sealed run {run_id} model"
        ) != _sha256(checkpoint.get("current_model_hash"), f"sealed run {run_id} checkpoint model"):
            raise ArtifactError(f"sealed run {run_id} checkpoint/model binding mismatch")
        if _strict_int(
            row.get("checkpoint_round"), f"sealed run {run_id} checkpoint_round", minimum=1
        ) != _strict_int(checkpoint.get("round"), f"sealed run {run_id} checkpoint round"):
            raise ArtifactError(f"sealed run {run_id} checkpoint round binding mismatch")
        _strict_int(row.get("expansion_training_seed"), f"sealed run {run_id} training seed")
        _sha256(row.get("source_pretrain_hash"), f"sealed run {run_id} source-pretrain model")
        _sha256(
            row.get("source_pretrain_checkpoint_sha256"),
            f"sealed run {run_id} source-pretrain checkpoint",
        )
        audit = _mapping(row.get("audit"))
        _validate_audit_counts(
            audit,
            f"sealed run {run_id}",
            expected_bank_role="sealed_final_test",
            expected_bank_fingerprint=bank_fingerprint,
        )
        if (
            _strict_int(audit.get("context_count"), f"sealed run {run_id} context_count")
            != _strict_int(context_count, "sealed protocol context_count")
            or _strict_int(
                audit.get("plans_per_context"), f"sealed run {run_id} plans_per_context"
            ) != _strict_int(protocol.get("plans_per_context"), "sealed protocol plans_per_context")
            or _strict_int(audit.get("seed"), f"sealed run {run_id} audit seed")
            != _strict_int(protocol.get("audit_seed"), "sealed protocol audit seed")
            or not math.isclose(
                float(audit.get("progress_threshold", math.nan)),
                float(protocol.get("progress_threshold", math.nan)),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ArtifactError(f"sealed run {run_id} audit differs from preregistered protocol")
        _validate_sealed_aggregate(row.get("aggregate"), audit, f"sealed run {run_id}")
        runtime_available = bool(_mapping(row.get("runtime")).get("available"))
        _validate_runtime_record(
            row.get("runtime"), f"sealed run {run_id}",
            expected_available=runtime_available,
        )
        by_id[run_id] = row

    selected = payload.get("selected_main_run_ids")
    if not isinstance(selected, Mapping) or set(selected) != set(RUN_LABELS):
        raise ArtifactError(f"selected_main_run_ids must have exact keys {RUN_LABELS}")
    per_method: dict[str, dict[str, Any]] = {}
    method_ids = {
        "Full": "full",
        "-AFE": "minus_afe",
        "-Progress": "minus_progress",
        "-SOCP": "minus_socp",
    }
    for label in RUN_LABELS:
        run_id = str(selected[label])
        if run_id not in by_id:
            raise ArtifactError(f"selected sealed run {run_id!r} for {label} does not exist")
        row = by_id[run_id]
        run = runs[label]
        if row.get("method") != method_ids[label] or row.get("selected_main") is not True:
            raise ArtifactError(f"{label}: sealed run method/selected-main identity mismatch")
        if label == "Full" and row.get("independent_full_replica") is not True:
            raise ArtifactError("selected Full sealed run is not an independent Full replica")
        if _sha256(row.get("checkpoint_file_sha256"), f"{label} sealed checkpoint") != run.checkpoint_sha256:
            raise ArtifactError(f"{label}: sealed audit checkpoint differs from displayed run")
        if Path(str(row.get("checkpoint_path", ""))).resolve() != run.checkpoint_path.resolve():
            raise ArtifactError(f"{label}: sealed audit checkpoint path differs from displayed run")
        if _sha256(row.get("model_state_sha256"), f"{label} sealed model") != run.model_state_sha256:
            raise ArtifactError(f"{label}: sealed audit model differs from displayed run")
        if _strict_int(row.get("checkpoint_round"), f"{label} sealed checkpoint_round") != run.final_round:
            raise ArtifactError(f"{label}: sealed checkpoint round differs from displayed run")
        if row.get("runtime_safety_claim") is not CONTROL_ARMS.get(label, {}).get(
            "runtime_safety_claim", True
        ):
            raise ArtifactError(f"{label}: sealed runtime-safety claim is inconsistent")
        if _sha256(row.get("source_pretrain_hash"), f"{label} sealed source model") != run.recipe.get("source_model_hash"):
            raise ArtifactError(f"{label}: sealed source-pretrain model mismatch")
        if _sha256(
            row.get("source_pretrain_checkpoint_sha256"),
            f"{label} sealed source checkpoint",
        ) != run.recipe.get("source_checkpoint_sha256"):
            raise ArtifactError(f"{label}: sealed source-pretrain checkpoint mismatch")
        seed = _strict_int(row.get("expansion_training_seed"), f"{label} training seed")
        if seed != _strict_int(_matched_protocol(run).get("seed"), f"{label} protocol seed"):
            raise ArtifactError(f"{label}: sealed training seed differs from expansion protocol")
        per_method[label] = row

    independent = _mapping(payload.get("independent_full_aggregate"))
    if independent.get("schema_version") != "afe_sealed_independent_full_aggregate_v1":
        raise ArtifactError("independent Full aggregate has the wrong schema")
    if independent.get("context_bank_fingerprint") != bank_fingerprint:
        raise ArtifactError("independent Full aggregate used a different sealed bank")
    if independent.get("context_bank_role") != "sealed_final_test":
        raise ArtifactError("independent Full aggregate bank role is not sealed_final_test")
    independent_count = _strict_int(
        independent.get("independent_training_seed_count"),
        "independent Full training-seed count",
        minimum=2,
    )
    seeds = independent.get("training_seeds")
    if not isinstance(seeds, list) or len(seeds) != independent_count:
        raise ArtifactError("independent Full aggregate training seed list is inconsistent")
    strict_seeds = [_strict_int(seed, "independent Full training seed") for seed in seeds]
    if len(set(strict_seeds)) != len(strict_seeds):
        raise ArtifactError("independent Full training seeds are not unique")
    if independent.get("replication_unit") != "independently_trained_model":
        raise ArtifactError("independent Full aggregate has the wrong replication unit")
    if independent.get("plan_samples_pooled_across_training_seeds") is not False:
        raise ArtifactError("independent Full aggregate illegally pools plan samples")
    if independent.get("protocol_fingerprint") != protocol_fingerprint:
        raise ArtifactError("independent Full aggregate protocol fingerprint mismatch")
    replicas = [
        row for row in by_id.values()
        if row.get("method") == "full" and row.get("independent_full_replica") is True
    ]
    if len(replicas) != independent_count:
        raise ArtifactError("independent Full replica count disagrees with per_run")
    replica_seeds = [_strict_int(row.get("expansion_training_seed"), "Full replica seed") for row in replicas]
    replica_ids = [str(row["run_id"]) for row in replicas]
    replica_models = [str(row["model_state_sha256"]) for row in replicas]
    replica_pretrains = [str(row["source_pretrain_hash"]) for row in replicas]
    if sorted(strict_seeds) != sorted(replica_seeds):
        raise ArtifactError("independent Full training seeds disagree with per_run")
    if set(independent.get("run_ids", ())) != set(replica_ids):
        raise ArtifactError("independent Full run IDs disagree with per_run")
    if set(independent.get("model_state_sha256s", ())) != set(replica_models):
        raise ArtifactError("independent Full model hashes disagree with per_run")
    if set(independent.get("source_pretrain_hashes", ())) != set(replica_pretrains):
        raise ArtifactError("independent Full pretrain hashes disagree with per_run")
    if (
        len(set(replica_models)) != independent_count
        or len(set(replica_pretrains)) != independent_count
        or len(set(replica_seeds)) != independent_count
    ):
        raise ArtifactError("independent Full replicas are not independently identified")
    overall = _mapping(independent.get("aggregate_over_gammas"))
    for field in ("validity", "progress_validity", "valid_mode_coverage"):
        interval = _validate_interval(overall.get(field), f"independent Full aggregate {field}")
        if _strict_int(
            interval.get("independent_training_seed_count"),
            f"independent Full aggregate {field} seed count",
        ) != independent_count:
            raise ArtifactError(f"independent Full aggregate {field} seed count mismatch")
    if overall.get("V") != overall.get("validity") or overall.get("Vprog") != overall.get("progress_validity"):
        raise ArtifactError("independent Full V/Vprog aliases are inconsistent")
    runtime_summary = _mapping(independent.get("runtime"))
    available_runtime_count = _strict_int(
        runtime_summary.get("available_run_count"), "independent Full runtime available count"
    )
    missing_runtime_count = _strict_int(
        runtime_summary.get("missing_run_count"), "independent Full runtime missing count"
    )
    if available_runtime_count + missing_runtime_count != independent_count:
        raise ArtifactError("independent Full runtime run counts are inconsistent")
    if available_runtime_count != independent_count:
        raise ArtifactError(
            "final report requires fallback/failclosed histories for every independent Full run"
        )
    for field in ("fallback_frequency", "failclosed_frequency"):
        interval = _validate_interval(overall.get(field), f"independent Full aggregate {field}")
        if _strict_int(
            interval.get("independent_training_seed_count"),
            f"independent Full aggregate {field} seed count",
        ) != independent_count:
            raise ArtifactError(f"independent Full aggregate {field} seed count mismatch")
        if runtime_summary.get(field) != overall.get(field):
            raise ArtifactError(f"independent Full runtime {field} alias is inconsistent")
    per_gamma = independent.get("per_gamma")
    if not isinstance(per_gamma, list) or len(per_gamma) != len(GAMMAS):
        raise ArtifactError("independent Full aggregate must contain all seven gamma rows")
    for gamma, row_raw in zip(GAMMAS, per_gamma):
        row = _mapping(row_raw)
        if not math.isclose(float(row.get("gamma", math.nan)), gamma, abs_tol=1e-8):
            raise ArtifactError("independent Full aggregate gamma ordering is inconsistent")
        for field in ("validity", "progress_validity", "valid_mode_coverage"):
            interval = _validate_interval(
                row.get(field), f"independent Full gamma={gamma:g} {field}"
            )
            if _strict_int(
                interval.get("independent_training_seed_count"),
                f"independent Full gamma={gamma:g} {field} seed count",
            ) != independent_count:
                raise ArtifactError(
                    f"independent Full gamma={gamma:g} {field} seed count mismatch"
                )
        if row.get("V") != row.get("validity") or row.get("Vprog") != row.get("progress_validity"):
            raise ArtifactError(f"independent Full gamma={gamma:g} V/Vprog aliases are inconsistent")
    return SealedValidity(payload, protocol, per_method, independent)


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_ci(value: float, interval: tuple[float, float]) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"


def _table_rows(
    methods: Mapping[str, RunArtifacts | Sequence[RolloutRow]],
    sealed: SealedValidity | None = None,
) -> list[dict[str, Any]]:
    output = []
    for name, artifact in methods.items():
        run = artifact if isinstance(artifact, RunArtifacts) else None
        all_rollouts = run.final_rollouts if run is not None else list(artifact)
        for gamma in (*GAMMAS, None):
            rows = all_rollouts if gamma is None else [row for row in all_rollouts if math.isclose(row.gamma, gamma, abs_tol=1e-8)]
            roll = _rollout_summary(rows)
            if run is not None and sealed is None:
                raise ArtifactError("scientific run tables require sealed validity")
            sealed_row = sealed.per_method[name] if run is not None and sealed is not None else None
            audit_source = _mapping(sealed_row.get("audit")) if sealed_row is not None else {}
            audit = _audit_summary(audit_source, gamma) if run is not None else {
                "n": 0, "V": math.nan, "V_ci": (math.nan, math.nan),
                "Vprog": math.nan, "Vprog_ci": (math.nan, math.nan),
            }
            if sealed_row is not None:
                sealed_runtime = _mapping(sealed_row.get("runtime"))
                runtime = {
                    "fallback": _finite(sealed_runtime.get("fallback_frequency")),
                    "failclosed": _finite(sealed_runtime.get("failclosed_frequency")),
                    "claim": bool(sealed_row.get("runtime_safety_claim")),
                }
                if gamma is None:
                    valid_modes = _strict_int(
                        _mapping(sealed_row.get("aggregate")).get("valid_mode_coverage_count"),
                        f"{name}.valid_mode_coverage_count",
                    )
                else:
                    valid_modes = _strict_int(
                        next(
                            row["safe_mode_coverage"]
                            for row in _audit_rows(audit_source)
                            if math.isclose(float(row["gamma"]), gamma, abs_tol=1e-8)
                        ),
                        f"{name}[{gamma:g}].safe_mode_coverage",
                    )
            else:
                valid_modes = int(roll.get("modes", 0))
                runtime = {
                    "fallback": math.nan, "failclosed": math.nan, "claim": None,
                }
            sr_ci = _wilson(int(roll.get("successes", 0)), int(roll.get("n", 0)))
            cr_ci = _wilson(int(roll.get("collisions", 0)), int(roll.get("n", 0)))
            output.append({
                "method": name, "gamma": "all" if gamma is None else f"{gamma:g}",
                "audit_n": audit["n"], "V": _fmt_ci(audit["V"], audit["V_ci"]),
                "Vprog": _fmt_ci(audit["Vprog"], audit["Vprog_ci"]),
                "rollout_n": int(roll.get("n", 0)),
                "SR": _fmt_ci(_finite(roll.get("success_rate")), sr_ci),
                "CR": _fmt_ci(_finite(roll.get("collision_rate")), cr_ci),
                "clearance": _fmt(_finite(roll.get("clearance"))),
                "time": _fmt(_finite(roll.get("time"))),
                "modes": str(valid_modes),
                "fallback": (
                    _fmt(runtime["fallback"])
                    if runtime["claim"] is not False
                    else "offline/no claim"
                ),
                "failclosed": (
                    _fmt(runtime["failclosed"])
                    if runtime["claim"] is not False
                    else "offline/no claim"
                ),
            })
    return output


def write_tables(
    output_dir: Path,
    methods: Mapping[str, RunArtifacts | Sequence[RolloutRow]],
    sealed: SealedValidity,
) -> None:
    rows = _table_rows(methods, sealed)
    columns = ("Method", "γ", "N audit", "V (95% CI)", "Vprog (95% CI)", "N rollout",
               "SR (95% CI)", "CR (95% CI)", "clearance m", "time s", "modes", "fallback", "fail-closed")
    keys = ("method", "gamma", "audit_n", "V", "Vprog", "rollout_n", "SR", "CR", "clearance", "time", "modes", "fallback", "failclosed")
    markdown = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    markdown.extend("| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows)
    markdown.extend(["", "V/Vprog intervals use the sealed-final bank and are conditional plan-sampling Wilson intervals for one fixed "
                     "context bank and one trained model; the separate Full aggregate is across independent training seeds. "
                     "All scientific sampling uses ordinary T=1. Fallback and fail-closed frequencies are measured during verifier-assisted expansion; "
                     "the offline -SOCP control makes no runtime-safety claim, so its runtime cells are intentionally suppressed.", ""])
    (output_dir / "table.md").write_text("\n".join(markdown))
    escape = lambda text: str(text).replace("%", r"\%").replace("_", r"\_")
    latex = [r"\begin{tabular}{llrrrrrrrrrrr}", r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    for row in rows:
        latex.append(" & ".join(escape(row[key]) for key in keys) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "table.tex").write_text("\n".join(latex))


def _query_total(run: RunArtifacts, gamma: float | None = None) -> dict[str, float]:
    records = list(_mapping(run.final_bundle.get("store_state")).get("records", ()))
    if records:
        if gamma is not None:
            records = [
                raw for raw in records
                if math.isclose(float(_mapping(raw).get("gamma", math.nan)), gamma, abs_tol=1e-8)
            ]
        flow_records = [
            raw
            for raw in records
            if str(_mapping(raw).get("source")) == "flow"
        ]
        backup_records = [
            raw
            for raw in records
            if str(_mapping(raw).get("source")) == "safemppi_backup"
        ]
        positive = 0
        for raw in flow_records:
            safety = _mapping(_mapping(raw).get("safety"))
            positive += bool(safety.get("strict_bounds")) and bool(safety.get("socp_certified"))
        backup_positive = 0
        for raw in backup_records:
            safety = _mapping(_mapping(raw).get("safety"))
            backup_positive += bool(safety.get("strict_bounds")) and bool(
                safety.get("socp_certified")
            )
        return {
            "n": len(flow_records), "positive": positive,
            "acceptance": positive / len(flow_records) if flow_records else math.nan,
            "backup_n": len(backup_records),
            "backup_positive": backup_positive,
            "backup_acceptance": (
                backup_positive / len(backup_records)
                if backup_records else math.nan
            ),
        }
    if gamma is not None:
        return {
            "n": 0,
            "positive": 0,
            "acceptance": math.nan,
            "backup_n": 0,
            "backup_positive": 0,
            "backup_acceptance": math.nan,
        }
    query_rows = [_mapping(row.get("query")) for row in run.history if row.get("query")]
    n = sum(int(row.get("new_verifier_calls", 0)) for row in query_rows)
    positive = sum(int(row.get("new_positive_queries", 0)) for row in query_rows)
    backup_n = sum(int(row.get("backup_verifier_calls", 0)) for row in query_rows)
    backup_positive = sum(
        int(row.get("backup_positive_queries", 0)) for row in query_rows
    )
    return {
        "n": n,
        "positive": positive,
        "acceptance": positive / n if n else math.nan,
        "backup_n": backup_n,
        "backup_positive": backup_positive,
        "backup_acceptance": (
            backup_positive / backup_n if backup_n else math.nan
        ),
    }


def _fmt_seed_interval(raw: Mapping[str, Any]) -> str:
    return (
        f"{float(raw['mean']):.3f} "
        f"[{float(raw['low']):.3f}, {float(raw['high']):.3f}]"
    )


def write_validity_report(
    output_dir: Path,
    runs: Mapping[str, RunArtifacts],
    sealed: SealedValidity,
) -> None:
    full = runs["Full"]
    selected_full = sealed.per_method["Full"]
    sealed_audit = _mapping(selected_full["audit"])
    audit = _audit_summary(sealed_audit, None)
    roll = _rollout_summary(full.final_rollouts)
    runtime = _mapping(selected_full["runtime"])
    valid_coverage = _mapping(selected_full["aggregate"])
    query = _query_total(full)
    independent = sealed.independent_full_aggregate
    independent_overall = _mapping(independent["aggregate_over_gammas"])
    lines = [
        "# Final run validity report", "",
        "Every validity value in this report comes from the immutable `afe_sealed_validity_v1` artifact. "
        "Round-monitoring audits are used only in the training-internals figure and never supply final validity numbers.", "",
        "## Query acceptance", "",
        f"The uncertainty-tilted acquisition queried {int(query['n'])} unique fully verified planned windows; "
        f"{int(query['positive'])} were safe (acceptance {_fmt(query['acceptance'])}). This is query efficiency, not model validity.", "",
        f"The runtime SafeMPPI backup made {int(query['backup_n'])} separate full-verifier calls; "
        f"{int(query['backup_positive'])} passed (acceptance {_fmt(query['backup_acceptance'])}). "
        "These rows update cumulative A but are excluded from acquisition acceptance and CFM replay.", "",
        "## Sealed held-out validity", "",
        f"On the untouched, evaluation-only, ordinary/untilted T=1 sealed-final bank, the selected Full run has "
        f"V={_fmt_ci(audit['V'], audit['V_ci'])} and Vprog={_fmt_ci(audit['Vprog'], audit['Vprog_ci'])} "
        f"over {audit['n']} explicitly counted plans. Its aggregate valid-mode coverage is "
        f"{int(valid_coverage['valid_mode_coverage_count'])} modes "
        f"({float(valid_coverage['valid_mode_coverage_fraction']):.3f}).", "",
        "Progress is reported only after the independent safety label; it is not part of that label. "
        "The sealed samples did not update A and were not replayed.", "",
        "## Independent-training-seed Full aggregate", "",
        f"Across {int(independent['independent_training_seed_count'])} independently trained Full models, "
        f"V={_fmt_seed_interval(_mapping(independent_overall['validity']))}, "
        f"Vprog={_fmt_seed_interval(_mapping(independent_overall['progress_validity']))}, and "
        f"valid-mode coverage={_fmt_seed_interval(_mapping(independent_overall['valid_mode_coverage']))}. "
        "These are Student-t intervals across independently trained model estimates; plan rows are not pooled.", "",
        "### Per-safety-level validity", "",
        "| γ | queried + / n | query acceptance | selected Full V | selected Full Vprog | independent Full V | independent Full Vprog | valid modes | ordinary SR | ordinary CR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    independent_by_gamma = {
        float(row["gamma"]): _mapping(row) for row in independent["per_gamma"]
    }
    for gamma in GAMMAS:
        gamma_audit = _audit_summary(sealed_audit, gamma)
        audit_row = next(
            row for row in _audit_rows(sealed_audit)
            if math.isclose(float(row["gamma"]), gamma, abs_tol=1e-8)
        )
        gamma_roll = _rollout_summary([
            row for row in full.final_rollouts
            if math.isclose(row.gamma, gamma, abs_tol=1e-8)
        ])
        gamma_query = _query_total(full, gamma)
        independent_row = independent_by_gamma[float(gamma)]
        lines.append(
            f"| {gamma:g} | {int(gamma_query['positive'])} / {int(gamma_query['n'])} | "
            f"{_fmt(gamma_query['acceptance'])} | {_fmt_ci(gamma_audit['V'], gamma_audit['V_ci'])} | "
            f"{_fmt_ci(gamma_audit['Vprog'], gamma_audit['Vprog_ci'])} | "
            f"{_fmt_seed_interval(_mapping(independent_row['validity']))} | "
            f"{_fmt_seed_interval(_mapping(independent_row['progress_validity']))} | "
            f"{int(audit_row['safe_mode_coverage'])} | "
            f"{_fmt(_finite(gamma_roll.get('success_rate')))} | "
            f"{_fmt(_finite(gamma_roll.get('collision_rate')))} |"
        )
    lines.extend(["", "### Sealed selected-run comparison", "",
                  "| Method | V | Vprog | valid-mode coverage | fallback | fail-closed |",
                  "|---|---:|---:|---:|---:|---:|"])
    for label in RUN_LABELS:
        row = sealed.per_method[label]
        counts = _audit_summary(_mapping(row["audit"]), None)
        aggregate = _mapping(row["aggregate"])
        arm_runtime = _mapping(row["runtime"])
        runtime_available = bool(arm_runtime["available"])
        lines.append(
            f"| {label} | {_fmt_ci(counts['V'], counts['V_ci'])} | "
            f"{_fmt_ci(counts['Vprog'], counts['Vprog_ci'])} | "
            f"{int(aggregate['valid_mode_coverage_count'])} "
            f"({float(aggregate['valid_mode_coverage_fraction']):.3f}) | "
            f"{_fmt(float(arm_runtime['fallback_frequency'])) if runtime_available else 'offline/no claim'} | "
            f"{_fmt(float(arm_runtime['failclosed_frequency'])) if runtime_available else 'offline/no claim'} |"
        )
    lines.extend([
        "",
        "## Ordinary closed-loop behavior", "",
        f"At T=1 without acquisition tilting or a safety filter, SR={_fmt(_finite(roll.get('success_rate')))}, "
        f"CR={_fmt(_finite(roll.get('collision_rate')))}, and successful-mode coverage={int(roll.get('modes', 0))} "
        f"over {int(roll.get('n', 0))} final-model rollouts.", "",
        "## Runtime safety accounting", "",
        f"For the selected Full run, certified-fallback frequency was {_fmt(float(runtime['fallback_frequency']))} "
        f"({int(runtime['fallback_steps'])}/{int(runtime['control_decisions'])} control decisions) and "
        f"fail-closed frequency was {_fmt(float(runtime['failclosed_frequency']))} "
        f"({int(runtime['failclosed_episodes'])}/{int(runtime['episode_count'])} episodes). "
        "This runtime statement does not apply to the ordinary unfiltered rollouts above.", "",
        f"Across independent Full runs, fallback frequency was "
        f"{_fmt_seed_interval(_mapping(independent_overall['fallback_frequency']))} and fail-closed frequency was "
        f"{_fmt_seed_interval(_mapping(independent_overall['failclosed_frequency']))}.", "",
        "There is no curriculum learning: every round allocates the same episode count to all seven gammas, verified positives are replayed uniformly, and sigma is used only for acquisition.", "",
        "The rollout gallery uses T=0.5 only for legibility. It is not evidence for any metric in this report.", "",
        "These empirical estimates do not establish a validity-mass theorem for the post-update CFM; such a theorem would require an additional density-floor or EBM assumption.", "",
    ])
    (output_dir / "final_validity_report.md").write_text("\n".join(lines))


def generate_reports(
    *,
    full_path: str | Path,
    full_viz_path: str | Path,
    full_checkpoint_path: str | Path | None = None,
    no_socp_path: str | Path,
    no_socp_viz_path: str | Path,
    no_socp_checkpoint_path: str | Path | None = None,
    no_progress_path: str | Path,
    no_progress_viz_path: str | Path,
    no_progress_checkpoint_path: str | Path | None = None,
    no_afe_path: str | Path,
    no_afe_viz_path: str | Path,
    no_afe_checkpoint_path: str | Path | None = None,
    sealed_validity_path: str | Path,
    id_demos_path: str | Path,
    output_dir: str | Path,
    baseline_path: str | Path | None = None,
    pretrained_viz_path: str | Path | None = None,
    expert_path: str | Path | None = None,
    mizuta_path: str | Path | None = None,
    mizuta_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create final reports from four matched runs and sealed validity."""
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, RunArtifacts] = {
        "Full": load_run(
            full_path, "Full", require_full=True,
            selected_checkpoint=full_checkpoint_path,
        ),
        "-AFE": load_run(
            no_afe_path, "-AFE", selected_checkpoint=no_afe_checkpoint_path,
        ),
        "-Progress": load_run(
            no_progress_path, "-Progress",
            selected_checkpoint=no_progress_checkpoint_path,
        ),
        "-SOCP": load_run(
            no_socp_path, "-SOCP", selected_checkpoint=no_socp_checkpoint_path,
        ),
    }
    _validate_matched_runs(runs)
    gallery_artifacts = {
        "Full": load_gallery(full_viz_path, "Full", runs["Full"]),
        "-AFE": load_gallery(no_afe_viz_path, "-AFE", runs["-AFE"]),
        "-Progress": load_gallery(
            no_progress_viz_path, "-Progress", runs["-Progress"]
        ),
        "-SOCP": load_gallery(no_socp_viz_path, "-SOCP", runs["-SOCP"]),
    }
    galleries: dict[str, Sequence[RolloutRow] | None] = {
        label: artifact.rows for label, artifact in gallery_artifacts.items()
    }
    sealed = load_sealed_validity(sealed_validity_path, runs)
    id_rows = load_id_demo_paths(id_demos_path)
    pretrained_metrics = expert_metrics = None
    if baseline_path is not None:
        try:
            pretrained_metrics = load_rollouts(baseline_path, section="rollouts")
            expert_metrics = load_rollouts(baseline_path, section="expert")
        except ArtifactError:
            pretrained_metrics = load_rollouts(baseline_path)
        _require_scientific_temperature(pretrained_metrics, "Pretrained")
    pretrained_gallery = None
    if pretrained_viz_path is not None:
        pretrained_gallery = load_rollouts(pretrained_viz_path)
        _require_visual_temperature(pretrained_gallery, "Pretrained")
    expert_gallery = expert_metrics
    if expert_path is not None:
        expert_gallery = load_rollouts(expert_path)
        expert_metrics = expert_gallery
    mizuta_gallery = load_rollouts(mizuta_path) if mizuta_path is not None else None
    if mizuta_gallery is not None:
        _require_visual_temperature(mizuta_gallery, r"CFM-MPPI*/Mizuta")
    mizuta_metrics = load_rollouts(mizuta_metrics_path) if mizuta_metrics_path is not None else None
    if mizuta_metrics is not None:
        _require_scientific_temperature(mizuta_metrics, r"CFM-MPPI*/Mizuta")
    rollout_figure(out / "rollouts.png", id_rows=id_rows, expert=expert_gallery,
                   pretrained=pretrained_gallery, mizuta=mizuta_gallery, galleries=galleries)
    internals_figure(out / "internals.png", runs["Full"])
    scatter_methods: dict[str, RunArtifacts | Sequence[RolloutRow] | None] = {}
    if expert_metrics: scatter_methods["Expert"] = expert_metrics
    if pretrained_metrics: scatter_methods["Pretrained"] = pretrained_metrics
    if mizuta_metrics: scatter_methods[r"CFM-MPPI$^{*}$"] = mizuta_metrics
    for name in ("Full", "-SOCP", "-Progress", "-AFE"):
        if name in runs: scatter_methods[name] = runs[name]
    scatter_figure(out / "scatter.png", scatter_methods)
    table_methods: dict[str, RunArtifacts | Sequence[RolloutRow]] = {}
    if expert_metrics: table_methods["Expert"] = expert_metrics
    if pretrained_metrics: table_methods["Pretrained"] = pretrained_metrics
    if mizuta_metrics: table_methods[r"CFM-MPPI$^{*}$"] = mizuta_metrics
    table_methods.update(runs)
    write_tables(out, table_methods, sealed)
    write_validity_report(out, runs, sealed)
    manifest = {
        "rollouts": str(out / "rollouts.png"), "internals": str(out / "internals.png"),
        "scatter": str(out / "scatter.png"), "table_md": str(out / "table.md"),
        "table_tex": str(out / "table.tex"), "validity": str(out / "final_validity_report.md"),
        "sealed_validity_source": str(Path(sealed_validity_path).resolve()),
        "sealed_validity_sha256": _sha256_file(Path(sealed_validity_path).resolve()),
        "sealed_bank_fingerprint": sealed.payload["bank_fingerprint"],
        "protocol_fingerprint": sealed.payload["protocol_fingerprint"],
        "run_provenance": {
            label: {
                "checkpoint": str(run.checkpoint_path),
                "checkpoint_sha256": run.checkpoint_sha256,
                "model_state_sha256": run.model_state_sha256,
                "gallery_checkpoint_sha256": gallery_artifacts[label].checkpoint_sha256,
                "gallery_model_state_sha256": gallery_artifacts[label].model_state_sha256,
            }
            for label, run in runs.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True, help="Full AFE run directory or bundle")
    parser.add_argument(
        "--full-checkpoint", type=Path, required=True,
        help="explicit selected Full round checkpoint; later exploratory rounds are ignored",
    )
    parser.add_argument("--full-viz", type=Path, required=True, help="Full T=0.5 gallery rollouts")
    parser.add_argument(
        "--sealed-validity", type=Path, required=True,
        help="one-shot afe_sealed_validity_v1 artifact used for every final validity claim",
    )
    parser.add_argument("--id-demos", type=Path, required=True, help="balanced planned-demo PT artifact")
    parser.add_argument("--outdir", type=Path, required=True)
    for option in ("no-socp", "no-progress", "no-afe"):
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(
            f"--{option}-checkpoint", type=Path,
            help="optional explicit selected control checkpoint cutoff",
        )
        parser.add_argument(f"--{option}-viz", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="optional Stage-04 baseline bundle")
    parser.add_argument("--pretrained-viz", type=Path, help="optional Pretrained T=0.5 gallery")
    parser.add_argument("--expert", type=Path, help="optional expert rollout artifact")
    parser.add_argument("--mizuta", "--mizuta-viz", dest="mizuta", type=Path,
                        help="optional CFM-MPPI*/Mizuta T=0.5 gallery artifact")
    parser.add_argument("--mizuta-metrics", type=Path,
                        help="optional independent CFM-MPPI*/Mizuta T=1 rollout artifact")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = generate_reports(
        full_path=args.full, full_viz_path=args.full_viz,
        full_checkpoint_path=args.full_checkpoint,
        id_demos_path=args.id_demos,
        sealed_validity_path=args.sealed_validity,
        output_dir=args.outdir, no_socp_path=args.no_socp,
        no_socp_viz_path=args.no_socp_viz,
        no_socp_checkpoint_path=args.no_socp_checkpoint,
        no_progress_path=args.no_progress,
        no_progress_viz_path=args.no_progress_viz,
        no_progress_checkpoint_path=args.no_progress_checkpoint,
        no_afe_path=args.no_afe,
        no_afe_viz_path=args.no_afe_viz,
        no_afe_checkpoint_path=args.no_afe_checkpoint,
        baseline_path=args.baseline,
        pretrained_viz_path=args.pretrained_viz, expert_path=args.expert,
        mizuta_path=args.mizuta, mizuta_metrics_path=args.mizuta_metrics,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
