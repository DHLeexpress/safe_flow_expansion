"""Optimizer-dose and symmetry-balanced expert support for low7 AFE.

This module is additive to the reviewed V2 lineage-mass protocol.  It owns only
the continued-training objective.  Expert rows never enter the acquisition
store, RBF memory, verifier accounting, execution, or evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import afe_core as AC
import afe_context as CX
import grid_expand_afe2 as AFE2
import grid_feats as GF
from codex_challenging.afe_restart.scene import make_id_scene


DATA_SCHEMA = "afe_planned_demo_v3_low7_uniform_pairs"
TRAIN_SCHEMA = "afe_fresh_pretrain_v2_low7_uniform_pairs"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def _resolve_from(parent: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _canonical_gamma(value: float) -> float:
    matches = [gamma for gamma in GAMMAS if abs(float(value) - gamma) <= 5.0e-7]
    if len(matches) != 1:
        raise ValueError(f"demo gamma {value!r} is not declared")
    return matches[0]


def reflect_xy(value: np.ndarray) -> np.ndarray:
    """Swap every final x/y coordinate pair."""

    array = np.asarray(value)
    if array.shape[-1:] != (2,):
        raise ValueError(f"reflection requires a final coordinate dimension of two: {array.shape}")
    return np.ascontiguousarray(array[..., ::-1])


def reflect_state(state: np.ndarray) -> np.ndarray:
    array = np.asarray(state)
    if array.shape != (4,):
        raise ValueError(f"reflected DI state must have shape (4,), got {array.shape}")
    return np.ascontiguousarray(array[[1, 0, 3, 2]])


def polar_reflection_indices(n_theta: int = GF.N_THETA) -> np.ndarray:
    """Map a reflected polar ray to its source ray under x<->y."""

    theta = -np.pi + (np.arange(n_theta) + 0.5) * 2.0 * np.pi / n_theta
    reflected_source = np.mod(np.pi / 2.0 - theta + np.pi, 2.0 * np.pi) - np.pi
    distance = np.abs(
        np.angle(np.exp(1j * (theta[None, :] - reflected_source[:, None])))
    )
    indices = np.argmin(distance, axis=1)
    if len(set(int(value) for value in indices)) != n_theta:
        raise RuntimeError("x/y polar reflection is not a permutation")
    return indices.astype(np.int64)


def reflect_polar_grid(grid: np.ndarray) -> np.ndarray:
    array = np.asarray(grid)
    if array.shape[-2] != GF.N_THETA:
        raise ValueError("grid theta dimension disagrees with canonical featurization")
    return np.ascontiguousarray(array[..., polar_reflection_indices(), :])


@dataclass(frozen=True)
class DemoProvenance:
    pretrain_manifest: str
    pretrain_manifest_sha256: str
    split_manifest: str
    split_manifest_sha256: str
    combined_manifest: str
    combined_manifest_sha256: str
    dataset: str
    dataset_sha256: str
    source_query_hash_digest: str
    source_checkpoint: str
    source_checkpoint_sha256: str
    source_checkpoint_model_sha256: str
    train_pairs: int
    validation_pairs: int
    train_windows: int
    validation_windows: int
    pair_leakage: int
    fixed_symmetry_audit: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class DemoReference:
    grid: torch.Tensor
    low7: torch.Tensor
    hist: torch.Tensor
    plans: torch.Tensor
    state: torch.Tensor
    goal: torch.Tensor
    gamma: torch.Tensor
    pair_ids: torch.Tensor
    trajectory_ids: torch.Tensor
    train_rows_by_gamma_trajectory: dict[float, dict[int, np.ndarray]]
    provenance: DemoProvenance

    def sample_original_rows(self, pair_count: int, rng: np.random.Generator):
        """Sample equal-gamma, equal-trajectory-mass originals with replacement."""

        pair_count = int(pair_count)
        if pair_count < 1:
            return np.zeros(0, dtype=np.int64), {
                "pair_count": 0,
                "gamma_counts": {f"{gamma:g}": 0 for gamma in GAMMAS},
                "unique_trajectories": 0,
            }
        gamma_order = np.asarray(
            [GAMMAS[index % len(GAMMAS)] for index in range(pair_count)],
            dtype=np.float64,
        )
        gamma_order = gamma_order[rng.permutation(pair_count)]
        # Build each gamma's trajectory sequence independently by complete,
        # randomly permuted cycles.  Consequently every trajectory represented
        # in that gamma differs in draw count by at most one.  This is reference
        # objective variance control only; it never conditions OOD gathering.
        gamma_slots = {
            gamma: int(np.sum(gamma_order == gamma)) for gamma in GAMMAS
        }
        trajectory_draws: dict[float, list[int]] = {}
        for gamma, count in gamma_slots.items():
            keys = np.asarray(
                sorted(self.train_rows_by_gamma_trajectory[gamma]), dtype=np.int64
            )
            if not len(keys):
                raise RuntimeError(f"authenticated TRAIN reference has no gamma={gamma} trajectories")
            draws: list[int] = []
            while len(draws) < count:
                draws.extend(int(keys[index]) for index in rng.permutation(len(keys)))
            trajectory_draws[gamma] = draws[:count]

        rows: list[int] = []
        trajectory_offsets = {gamma: 0 for gamma in GAMMAS}
        gamma_counts = {f"{gamma:g}": 0 for gamma in GAMMAS}
        trajectory_counts = {f"{gamma:g}": {} for gamma in GAMMAS}
        for gamma in gamma_order:
            gamma = float(gamma)
            trajectory_map = self.train_rows_by_gamma_trajectory[gamma]
            offset = trajectory_offsets[gamma]
            trajectory_id = trajectory_draws[gamma][offset]
            trajectory_offsets[gamma] = offset + 1
            candidates = trajectory_map[trajectory_id]
            rows.append(int(candidates[int(rng.integers(0, len(candidates)))]))
            gamma_counts[f"{gamma:g}"] += 1
            key = str(int(trajectory_id))
            counts = trajectory_counts[f"{gamma:g}"]
            counts[key] = counts.get(key, 0) + 1
        per_gamma_spread = {
            gamma: (
                max(counts.values()) - min(counts.values()) if counts else 0
            )
            for gamma, counts in trajectory_counts.items()
        }
        if max(gamma_counts.values()) - min(gamma_counts.values()) > 1:
            raise RuntimeError("balanced demo gamma counts differ by more than one")
        if max(per_gamma_spread.values(), default=0) > 1:
            raise RuntimeError("balanced demo trajectory counts differ by more than one")
        trajectory_balance = {
            gamma: {
                "represented_trajectories": len(counts),
                "minimum_draw_count": min(counts.values(), default=0),
                "maximum_draw_count": max(counts.values(), default=0),
                "draw_count_spread": per_gamma_spread[gamma],
            }
            for gamma, counts in trajectory_counts.items()
        }
        return np.asarray(rows, dtype=np.int64), {
            "pair_count": pair_count,
            "gamma_counts": gamma_counts,
            "trajectory_balance_by_gamma": trajectory_balance,
            "trajectory_draws_sha256": sha256_json(trajectory_counts),
            "unique_trajectories": sum(len(value) for value in trajectory_counts.values()),
            "balance_contract": (
                "gamma draw counts differ by at most one; within each gamma, "
                "TRAIN trajectory draw counts differ by at most one"
            ),
        }

    def paired_batch(self, rows: Iterable[int], device: torch.device):
        """Return interleaved original/reflection examples and recomputed contexts."""

        rows = [int(value) for value in rows]
        if not rows:
            raise ValueError("a paired demo batch requires at least one original row")
        grids: list[np.ndarray] = []
        lows: list[np.ndarray] = []
        histories: list[np.ndarray] = []
        plans: list[np.ndarray] = []
        env = make_id_scene(start=np.asarray((0.3, 0.3), np.float32),
                            goal=np.asarray((4.7, 4.7), np.float32))
        for row in rows:
            original_grid = self.grid[row].numpy()
            original_low = self.low7[row].numpy()
            original_hist = self.hist[row].numpy()
            original_plan = self.plans[row].numpy()
            reflected_state = reflect_state(
                self.state[row].double().numpy()
            ).astype(np.float32)
            reflected_goal = reflect_xy(self.goal[row].numpy())
            reflected_hist = reflect_xy(original_hist)
            reflected_plan = reflect_xy(original_plan)
            reflected = CX.build_context(
                reflected_state,
                reflected_goal,
                _canonical_gamma(float(self.gamma[row])),
                reflected_hist,
                env,
                CX.LOW7_SCHEMA,
            )
            grids.extend((original_grid, np.asarray(reflected.grid, np.float32)))
            lows.extend((original_low, np.asarray(reflected.low5, np.float32)))
            histories.extend((original_hist, np.asarray(reflected.hist, np.float32)))
            plans.extend((original_plan, reflected_plan))
        return tuple(
            torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
            for values in (grids, lows, histories, plans)
        )


def _require_tensor(payload, key: str, shape_tail: tuple[int, ...]) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape[1:]) != shape_tail:
        raise RuntimeError(
            f"authenticated demo field {key} must have shape [N,{shape_tail}], "
            f"got {None if not isinstance(value, torch.Tensor) else tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"authenticated demo field {key} contains nonfinite values")
    return value.contiguous()


def _fixed_symmetry_audit(payload: dict[str, Any], train_rows: np.ndarray) -> dict[str, Any]:
    """Recompute both original and reflected contexts on a fixed TRAIN subset."""

    selected = [int(train_rows[index]) for index in np.linspace(
        0, len(train_rows) - 1, num=min(14, len(train_rows)), dtype=np.int64
    )]
    env = make_id_scene(start=np.asarray((0.3, 0.3), np.float32),
                        goal=np.asarray((4.7, 4.7), np.float32))
    max_grid_error = 0.0
    max_reflected_grid_error = 0.0
    max_low_error = 0.0
    max_hist_error = 0.0
    gamma_counts = {f"{gamma:g}": 0 for gamma in GAMMAS}
    for row in selected:
        state = payload["verifier_state"][row].double().numpy().astype(np.float32)
        goal = payload["window_goal"][row].numpy()
        gamma = _canonical_gamma(float(payload["gamma"][row]))
        hist = payload["hist"][row].numpy()
        original = CX.build_context(state, goal, gamma, hist, env, CX.LOW7_SCHEMA)
        reflected = CX.build_context(
            reflect_state(state), reflect_xy(goal), gamma, reflect_xy(hist), env,
            CX.LOW7_SCHEMA,
        )
        double_reflected = CX.build_context(
            reflect_state(reflect_state(state)),
            reflect_xy(reflect_xy(goal)),
            gamma,
            reflect_xy(reflect_xy(hist)),
            env,
            CX.LOW7_SCHEMA,
        )
        max_grid_error = max(
            max_grid_error,
            float(np.max(np.abs(np.asarray(original.grid) - payload["grid"][row].numpy()))),
            float(np.max(np.abs(np.asarray(double_reflected.grid) - payload["grid"][row].numpy()))),
        )
        max_low_error = max(
            max_low_error,
            float(np.max(np.abs(np.asarray(original.low5) - payload["low7"][row].numpy()))),
            float(np.max(np.abs(np.asarray(double_reflected.low5) - payload["low7"][row].numpy()))),
        )
        max_hist_error = max(
            max_hist_error,
            float(np.max(np.abs(np.asarray(original.hist) - payload["hist"][row].numpy()))),
            float(np.max(np.abs(np.asarray(double_reflected.hist) - payload["hist"][row].numpy()))),
        )
        max_reflected_grid_error = max(
            max_reflected_grid_error,
            float(np.max(np.abs(
                np.asarray(reflected.grid) - reflect_polar_grid(payload["grid"][row].numpy())
            ))),
        )
        gamma_counts[f"{gamma:g}"] += 1
    tolerance = 2.0e-5
    if max(max_grid_error, max_low_error, max_hist_error, max_reflected_grid_error) > tolerance:
        raise RuntimeError(
            "canonical demo reflection audit failed: "
            f"grid={max_grid_error} reflected_grid={max_reflected_grid_error} "
            f"low7={max_low_error} hist={max_hist_error}"
        )
    verifier_rows = []
    train_set = set(int(value) for value in train_rows)
    for gamma in GAMMAS:
        candidates = torch.where(torch.isclose(
            payload["gamma"].float(), torch.tensor(gamma), atol=5.0e-7
        ))[0].tolist()
        row = next(int(value) for value in candidates if int(value) in train_set)
        state = payload["verifier_state"][row].numpy().astype(np.float32)
        plan = payload["U"][row].numpy()
        goal = payload["window_goal"][row].numpy().astype(np.float64)
        original = AC.verify_plan(state, plan, env, gamma, goal, n_theta=180)
        reflected = AC.verify_plan(
            reflect_state(state), reflect_xy(plan), env, gamma, reflect_xy(goal),
            n_theta=180,
        )
        if (
            original["y"] != reflected["y"]
            or original["reason"] != reflected["reason"]
            or not np.isclose(original["prog"], reflected["prog"], atol=1.0e-7)
            or not np.isclose(original["margin"], reflected["margin"], atol=1.0e-7)
        ):
            raise RuntimeError(
                f"nominal-scene verifier reflection invariance failed at row {row}"
            )
        verifier_rows.append({
            "row": row,
            "gamma": gamma,
            "safe": bool(original["y"]),
            "reason": original["reason"],
            "margin_abs_error": abs(original["margin"] - reflected["margin"]),
            "progress_abs_error": abs(original["prog"] - reflected["prog"]),
        })
    return {
        "subset_rows": selected,
        "subset_train_only": True,
        "gamma_counts": gamma_counts,
        "reflection_involution": True,
        "canonical_context_recomputed": True,
        "max_cached_grid_error": max_grid_error,
        "max_reflected_grid_symmetry_error": max_reflected_grid_error,
        "max_low7_error": max_low_error,
        "max_history_error": max_hist_error,
        "tolerance": tolerance,
        "nominal_verifier_invariance": verifier_rows,
    }


def load_authenticated_demo_reference(
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    *,
    load_tensors: bool,
) -> tuple[DemoReference | None, DemoProvenance]:
    """Authenticate the exact TRAIN split linked to the promoted low7 candidate."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_sha256 = str(checkpoint_sha256).lower()
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("demo/checkpoint file linkage hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage_schema") != TRAIN_SCHEMA:
        raise RuntimeError("demo support requires the authenticated low7 pretraining schema")
    checkpoint_model_sha = model_state_hash_from_payload(checkpoint)
    if checkpoint_model_sha != checkpoint.get("model_state_sha256"):
        raise RuntimeError("checkpoint embedded model-state linkage is invalid")
    pretrain_manifest_path = checkpoint_path.parent.parent / "manifest.json"
    split_manifest_path = checkpoint_path.parent.parent / "logs" / "pair_split.json"
    for path in (pretrain_manifest_path, split_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"authenticated demo provenance artifact is absent: {path}")
    pretrain = _load_json(pretrain_manifest_path)
    split = _load_json(split_manifest_path)
    if pretrain.get("schema_version") != TRAIN_SCHEMA:
        raise RuntimeError("pretraining manifest schema mismatch")
    if pretrain.get("artifacts", {}).get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("pretraining manifest does not link the selected checkpoint")
    if Path(pretrain["artifacts"]["checkpoint"]).resolve() != checkpoint_path:
        raise RuntimeError("pretraining manifest checkpoint path mismatch")
    embedded_split = pretrain.get("data", {}).get("split")
    if split != embedded_split:
        raise RuntimeError("pair split file and pretraining manifest disagree")
    if int(split.get("pair_leakage", -1)) != 0:
        raise RuntimeError("demo source has train/validation pair leakage")
    train_pairs = {int(value) for value in split.get("train_pair_ids", ())}
    validation_pairs = {int(value) for value in split.get("validation_pair_ids", ())}
    if not train_pairs or not validation_pairs or train_pairs & validation_pairs:
        raise RuntimeError("demo train/validation pair partition is invalid")
    combined_manifest_path = _resolve_from(
        pretrain_manifest_path.parent, pretrain["data"]["manifest"]
    )
    if Path(checkpoint.get("source_manifest", "")).resolve() != combined_manifest_path:
        raise RuntimeError("checkpoint source-manifest linkage mismatch")
    combined = _load_json(combined_manifest_path)
    if combined.get("schema_version") != DATA_SCHEMA:
        raise RuntimeError("combined demo archive schema mismatch")
    dataset_path = _resolve_from(combined_manifest_path.parent, combined["dataset"])
    dataset_sha = str(combined["dataset_sha256"]).lower()
    if dataset_sha != str(pretrain["data"]["sha256"]).lower():
        raise RuntimeError("pretrain and combined manifests disagree on demo archive hash")
    if sha256_file(dataset_path) != dataset_sha:
        raise RuntimeError("authenticated demo archive hash mismatch")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != DATA_SCHEMA:
        raise RuntimeError("demo archive payload schema mismatch")
    query_hashes = tuple(str(value) for value in payload.get("query_hashes", ()))
    source_query_digest = hashlib.sha256("".join(query_hashes).encode()).hexdigest()
    if source_query_digest != checkpoint.get("source_query_hash_digest"):
        raise RuntimeError("demo archive source-query digest does not link to checkpoint")
    pair_ids = _require_tensor(payload, "window_pair_ids", ()).long().reshape(-1)
    train_mask = torch.isin(pair_ids, torch.tensor(sorted(train_pairs), dtype=torch.long))
    validation_mask = torch.isin(
        pair_ids, torch.tensor(sorted(validation_pairs), dtype=torch.long)
    )
    if bool((train_mask & validation_mask).any()) or not bool((train_mask | validation_mask).all()):
        raise RuntimeError("demo archive rows are not partitioned by the authenticated split")
    train_rows = torch.where(train_mask)[0].numpy()
    validation_rows = torch.where(validation_mask)[0].numpy()
    audit = _fixed_symmetry_audit(payload, train_rows)
    provenance = DemoProvenance(
        pretrain_manifest=str(pretrain_manifest_path),
        pretrain_manifest_sha256=sha256_file(pretrain_manifest_path),
        split_manifest=str(split_manifest_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        combined_manifest=str(combined_manifest_path),
        combined_manifest_sha256=sha256_file(combined_manifest_path),
        dataset=str(dataset_path),
        dataset_sha256=dataset_sha,
        source_query_hash_digest=source_query_digest,
        source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=checkpoint_sha256,
        source_checkpoint_model_sha256=checkpoint_model_sha,
        train_pairs=len(train_pairs),
        validation_pairs=len(validation_pairs),
        train_windows=len(train_rows),
        validation_windows=len(validation_rows),
        pair_leakage=0,
        fixed_symmetry_audit=audit,
    )
    if not load_tensors:
        return None, provenance
    grid = _require_tensor(payload, "grid", (3, 32, 32)).float()
    low7 = _require_tensor(payload, "low7", (7,)).float()
    hist = _require_tensor(payload, "hist", (GF.K_HIST, 2)).float()
    plans = _require_tensor(payload, "U", (GF.H_PRED, 2)).float()
    state = _require_tensor(payload, "verifier_state", (4,)).double()
    goal = _require_tensor(payload, "window_goal", (2,)).float()
    gamma = _require_tensor(payload, "gamma", ()).float().reshape(-1)
    trajectory_ids = _require_tensor(payload, "window_trajectory_ids", ()).long().reshape(-1)
    hierarchy: dict[float, dict[int, list[int]]] = {value: {} for value in GAMMAS}
    for row in train_rows:
        row = int(row)
        gamma_key = _canonical_gamma(float(gamma[row]))
        trajectory = int(trajectory_ids[row])
        hierarchy[gamma_key].setdefault(trajectory, []).append(row)
    packed = {
        gamma_key: {
            trajectory: np.asarray(rows, dtype=np.int64)
            for trajectory, rows in trajectory_map.items()
        }
        for gamma_key, trajectory_map in hierarchy.items()
    }
    if any(not value for value in packed.values()):
        raise RuntimeError("authenticated TRAIN split has an empty gamma demo population")
    return DemoReference(
        grid=grid,
        low7=low7,
        hist=hist,
        plans=plans,
        state=state,
        goal=goal,
        gamma=gamma,
        pair_ids=pair_ids,
        trajectory_ids=trajectory_ids,
        train_rows_by_gamma_trajectory=packed,
        provenance=provenance,
    ), provenance


def model_state_hash_from_payload(payload: dict[str, Any]) -> str:
    """Hash checkpoint state without constructing a second model."""

    digest = hashlib.sha256()
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint lacks state_dict")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def partition_epoch_by_mass(
    epoch_ids: list[int], mass_by_id: dict[int, float], steps: int
) -> tuple[list[list[int]], list[float], float]:
    """Greedily partition one exact epoch into S near-equal-mass macro-batches."""

    steps = int(steps)
    ids = [int(value) for value in epoch_ids]
    if steps < 1:
        raise ValueError("optimizer macro-step count must be positive")
    if len(ids) < steps:
        raise RuntimeError(
            f"eligible D+ count {len(ids)} cannot support {steps} nonempty optimizer steps"
        )
    if len(set(ids)) != len(ids) or set(ids) != set(mass_by_id):
        raise ValueError("macro partition population/mass mismatch")
    batches: list[list[int]] = [[] for _ in range(steps)]
    masses = [0.0] * steps
    # Largest-first placement minimizes worst-bin imbalance.  Epoch position is
    # the deterministic tie key and remains the order within each macro-batch.
    position = {query_id: index for index, query_id in enumerate(ids)}
    for query_id in sorted(ids, key=lambda value: (-mass_by_id[value], position[value])):
        batch_index = min(range(steps), key=lambda value: (masses[value], value))
        batches[batch_index].append(query_id)
        masses[batch_index] += float(mass_by_id[query_id])
    for batch in batches:
        batch.sort(key=position.__getitem__)
    if any(not batch for batch in batches):
        raise RuntimeError("macro partition produced an empty optimizer step")
    flat = [query_id for batch in batches for query_id in batch]
    if len(flat) != len(ids) or set(flat) != set(ids):
        raise RuntimeError("macro partition lost or duplicated eligible positives")
    target = 1.0 / steps
    residual = max(abs(value - target) for value in masses)
    return batches, masses, float(residual)


def demo_example_count(positive_count: int, demo_frac: float) -> int:
    demo_frac = float(demo_frac)
    if demo_frac == 0.0:
        return 0
    if not 0.0 < demo_frac < 1.0:
        raise ValueError("demo objective mass must lie in [0,1)")
    target = int(round(int(positive_count) * demo_frac / (1.0 - demo_frac)))
    return max(2, 2 * int(round(target / 2.0)))


def partition_pairs(pair_rows: np.ndarray, steps: int) -> list[np.ndarray]:
    if len(pair_rows) < steps:
        raise RuntimeError("demo pair count cannot supply every optimizer macro-step")
    base, larger = divmod(len(pair_rows), steps)
    batches = []
    offset = 0
    for index in range(steps):
        size = base + int(index < larger)
        batches.append(np.asarray(pair_rows[offset:offset + size], dtype=np.int64))
        offset += size
    if offset != len(pair_rows) or any(len(value) == 0 for value in batches):
        raise RuntimeError("failed to partition complete demo pairs")
    return batches


def paired_demo_cfm_loss(policy, grid, low, hist, controls, generator):
    """CFM loss with identical tau and reflected x0 within every demo pair."""

    if len(controls) % 2 or len(controls) < 2:
        raise ValueError("paired demo loss requires interleaved original/reflection rows")
    pair_count = len(controls) // 2
    x1 = (controls / float(policy.u_max)).reshape(len(controls), int(policy.d))
    x0_original = torch.randn(
        (pair_count, int(policy.d)), device=x1.device, dtype=x1.dtype,
        generator=generator,
    )
    x0_reflected = reflect_action_tensor(x0_original)
    x0 = torch.stack((x0_original, x0_reflected), dim=1).reshape_as(x1)
    tau_original = torch.rand(
        pair_count, device=x1.device, dtype=x1.dtype, generator=generator
    ).clamp_(1.0e-4, 1.0)
    tau = torch.repeat_interleave(tau_original, 2)
    x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
    context = policy.ctx_from(grid, low, hist)
    prediction = policy(x_tau, tau, policy._expand_ctx(context, len(controls)))
    per_sample = ((prediction - (x1 - x0)) ** 2).mean(dim=1)
    return per_sample.mean()


def reflect_action_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] % 2:
        raise ValueError("flattened action reflection requires shape [B,2H]")
    return value.reshape(len(value), -1, 2).flip(-1).reshape_as(value)


def _gradient_norm(gradients, device) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().to(torch.float64).square().sum()
    return total.sqrt()


def _parameter_norm(parameters) -> float:
    return float(torch.stack([
        parameter.detach().to(torch.float64).square().sum()
        for parameter in parameters
    ]).sum().sqrt()) if parameters else 0.0


def update_round_support(
    policy,
    optimizer,
    store,
    cfg,
    device,
    replay_rng,
    round_i: int,
    demo_reference: DemoReference | None,
):
    """Run exactly S macro Adam updates over one unique hierarchical D+ epoch."""

    eligible = store.positive_ids(round_i=round_i, replay_window=cfg.replay_window)
    if not eligible:
        return None
    if cfg.replay_sampling != "round_gamma_replica_context":
        raise RuntimeError("support profile requires hierarchical epoch ordering")
    if cfg.replay_loss_weighting != "gamma_episode_context_query_equal_mass":
        raise RuntimeError("support profile requires hierarchical equal-mass replay")
    epoch_ids = store.positive_epoch_ids(
        replay_rng, eligible_ids=eligible, sampling=cfg.replay_sampling
    )
    mass_by_id, mass_diagnostics = store.positive_hierarchy_equal_mass(
        cfg.gammas, eligible_ids=eligible
    )
    steps = int(cfg.optimizer_steps_per_round)
    macro_batches, macro_masses, mass_residual = partition_epoch_by_mass(
        epoch_ids, mass_by_id, steps
    )
    demo_frac = float(cfg.demo_frac)
    if demo_frac and demo_reference is None:
        raise RuntimeError("nonzero demo objective has no authenticated TRAIN reference")
    example_count = demo_example_count(len(eligible), demo_frac)
    demo_rng = None
    demo_batches = None
    demo_sampling = {
        "pair_count": 0,
        "gamma_counts": {f"{gamma:g}": 0 for gamma in GAMMAS},
        "unique_trajectories": 0,
    }
    demo_generator = None
    if demo_frac:
        demo_rng = np.random.default_rng(AFE2.named_seed(cfg.seed, "demo_rows", round_i))
        original_rows, demo_sampling = demo_reference.sample_original_rows(
            example_count // 2, demo_rng
        )
        demo_batches = partition_pairs(original_rows, steps)
        demo_generator = torch.Generator(device=device).manual_seed(
            AFE2.named_seed(cfg.seed, "demo_cfm", round_i)
        )

    policy.train()
    groups = {name: list(module.parameters()) for name, module in policy.module_groups().items()}
    before_norm = {name: _parameter_norm(parameters) for name, parameters in groups.items()}
    snapshot = {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("support update has no trainable parameters")
    drawn: dict[int, int] = {}
    positive_losses: list[float] = []
    demo_losses: list[float] = []
    mixed_losses: list[float] = []
    preclip_norms: list[float] = []
    clipped = 0
    gradient_cosines: list[float] = []
    applied_weights: list[float] = []
    group_norms = {name: [] for name in groups}
    probe = None
    value_before = None
    functional_steps: list[float] = []

    for step_index, query_ids in enumerate(macro_batches):
        grid, low, hist, controls, ids = store.positive_batch(query_ids)
        for query_id in ids:
            drawn[int(query_id)] = drawn.get(int(query_id), 0) + 1
        grid, low, hist, controls = (
            value.to(device) for value in (grid, low, hist, controls)
        )
        if probe is None:
            count = min(len(controls), 128)
            probe_x = 0.5 * (controls[:count] / policy.u_max).reshape(count, policy.d)
            probe_t = torch.full((count,), 0.5, device=device)
            probe_context = policy.ctx_from(
                grid[:count], low[:count], hist[:count]
            ).detach()
            with torch.no_grad():
                value_before = policy(
                    probe_x, probe_t, policy._expand_ctx(probe_context, count)
                ).detach()
            probe = (probe_x, probe_t, probe_context, count)
        weights = torch.as_tensor(
            [len(ids) * steps * mass_by_id[int(query_id)] for query_id in ids],
            dtype=controls.dtype,
            device=device,
        )
        applied_weights.extend(float(value) for value in weights.detach().cpu())
        positive_loss = policy.cfm_loss(
            controls,
            policy.ctx_from(grid, low, hist),
            weights=weights,
        )
        optimizer.zero_grad(set_to_none=True)
        demo_loss = None
        cosine = None
        if demo_frac == 0.0:
            # Exact no-demo compatibility path: no demo RNG, data access, or extra
            # autograd operation can perturb the parameter update.
            positive_loss.backward()
            mixed_loss = positive_loss
        else:
            demo_grid, demo_low, demo_hist, demo_controls = demo_reference.paired_batch(
                demo_batches[step_index], device
            )
            demo_loss = paired_demo_cfm_loss(
                policy,
                demo_grid,
                demo_low,
                demo_hist,
                demo_controls,
                demo_generator,
            )
            positive_gradients = torch.autograd.grad(
                positive_loss, trainable, allow_unused=True
            )
            demo_gradients = torch.autograd.grad(
                demo_loss, trainable, allow_unused=True
            )
            dot = torch.zeros((), dtype=torch.float64, device=device)
            for positive_gradient, demo_gradient in zip(
                positive_gradients, demo_gradients
            ):
                if positive_gradient is not None and demo_gradient is not None:
                    dot = dot + (
                        positive_gradient.detach().to(torch.float64)
                        * demo_gradient.detach().to(torch.float64)
                    ).sum()
            positive_norm = _gradient_norm(positive_gradients, device)
            demo_norm = _gradient_norm(demo_gradients, device)
            denominator = positive_norm * demo_norm
            cosine = (
                float((dot / denominator).clamp(-1.0, 1.0))
                if float(denominator) > 1.0e-12 else 0.0
            )
            for parameter, positive_gradient, demo_gradient in zip(
                trainable, positive_gradients, demo_gradients
            ):
                if positive_gradient is None and demo_gradient is None:
                    parameter.grad = None
                else:
                    positive_part = (
                        torch.zeros_like(demo_gradient)
                        if positive_gradient is None else positive_gradient
                    )
                    demo_part = (
                        torch.zeros_like(positive_gradient)
                        if demo_gradient is None else demo_gradient
                    )
                    parameter.grad = (
                        (1.0 - demo_frac) * positive_part.detach()
                        + demo_frac * demo_part.detach()
                    )
            mixed_loss = (1.0 - demo_frac) * positive_loss + demo_frac * demo_loss
        for name, parameters in groups.items():
            group_norms[name].append(float(sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in parameters if parameter.grad is not None
            )) ** 0.5)
        preclip = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip))
        preclip_norms.append(preclip)
        clipped += int(preclip > float(cfg.grad_clip))
        optimizer.step()
        positive_losses.append(float(positive_loss.detach()))
        if demo_loss is not None:
            demo_losses.append(float(demo_loss.detach()))
            gradient_cosines.append(float(cosine))
        mixed_losses.append(float(mixed_loss.detach()))
        probe_x, probe_t, probe_context, count = probe
        with torch.no_grad():
            value_after = policy(
                probe_x, probe_t, policy._expand_ctx(probe_context, count)
            )
            functional_steps.append(float(
                (value_after - value_before).norm(dim=1).mean()
                / value_before.norm(dim=1).mean().clamp_min(1.0e-9)
            ))

    total_draws = sum(drawn.values())
    duplicates = total_draws - len(drawn)
    coverage = len(drawn) / len(eligible)
    if len(positive_losses) != steps:
        raise RuntimeError("support update performed the wrong number of Adam steps")
    if duplicates != 0 or coverage != 1.0 or set(drawn) != set(eligible):
        raise RuntimeError("support update violated exact unique D+ coverage")
    replay_mass = np.asarray([mass_by_id[int(value)] for value in eligible], np.float64)
    replay_ess = float(replay_mass.sum() ** 2 / np.square(replay_mass).sum())
    relative_change = {}
    for name, parameters in groups.items():
        delta = _parameter_norm([
            parameter.detach() - original
            for parameter, original in zip(parameters, snapshot[name])
        ])
        relative_change[name] = delta / max(before_norm[name], 1.0e-12)
    fresh = sum(
        count for query_id, count in drawn.items()
        if int(store.q_round[query_id]) == int(round_i)
    )
    eligible_round_counts: dict[str, int] = {}
    for query_id in eligible:
        key = str(int(store.q_round[query_id]))
        eligible_round_counts[key] = eligible_round_counts.get(key, 0) + 1
    return {
        "steps": steps,
        "optimizer_steps": steps,
        "stop": "support_macro_epoch_complete",
        "cfm": float(np.mean(positive_losses)),
        "cfm_first": positive_losses[0],
        "cfm_last": positive_losses[-1],
        "demo_cfm": float(np.mean(demo_losses)) if demo_losses else None,
        "mixed_objective": float(np.mean(mixed_losses)),
        "demo_positive_gradient_cosine": (
            float(np.mean(gradient_cosines)) if gradient_cosines else None
        ),
        "demo_positive_gradient_cosine_steps": gradient_cosines,
        "demo_frac": demo_frac,
        "demo_examples": int(example_count),
        "demo_original_count": int(example_count // 2),
        "demo_reflected_count": int(example_count // 2),
        "demo_sampling": demo_sampling,
        "demo_batch_sizes": (
            [2 * len(value) for value in demo_batches] if demo_batches is not None else []
        ),
        "fstep_final": functional_steps[-1],
        "fstep_max": max(functional_steps),
        "grad_norm": {
            name: float(np.mean(values)) for name, values in group_norms.items()
        },
        "rel_param_change": relative_change,
        "drawn_ids": drawn,
        "n_distinct": len(drawn),
        "replay_window": cfg.replay_window,
        "replay_eligible": len(eligible),
        "replay_sampling": cfg.replay_sampling,
        "replay_update_mode": "fixed_macro_steps_exact_epoch",
        "replay_loss_weighting": cfg.replay_loss_weighting,
        "replay_raw_mass_sum": float(replay_mass.sum()),
        "replay_weight_ess": replay_ess,
        "replay_weight_ess_fraction": replay_ess / len(replay_mass),
        "replay_mass_diagnostics": mass_diagnostics,
        "replay_macro_mass": macro_masses,
        "replay_macro_mass_target": 1.0 / steps,
        "replay_macro_mass_max_residual": mass_residual,
        "replay_applied_weight_min": min(applied_weights),
        "replay_applied_weight_max": max(applied_weights),
        "replay_applied_weight_mean": float(np.mean(applied_weights)),
        "preclip_grad_norm_mean": float(np.mean(preclip_norms)),
        "preclip_grad_norm_by_step": preclip_norms,
        "grad_clipped_steps": clipped,
        "grad_clipped_fraction": clipped / steps,
        "optimizer_draws": total_draws,
        "replay_duplicate_draws": duplicates,
        "replay_epoch_coverage": coverage,
        "replay_batch_sizes": [len(value) for value in macro_batches],
        "replay_batch_size_min": min(len(value) for value in macro_batches),
        "replay_batch_size_max": max(len(value) for value in macro_batches),
        "replay_fresh_draws": int(fresh),
        "replay_fresh_distinct": int(fresh),
        "replay_fresh_fraction": fresh / total_draws,
        "replay_eligible_round_counts": eligible_round_counts,
        "replay_draw_round_counts": dict(eligible_round_counts),
    }
