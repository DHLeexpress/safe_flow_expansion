"""Flow-policy sampling and frozen-feature adapters.

Sampling accepts an explicit :class:`torch.Generator`; unlike the legacy
``sample_window`` helper it never mutates or relies on global RNG state.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .schemas import QueryContext


def _as_batch_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    # QueryContext arrays are intentionally read-only.  Copy before exposing
    # their storage to torch so no tensor kernel can mutate ledger identity.
    value = torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=device)
    return value.unsqueeze(0)


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def require_promoted_fresh_pretrain(
    model: torch.nn.Module,
    checkpoint: Mapping[str, object],
) -> str:
    """Reject diagnostic/failed-gate or legacy checkpoints before expansion.

    Stage 03 writes the production checkpoint names only after its ordinary
    temperature-one evaluation observes successful R- and U-first behavior at
    every gamma.  Consumers recheck the embedded promotion evidence instead of
    inferring success from a filename or a prior process exit code.
    """

    state_hash = model_state_hash(model)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("pretrained checkpoint is missing its model config")
    source_manifest = checkpoint.get("source_manifest")
    required = (
        checkpoint.get("stage_schema") == "afe_fresh_pretrain_v1"
        and checkpoint.get("fresh_from_scratch") is True
        and checkpoint.get("endpoint_free") is True
        and checkpoint.get("expansion_promotion") is True
        and checkpoint.get("id_mode_diversity_gate_passed") is True
        and float(checkpoint.get("id_evaluation_temperature", float("nan"))) == 1.0
        and checkpoint.get("id_evaluation_uncertainty_tilting") is False
        and checkpoint.get("model_state_sha256") == state_hash
        and _is_sha256(checkpoint.get("source_query_hash_digest"))
        and isinstance(source_manifest, str)
        and bool(source_manifest.strip())
        and _is_sha256(checkpoint.get("id_metrics_sha256"))
        and checkpoint.get("frozen_feature_snapshot") is not True
        and config.get("arch") == "hp-repr"
        and config.get("schema_version") == "w8sg-hp-v2-low5-only"
        and config.get("raw_start_goal") is False
        and int(config.get("H_pred", -1)) == 10
        and tuple(config.get("grid_shape", ())) == (1, 32, 32)
        and int(config.get("K_hist", -1)) == 16
        and float(config.get("u_max", float("nan"))) == 1.0
        and config.get("use_gru") is False
        and config.get("boundary_adapter") is False
        and int(config.get("repr_dim", -1)) == 32
        and int(config.get("ctx_dim", -1)) == 37
        and int(getattr(model, "repr_dim", -1)) == 32
        and int(getattr(model, "ctx_dim", -1)) == 37
        and int(getattr(model, "H_pred", -1)) == 10
        and int(getattr(model, "T", -1)) == 10
        and tuple(getattr(model, "grid_shape", ())) == (1, 32, 32)
        and int(getattr(model, "K_hist", -1)) == 16
        and float(getattr(model, "u_max", float("nan"))) == 1.0
        and getattr(model, "use_gru", None) is False
        and getattr(model, "boundary_adapter", None) is False
    )
    if not required:
        raise RuntimeError(
            "consumer requires a fresh endpoint-free Stage-03 checkpoint "
            "promoted only after the ordinary T=1 all-gamma R/U gate"
        )
    return state_hash


def context_tensors(context: QueryContext, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _as_batch_tensor(context.grid, device),
        _as_batch_tensor(context.low5, device),
        _as_batch_tensor(context.hist, device),
    )


@torch.inference_mode()
def sample_plans(
    model: torch.nn.Module,
    context: QueryContext,
    count: int,
    *,
    temperature: float,
    nfe: int,
    generator: torch.Generator,
) -> np.ndarray:
    """Draw complete ``[count,10,2]`` plans using seeded Euler integration."""
    if count <= 0:
        raise ValueError("count must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    device = next(model.parameters()).device
    grid, low5, hist = context_tensors(context, device)
    encoded = model.ctx_from(grid, low5, hist)
    encoded = model._expand_ctx(encoded[0], count)
    x = temperature * torch.randn(
        count, int(model.d), generator=generator, device=device, dtype=encoded.dtype,
    )
    for step in range(nfe):
        tau = torch.full((count,), step / nfe, device=device, dtype=encoded.dtype)
        x = x + model(x, tau, encoded) / nfe
    plans = (x.reshape(count, int(model.T), 2) * float(model.u_max)).clamp(
        -float(model.u_max), float(model.u_max),
    )
    return plans.detach().cpu().numpy().astype(np.float32, copy=False)


@dataclass
class FrozenFeatureModel:
    """Immutable hash-checked copy of the pretrained representation."""

    model: torch.nn.Module
    s: float = 0.9
    expected_dim: int = 32

    @classmethod
    def from_pretrained(
        cls, model: torch.nn.Module, *, s: float = 0.9, expected_dim: int = 32,
    ) -> "FrozenFeatureModel":
        frozen = copy.deepcopy(model).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        instance = cls(frozen, float(s), int(expected_dim))
        instance._initial_hash = model_state_hash(frozen)
        return instance

    def __post_init__(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._initial_hash = getattr(self, "_initial_hash", model_state_hash(self.model))

    @property
    def state_hash(self) -> str:
        current = model_state_hash(self.model)
        if current != self._initial_hash:
            raise RuntimeError("frozen feature model changed during expansion")
        return current

    @torch.inference_mode()
    def encode(self, context: QueryContext, plans: np.ndarray | torch.Tensor) -> np.ndarray:
        self.state_hash
        device = next(self.model.parameters()).device
        grid, low5, hist = context_tensors(context, device)
        controls = torch.as_tensor(plans, dtype=torch.float32, device=device)
        if controls.ndim == 2:
            controls = controls.unsqueeze(0)
        features = self.model.phi_s_at(controls, grid, low5, hist, s=self.s)
        if features.ndim != 2 or features.shape[1] != self.expected_dim:
            raise RuntimeError(
                f"expected frozen feature shape [B,{self.expected_dim}], got {tuple(features.shape)}"
            )
        norms = torch.linalg.vector_norm(features.double(), dim=1, keepdim=True)
        if bool((norms <= 1e-12).any()):
            raise RuntimeError("cannot normalize a zero frozen feature")
        normalized = features.double() / norms
        return normalized.cpu().numpy()


def batch_context_arrays(records: Sequence[object], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Stack ledger contexts for the proximal CFM loss adapter."""
    grids = torch.as_tensor(
        np.stack([np.asarray(record.context.grid) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    low5 = torch.as_tensor(
        np.stack([np.asarray(record.context.low5) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    hist = torch.as_tensor(
        np.stack([np.asarray(record.context.hist) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    plans = torch.as_tensor(
        np.stack([np.asarray(record.plan) for record in records]),
        dtype=torch.float32,
        device=device,
    )
    return grids, low5, hist, plans


def _ledger_identity(record: object) -> str:
    """Return the immutable query identity used to key CFM common randomness."""

    validate = getattr(record, "validate_identity", None)
    if callable(validate):
        validate()
    if isinstance(record, dict):
        candidates = (
            record.get("query_hash"),
            record.get("source_query_hash"),
            record.get("training_target_hash"),
        )
    else:
        candidates = (
            getattr(record, "query_hash", None),
            getattr(record, "source_query_hash", None),
            getattr(record, "training_target_hash", None),
        )
    identity = next((str(value).lower() for value in candidates if value), "")
    if len(identity) != 64:
        raise ValueError(
            "ledger CFM replay requires a SHA-256 query identity per record"
        )
    try:
        bytes.fromhex(identity)
    except ValueError as exc:
        raise ValueError(
            "ledger CFM replay contains a non-hex query identity"
        ) from exc
    return identity


def ledger_common_random_arrays(
    records: Sequence[object],
    *,
    round_seed: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-query fixed x0 and tau for one proximal round.

    Each row is keyed by (round_seed, exact_query_hash). Consequently the same
    record receives bit-identical bridge noise and flow time on every objective
    evaluation, regardless of shuffled order or microbatch size. This turns
    tolerance checks into checks on one fixed Monte Carlo CFM objective instead
    of comparisons between newly redrawn objectives.
    """

    if dimension <= 0:
        raise ValueError("CFM common-random dimension must be positive")
    seed_bytes = (int(round_seed) & ((1 << 64) - 1)).to_bytes(8, "big")
    x0_rows: list[np.ndarray] = []
    tau_rows: list[np.float32] = []
    for record in records:
        identity = _ledger_identity(record)
        digest = hashlib.sha256(
            b"afe-ledger-cfm-crn-v1\x00" + seed_bytes + bytes.fromhex(identity)
        ).digest()
        record_seed = int.from_bytes(digest[:16], "big")
        rng = np.random.Generator(np.random.PCG64(record_seed))
        x0_rows.append(rng.standard_normal(dimension, dtype=np.float32))
        tau_rows.append(
            np.float32(max(float(rng.random(dtype=np.float32)), 1.0e-4))
        )
    return (
        np.stack(x0_rows, axis=0).astype(np.float32, copy=False),
        np.asarray(tau_rows, dtype=np.float32),
    )


def ledger_cfm_loss(
    model: torch.nn.Module,
    records: Sequence[object],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """CFM loss with explicit RNG, suitable for the proximal solver."""
    device = next(model.parameters()).device
    grid, low5, hist, plans = batch_context_arrays(records, device)
    context = model.ctx_from(grid, low5, hist)
    batch = plans.shape[0]
    x1 = (plans / float(model.u_max)).reshape(batch, int(model.d))
    x0_array, tau_array = ledger_common_random_arrays(
        records,
        round_seed=generator.initial_seed(),
        dimension=int(model.d),
    )
    x0 = torch.as_tensor(x0_array, device=device, dtype=x1.dtype)
    tau = torch.as_tensor(tau_array, device=device, dtype=x1.dtype)
    x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
    target = x1 - x0
    prediction = model(x_tau, tau, model._expand_ctx(context, batch))
    return ((prediction - target) ** 2).mean()


# The proximal solver records this declaration in its telemetry. It is safe
# because ledger_common_random_arrays ignores mutable generator state and keys
# every auxiliary draw by the immutable ledger identity and round seed.
ledger_cfm_loss.objective_randomness = (
    "fixed_per_query_sha256_pcg64_common_random_numbers_v1"
)
