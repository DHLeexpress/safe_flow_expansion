"""Round-local acquisition-temperature calibration for AFE uncertainty models."""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np
import torch

import afe2_calibration as BC
import afe_core as AC
import afe_context as CX
import afe_rbf_core as RC
import grid_expand_afe2 as AFE2


def round_context_ids(store, round_i: int) -> list[int]:
    """Unique stored control contexts visited in one expansion round."""

    output = [
        context_id for context_id, meta in enumerate(store.ctx_meta)
        if int(meta[0]) == int(round_i)
    ]
    if not output:
        raise ValueError(f"round {round_i} contains no stored acquisition contexts")
    return output


def round_gamma_episode_balanced_context_ids(
    store,
    round_i: int,
    gammas,
    cap_per_gamma: int,
    seed: int,
    *,
    equalize_gammas: bool = False,
) -> list[int]:
    """Select a deterministic, episode-balanced context cap in each gamma cell.

    Contexts are first restricted to one round, then grouped by conditioning
    gamma and episode id.  Within each gamma, a seeded round-robin over episodes
    prevents a long-lived replica from supplying most of the beta-calibration
    contexts.  The returned ids are sorted so downstream chunking is stable.
    """

    cap_per_gamma = int(cap_per_gamma)
    if cap_per_gamma < 1:
        raise ValueError("adaptive-beta context cap per gamma must be positive")

    gamma_storage_map = CX.declared_gamma_storage_map(gammas)
    gamma_keys = list(gamma_storage_map.values())
    grouped: dict[float, dict[int, list[int]]] = {
        gamma: defaultdict(list) for gamma in gamma_keys
    }
    for context_id, meta in enumerate(store.ctx_meta):
        if int(meta[0]) != int(round_i):
            continue
        raw_gamma = float(store.ctx_low5[context_id][-1])
        try:
            gamma = CX.canonical_declared_gamma(raw_gamma, gamma_storage_map)
        except ValueError as exc:
            raise ValueError(
                f"round {round_i} contains undeclared conditioning gamma {raw_gamma}"
            ) from exc
        grouped[gamma][int(meta[1])].append(int(context_id))

    available_by_gamma = {
        gamma: sum(len(values) for values in grouped[gamma].values())
        for gamma in gamma_keys
    }
    if equalize_gammas:
        if any(count == 0 for count in available_by_gamma.values()):
            raise RuntimeError(
                "equalized adaptive-beta calibration requires every declared gamma"
            )
        target_per_gamma = min(cap_per_gamma, min(available_by_gamma.values()))
    else:
        target_per_gamma = cap_per_gamma

    selected: list[int] = []
    for gamma_index, gamma in enumerate(gamma_keys):
        episodes = grouped[gamma]
        if not episodes:
            continue
        available = available_by_gamma[gamma]
        if available <= target_per_gamma:
            selected.extend(
                context_id
                for episode_id in sorted(episodes)
                for context_id in episodes[episode_id]
            )
            continue

        rng = np.random.default_rng(AFE2.named_seed(
            int(seed),
            "adaptive_beta_context_cap",
            int(round_i),
            int(gamma_index),
            float(gamma),
        ))
        episode_ids = np.asarray(sorted(episodes), dtype=np.int64)
        episode_ids = episode_ids[rng.permutation(len(episode_ids))].tolist()
        queues = {
            int(episode_id): np.asarray(
                episodes[int(episode_id)], dtype=np.int64
            )[rng.permutation(len(episodes[int(episode_id)]))].tolist()
            for episode_id in episode_ids
        }
        offsets = {int(episode_id): 0 for episode_id in episode_ids}
        selected_gamma: list[int] = []
        while len(selected_gamma) < target_per_gamma:
            progressed = False
            for episode_id in episode_ids:
                offset = offsets[int(episode_id)]
                queue = queues[int(episode_id)]
                if offset >= len(queue):
                    continue
                selected_gamma.append(int(queue[offset]))
                offsets[int(episode_id)] = offset + 1
                progressed = True
                if len(selected_gamma) >= target_per_gamma:
                    break
            if not progressed:
                raise RuntimeError("episode-balanced context selection exhausted early")
        selected.extend(selected_gamma)

    if len(selected) != len(set(selected)):
        raise RuntimeError("adaptive-beta context selection sampled with replacement")
    return sorted(selected)


@torch.no_grad()
def feature_pools(policy, store, cfg, device, round_i: int) -> tuple[list[torch.Tensor], dict]:
    """Generate beta-neutral K-pools at every context from one completed round."""

    cap_per_gamma = getattr(cfg, "adaptive_beta_contexts_per_gamma", None)
    context_ids = (
        round_context_ids(store, round_i)
        if cap_per_gamma is None
        else round_gamma_episode_balanced_context_ids(
            store,
            round_i,
            cfg.gammas,
            int(cap_per_gamma),
            cfg.seed,
            equalize_gammas=bool(getattr(
                cfg, "adaptive_beta_equalize_gammas", False
            )),
        )
    )
    if not context_ids:
        raise ValueError(f"round {round_i} contains no selected acquisition contexts")
    pools: list[torch.Tensor] = []
    gamma_counts: dict[str, int] = {}
    chunk_size = 16
    for offset in range(0, len(context_ids), chunk_size):
        sids = context_ids[offset:offset + chunk_size]
        grid = store.grid3_of(sids).to(device)
        low = torch.stack([
            torch.from_numpy(store.ctx_low5[sid]) for sid in sids
        ]).to(device)
        hist = torch.stack([
            torch.from_numpy(store.ctx_hist[sid].astype(np.float32)) for sid in sids
        ]).to(device)
        context = policy.ctx_from(grid, low, hist)
        repeated = context.repeat_interleave(cfg.K, dim=0)
        with AC.isolated_random_state(AFE2.named_seed(
            cfg.seed, "adaptive_beta_candidates", round_i, offset
        )):
            controls = policy.sample(
                len(sids) * cfg.K,
                repeated,
                nfe=cfg.nfe,
                temp=cfg.temp,
            )
        features = RC.l2_normalize(
            policy.phi_s(controls, repeated, s=cfg.s)
        ).reshape(len(sids), cfg.K, -1)
        for local_index, sid in enumerate(sids):
            pools.append(features[local_index].detach())
            gamma = str(round(float(store.ctx_low5[sid][-1]), 2))
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + 1
    return pools, gamma_counts


@torch.no_grad()
def score_vectors(estimator, pools, cfg, round_i: int) -> list[np.ndarray]:
    """Sequential score vectors under beta-neutral random pending orders."""

    vectors: list[np.ndarray] = []
    for pool_index, features in enumerate(pools):
        rng = np.random.default_rng(AFE2.named_seed(
            cfg.seed, "adaptive_beta_order", round_i, pool_index
        ))
        order = torch.as_tensor(
            rng.permutation(cfg.K), device=features.device, dtype=torch.long
        )
        vectors.extend([
            values.detach().cpu().numpy()
            for values in estimator.sequential_score_vectors(
                features, order, min(cfg.B, cfg.K)
            )
        ])
    if not vectors:
        raise RuntimeError("round-local beta calibration produced no score vectors")
    return vectors


def calibrate_from_pools(estimator, pools, cfg, round_i: int, target: float) -> dict:
    """Solve beta for one estimator on already-generated current-policy pools."""

    started = time.perf_counter()
    vectors = score_vectors(estimator, pools, cfg, round_i)
    solution = BC.solve_beta_ragged(vectors, target=target)
    return {
        "status": "CALIBRATED_AFE_ROUND_LOCAL_ESS_V1",
        "round": int(round_i),
        "target": float(target),
        "beta": float(solution["beta"]),
        "solution": solution,
        "score_vector_sha256": BC.score_vectors_sha256(vectors),
        "score_vector_count": len(vectors),
        "context_count": len(pools),
        "verifier_queries": 0,
        "seconds": float(time.perf_counter() - started),
    }


def rbf_counterfactual_sweep(
    pools,
    buffer_features: torch.Tensor,
    cfg,
    round_i: int,
    target: float,
    *,
    lengthscale: float,
    multipliers=(0.5, 1.0, 2.0),
    caps=(128, 512),
) -> list[dict]:
    """Offline score-scale sweep; never selects or verifies an action."""

    rows = []
    full = RC.l2_normalize(buffer_features.detach())
    for cap in caps:
        count = min(int(cap), int(full.shape[0]))
        if count < 2:
            continue
        indices = torch.linspace(
            0, full.shape[0] - 1, steps=count, device=full.device
        ).round().to(torch.long)
        subset = full[indices]
        for multiplier in multipliers:
            gp = RC.RBFGPSigma(
                lengthscale=float(lengthscale) * float(multiplier),
                lam=cfg.gp_lam,
            )
            gp.set_buffer(subset)
            calibrated = calibrate_from_pools(
                gp, pools, cfg, round_i, target
            )
            rows.append({
                "cap": count,
                "lengthscale_multiplier": float(multiplier),
                "lengthscale": float(lengthscale) * float(multiplier),
                "beta": calibrated["beta"],
                "achieved": calibrated["solution"]["achieved"],
                "sigma_span_med": calibrated["solution"]["sigma_span_med"],
                "flat_pool_fraction": calibrated["solution"]["flat_pool_fraction"],
                "score_vector_sha256": calibrated["score_vector_sha256"],
            })
    return rows
