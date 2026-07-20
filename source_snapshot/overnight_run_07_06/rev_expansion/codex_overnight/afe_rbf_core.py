"""Small, explicit RBF-GP acquisition pieces for single-arm Safe Flow Expansion.

The acquisition buffer and the CFM replay store deliberately have different
semantics:

* ``D+`` remains cumulative and contains every full-window verifier positive.
* the exact GP contains at most ``cap`` positives from a declared recent-round
  window, selected without replacement and balanced across round/gamma cells.

The GP is frozen while a round's closed-loop replicas are gathered.  This makes
parallel replicas order-independent and defines sigma as *novelty relative to the
declared recent-round buffer*, not as a calibrated probability of validity.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

import afe_context as CX


def l2_normalize(values: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def mean_pairwise_lengthscale(features: torch.Tensor) -> float:
    """Mean off-diagonal distance of the supplied normalized pretrained features."""

    values = l2_normalize(features.detach().to(torch.float64))
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("length-scale calibration requires at least two feature rows")
    distances = torch.pdist(values, p=2)
    ell = float(distances.mean())
    if not math.isfinite(ell) or ell <= 0.0:
        raise ValueError("pretrained feature distances do not define a positive RBF length scale")
    return ell


class RBFGPSigma:
    """Exact RBF-GP posterior standard deviation on an explicitly capped buffer."""

    def __init__(self, lengthscale: float, lam: float = 1.0e-2):
        if not math.isfinite(lengthscale) or lengthscale <= 0.0:
            raise ValueError("RBF length scale must be finite and positive")
        if not math.isfinite(lam) or lam <= 0.0:
            raise ValueError("GP noise must be finite and positive")
        self.ell = float(lengthscale)
        self.lam = float(lam)
        self.X: torch.Tensor | None = None
        self.L: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return 0 if self.X is None else int(self.X.shape[0])

    @staticmethod
    def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (
            (a * a).sum(dim=1, keepdim=True)
            + (b * b).sum(dim=1)[None]
            - 2.0 * a @ b.T
        ).clamp_min(0.0)

    def _kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sqdist(a, b) / (2.0 * self.ell * self.ell))

    @torch.no_grad()
    def set_buffer(self, features: torch.Tensor | None) -> None:
        if features is None or int(features.shape[0]) == 0:
            self.X = None
            self.L = None
            return
        self.X = l2_normalize(features.detach())
        kernel = self._kernel(self.X, self.X).to(torch.float64)
        eye = torch.eye(kernel.shape[0], dtype=torch.float64, device=kernel.device)
        jitter = self.lam
        last_error = None
        for _ in range(6):
            try:
                self.L = torch.linalg.cholesky(kernel + jitter * eye)
                return
            except RuntimeError as error:
                last_error = error
                jitter *= 10.0
        raise RuntimeError("RBF-GP Cholesky failed after jitter retries") from last_error

    @torch.no_grad()
    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        query = l2_normalize(features.detach())
        if self.X is None:
            return torch.ones(query.shape[0], dtype=query.dtype, device=query.device)
        cross = self._kernel(query, self.X)
        solved = torch.cholesky_solve(cross.T.to(torch.float64), self.L)
        reduction = (cross * solved.T.to(cross.dtype)).sum(dim=1)
        return (1.0 - reduction).clamp_min(0.0).sqrt()

    @torch.no_grad()
    def posterior_covariance(
        self,
        features: torch.Tensor,
        *,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        """Joint GP posterior covariance for one candidate batch."""

        query = l2_normalize(features.detach())
        covariance = self._kernel(query, query)
        if self.X is not None:
            cross = self._kernel(query, self.X)
            solved = torch.cholesky_solve(cross.T.to(torch.float64), self.L)
            covariance = covariance - cross @ solved.to(cross.dtype)
        covariance = 0.5 * (covariance + covariance.T)
        if include_observation_noise:
            covariance = covariance + self.lam * torch.eye(
                covariance.shape[0], dtype=covariance.dtype, device=covariance.device
            )
        return covariance

    @torch.no_grad()
    def conditional_variance(self, features: torch.Tensor, jitter: float = 1.0e-6) -> torch.Tensor:
        """Var(f_i | f_{-i}, GP buffer), matching the peptide implementation.

        For joint posterior covariance ``C``, the Schur-complement identity is
        ``Var(f_i | f_{-i}) = 1 / [C^{-1}]_ii``.  Conditioning on the rest of
        the K-pool makes near-duplicate candidates suppress one another even
        when their marginal variances are similar.
        """

        covariance = self.posterior_covariance(features)
        eye = torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        )
        covariance = covariance + float(jitter) * eye
        factor = torch.linalg.cholesky(covariance.to(torch.float64))
        inverse_factor = torch.linalg.solve_triangular(
            factor,
            eye.to(torch.float64),
            upper=False,
        )
        inverse_diagonal = inverse_factor.square().sum(dim=0).clamp_min(1.0e-12)
        conditional = (1.0 / inverse_diagonal).to(features.dtype)
        # Same prior-variance normalization used by the reference peptide code.
        return (conditional / (1.0 + self.lam)).clamp(0.0, 1.0)

    @staticmethod
    def _condition_covariance(
        covariance: torch.Tensor,
        remaining: torch.Tensor,
        chosen_local: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Condition a pending-point covariance on one newly selected location."""

        keep = torch.ones(
            covariance.shape[0], dtype=torch.bool, device=covariance.device
        )
        keep[int(chosen_local)] = False
        if int(keep.sum()) == 0:
            return covariance.new_zeros((0, 0)), remaining[keep]
        cross = covariance[keep, int(chosen_local)]
        denominator = covariance[int(chosen_local), int(chosen_local)].clamp_min(1.0e-12)
        conditioned = covariance[keep][:, keep] - torch.outer(cross, cross) / denominator
        conditioned = 0.5 * (conditioned + conditioned.T)
        return conditioned, remaining[keep]

    @torch.no_grad()
    def sequential_score_vectors(
        self,
        features: torch.Tensor,
        order: torch.Tensor,
        steps: int,
    ) -> list[torch.Tensor]:
        """Scores seen under a fixed pending-point order for beta calibration.

        ``order`` is deliberately chosen independently of beta.  At step ``b``,
        the returned vector is the posterior variance of every still-pending
        candidate conditioned only on the GP buffer and the first ``b`` selected
        locations.  Unqueried candidates are never treated as observations.
        """

        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError("sequential acquisition requires a K-by-D feature matrix")
        if steps < 1 or steps > features.shape[0]:
            raise ValueError("sequential acquisition steps must lie in [1, K]")
        order = order.to(device=features.device, dtype=torch.long).flatten()
        if order.numel() != features.shape[0] or sorted(order.tolist()) != list(
            range(features.shape[0])
        ):
            raise ValueError("sequential calibration order must be a permutation of K")

        covariance = self.posterior_covariance(features)
        remaining = torch.arange(features.shape[0], device=features.device)
        vectors: list[torch.Tensor] = []
        for step in range(steps):
            scores = (torch.diagonal(covariance) / (1.0 + self.lam)).clamp(0.0, 1.0)
            vectors.append(scores)
            chosen = int(order[step])
            locations = torch.nonzero(remaining == chosen, as_tuple=False).flatten()
            if locations.numel() != 1:
                raise RuntimeError("sequential calibration order selected a missing candidate")
            covariance, remaining = self._condition_covariance(
                covariance, remaining, int(locations[0])
            )
        return vectors

    @torch.no_grad()
    def sequential_acquire(
        self,
        features: torch.Tensor,
        steps: int,
        beta: float,
    ) -> tuple[list[int], list[dict[str, object]]]:
        """Draw a B-budget batch sequentially from an RBF posterior.

        After each draw, only that selected location is added to the pending-point
        conditioning set.  This is the budget-consistent counterpart of the
        peptide implementation's all-batch conditional variance when ``B < K``.
        """

        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("sequential acquisition beta must be finite and positive")
        if steps < 1 or steps > features.shape[0]:
            raise ValueError("sequential acquisition steps must lie in [1, K]")

        covariance = self.posterior_covariance(features)
        remaining = torch.arange(features.shape[0], device=features.device)
        selected: list[int] = []
        trace: list[dict[str, object]] = []
        for _ in range(steps):
            scores = (torch.diagonal(covariance) / (1.0 + self.lam)).clamp(0.0, 1.0)
            weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
            probability = weights / weights.sum()
            chosen_local = int(torch.multinomial(probability, 1).item())
            chosen_global = int(remaining[chosen_local])
            normalized_ess = float(
                1.0 / (probability.to(torch.float64).square().sum() * probability.numel())
            )
            normalized_entropy = (
                float(
                    -(probability.to(torch.float64)
                      * (probability.to(torch.float64) + 1.0e-30).log()).sum()
                    / math.log(probability.numel())
                )
                if probability.numel() > 1
                else 1.0
            )
            trace.append({
                "scores": scores,
                "remaining": remaining,
                "chosen": chosen_global,
                "chosen_score": float(scores[chosen_local]),
                "ess_norm": normalized_ess,
                "entropy_norm": normalized_entropy,
            })
            selected.append(chosen_global)
            covariance, remaining = self._condition_covariance(
                covariance, remaining, chosen_local
            )
        return selected, trace

    @torch.no_grad()
    def diagnostics(self) -> dict[str, object]:
        if self.X is None:
            return {
                "n": 0,
                "kernel_effective_rank": 0.0,
                "kernel_eigenvalues": [],
                "buffer_sigma_mean": None,
            }
        kernel = self._kernel(self.X, self.X).to(torch.float64)
        eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0.0)
        effective_rank = float(
            eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1.0e-12)
        )
        return {
            "n": self.n,
            "kernel_effective_rank": effective_rank,
            "kernel_eigenvalues": [float(value) for value in eigenvalues.cpu()],
            "buffer_sigma_mean": float(self.sigma(self.X).mean()),
        }


def previous_round_positive_ids(store, round_i: int, cap: int, gammas, seed: int) -> list[int]:
    """Gamma-balanced, without-replacement compression of round ``round_i`` positives."""

    if cap <= 0:
        raise ValueError("GP buffer cap must be positive")
    gamma_storage_map = CX.declared_gamma_storage_map(gammas)
    groups: dict[float, list[int]] = defaultdict(list)
    for query_id in store.pos_ids:
        if int(store.q_round[query_id]) == int(round_i):
            gamma = CX.canonical_declared_gamma(
                store.q_gamma[query_id], gamma_storage_map
            )
            groups[gamma].append(int(query_id))
    all_ids = [query_id for values in groups.values() for query_id in values]
    if len(all_ids) <= cap:
        return sorted(all_ids)

    rng = np.random.default_rng(int(seed))
    gamma_keys = list(gamma_storage_map.values())
    quota, extra = divmod(cap, len(gamma_keys))
    selected: list[int] = []
    selected_set: set[int] = set()
    for index, gamma in enumerate(gamma_keys):
        candidates = np.asarray(groups.get(gamma, []), dtype=np.int64)
        take = min(len(candidates), quota + int(index < extra))
        if take:
            chosen = rng.choice(candidates, size=take, replace=False).tolist()
            selected.extend(int(value) for value in chosen)
            selected_set.update(int(value) for value in chosen)
    if len(selected) < cap:
        remaining = np.asarray(
            [query_id for query_id in all_ids if query_id not in selected_set],
            dtype=np.int64,
        )
        take = min(cap - len(selected), len(remaining))
        if take:
            selected.extend(int(value) for value in rng.choice(remaining, size=take, replace=False))
    if len(selected) != cap or len(set(selected)) != cap:
        raise RuntimeError("failed to construct the declared without-replacement GP buffer")
    return sorted(selected)


def recent_round_positive_ids(
    store,
    round_i: int,
    replay_window: int,
    cap: int,
    gammas,
    seed: int,
) -> list[int]:
    """Compress recent positives, balanced across non-empty round/gamma cells.

    ``replay_window=1`` deliberately delegates to the original selector so the
    established previous-round buffer is bit-for-bit unchanged.
    """

    replay_window = int(replay_window)
    if replay_window < 1:
        raise ValueError("GP replay window must be at least one round")
    if replay_window == 1:
        return previous_round_positive_ids(store, round_i, cap, gammas, seed)
    if cap <= 0:
        raise ValueError("GP buffer cap must be positive")

    last_round = int(round_i)
    first_round = last_round - replay_window + 1
    gamma_storage_map = CX.declared_gamma_storage_map(gammas)
    groups: dict[tuple[int, float], list[int]] = defaultdict(list)
    all_ids: list[int] = []
    for query_id in store.pos_ids:
        query_round = int(store.q_round[query_id])
        if first_round <= query_round <= last_round:
            query_id = int(query_id)
            gamma = CX.canonical_declared_gamma(
                store.q_gamma[query_id], gamma_storage_map
            )
            groups[(query_round, gamma)].append(query_id)
            all_ids.append(query_id)
    if len(all_ids) <= cap:
        return sorted(all_ids)

    gamma_keys = list(gamma_storage_map.values())
    cell_keys = [
        (query_round, gamma)
        for query_round in range(first_round, last_round + 1)
        for gamma in gamma_keys
        if groups.get((query_round, gamma))
    ]
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    selected_set: set[int] = set()
    if cell_keys:
        quota, extra = divmod(cap, len(cell_keys))
        for index, cell in enumerate(cell_keys):
            candidates = np.asarray(groups[cell], dtype=np.int64)
            take = min(len(candidates), quota + int(index < extra))
            if take:
                chosen = rng.choice(candidates, size=take, replace=False).tolist()
                selected.extend(int(value) for value in chosen)
                selected_set.update(int(value) for value in chosen)
    if len(selected) < cap:
        remaining = np.asarray(
            [query_id for query_id in all_ids if query_id not in selected_set],
            dtype=np.int64,
        )
        take = min(cap - len(selected), len(remaining))
        if take:
            selected.extend(int(value) for value in rng.choice(remaining, size=take, replace=False))
    if len(selected) != cap or len(set(selected)) != cap:
        raise RuntimeError("failed to construct the declared without-replacement GP buffer")
    return sorted(selected)


def recent_round_positive_ids_hierarchical(
    store,
    round_i: int,
    replay_window: int,
    cap: int,
    seed: int,
) -> list[int]:
    """Without-replacement round/gamma/replica/context-balanced GP buffer.

    This is an opt-in V2 selector.  Recursive round-robin interleaving prevents
    a long replica or a context with several positive queries from dominating
    the fixed-size RBF memory.  The legacy selectors above remain unchanged.
    """

    replay_window = int(replay_window)
    cap = int(cap)
    if replay_window < 1:
        raise ValueError("GP replay window must be at least one round")
    if cap < 1:
        raise ValueError("GP buffer cap must be positive")
    first_round = max(0, int(round_i) - replay_window + 1)
    eligible = [
        int(query_id)
        for query_id in store.pos_ids
        if first_round <= int(store.q_round[query_id]) <= int(round_i)
    ]
    if len(eligible) <= cap:
        return sorted(eligible)
    hierarchy = store.positive_replay_hierarchy(eligible_ids=eligible)
    rng = np.random.default_rng(int(seed))

    def shuffled(values):
        ordered = sorted(values)
        return [ordered[index] for index in rng.permutation(len(ordered))]

    def interleave(generators):
        active = [(key, generators[key]) for key in shuffled(generators)]
        while active:
            next_active = []
            for key, generator in active:
                try:
                    yield next(generator)
                    next_active.append((key, generator))
                except StopIteration:
                    pass
            active = next_active

    def query_generator(query_ids):
        yield from shuffled(query_ids)

    def context_generator(contexts):
        yield from interleave({
            context_id: query_generator(query_ids)
            for context_id, query_ids in contexts.items()
        })

    def replica_generator(replicas):
        yield from interleave({
            replica: context_generator(contexts)
            for replica, contexts in replicas.items()
        })

    def gamma_generator(gammas):
        yield from interleave({
            gamma: replica_generator(replicas)
            for gamma, replicas in gammas.items()
        })

    order = interleave({
        query_round: gamma_generator(gammas)
        for query_round, gammas in hierarchy.items()
    })
    selected = [int(query_id) for _, query_id in zip(range(cap), order)]
    if len(selected) != cap or len(set(selected)) != cap:
        raise RuntimeError("failed to build hierarchical GP buffer without replacement")
    return sorted(selected)


_VERIFY_ENV = None
_VERIFY_GOAL = None
_VERIFY_REACH = None
_VERIFY_N_THETA = None


def initialize_verifier_worker(scene_profile: str, reach: float, n_theta: int) -> None:
    """Process-pool initializer; each CPU worker builds one immutable scene."""

    global _VERIFY_ENV, _VERIFY_GOAL, _VERIFY_REACH, _VERIFY_N_THETA
    from afe2_scene_profiles import build_scene, get_scene_profile
    import grid_metrics2 as metrics2

    torch.set_num_threads(1)
    profile = get_scene_profile(scene_profile)
    _VERIFY_ENV = build_scene(profile)
    _VERIFY_GOAL = _VERIFY_ENV.goal.detach().cpu().numpy()
    _VERIFY_REACH = float(reach)
    _VERIFY_N_THETA = int(n_theta)
    metrics2.GOAL_XY = np.asarray(profile.goal, dtype=float)


def verify_in_worker(task):
    """Return the task identity plus one terminal-aware deterministic verifier result."""

    if _VERIFY_ENV is None:
        raise RuntimeError("verifier worker was not initialized")
    from afe_core import verify_plan_with_terminal

    episode_id, candidate_id, state, controls, gamma = task
    result = verify_plan_with_terminal(
        state,
        controls,
        _VERIFY_ENV,
        float(gamma),
        _VERIFY_GOAL,
        reach=_VERIFY_REACH,
        n_theta=_VERIFY_N_THETA,
    )
    return int(episode_id), int(candidate_id), result
