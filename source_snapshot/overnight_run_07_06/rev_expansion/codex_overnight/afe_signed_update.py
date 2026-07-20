"""Signed AFE replay update with separate recent positive and negative batches."""
from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import torch

import grid_expand_afe2 as AFE2


def _query_batch(store, query_ids: list[int], batch: int, rng):
    """Uniform-with-replacement batch reconstruction for an explicit population."""

    positions = rng.integers(0, len(query_ids), batch)
    ids = [int(query_ids[int(position)]) for position in positions]
    sids = [int(store.q_sid[query_id]) for query_id in ids]
    grid = store.grid3_of(sids)
    low = torch.stack([
        torch.as_tensor(store.ctx_low5[sid], dtype=torch.float32) for sid in sids
    ])
    hist = torch.stack([
        torch.as_tensor(store.ctx_hist[sid], dtype=torch.float32) for sid in sids
    ])
    controls = torch.stack([
        torch.as_tensor(store.q_U[query_id], dtype=torch.float32) for query_id in ids
    ])
    return grid, low, hist, controls, ids


def _negative_ids(store, *, round_i, replay_window) -> list[int]:
    """Queries from terminal NVP contexts; SOCP errors were never stored.

    Full-H positives remain the sole positive CFM population.  The two populations
    may overlap: a full-H SOCP-positive plan can still be evidence against
    closed-loop viability when every queried plan at that context fails the
    declared first-step execution gate.  This is the task-specific NVP hypothesis
    under test, not the original AFE partition over all verifier rejections.
    """

    labels = getattr(store, "q_nvp_negative", None)
    if labels is None:
        raise RuntimeError("signed NVP replay requires persistent q_nvp_negative labels")

    if replay_window is None:
        return [
            query_id for query_id, label in enumerate(labels)
            if int(label) == 1
        ]
    if round_i is None:
        raise ValueError("windowed negative replay requires the current round")
    replay_window = int(replay_window)
    if replay_window < 1:
        raise ValueError("negative replay window must be at least one round")
    first_round = max(1, int(round_i) - replay_window + 1)
    return [
        query_id for query_id, label in enumerate(labels)
        if int(label) == 1
        and first_round <= int(store.q_round[query_id]) <= int(round_i)
    ]


def _squared_norm(gradients: Iterable[torch.Tensor | None], device) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().to(torch.float64).square().sum()
    return total


def _gradient_norm(gradients: Iterable[torch.Tensor | None], device) -> torch.Tensor:
    return _squared_norm(gradients, device).sqrt()


def _group_norms(groups, parameters, gradients, device) -> dict[str, float]:
    gradient_by_parameter = {
        id(parameter): gradient
        for parameter, gradient in zip(parameters, gradients)
    }
    output = {}
    for name, group_parameters in groups.items():
        values = [
            gradient_by_parameter.get(id(parameter))
            for parameter in group_parameters
            if parameter.requires_grad
        ]
        output[name] = float(_gradient_norm(values, device))
    return output


def _parameter_norm(parameters) -> float:
    if not parameters:
        return 0.0
    return float(torch.stack([
        parameter.detach().to(torch.float64).square().sum()
        for parameter in parameters
    ]).sum().sqrt())


def _draw_diagnostics(store, eligible_ids, drawn_ids, round_i) -> dict:
    eligible_round_counts: dict[str, int] = {}
    for query_id in eligible_ids:
        key = str(int(store.q_round[query_id]))
        eligible_round_counts[key] = eligible_round_counts.get(key, 0) + 1
    draw_round_counts: dict[str, int] = {}
    for query_id, count in drawn_ids.items():
        key = str(int(store.q_round[query_id]))
        draw_round_counts[key] = draw_round_counts.get(key, 0) + int(count)
    fresh_draws = (
        0 if round_i is None else
        sum(
            int(count) for query_id, count in drawn_ids.items()
            if int(store.q_round[query_id]) == int(round_i)
        )
    )
    fresh_distinct = (
        0 if round_i is None else
        sum(
            int(int(store.q_round[query_id]) == int(round_i))
            for query_id in drawn_ids
        )
    )
    total_draws = int(sum(drawn_ids.values()))
    return {
        "eligible_round_counts": eligible_round_counts,
        "draw_round_counts": draw_round_counts,
        "fresh_draws": int(fresh_draws),
        "fresh_distinct": int(fresh_distinct),
        "fresh_fraction": float(fresh_draws / total_draws) if total_draws else 0.0,
    }


def update_round_signed(
    policy,
    opt,
    store,
    cfg,
    device,
    rng,
    round_i=None,
    *,
    alpha: float = 0.0,
    negative_rng=None,
    eps: float = 1.0e-12,
):
    """Apply ``g+ - rho*g-`` with paper gradient-norm normalization.

    Alpha zero is an exact compatibility gate: it delegates before inspecting
    or drawing from the negative archive.
    """

    if alpha == 0:
        return AFE2.update_round(policy, opt, store, cfg, device, rng, round_i)

    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("signed-update alpha must be finite and non-negative")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("signed-update epsilon must be finite and positive")
    if getattr(cfg, "arm", "afe") != "afe":
        raise ValueError("signed replay is defined only for the AFE arm")

    replay_window = getattr(cfg, "replay_window", None)
    eligible_positive = store.positive_ids(
        round_i=round_i,
        replay_window=replay_window,
    )
    if not eligible_positive:
        return None
    replay_sampling = getattr(
        cfg,
        "replay_sampling",
        getattr(cfg, "positive_replay_sampling", "query_uniform"),
    )
    replay_hierarchy = (
        store.positive_replay_hierarchy(eligible_ids=eligible_positive)
        if replay_sampling == "round_gamma_replica_context"
        else None
    )
    eligible_negative = _negative_ids(
        store,
        round_i=round_i,
        replay_window=replay_window,
    )
    if not eligible_negative:
        result = AFE2.update_round(policy, opt, store, cfg, device, rng, round_i)
        if result is None:
            return None
        result = dict(result)
        result.update({
            "alpha": alpha,
            "signed_active": False,
            "negative_replay_eligible": 0,
            "negative_drawn_ids": {},
            "negative_n_distinct": 0,
        })
        return result

    batch = int(cfg.batch)
    if batch < 1:
        raise ValueError("signed replay batch must be positive")
    n_steps = int(cfg.afe_steps)
    if n_steps < 1:
        raise ValueError("signed replay steps must be positive")
    if negative_rng is None:
        negative_rng = np.random.default_rng(AFE2.named_seed(
            getattr(cfg, "seed", 0), "negative_replay", round_i
        ))

    policy.train()
    groups = {
        name: list(module.parameters())
        for name, module in policy.module_groups().items()
    }
    before_norm = {
        name: _parameter_norm(parameters)
        for name, parameters in groups.items()
    }
    snapshot = {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in groups.items()
    }
    trainable = [
        parameter for parameter in policy.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("signed replay requires at least one trainable parameter")

    positive_draws: dict[int, int] = {}
    negative_draws: dict[int, int] = {}
    positive_losses: list[float] = []
    negative_losses: list[float] = []
    signed_objectives: list[float] = []
    functional_steps: list[float] = []
    step_diagnostics: list[dict[str, float | dict[str, float]]] = []
    signed_group_history = {name: [] for name in groups}
    positive_group_history = {name: [] for name in groups}
    negative_group_history = {name: [] for name in groups}
    probe = None
    value_before = None

    for _ in range(n_steps):
        positive_batch = (
            store.sample_pos(batch, rng, eligible_ids=eligible_positive)
            if replay_sampling == "query_uniform"
            else store.sample_pos(
                batch,
                rng,
                eligible_ids=eligible_positive,
                sampling=replay_sampling,
                hierarchy=replay_hierarchy,
            )
        )
        if positive_batch is None:
            raise RuntimeError("positive replay population became empty during update")
        grid_pos, low_pos, hist_pos, controls_pos, positive_ids = positive_batch
        grid_neg, low_neg, hist_neg, controls_neg, negative_ids = _query_batch(
            store, eligible_negative, batch, negative_rng
        )
        for query_id in positive_ids:
            positive_draws[int(query_id)] = positive_draws.get(int(query_id), 0) + 1
        for query_id in negative_ids:
            negative_draws[int(query_id)] = negative_draws.get(int(query_id), 0) + 1

        grid_pos = grid_pos.to(device)
        low_pos = low_pos.to(device)
        hist_pos = hist_pos.to(device)
        controls_pos = controls_pos.to(device)
        grid_neg = grid_neg.to(device)
        low_neg = low_neg.to(device)
        hist_neg = hist_neg.to(device)
        controls_neg = controls_neg.to(device)

        if probe is None:
            probe_count = min(controls_pos.shape[0], 128)
            probe_x = 0.5 * (controls_pos[:probe_count] / policy.u_max).reshape(
                probe_count, policy.d
            )
            probe_t = torch.full((probe_count,), 0.5, device=device)
            probe_context = policy.ctx_from(
                grid_pos[:probe_count], low_pos[:probe_count], hist_pos[:probe_count]
            ).detach()
            with torch.no_grad():
                value_before = policy.forward(
                    probe_x,
                    probe_t,
                    policy._expand_ctx(probe_context, probe_count),
                ).detach()
            probe = (probe_x, probe_t, probe_context, probe_count)

        positive_loss = policy.cfm_loss(
            controls_pos,
            policy.ctx_from(grid_pos, low_pos, hist_pos),
        )
        negative_loss = policy.cfm_loss(
            controls_neg,
            policy.ctx_from(grid_neg, low_neg, hist_neg),
        )
        positive_gradients = torch.autograd.grad(
            positive_loss, trainable, allow_unused=True
        )
        negative_gradients = torch.autograd.grad(
            negative_loss, trainable, allow_unused=True
        )
        positive_norm = _gradient_norm(positive_gradients, device)
        negative_norm = _gradient_norm(negative_gradients, device)
        rho_tensor = alpha * positive_norm / (negative_norm + eps)
        rho = float(rho_tensor)

        dot = torch.zeros((), dtype=torch.float64, device=device)
        combined_gradients: list[torch.Tensor | None] = []
        for positive_gradient, negative_gradient in zip(
            positive_gradients, negative_gradients
        ):
            if positive_gradient is not None and negative_gradient is not None:
                dot = dot + (
                    positive_gradient.detach().to(torch.float64)
                    * negative_gradient.detach().to(torch.float64)
                ).sum()
            if positive_gradient is None and negative_gradient is None:
                combined_gradients.append(None)
            elif positive_gradient is None:
                combined_gradients.append(-rho * negative_gradient.detach())
            elif negative_gradient is None:
                combined_gradients.append(positive_gradient.detach())
            else:
                combined_gradients.append(
                    positive_gradient.detach() - rho * negative_gradient.detach()
                )

        denominator = positive_norm * negative_norm
        cosine = (
            float((dot / denominator).clamp(-1.0, 1.0))
            if float(denominator) > eps else 0.0
        )
        combined_norm = _gradient_norm(combined_gradients, device)
        positive_group = _group_norms(
            groups, trainable, positive_gradients, device
        )
        negative_group = _group_norms(
            groups, trainable, negative_gradients, device
        )
        combined_group = _group_norms(
            groups, trainable, combined_gradients, device
        )

        opt.zero_grad()
        for parameter, gradient in zip(trainable, combined_gradients):
            parameter.grad = None if gradient is None else gradient.clone()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        post_clip_gradients = [parameter.grad for parameter in trainable]
        post_clip_norm = _gradient_norm(post_clip_gradients, device)
        opt.step()

        positive_value = float(positive_loss.detach())
        negative_value = float(negative_loss.detach())
        signed_value = positive_value - rho * negative_value
        positive_losses.append(positive_value)
        negative_losses.append(negative_value)
        signed_objectives.append(signed_value)
        for name in groups:
            positive_group_history[name].append(positive_group[name])
            negative_group_history[name].append(negative_group[name])
            signed_group_history[name].append(combined_group[name])
        step_diagnostics.append({
            "positive_grad_norm": float(positive_norm),
            "negative_grad_norm": float(negative_norm),
            "scaled_negative_grad_norm": float(rho_tensor * negative_norm),
            "signed_grad_norm": float(combined_norm),
            "post_clip_grad_norm": float(post_clip_norm),
            "gradient_cosine": cosine,
            "rho": rho,
            "positive_grad_norm_by_group": positive_group,
            "negative_grad_norm_by_group": negative_group,
            "signed_grad_norm_by_group": combined_group,
        })

        probe_x, probe_t, probe_context, probe_count = probe
        with torch.no_grad():
            value_after = policy.forward(
                probe_x,
                probe_t,
                policy._expand_ctx(probe_context, probe_count),
            )
            functional_step = float(
                (value_after - value_before).norm(dim=1).mean()
                / value_before.norm(dim=1).mean().clamp_min(1.0e-9)
            )
        functional_steps.append(functional_step)

    relative_change = {}
    for name, parameters in groups.items():
        delta = _parameter_norm([
            parameter.detach() - original
            for parameter, original in zip(parameters, snapshot[name])
        ])
        relative_change[name] = delta / max(before_norm[name], 1.0e-12)

    positive_stats = _draw_diagnostics(
        store, eligible_positive, positive_draws, round_i
    )
    negative_stats = _draw_diagnostics(
        store, eligible_negative, negative_draws, round_i
    )

    def mean_step(key: str) -> float:
        return float(np.mean([row[key] for row in step_diagnostics]))

    return {
        "steps": len(positive_losses),
        "stop": "all_steps",
        "cfm": float(np.mean(positive_losses)),
        "cfm_first": positive_losses[0],
        "cfm_last": positive_losses[-1],
        "negative_cfm": float(np.mean(negative_losses)),
        "negative_cfm_first": negative_losses[0],
        "negative_cfm_last": negative_losses[-1],
        "signed_objective": float(np.mean(signed_objectives)),
        "fstep_final": functional_steps[-1],
        "fstep_max": max(functional_steps),
        "grad_norm": {
            name: float(np.mean(values))
            for name, values in signed_group_history.items()
        },
        "positive_grad_norm_by_group": {
            name: float(np.mean(values))
            for name, values in positive_group_history.items()
        },
        "negative_grad_norm_by_group": {
            name: float(np.mean(values))
            for name, values in negative_group_history.items()
        },
        "rel_param_change": relative_change,
        "alpha": alpha,
        "signed_active": True,
        "positive_grad_norm": mean_step("positive_grad_norm"),
        "negative_grad_norm": mean_step("negative_grad_norm"),
        "scaled_negative_grad_norm": mean_step("scaled_negative_grad_norm"),
        "signed_grad_norm": mean_step("signed_grad_norm"),
        "post_clip_grad_norm": mean_step("post_clip_grad_norm"),
        "gradient_cosine": mean_step("gradient_cosine"),
        "rho": mean_step("rho"),
        "signed_step_diagnostics": step_diagnostics,
        "drawn_ids": positive_draws,
        "n_distinct": len(positive_draws),
        "negative_drawn_ids": negative_draws,
        "negative_n_distinct": len(negative_draws),
        "replay_window": replay_window,
        "replay_sampling": replay_sampling,
        "replay_eligible": len(eligible_positive),
        "replay_fresh_draws": positive_stats["fresh_draws"],
        "replay_fresh_distinct": positive_stats["fresh_distinct"],
        "replay_eligible_round_counts": positive_stats["eligible_round_counts"],
        "replay_draw_round_counts": positive_stats["draw_round_counts"],
        "replay_fresh_fraction": positive_stats["fresh_fraction"],
        "negative_replay_eligible": len(eligible_negative),
        "negative_replay_fresh_draws": negative_stats["fresh_draws"],
        "negative_replay_fresh_distinct": negative_stats["fresh_distinct"],
        "negative_replay_eligible_round_counts": negative_stats["eligible_round_counts"],
        "negative_replay_draw_round_counts": negative_stats["draw_round_counts"],
        "negative_replay_fresh_fraction": negative_stats["fresh_fraction"],
    }
