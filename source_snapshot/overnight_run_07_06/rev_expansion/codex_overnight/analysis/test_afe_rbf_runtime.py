from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import grid_expand_afe_rbf as R
from flow_policy import FlowPolicy


def test_proposal_stream_is_episode_stable_and_eval_round_independent() -> None:
    policy = SimpleNamespace(d=4)
    cfg = SimpleNamespace(K=3, seed=910)
    active = [{"episode_id": 0}, {"episode_id": 7}]

    together = R._proposal_noise(
        policy, active, cfg, "gather", 2, 11, "cpu"
    )
    survivor = R._proposal_noise(
        policy, [active[1]], cfg, "gather", 2, 11, "cpu"
    )
    assert torch.equal(together[cfg.K:], survivor)

    eval_round_0 = R._proposal_noise(
        policy, active, cfg, "controller_eval", 0, 11, "cpu"
    )
    eval_round_9 = R._proposal_noise(
        policy, active, cfg, "controller_eval", 9, 11, "cpu"
    )
    gather_round_9 = R._proposal_noise(
        policy, active, cfg, "gather", 9, 11, "cpu"
    )
    assert torch.equal(eval_round_0, eval_round_9)
    assert not torch.equal(together, gather_round_9)


def test_flow_sampler_accepts_explicit_initial_noise_without_global_rng() -> None:
    policy = FlowPolicy(T=2, ctx_dim=1, width=8, depth=1, u_max=1.0)
    context = torch.zeros(3, 1)
    initial_noise = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0

    torch.manual_seed(1)
    first = policy.sample(3, context, nfe=2, initial_noise=initial_noise)
    torch.manual_seed(999)
    second = policy.sample(3, context, nfe=2, initial_noise=initial_noise)

    assert torch.equal(first, second)
