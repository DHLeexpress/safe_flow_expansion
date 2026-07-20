"""SafeMPPI as a proposal source for same-verifier certified backup plans."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import grid_scene
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter


@dataclass(frozen=True)
class BackupProposal:
    plan: np.ndarray
    kind: str
    internal_feasible: bool | None

    def __post_init__(self) -> None:
        plan = np.asarray(self.plan, dtype=np.float32)
        if plan.shape != (10, 2) or not np.isfinite(plan).all():
            raise ValueError(f"backup plan must be finite [10,2], got {plan.shape}")
        object.__setattr__(self, "plan", plan.copy())


class SafeMPPIBackup:
    """Generate proposals only; this class makes no certification claim."""

    def __init__(
        self,
        *,
        smooth_weight: float = 0.12,
        retreat_weight: float = 0.0,
        max_debug_candidates: int = 0,
        noise_var_mult: float = 3.0,
    ) -> None:
        if not np.isfinite(noise_var_mult) or noise_var_mult <= 0.0:
            raise ValueError("noise_var_mult must be finite and positive")
        if int(max_debug_candidates) < 0:
            raise ValueError("max_debug_candidates must be nonnegative")
        config = grid_scene.mode1_config(noise_var_mult=float(noise_var_mult))
        config["smooth_weight"] = float(smooth_weight)
        config["goal_retreat_exp_weight"] = float(retreat_weight)
        config["debug_max_rollouts"] = max(int(max_debug_candidates), 1)
        self.adapter = SafeMPPIAdapter(**config)
        self.max_debug_candidates = int(max_debug_candidates)
        self.noise_var_mult = float(noise_var_mult)

    def reset(self) -> None:
        """Clear episode-local warm-start state without disabling warm starts.

        ``mode1_config`` enables SafeMPPI warm starts.  They are useful between
        replans in one episode, but carrying them into another gamma/episode
        makes the fallback proposal depend on experiment ordering and cannot be
        reconstructed from the query context and seed.
        """

        self.adapter._u_prev = None
        self.adapter._p_prev = None
        self.adapter._size_prev = None

    @torch.inference_mode()
    def propose(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        env,
        gamma: float,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[list[BackupProposal], dict[str, object]]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        goal_tensor = torch.as_tensor(goal, dtype=torch.float32, device=device)
        planner_obstacles = grid_scene.planner_obstacles(env).to(device=device, dtype=torch.float32)
        _action, info = self.adapter.plan(
            state_tensor,
            goal_tensor,
            planner_obstacles,
            gamma=float(gamma),
            seed=int(seed),
            return_rollouts=True,
        )
        proposals: list[BackupProposal] = []
        seen: set[bytes] = set()

        def add(plan: object, kind: str, feasible: bool | None) -> None:
            value = np.asarray(plan, dtype=np.float32)
            if value.shape != (10, 2) or not np.isfinite(value).all():
                return
            key = value.tobytes()
            if key in seen:
                return
            seen.add(key)
            proposals.append(BackupProposal(value, kind, feasible))

        add(info.get("mean_sequence"), "weighted_mean", None)
        add(info.get("best_sequence"), "internal_best", bool(info.get("best_feasible_internal", False)))
        if self.max_debug_candidates:
            debug = info.get("debug_rollouts") or {}
            controls = np.asarray(debug.get("controls", []), dtype=np.float32)
            feasibility = np.asarray(debug.get("feasible", []), dtype=bool)
            # Debug rollouts are optional audit queries only.  Setting the
            # count to zero restores the cost-selected SafeMPPI proposal set
            # (weighted mean plus internal best) exactly and prevents a raw
            # Monte-Carlo sample from becoming a runtime fallback action.
            order = (
                np.argsort(~feasibility, kind="stable")
                if len(feasibility) == len(controls)
                else range(len(controls))
            )
            debug_added = 0
            for position in order:
                if debug_added >= self.max_debug_candidates:
                    break
                internal = (
                    bool(feasibility[position])
                    if len(feasibility) == len(controls)
                    else None
                )
                before = len(proposals)
                add(controls[position], "debug_candidate", internal)
                debug_added += int(len(proposals) > before)
        telemetry = {
            "internal_all_infeasible": bool(info.get("all_samples_infeasible_internal", False)),
            "internal_best_feasible": bool(info.get("best_feasible_internal", False)),
            "proposal_count": len(proposals),
            "polytope_size": info.get("polytope_size"),
        }
        return proposals, telemetry
