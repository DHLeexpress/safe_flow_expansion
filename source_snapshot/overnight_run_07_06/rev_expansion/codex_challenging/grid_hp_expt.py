#!/usr/bin/env python3
"""Original H_P flow-policy structure for the challenging walled scene.

The policy context is exactly the established reference structure:

    context = raw_condition(current state, goal, gamma) + E(H_P)

The established checkpoint uses ``low5`` (relative goal, velocity, gamma).
New checkpoints may opt into ``low7`` by appending the world-frame vector to
the closest physical obstacle boundary.  No raw episode start or absolute goal
coordinates are appended in either case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for _path in (WORK, HERE.parent, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _paths  # noqa: F401,E402
# ``_paths`` intentionally prepends shared experiment directories.  Restore
# this stage directory to the front so same-named local adaptations cannot be
# shadowed by their fixed-goal references.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import grid_feats as GF  # noqa: E402
import grid_policy2 as GP2  # noqa: E402


class GridHPFlowPolicy(GP2.GridGRUFlowPolicy2):
    """Reference H_P visual encoder with goal-aware ``low5`` context."""

    def __init__(
        self,
        width: int = 256,
        depth: int = 2,
        u_max: float = 1.0,
        use_gru: bool = False,
        repr_dim: int | None = 20,
        grid_hw: tuple[int, int] = (32, 32),
        trunk_hidden: tuple[int, ...] = (128, 64),
        enc_depth: int = 2,
        raw_condition_dim: int = 5,
        conditioning_schema: str = "low5",
        boundary_adapter: bool = False,
        boundary_adapter_hidden: int = 0,
        boundary_origin_gate: tuple[float, float, float, float] = (1.25, 0.65, 0.50, 0.47),
        boundary_goal_gate: tuple[float, float, float, float] = (3.95, 4.05, 0.55, 0.55),
        reflection_group_average: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            grid_shape=(1, grid_hw[0], grid_hw[1]),
            width=width,
            depth=depth,
            u_max=u_max,
            use_gru=use_gru,
            encode_low=False,
            use_grid=True,
            **kwargs,
        )
        self.repr_dim = repr_dim
        self.grid_hw = tuple(grid_hw)
        self.trunk_hidden = tuple(trunk_hidden)
        self.enc_depth = int(enc_depth)
        self.depth = int(depth)
        self.raw_condition_dim = int(raw_condition_dim)
        if self.raw_condition_dim < 5:
            raise ValueError("raw_condition_dim must include at least the established low5")
        self.conditioning_schema = str(conditioning_schema)
        if (self.raw_condition_dim, self.conditioning_schema) not in {
            (5, "low5"),
            (7, "low7_closest_boundary"),
            (7, "low7_closest_boundary_tie_mean"),
        }:
            raise ValueError(
                "conditioning dimension and schema must declare low5 or "
                "low7_closest_boundary"
            )
        self.reflection_group_average = bool(reflection_group_average)
        if self.reflection_group_average and (
            self.conditioning_schema != "low7_closest_boundary_tie_mean"
            or use_gru
            or boundary_adapter
        ):
            raise ValueError(
                "reflection group averaging requires tie-mean low7 conditioning "
                "conditioning without a GRU or boundary adapter"
            )
        self.low_raw_dim = self.raw_condition_dim + (self.gru_dim if use_gru else 0)

        # Same shallow CNN/AAP visual encoder as the approved reference, using
        # only channel 2: the clipped nominal H_P level set.
        pool_hw = (8, 8) if max(grid_hw) >= 24 else (4, 3)
        channels = [1, 8, 16]
        while len(channels) - 1 < self.enc_depth:
            channels.append(16)
        conv: list[nn.Module] = []
        for index in range(self.enc_depth):
            conv.extend(
                [
                    nn.Conv2d(channels[index], channels[index + 1], 3, padding=1),
                    nn.SiLU(),
                ]
            )
        self.enc_grid = nn.Sequential(
            *conv,
            nn.AdaptiveAvgPool2d(pool_hw),
            nn.Flatten(),
            nn.Linear(channels[self.enc_depth] * pool_hw[0] * pool_hw[1], 32),
            nn.SiLU(),
        )

        self.ctx_dim = self.low_raw_dim + 32
        in_dim = self.d + self.ctx_dim + self.t_dim
        if repr_dim is None:
            layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers.extend([nn.Linear(width, width), nn.SiLU()])
            self.trunk = nn.Sequential(*layers)
        else:
            dims = [in_dim, *self.trunk_hidden, repr_dim]
            layers = []
            for input_dim, output_dim in zip(dims[:-1], dims[1:]):
                layers.extend([nn.Linear(input_dim, output_dim), nn.SiLU()])
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(repr_dim, self.d)

        self.boundary_adapter = False
        self.boundary_adapter_hidden = int(boundary_adapter_hidden)
        self.boundary_origin_gate = tuple(float(value) for value in boundary_origin_gate)
        self.boundary_goal_gate = tuple(float(value) for value in boundary_goal_gate)
        if boundary_adapter:
            self.enable_boundary_adapter(self.boundary_adapter_hidden)

    def hp_token(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)
        if grid.ndim != 4 or grid.shape[1] != 3:
            raise ValueError(f"grid must have shape [B,3,H,W] or [3,H,W], got {tuple(grid.shape)}")
        return self.enc_grid(grid[:, 2:3].float())

    def _low_raw(self, low: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """Return every declared raw condition; the parent keeps only low5."""

        if low.ndim == 1:
            low = low.unsqueeze(0)
        if low.ndim != 2 or low.shape[1] != self.raw_condition_dim:
            raise ValueError(
                f"raw condition must have shape [B,{self.raw_condition_dim}], "
                f"got {tuple(low.shape)}"
            )
        low = low.float()
        if not self.use_gru:
            return low
        if hist.ndim == 2:
            hist = hist.unsqueeze(0)
        _, hidden = self.gru(hist.float())
        # Preserve the original ordering around the GRU token and retain any
        # explicitly declared conditions after gamma.
        return torch.cat((low[:, :4], hidden[-1], low[:, 4:]), dim=1)

    def ctx_from(
        self,
        grid: torch.Tensor,
        low5: torch.Tensor,
        hist: torch.Tensor,
    ) -> torch.Tensor:
        """Build the declared raw-condition + ``E(H_P)`` context."""
        context = torch.cat((self._low_raw(low5, hist), self.hp_token(grid)), dim=1)
        if not self.reflection_group_average:
            return context
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)
        if low5.ndim == 1:
            low5 = low5.unsqueeze(0)
        if hist.ndim == 2:
            hist = hist.unsqueeze(0)
        n_theta = int(grid.shape[-2])
        if n_theta % 4:
            raise ValueError("x/y reflection requires a polar grid divisible by four")
        indices = torch.remainder(
            n_theta // 4 - torch.arange(n_theta, device=grid.device) - 1,
            n_theta,
        )
        reflected_grid = grid.index_select(-2, indices)
        reflected_low = low5[:, (1, 0, 3, 2, 5, 4, 6)]
        reflected_context = torch.cat(
            (
                self._low_raw(reflected_low, hist.flip(-1)),
                self.hp_token(reflected_grid),
            ),
            dim=1,
        )
        return torch.cat((context, reflected_context), dim=1)

    @torch.no_grad()
    def sample_window(
        self,
        grid: torch.Tensor,
        low5: torch.Tensor,
        hist: torch.Tensor,
        n: int = 1,
        temp: float = 1.0,
        nfe: int = 12,
        churn: float = 0.0,
    ) -> torch.Tensor:
        context = self.ctx_from(grid, low5, hist)
        if context.shape[0] == 1:
            context = context[0]
        return self.sample(n, context, nfe=nfe, temp=temp, churn=churn)

    def phi_s_at(
        self,
        controls: torch.Tensor,
        grid: torch.Tensor,
        low5: torch.Tensor,
        hist: torch.Tensor,
        s: float = 0.9,
    ) -> torch.Tensor:
        context = self.ctx_from(grid, low5, hist)
        if context.shape[0] == 1:
            context = context[0]
        return self.phi_s(controls, context, s=s)

    def enable_boundary_adapter(self, hidden: int = 0) -> None:
        if self.boundary_adapter:
            return
        feature_dim = self.repr_dim if self.repr_dim is not None else self.width
        self.boundary_adapter_hidden = int(hidden)

        def make_adapter() -> nn.Module:
            if self.boundary_adapter_hidden > 0:
                module = nn.Sequential(
                    nn.Linear(feature_dim, self.boundary_adapter_hidden),
                    nn.SiLU(),
                    nn.Linear(self.boundary_adapter_hidden, self.d, bias=False),
                )
                nn.init.zeros_(module[-1].weight)
                return module
            module = nn.Linear(feature_dim, self.d, bias=False)
            nn.init.zeros_(module.weight)
            return module

        self.adapter_origin = make_adapter()
        self.adapter_goal = make_adapter()
        self.boundary_adapter = True

    def _boundary_gates(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Boundary adapters are used only for the canonical (5,5) expansion
        # task; recover position from its relative-goal low5 feature.
        x = 5.0 - context[:, 0] * GF.R_GOAL
        y = 5.0 - context[:, 1] * GF.R_GOAL
        xmax, ymax, xwidth, ywidth = self.boundary_origin_gate
        origin = ((ymax - y) / ywidth).clamp(0.0, 1.0) * ((xmax - x) / xwidth).clamp(0.0, 1.0)
        xmin, ymin, xwidth, ywidth = self.boundary_goal_gate
        goal_gate = ((y - ymin) / ywidth).clamp(0.0, 1.0) * ((x - xmin) / xwidth).clamp(0.0, 1.0)
        return origin, goal_gate

    def forward(self, x, tau, ctx, return_features: bool = False):
        if self.reflection_group_average:
            if ctx.ndim != 2 or ctx.shape[1] != 2 * self.ctx_dim:
                raise ValueError(
                    "group-averaged context must contain original and reflected branches"
                )
            original_context, reflected_context = ctx.split(self.ctx_dim, dim=1)
            reflected_x = x.reshape(len(x), self.T, 2).flip(-1).reshape_as(x)
            combined_features = self.features(
                torch.cat((x, reflected_x), dim=0),
                torch.cat((tau, tau), dim=0),
                torch.cat((original_context, reflected_context), dim=0),
            )
            original_features, reflected_features = combined_features.split(len(x))
            original_velocity = self.head(original_features)
            reflected_velocity = self.head(reflected_features).reshape(
                len(x), self.T, 2
            ).flip(-1).reshape_as(original_velocity)
            velocity = 0.5 * (original_velocity + reflected_velocity)
            features = 0.5 * (original_features + reflected_features)
            return (velocity, features) if return_features else velocity
        features = self.features(x, tau, ctx)
        velocity = self.head(features)
        if self.boundary_adapter:
            origin_gate, goal_gate = self._boundary_gates(ctx)
            velocity = (
                velocity
                + origin_gate[:, None] * self.adapter_origin(features)
                + goal_gate[:, None] * self.adapter_goal(features)
            )
        return (velocity, features) if return_features else velocity

    @torch.no_grad()
    def phi_s(self, controls: torch.Tensor, ctx: torch.Tensor, s: float = 0.9) -> torch.Tensor:
        if not self.reflection_group_average:
            return super().phi_s(controls, ctx, s=s)
        batch = controls.shape[0]
        x1 = (controls / self.u_max).reshape(batch, self.d)
        ctx = self._expand_ctx(ctx, batch)
        if len(self.noise_templates) % 2:
            raise RuntimeError("group-averaged feature templates must have even size")
        base_templates = self.noise_templates[: len(self.noise_templates) // 2]
        templates = torch.cat(
            (
                base_templates,
                base_templates.reshape(-1, self.T, 2).flip(-1).reshape(
                    -1, self.d
                ),
            ),
            dim=0,
        )
        features = []
        for template in templates:
            x_s = (1.0 - s) * template[None] + s * x1
            tau = torch.full((batch,), s, device=x1.device, dtype=x1.dtype)
            features.append(self.forward(x_s, tau, ctx, return_features=True)[1])
        return torch.stack(features, dim=0).mean(dim=0)

    def config(self) -> dict:
        return {
            "arch": "hp-repr" if self.repr_dim is not None else "hp-reduced-32",
            "schema_version": (
                "w8sg-hp-v4-low7-closest-boundary-tie-mean"
                if self.conditioning_schema == "low7_closest_boundary_tie_mean"
                else "w8sg-hp-v3-low7-closest-boundary"
                if self.conditioning_schema == "low7_closest_boundary"
                else "w8sg-hp-v2-low5-only"
            ),
            "raw_start_goal": False,
            "H_pred": self.H_pred,
            "grid_shape": (1, self.grid_hw[0], self.grid_hw[1]),
            "K_hist": self.K_hist,
            "width": self.width,
            "depth": self.depth,
            "u_max": self.u_max,
            "ctx_dim": self.ctx_dim,
            "raw_condition_dim": self.raw_condition_dim,
            "conditioning_schema": self.conditioning_schema,
            "use_gru": self.use_gru,
            "repr_dim": self.repr_dim,
            "grid_hw": list(self.grid_hw),
            "trunk_hidden": list(self.trunk_hidden),
            "enc_depth": self.enc_depth,
            "boundary_adapter": bool(self.boundary_adapter),
            "boundary_adapter_hidden": self.boundary_adapter_hidden,
            "boundary_origin_gate": list(self.boundary_origin_gate),
            "boundary_goal_gate": list(self.boundary_goal_gate),
            "reflection_group_average": self.reflection_group_average,
        }


def save_hp(policy: GridHPFlowPolicy, path: str | Path, extra: dict | None = None) -> None:
    payload = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_hp(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("raw_start_goal", False):
        raise ValueError(f"{path} still contains rejected raw start/goal inputs")
    policy = GridHPFlowPolicy(
        width=config["width"],
        depth=config.get("depth", 2),
        u_max=config["u_max"],
        use_gru=config.get("use_gru", False),
        repr_dim=config.get("repr_dim"),
        grid_hw=tuple(config.get("grid_hw", (32, 32))),
        trunk_hidden=tuple(config.get("trunk_hidden", (128, 64))),
        enc_depth=config.get("enc_depth", 2),
        raw_condition_dim=config.get("raw_condition_dim", 5),
        conditioning_schema=config.get("conditioning_schema", "low5"),
        boundary_adapter=config.get("boundary_adapter", False),
        boundary_adapter_hidden=config.get("boundary_adapter_hidden", 0),
        boundary_origin_gate=tuple(config.get("boundary_origin_gate", (1.25, 0.65, 0.50, 0.47))),
        boundary_goal_gate=tuple(config.get("boundary_goal_gate", (3.95, 4.05, 0.55, 0.55))),
        reflection_group_average=config.get("reflection_group_average", False),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    return policy.to(device).eval(), checkpoint


if __name__ == "__main__":
    batch = 8
    model = GridHPFlowPolicy()
    grid = torch.rand(batch, 3, 32, 32)
    low5 = torch.randn(batch, 5)
    hist = torch.randn(batch, GF.K_HIST, 2)
    controls = torch.randn(batch, GF.H_PRED, 2).clamp(-1, 1)
    context = model.ctx_from(grid, low5, hist)
    loss = model.cfm_loss(controls, context)
    loss.backward()
    print(
        f"ctx={tuple(context.shape)} loss={float(loss.detach()):.4f} "
        f"params={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
