"""Structured tiny controllers: 20-50 trainable parameters.

Candidate 1 (global):   u_l = -a_l * c_t * normalized_gradient_l
Candidate 2 (per-tensor): u_l = -a_l * c_{t,l} * normalized_gradient_l

``a_l`` are validation-tuned fixed role coefficients. ``c`` is a tiny learned
scalar controller over cheap tensor statistics.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.static_role import (
    StaticRoleNormGrad,
    expand_role_lrs,
    roles_for_case,
)
from optdistil.multitensor.stochastic import (
    MultiTensorCase,
    _observe_ema,
    batch_sequence,
    summarize_ratios,
)
from optdistil.multitensor.teachers import MultiTensorTeacher

GLOBAL_STAT_DIM = 4
PER_TENSOR_STAT_DIM = 6


class GlobalScalarController(nn.Module):
    """c_t = tiny MLP over global cheap statistics. Params: ~27 + role coeffs."""

    def __init__(self, *, hidden: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_STAT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, stats: Tensor) -> Tensor:
        if stats.ndim != 1 or stats.numel() != GLOBAL_STAT_DIM:
            raise ValueError("stats must be a 1-D vector of length GLOBAL_STAT_DIM")
        return self.net(stats).squeeze(-1)


class PerTensorController(nn.Module):
    """c_{t,l} = shared tiny MLP over per-tensor cheap statistics."""

    def __init__(self, *, hidden: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PER_TENSOR_STAT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, stats: Tensor) -> Tensor:
        if stats.ndim != 1 or stats.numel() != PER_TENSOR_STAT_DIM:
            raise ValueError("stats must be a 1-D vector of length PER_TENSOR_STAT_DIM")
        return self.net(stats).squeeze(-1)


def global_cheap_stats(
    grads: Sequence[Tensor],
    momentums: Sequence[Tensor],
    second_moments: Sequence[Tensor],
    params: Sequence[Tensor],
    *,
    step: int,
    total_steps: int,
    eps: float = 1e-8,
) -> Tensor:
    total_grad_sq = torch.zeros((), dtype=torch.float32)
    total_param_sq = torch.zeros((), dtype=torch.float32)
    total_mom_sq = torch.zeros((), dtype=torch.float32)
    total_v = torch.zeros((), dtype=torch.float32)
    count = 0
    for grad, mom, v, param in zip(grads, momentums, second_moments, params, strict=True):
        total_grad_sq = total_grad_sq + grad.reshape(-1).float().square().sum()
        total_param_sq = total_param_sq + param.reshape(-1).float().square().sum()
        total_mom_sq = total_mom_sq + mom.reshape(-1).float().square().sum()
        total_v = total_v + v.reshape(-1).float().clamp_min(0).sum()
        count += grad.numel()
    count = max(count, 1)
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    stats = torch.stack(
        (
            (total_grad_sq / count).sqrt().clamp_min(eps),
            (total_param_sq / count).sqrt().clamp_min(eps),
            (total_mom_sq / count).sqrt().clamp_min(eps),
            torch.tensor(progress, dtype=torch.float32),
        )
    )
    _ = total_v
    return stats


def per_tensor_cheap_stats(
    grad: Tensor,
    momentum: Tensor,
    second_moment: Tensor,
    parameter: Tensor,
    *,
    global_grad_rms: Tensor,
    progress: float,
    eps: float = 1e-8,
) -> Tensor:
    return torch.stack(
        (
            grad.square().mean().add(eps).sqrt().float(),
            parameter.square().mean().add(eps).sqrt().float(),
            momentum.square().mean().add(eps).sqrt().float(),
            second_moment.clamp_min(0).mean().add(eps).sqrt().float(),
            global_grad_rms.float(),
            torch.tensor(progress, dtype=torch.float32),
        )
    )


@dataclass
class StructuredTinyOptimizer(MultiTensorTeacher):
    """u_l = -a_l * c_l(stats) * g_l / ||g_l|| with a tiny shared controller."""

    role_lrs: dict[str, float]
    controller: nn.Module
    mode: str  # "global" | "per_tensor"
    eps: float = 1e-8
    output_bias: float = 0.0

    def parameter_count(self) -> int:
        controller_params = sum(p.numel() for p in self.controller.parameters())
        return controller_params + len(self.role_lrs)

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        raise NotImplementedError("use step_with_stats")

    @torch.no_grad()
    def step_with_stats(
        self,
        params: ParamCollection,
        grads: Sequence[Tensor],
        momentums: Sequence[Tensor],
        second_moments: Sequence[Tensor],
        *,
        step: int,
        total_steps: int,
        roles: Sequence[str] | None = None,
    ) -> list[Tensor]:
        roles = list(roles) if roles is not None else [
            f"t{i}" for i in range(len(params))
        ]
        if len(roles) != len(params):
            raise ValueError("roles length must match parameters")
        self.controller.eval()
        progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
        gstats = global_cheap_stats(
            grads,
            momentums,
            second_moments,
            params,
            step=step,
            total_steps=total_steps,
            eps=self.eps,
        )
        updates: list[Tensor] = []
        if self.mode == "global":
            c = float(self.controller(gstats).clamp(-2.0, 2.0)) + self.output_bias
            for role, grad in zip(roles, grads, strict=True):
                a = float(self.role_lrs[role])
                norm = grad.reshape(-1).float().norm().clamp_min(self.eps)
                updates.append((-a * c * grad / norm.to(grad.dtype)).to(grad.dtype))
            return updates

        if self.mode != "per_tensor":
            raise ValueError(f"unknown structured mode: {self.mode}")
        global_grad_rms = gstats[0]
        for role, grad, mom, v, param in zip(
            roles, grads, momentums, second_moments, params, strict=True
        ):
            stats = per_tensor_cheap_stats(
                grad,
                mom,
                v,
                param,
                global_grad_rms=global_grad_rms,
                progress=progress,
                eps=self.eps,
            )
            c = float(self.controller(stats).clamp(-2.0, 2.0)) + self.output_bias
            a = float(self.role_lrs[role])
            norm = grad.reshape(-1).float().norm().clamp_min(self.eps)
            updates.append((-a * c * grad / norm.to(grad.dtype)).to(grad.dtype))
        return updates


def collect_structured_features(
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
    mode: str,
) -> list[dict[str, Tensor]]:
    """Collect controller inputs and NormGrad directions for supervised fitting."""
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    samples: list[dict[str, Tensor]] = []
    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)
        progress = min(max(step / max(steps, 1), 0.0), 1.0)
        gstats = global_cheap_stats(
            grads, momentums, second_moments, params, step=step, total_steps=steps
        )
        directions: list[Tensor] = []
        for grad in grads:
            g = grad.reshape(-1).float()
            directions.append((-g / g.norm().clamp_min(1e-8)).reshape(grad.shape))
        entry: dict[str, Tensor] = {
            "global_stats": gstats.detach(),
            "directions": [d.detach() for d in directions],
            "grads": [g.detach() for g in grads],
            "roles": tuple(roles_for_case(case)),
        }
        if mode == "per_tensor":
            entry["tensor_stats"] = [
                per_tensor_cheap_stats(
                    grad,
                    mom,
                    v,
                    param,
                    global_grad_rms=gstats[0],
                    progress=progress,
                ).detach()
                for grad, mom, v, param in zip(
                    grads, momentums, second_moments, params, strict=True
                )
            ]
        samples.append(entry)
        # Advance with a modest local-NormGrad step so statistics stay on a real path.
        step_lr = 0.05
        params = params.add([step_lr * direction for direction in directions])
        if not params.is_finite():
            break
    return samples


def fit_structured_from_privileged(
    cases: Sequence[MultiTensorCase],
    privileged_scales_per_case: Sequence[Sequence[float]],
    *,
    batch_size: int,
    steps: int,
    mode: str,
    role_lrs: dict[str, float],
    seed: int,
    epochs: int = 40,
    lr: float = 3e-3,
) -> StructuredTinyOptimizer:
    """Fit controller magnitudes to privileged NormGrad step scales.

    Privileged theta-update magnitude along the local NormGrad direction is
    ``lr_priv / s_l``. We match ``a_l * c`` to that magnitude using the
    controller's predicted ``c``.
    """
    if len(cases) != len(privileged_scales_per_case):
        raise ValueError("cases and scales length mismatch")
    torch.manual_seed(seed)
    if mode == "global":
        controller: nn.Module = GlobalScalarController()
    elif mode == "per_tensor":
        controller = PerTensorController()
    else:
        raise ValueError(f"unknown mode: {mode}")

    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    controller.train()
    history: list[float] = []

    # Target: c such that a_role * c = target_scale_l (privileged magnitude).
    # We regress c onto target_scale_l / a_role.
    targets: list[tuple[dict[str, Tensor], list[float]]] = []
    for case, scales in zip(cases, privileged_scales_per_case, strict=True):
        roles = roles_for_case(case)
        samples = collect_structured_features(case, batch_size=batch_size, steps=steps, mode=mode)
        for sample in samples:
            # Privileged magnitude along unit NormGrad direction is 1/s_l
            # (teacher lr absorbed into role coeffs a_l after tuning).
            desired = [1.0 / float(s) for s in scales]
            if mode == "global":
                # Shared c approximates the mean of desired/a.
                ratios = [
                    desired[i] / float(role_lrs[roles[i]]) for i in range(len(desired))
                ]
                target_c = sum(ratios) / len(ratios)
                targets.append((sample, [target_c]))
            else:
                target_cs = [
                    desired[i] / max(float(role_lrs[roles[i]]), 1e-12)
                    for i in range(len(desired))
                ]
                targets.append((sample, target_cs))

    for _ in range(epochs):
        total = 0.0
        for sample, target_cs in targets:
            optimizer.zero_grad(set_to_none=True)
            if mode == "global":
                pred = controller(sample["global_stats"])
                loss = (pred - torch.tensor(target_cs[0])).square()
            else:
                preds = torch.stack(
                    [controller(stats) for stats in sample["tensor_stats"]]  # type: ignore[index]
                )
                loss = (preds - torch.tensor(target_cs)).square().mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append(total / max(len(targets), 1))

    controller.eval()
    structured = StructuredTinyOptimizer(
        role_lrs=dict(role_lrs),
        controller=controller,
        mode=mode,
        output_bias=0.0,
    )
    _ = history
    return structured


@torch.no_grad()
def rollout_structured(
    optimizer: StructuredTinyOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> tuple[float, float, bool]:
    """Return (loss_ratio, aulc, finite)."""
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    roles = roles_for_case(case)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)
        updates = optimizer.step_with_stats(
            params,
            grads,
            momentums,
            second_moments,
            step=step,
            total_steps=steps,
            roles=roles,
        )
        params = params.add(updates)
        if not params.is_finite():
            losses.append(math.inf)
            ratio = math.inf
            return ratio, math.inf, False
        losses.append(float(case.task.loss(params)))
    final = losses[-1]
    ratio = final / max(abs(initial_loss), 1e-12)
    area = 0.0
    for left, right in itertools.pairwise(losses):
        area += 0.5 * (left + right)
    aulc = area / ((len(losses) - 1) * max(abs(initial_loss), 1e-12))
    return ratio, aulc, True


def evaluate_structured_split(
    optimizer: StructuredTinyOptimizer,
    split: dict[float, list[MultiTensorCase]],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    for cases in split.values():
        for case in cases:
            ratio, aulc, finite = rollout_structured(
                optimizer, case, batch_size=batch_size, steps=steps
            )
            ratios.append(ratio)
            aulcs.append(aulc if finite else math.inf)
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
        "parameter_count": optimizer.parameter_count(),
        "mode": optimizer.mode,
    }


def expand_role_lrs_for_optimizer(role_lrs: dict[str, float], roles: Sequence[str]) -> dict[str, float]:
    return expand_role_lrs(role_lrs, roles)  # type: ignore[return-value]


def static_role_teacher_from_lrs(
    role_lrs: dict[str, float],
    case: MultiTensorCase,
) -> StaticRoleNormGrad:
    return StaticRoleNormGrad(scales=expand_role_lrs(role_lrs, roles_for_case(case)))
