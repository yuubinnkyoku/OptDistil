"""Validation-tuned static tensor-role learning-rate optimizers."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.stochastic import MultiTensorCase, batch_sequence, summarize_ratios
from optdistil.multitensor.teachers import MultiTensorTeacher

DEFAULT_ROLE_CANDIDATES = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)


class StaticRoleNormGrad(MultiTensorTeacher):
    """u_l = -lr_role(l) * g_l / ||g_l|| with fixed per-role or per-tensor scales."""

    def __init__(
        self,
        *,
        scales: Sequence[float],
        eps: float = 1e-8,
    ) -> None:
        if not scales:
            raise ValueError("scales must be non-empty")
        values = [float(s) for s in scales]
        if any((not math.isfinite(s)) or s <= 0.0 for s in values):
            raise ValueError("scales must be positive and finite")
        self.scales = values
        self.eps = eps

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        if len(grads) != len(self.scales):
            raise ValueError("grad count must match scale count")
        updates: list[Tensor] = []
        for scale, grad in zip(self.scales, grads, strict=True):
            norm = grad.reshape(-1).float().norm().clamp_min(self.eps)
            updates.append((-scale * grad / norm.to(grad.dtype)).to(grad.dtype))
        return updates


def roles_for_case(case: MultiTensorCase) -> tuple[str, ...]:
    roles = getattr(case.task, "parameter_roles", None)
    if roles is None:
        return tuple(f"t{i}" for i in range(len(case.initial)))
    return tuple(roles)


def unique_roles(roles: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for role in roles:
        if role not in seen:
            seen.append(role)
    return seen


def expand_role_lrs(role_lrs: dict[str, float], roles: Sequence[str]) -> list[float]:
    values: list[float] = []
    for role in roles:
        if role not in role_lrs:
            raise KeyError(f"missing learning rate for role {role!r}")
        values.append(float(role_lrs[role]))
    return values


def expand_tensor_lrs(
    tensor_lrs: Sequence[float],
    *,
    architecture: str,
    default: dict[str, float] | None = None,
    roles: Sequence[str] | None = None,
) -> list[float]:
    if roles is None:
        # Two-layer has 4 tensors, residual has 5; roles are matrix/vector alternating.
        roles = ("matrix", "vector", "matrix", "vector") if architecture != "residual" else (
            "matrix",
            "vector",
            "matrix",
            "vector",
            "matrix",
        )
    if len(tensor_lrs) != len(roles):
        raise ValueError("tensor_lrs length must match roles")
    if any((not math.isfinite(lr)) or lr <= 0 for lr in tensor_lrs):
        raise ValueError("tensor lrs must be positive and finite")
    return [float(lr) for lr in tensor_lrs]


@torch.no_grad()
def rollout_static_role(
    teacher: StaticRoleNormGrad,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> float:
    params = case.initial.clone()
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    for indices in batches:
        grads = case.task.grad_on_samples(params, indices)
        updates = teacher.step(params, grads)
        params = params.add(updates)
        if not params.is_finite():
            losses.append(math.inf)
            return summarize_ratios([losses[-1] / max(abs(initial_loss), 1e-12)]).mean
        losses.append(float(case.task.loss(params)))
    final = losses[-1]
    return final / max(abs(initial_loss), 1e-12)


def tune_shared_role_lrs(
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    candidates: Sequence[float] = DEFAULT_ROLE_CANDIDATES,
) -> tuple[dict[str, float], float]:
    """Tune one LR per role on validation cases only."""
    if not validation_cases:
        raise ValueError("validation_cases must be non-empty")
    roles = unique_roles(roles_for_case(validation_cases[0]))
    scored: list[tuple[float, dict[str, float]]] = []
    for combo in itertools.product(candidates, repeat=len(roles)):
        role_lrs = {role: float(lr) for role, lr in zip(roles, combo, strict=True)}
        scores: list[float] = []
        for case in validation_cases:
            case_roles = roles_for_case(case)
            teacher = StaticRoleNormGrad(scales=expand_role_lrs(role_lrs, case_roles))
            scores.append(rollout_static_role(teacher, case, batch_size=batch_size, steps=steps))
        scored.append((statistics.fmean(scores), role_lrs))
    score, role_lrs = min(scored, key=lambda item: item[0])
    return role_lrs, score


def tune_architecture_tensor_lrs(
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    candidates: Sequence[float] = DEFAULT_ROLE_CANDIDATES,
    init_scales: Sequence[float] | None = None,
    rounds: int = 2,
) -> tuple[list[float], float]:
    """Coordinate-descent tune of one LR per tensor on validation cases.

    ``validation_cases`` must share one architecture. ``init_scales`` seeds the
    search (typically the shared-role solution expanded to tensors).
    """
    if not validation_cases:
        raise ValueError("validation_cases must be non-empty")
    architectures = {case.architecture for case in validation_cases}
    if len(architectures) != 1:
        raise ValueError("architecture-specific tuning requires a single architecture")
    roles = roles_for_case(validation_cases[0])
    if init_scales is None:
        current = [float(candidates[len(candidates) // 2])] * len(roles)
    else:
        if len(init_scales) != len(roles):
            raise ValueError("init_scales length must match tensor count")
        current = [float(s) for s in init_scales]

    def score_scales(scales: Sequence[float]) -> float:
        teacher = StaticRoleNormGrad(scales=scales)
        return statistics.fmean(
            rollout_static_role(teacher, case, batch_size=batch_size, steps=steps)
            for case in validation_cases
        )

    best_score = score_scales(current)
    for _ in range(max(1, rounds)):
        improved = False
        for index in range(len(current)):
            local_best = current[index]
            local_best_score = best_score
            for candidate in candidates:
                trial = list(current)
                trial[index] = float(candidate)
                score = score_scales(trial)
                if score < local_best_score - 1e-12:
                    local_best = float(candidate)
                    local_best_score = score
            if local_best != current[index]:
                current[index] = local_best
                best_score = local_best_score
                improved = True
        if not improved:
            break
    return current, best_score


def evaluate_static_role_split(
    teacher: StaticRoleNormGrad,
    split: dict[float, list[MultiTensorCase]],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    for cases in split.values():
        for case in cases:
            case_roles = roles_for_case(case)
            # Rebuild scale vector from the teacher's role mapping when lengths match.
            if len(teacher.scales) == len(case_roles):
                scales = teacher.scales
            else:
                raise ValueError("static-role teacher scale count does not match case")
            local = StaticRoleNormGrad(scales=scales, eps=teacher.eps)
            params = case.initial.clone()
            batches = batch_sequence(case, batch_size=batch_size, steps=steps)
            initial_loss = float(case.task.loss(params))
            losses = [initial_loss]
            for indices in batches:
                grads = case.task.grad_on_samples(params, indices)
                updates = local.step(params, grads)
                params = params.add(updates)
                if not params.is_finite():
                    losses.append(math.inf)
                    break
                losses.append(float(case.task.loss(params)))
            final = losses[-1]
            ratios.append(final / max(abs(initial_loss), 1e-12))
            if len(losses) > 1 and math.isfinite(losses[0]) and all(math.isfinite(x) for x in losses):
                area = 0.0
                for left, right in itertools.pairwise(losses):
                    area += 0.5 * (left + right)
                aulcs.append(area / ((len(losses) - 1) * max(abs(initial_loss), 1e-12)))
            else:
                aulcs.append(math.inf)
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
    }
