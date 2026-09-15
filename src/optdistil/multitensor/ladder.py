"""Minimum-computation ladder helpers: role-scaled SGD/AdamW/NormGrad tuning."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

import torch

from optdistil.multitensor.roles import random_balanced_binary_partition, swap_binary_role_labels
from optdistil.multitensor.static_role import (
    DEFAULT_ROLE_CANDIDATES,
    StaticRoleNormGrad,
    expand_role_lrs,
    roles_for_case,
    unique_roles,
)
from optdistil.multitensor.stochastic import (
    MultiTensorCase,
    batch_sequence,
    summarize_ratios,
)
from optdistil.multitensor.teachers import (
    MultiTensorTeacher,
    RoleScaledAdamW,
    RoleScaledSGD,
)

LadderBuilder = Callable[[list[float]], MultiTensorTeacher]


def _rollout_ratio(
    teacher: MultiTensorTeacher,
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
            return math.inf
        losses.append(float(case.task.loss(params)))
    return losses[-1] / max(abs(initial_loss), 1e-12)


def tune_role_grid(
    builder: LadderBuilder,
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    candidates: Sequence[float] = DEFAULT_ROLE_CANDIDATES,
    assigned_roles: Sequence[str] | None = None,
) -> tuple[dict[str, float], float]:
    """Grid-search one LR per unique assigned role on validation cases only."""
    if not validation_cases:
        raise ValueError("validation_cases must be non-empty")
    if assigned_roles is None:
        base = roles_for_case(validation_cases[0])
    else:
        base = list(assigned_roles)
    roles = unique_roles(base)
    scored: list[tuple[float, dict[str, float]]] = []
    for combo in itertools.product(candidates, repeat=len(roles)):
        role_lrs = {role: float(lr) for role, lr in zip(roles, combo, strict=True)}
        scores: list[float] = []
        for case in validation_cases:
            case_roles = (
                list(assigned_roles)
                if assigned_roles is not None
                else roles_for_case(case)
            )
            scales = expand_role_lrs(role_lrs, case_roles)
            teacher = builder(scales)
            scores.append(_rollout_ratio(teacher, case, batch_size=batch_size, steps=steps))
        scored.append((statistics.fmean(scores), role_lrs))
    score, role_lrs = min(scored, key=lambda item: item[0])
    return role_lrs, score


def build_normgrad(scales: Sequence[float]) -> MultiTensorTeacher:
    return StaticRoleNormGrad(scales=scales)


def build_sgd(scales: Sequence[float]) -> MultiTensorTeacher:
    return RoleScaledSGD(scales=scales)


def build_adamw(scales: Sequence[float]) -> MultiTensorTeacher:
    return RoleScaledAdamW(scales=scales)


def evaluate_ladder_method(
    teacher: MultiTensorTeacher,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratio = _rollout_ratio(teacher, case, batch_size=batch_size, steps=steps)
    return {"loss_ratio": ratio, "finite": math.isfinite(ratio)}


def evaluate_ladder_split(
    teacher_for_case: Callable[[MultiTensorCase], MultiTensorTeacher],
    cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratios: list[float] = []
    for case in cases:
        teacher = teacher_for_case(case)
        ratios.append(_rollout_ratio(teacher, case, batch_size=batch_size, steps=steps))
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
        "n": len(ratios),
    }


def fixed_role_teacher(
    builder: LadderBuilder,
    role_lrs: dict[str, float],
    case: MultiTensorCase,
    *,
    assigned_roles: Sequence[str] | None = None,
) -> MultiTensorTeacher:
    case_roles = (
        list(assigned_roles) if assigned_roles is not None else roles_for_case(case)
    )
    return builder(expand_role_lrs(role_lrs, case_roles))


def frozen_ratio_teacher(
    builder: LadderBuilder,
    base_role_lrs: dict[str, float],
    case: MultiTensorCase,
    *,
    global_scale: float,
    assigned_roles: Sequence[str] | None = None,
) -> MultiTensorTeacher:
    """Keep relative role ratios frozen; scale every LR by ``global_scale``."""
    if global_scale <= 0 or not math.isfinite(global_scale):
        raise ValueError("global_scale must be positive and finite")
    scaled = {role: float(lr) * global_scale for role, lr in base_role_lrs.items()}
    return fixed_role_teacher(builder, scaled, case, assigned_roles=assigned_roles)


def tune_global_scale(
    builder: LadderBuilder,
    base_role_lrs: dict[str, float],
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    candidates: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
) -> tuple[float, float]:
    scored: list[tuple[float, float]] = []
    for scale in candidates:
        scores = [
            _rollout_ratio(
                frozen_ratio_teacher(builder, base_role_lrs, case, global_scale=float(scale)),
                case,
                batch_size=batch_size,
                steps=steps,
            )
            for case in validation_cases
        ]
        scored.append((statistics.fmean(scores), float(scale)))
    score, scale = min(scored, key=lambda item: item[0])
    return scale, score


def permutation_specs(
    n_tensors: int,
    *,
    n_random: int = 3,
    seed_base: int = 910000,
) -> list[dict[str, Any]]:
    """True / swapped / random balanced partitions for falsification."""
    true_roles = ["matrix" if i % 2 == 0 else "vector" for i in range(n_tensors)]
    specs: list[dict[str, Any]] = [
        {"name": "true_matrix_vector", "roles": true_roles, "kind": "true"},
        {"name": "swapped_labels", "roles": swap_binary_role_labels(true_roles), "kind": "swap"},
    ]
    for index in range(n_random):
        specs.append(
            {
                "name": f"random_partition_{index}",
                "roles": random_balanced_binary_partition(n_tensors, seed=seed_base + index),
                "kind": "random",
                "seed": seed_base + index,
            }
        )
    return specs


def paired_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired sequences must be non-empty and equal length")
    diffs = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    wins = sum(1 for d in diffs if d < 0)
    return {
        "mean_diff": statistics.fmean(diffs),
        "median_diff": statistics.median(diffs),
        "wins_left_better": wins,
        "n": len(diffs),
        "diffs": diffs,
    }


@torch.no_grad()
def collect_ratios_for_method(
    teacher_for_case: Callable[[MultiTensorCase], MultiTensorTeacher],
    cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
) -> list[float]:
    return [
        _rollout_ratio(teacher_for_case(case), case, batch_size=batch_size, steps=steps)
        for case in cases
    ]


def make_case_list_from_split(split: dict[float, list[MultiTensorCase]]) -> list[MultiTensorCase]:
    return [case for cases in split.values() for case in cases]


def clone_case_with_roles(
    case: MultiTensorCase,
    *,
    assigned_roles: Sequence[str],
) -> tuple[MultiTensorCase, list[str]]:
    """Return a lightweight role-override view without changing the task object."""
    return case, list(assigned_roles)
