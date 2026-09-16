"""Group-count frontier, property partitions, and per-tensor oracle helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import torch

from optdistil.multitensor.ladder import LadderBuilder, _rollout_ratio, build_normgrad
from optdistil.multitensor.static_role import (
    DEFAULT_ROLE_CANDIDATES,
    roles_for_case,
)
from optdistil.multitensor.stochastic import MultiTensorCase


def case_roles(case: MultiTensorCase) -> list[str]:
    return list(roles_for_case(case))


def numel_partition(roles_or_shapes: Sequence[Any], *, n_groups: int = 2) -> list[str]:
    """Group tensors by numel quantile (group id = rank bucket)."""
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    numels: list[int] = []
    for item in roles_or_shapes:
        if isinstance(item, torch.Size):
            numels.append(int(torch.Size(item).numel()))
        elif hasattr(item, "numel"):
            numels.append(int(item.numel()))
        else:
            raise TypeError("expected shapes or tensors")
    order = sorted(range(len(numels)), key=lambda i: numels[i])
    labels = [f"g{i}" for i in range(len(numels))]
    n = len(numels)
    for rank, index in enumerate(order):
        bucket = min(n_groups - 1, (rank * n_groups) // max(n, 1))
        labels[index] = f"g{bucket}"
    return labels


def ndim_partition(shapes: Sequence[torch.Size]) -> list[str]:
    return [f"ndim{len(shape)}" for shape in shapes]


def property_partition_from_case(
    case: MultiTensorCase,
    *,
    mode: str,
) -> list[str]:
    n = len(case.initial)
    if mode == "true_roles":
        return case_roles(case)
    if mode == "ndim":
        return ndim_partition(case.initial.shapes())
    if mode == "numel2":
        return numel_partition(case.initial.shapes(), n_groups=2)
    if mode == "numel3":
        return numel_partition(case.initial.shapes(), n_groups=3)
    if mode == "index_mod2":
        return [f"g{i % 2}" for i in range(n)]
    if mode == "index_mod3":
        return [f"g{i % 3}" for i in range(n)]
    if mode == "single":
        return ["g0"] * n
    if mode == "per_tensor":
        return [f"t{i}" for i in range(n)]
    if mode == "first_half":
        mid = (n + 1) // 2
        return ["g0"] * mid + ["g1"] * (n - mid)
    raise ValueError(f"unknown partition mode: {mode}")


def expand_assigned(
    assigned: Sequence[str],
    role_lrs: dict[str, float],
) -> list[float]:
    return [float(role_lrs[role]) for role in assigned]


def tune_groups_cd(
    builder: LadderBuilder,
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    assigned_by_arch: dict[str, list[str]] | None = None,
    assigned_roles: Sequence[str] | None = None,
    candidates: Sequence[float] = DEFAULT_ROLE_CANDIDATES,
    rounds: int = 3,
    init_scales: Sequence[float] | None = None,
) -> tuple[list[str], dict[str, float], float]:
    """Coordinate-descent tune of one LR per group label on validation cases.

    Assignment is either fixed ``assigned_roles`` or taken per-architecture from
    ``assigned_by_arch`` / the task's parameter_roles.
    """
    if not validation_cases:
        raise ValueError("validation_cases must be non-empty")

    def assignment_for(case: MultiTensorCase) -> list[str]:
        if assigned_roles is not None:
            return list(assigned_roles)
        if assigned_by_arch is not None:
            return list(assigned_by_arch[case.architecture])
        return case_roles(case)

    first = assignment_for(validation_cases[0])
    groups: list[str] = []
    for case in validation_cases:
        for role in assignment_for(case):
            if role not in groups:
                groups.append(role)
    if init_scales is None:
        current = {g: float(candidates[len(candidates) // 2]) for g in groups}
    else:
        if len(init_scales) != len(groups):
            raise ValueError("init_scales length must match group count")
        current = {g: float(s) for g, s in zip(groups, init_scales, strict=True)}

    def score(role_lrs: dict[str, float]) -> float:
        scores = []
        for case in validation_cases:
            assigned = assignment_for(case)
            scales = expand_assigned(assigned, role_lrs)
            teacher = builder(scales)
            scores.append(_rollout_ratio(teacher, case, batch_size=batch_size, steps=steps))
        return statistics.fmean(scores)

    best_score = score(current)
    for _ in range(max(1, rounds)):
        improved = False
        for group in groups:
            local_best = current[group]
            local_best_score = best_score
            for candidate in candidates:
                trial = dict(current)
                trial[group] = float(candidate)
                value = score(trial)
                if value < local_best_score - 1e-12:
                    local_best = float(candidate)
                    local_best_score = value
            if local_best != current[group]:
                current[group] = local_best
                best_score = local_best_score
                improved = True
        if not improved:
            break
    return first, current, best_score  # current covers union of groups seen


def evaluate_assigned_lrs(
    builder: LadderBuilder,
    role_lrs: dict[str, float],
    test_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    assigned_by_arch: dict[str, list[str]] | None = None,
    assigned_roles: Sequence[str] | None = None,
) -> list[float]:
    ratios: list[float] = []
    for case in test_cases:
        if assigned_roles is not None:
            assigned = list(assigned_roles)
        elif assigned_by_arch is not None:
            assigned = list(assigned_by_arch[case.architecture])
        else:
            assigned = case_roles(case)
        teacher = builder(expand_assigned(assigned, role_lrs))
        ratios.append(_rollout_ratio(teacher, case, batch_size=batch_size, steps=steps))
    return ratios


def fan_in(shape: torch.Size) -> int:
    if len(shape) <= 1:
        return max(int(shape[0]) if shape else 1, 1)
    return int(shape[1])


def formula_lrs_for_case(
    case: MultiTensorCase,
    *,
    mode: str,
    base: float = 0.1,
) -> list[float]:
    """Property-based absolute LRs with no validation search."""
    shapes = case.initial.shapes()
    if mode == "uniform":
        return [float(base)] * len(shapes)
    if mode == "inv_sqrt_numel":
        return [float(base) / math.sqrt(max(numel, 1.0)) for numel in case.initial.numels()]
    if mode == "inv_numel":
        return [float(base) / float(max(numel, 1)) for numel in case.initial.numels()]
    if mode == "inv_sqrt_fan_in":
        return [float(base) / math.sqrt(fan_in(shape)) for shape in shapes]
    if mode == "ndim_scaled":
        # matrices (ndim=2) get base, vectors get base/3 — fixed formula, no search
        return [float(base) if len(shape) >= 2 else float(base) / 3.0 for shape in shapes]
    if mode == "numel_rank":
        # monotone map from numel rank to LR in [base/4, base]
        numels = case.initial.numels()
        order = sorted(range(len(numels)), key=lambda i: numels[i])
        ranks = [0.0] * len(numels)
        n = len(numels)
        for rank, index in enumerate(order):
            ranks[index] = rank / max(n - 1, 1)
        return [float(base) * (0.25 + 0.75 * r) for r in ranks]
    if mode == "lars_init_norm":
        # Frozen LARS-like: c_l ∝ ||θ_l|| at initialization (theory prior).
        norms = [float(t.reshape(-1).float().norm()) for t in case.initial]
        scale = max(sum(norms) / max(len(norms), 1), 1e-8)
        return [float(base) * (n / scale) for n in norms]
    if mode == "inv_sqrt_numel_normalized":
        raw = [1.0 / math.sqrt(max(numel, 1.0)) for numel in case.initial.numels()]
        mean_raw = sum(raw) / max(len(raw), 1)
        return [float(base) * (r / mean_raw) for r in raw]
    raise ValueError(f"unknown formula mode: {mode}")


def per_tensor_oracle(
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    candidates: Sequence[float] = DEFAULT_ROLE_CANDIDATES,
    rounds: int = 3,
) -> tuple[dict[str, list[str]], dict[str, float], float]:
    """One free LR per tensor index; assignment is per-architecture length."""
    assigned_by_arch: dict[str, list[str]] = {}
    for case in validation_cases:
        if case.architecture in assigned_by_arch:
            continue
        n = len(case.initial)
        assigned_by_arch[case.architecture] = [f"t{i}" for i in range(n)]
    _roles, role_lrs, score = tune_groups_cd(
        build_normgrad,
        validation_cases,
        batch_size=batch_size,
        steps=steps,
        assigned_by_arch=assigned_by_arch,
        candidates=candidates,
        rounds=rounds,
    )
    return assigned_by_arch, role_lrs, score


def complexity_budget(n_groups: int, *, n_candidates: int, n_tensors: int) -> dict[str, int]:
    """Report comparable tuning budgets for grid vs coordinate descent."""
    grid = n_candidates**max(n_groups, 1)
    cd = n_groups * n_candidates  # per round
    return {
        "n_groups": n_groups,
        "grid_evals_per_round": grid,
        "cd_evals_per_round": cd,
        "free_scalars": n_groups,
        "n_tensors": n_tensors,
    }
