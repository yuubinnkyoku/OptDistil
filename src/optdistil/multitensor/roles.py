"""Role-label utilities for falsification of matrix/vector specialness."""

from __future__ import annotations

import random
from collections.abc import Sequence


def invert_role_map(role_lrs: dict[str, float], roles: Sequence[str]) -> list[float]:
    return [float(role_lrs[role]) for role in roles]


def swap_binary_role_labels(roles: Sequence[str]) -> list[str]:
    """Swap the first two unique labels when exactly two roles exist."""
    unique: list[str] = []
    for role in roles:
        if role not in unique:
            unique.append(role)
    if len(unique) != 2:
        raise ValueError("swap requires exactly two unique roles")
    mapping = {unique[0]: unique[1], unique[1]: unique[0]}
    return [mapping[role] for role in roles]


def random_balanced_binary_partition(
    n_tensors: int,
    *,
    seed: int,
) -> list[str]:
    """Assign tensors to two groups A/B with sizes differing by at most one."""
    if n_tensors < 2:
        raise ValueError("n_tensors must be >= 2")
    labels = ["A"] * (n_tensors // 2) + ["B"] * (n_tensors - n_tensors // 2)
    random.Random(seed).shuffle(labels)
    return labels


def apply_role_assignment(
    true_roles: Sequence[str],
    assigned_roles: Sequence[str],
) -> list[str]:
    if len(true_roles) != len(assigned_roles):
        raise ValueError("assignment length must match tensor count")
    return [str(role) for role in assigned_roles]


def role_lrs_from_assigned(
    role_lrs: dict[str, float],
    assigned_roles: Sequence[str],
) -> list[float]:
    return invert_role_map(role_lrs, assigned_roles)
