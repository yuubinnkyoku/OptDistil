from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def normalized_gradient_directions(
    grads: Sequence[Tensor],
    *,
    eps: float = 1e-8,
) -> list[Tensor]:
    """Return per-tensor unit negative-gradient directions.

    These are exactly the directions used by tensor-wise NormGrad before multiplying
    by its learning rate. Keeping the direction unit-norm makes scalar coefficients
    directly interpretable as per-tensor update norms.
    """
    if not grads:
        raise ValueError("grads must be non-empty")
    directions: list[Tensor] = []
    for grad in grads:
        norm = grad.reshape(-1).float().norm().clamp_min(eps)
        directions.append(-grad / norm.to(dtype=grad.dtype, device=grad.device))
    return directions


def tensor_projection(
    updates: Sequence[Tensor],
    grads: Sequence[Tensor],
    *,
    eps: float = 1e-8,
) -> tuple[list[Tensor], list[float]]:
    """Project each update onto its tensor-local NormGrad direction.

    Returns projected updates and signed scalar coefficients. Since every direction
    is unit norm, each coefficient is the signed update magnitude along NormGrad.
    """
    if len(updates) != len(grads) or not updates:
        raise ValueError("updates and grads must be non-empty and have equal length")
    directions = normalized_gradient_directions(grads, eps=eps)
    projected: list[Tensor] = []
    coefficients: list[float] = []
    for update, direction in zip(updates, directions, strict=True):
        if update.shape != direction.shape:
            raise ValueError("update and gradient shapes must match per tensor")
        coefficient = float(
            torch.dot(update.reshape(-1).float(), direction.reshape(-1).float())
        )
        projected.append(direction * coefficient)
        coefficients.append(coefficient)
    return projected, coefficients


def global_projection(
    updates: Sequence[Tensor],
    grads: Sequence[Tensor],
    *,
    eps: float = 1e-8,
) -> tuple[list[Tensor], float]:
    """Project the complete update onto one shared tensor-wise NormGrad scale.

    Every tensor receives the same signed coefficient. This preserves the common
    NormGrad direction family while removing learned tensor-wise scale allocation.
    """
    if len(updates) != len(grads) or not updates:
        raise ValueError("updates and grads must be non-empty and have equal length")
    directions = normalized_gradient_directions(grads, eps=eps)
    numerator = 0.0
    denominator = 0.0
    for update, direction in zip(updates, directions, strict=True):
        if update.shape != direction.shape:
            raise ValueError("update and gradient shapes must match per tensor")
        numerator += float(
            torch.dot(update.reshape(-1).float(), direction.reshape(-1).float())
        )
        denominator += float(direction.reshape(-1).float().square().sum())
    coefficient = numerator / max(denominator, eps)
    return [direction * coefficient for direction in directions], coefficient


def apply_direction_scales(
    grads: Sequence[Tensor],
    scales: Sequence[float],
    *,
    eps: float = 1e-8,
) -> list[Tensor]:
    """Apply explicit signed per-tensor scales to NormGrad directions."""
    if len(grads) != len(scales) or not grads:
        raise ValueError("grads and scales must be non-empty and have equal length")
    directions = normalized_gradient_directions(grads, eps=eps)
    return [direction * float(scale) for direction, scale in zip(directions, scales, strict=True)]


def aggregate_role_scales(
    named_coefficients: Sequence[tuple[Sequence[str], Sequence[float]]],
) -> dict[str, float]:
    """Return validation-only median projection coefficient for each tensor name."""
    values: dict[str, list[float]] = {}
    for names, coefficients in named_coefficients:
        if len(names) != len(coefficients):
            raise ValueError("names and coefficients must have equal length")
        for name, coefficient in zip(names, coefficients, strict=True):
            values.setdefault(str(name), []).append(float(coefficient))
    if not values:
        raise ValueError("at least one named coefficient is required")
    return {name: statistics.median(coefficients) for name, coefficients in values.items()}


def scales_for_names(
    names: Sequence[str],
    role_scales: Mapping[str, float],
) -> list[float]:
    """Resolve fixed role scales for one parameter collection."""
    if not names:
        raise ValueError("names must be non-empty")
    if not role_scales:
        raise ValueError("role_scales must be non-empty")
    fallback = statistics.median(float(value) for value in role_scales.values())
    return [float(role_scales.get(str(name), fallback)) for name in names]


def equalized_role_scales(role_scales: Mapping[str, float]) -> dict[str, float]:
    """Replace all role scales by their median while preserving the key set."""
    if not role_scales:
        raise ValueError("role_scales must be non-empty")
    value = statistics.median(float(scale) for scale in role_scales.values())
    return {name: value for name in role_scales}


def swapped_role_scales(role_scales: Mapping[str, float]) -> dict[str, float]:
    """Swap common first/second-layer scales while leaving unmatched roles intact."""
    if not role_scales:
        raise ValueError("role_scales must be non-empty")
    swapped = {str(name): float(value) for name, value in role_scales.items()}
    for left, right in (("W1", "W2"), ("b1", "b2")):
        if left in swapped and right in swapped:
            swapped[left], swapped[right] = swapped[right], swapped[left]
    return swapped
