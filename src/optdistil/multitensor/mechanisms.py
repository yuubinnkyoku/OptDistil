from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence

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


def coordinate_descent_role_scales(
    role_names: Sequence[str],
    candidates: Sequence[float],
    evaluate: Callable[[Mapping[str, float]], float],
    *,
    initial_scale: float,
    passes: int = 3,
) -> tuple[dict[str, float], float, list[dict[str, object]]]:
    """Tune static role scales using only a caller-provided validation objective.

    The search is deterministic coordinate descent over absolute scale candidates.
    It is intentionally simple: the goal is a strong low-dimensional analytic control,
    not a learned optimizer. ``evaluate`` must use validation data only.
    """
    roles = tuple(dict.fromkeys(str(name) for name in role_names))
    values = tuple(float(value) for value in candidates)
    if not roles:
        raise ValueError("role_names must be non-empty")
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("candidates must contain positive scales")
    if initial_scale <= 0.0:
        raise ValueError("initial_scale must be positive")
    if passes <= 0:
        raise ValueError("passes must be positive")

    scales = {name: float(initial_scale) for name in roles}
    best_score = float(evaluate(scales))
    history: list[dict[str, object]] = [
        {"pass": -1, "role": "initial", "scale": initial_scale, "score": best_score}
    ]

    for pass_index in range(passes):
        changed = False
        for role in roles:
            current_scale = scales[role]
            role_best_scale = current_scale
            role_best_score = best_score
            for candidate in values:
                trial = dict(scales)
                trial[role] = candidate
                score = float(evaluate(trial))
                if score < role_best_score:
                    role_best_score = score
                    role_best_scale = candidate
            scales[role] = role_best_scale
            best_score = role_best_score
            changed |= role_best_scale != current_scale
            history.append(
                {
                    "pass": pass_index,
                    "role": role,
                    "scale": role_best_scale,
                    "score": role_best_score,
                }
            )
        if not changed:
            break

    return scales, best_score, history
