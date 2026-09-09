from __future__ import annotations

from typing import Protocol

import torch

FEATURE_NAMES = (
    "grad",
    "momentum",
    "rms",
    "parameter",
    "grad_sign",
    "log_abs_grad",
    "parameter_rms",
    "progress",
)

MATRIX_FEATURE_NAMES = (
    "grad",
    "momentum",
    "rms",
    "parameter",
    "row_grad_rms",
    "col_grad_rms",
    "parameter_rms",
    "progress",
)


class FeatureBuilder(Protocol):
    def __call__(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        second_moment: torch.Tensor,
        *,
        step: int,
        total_steps: int,
        eps: float = 1e-8,
    ) -> torch.Tensor: ...


def _validate_inputs(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    total_steps: int,
) -> None:
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape")
    if momentum.shape != parameter.shape or second_moment.shape != parameter.shape:
        raise ValueError("student state tensors must match parameter shape")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")


def build_elementwise_features(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build the baseline teacher-independent 8-feature observation."""
    _validate_inputs(parameter, grad, momentum, second_moment, total_steps)

    parameter = parameter.detach()
    grad = grad.detach()
    momentum = momentum.detach()
    second_moment = second_moment.detach()

    rms = second_moment.clamp_min(0).add(eps).sqrt()
    parameter_rms = parameter.square().mean().add(eps).sqrt()
    progress = min(max(step / total_steps, 0.0), 1.0)

    def flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1)

    flat_grad = flat(grad)
    return torch.stack(
        (
            flat_grad,
            flat(momentum),
            flat(rms),
            flat(parameter),
            flat(grad.sign()),
            flat(grad.abs().log1p()),
            parameter_rms.expand_as(flat_grad),
            torch.full_like(flat_grad, progress),
        ),
        dim=-1,
    )


def build_matrix_aware_features(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build an NPU-friendly 8-feature observation with cheap row/column statistics.

    The feature count stays identical to the baseline, so an ``8 -> 8 -> 8 -> 1``
    student remains exactly 153 parameters. Row and column RMS values use only reductions
    and broadcasts; they are intended as a cheap structural hint for matrix teachers such
    as Muon rather than a hidden copy of the teacher computation.
    """
    _validate_inputs(parameter, grad, momentum, second_moment, total_steps)
    if parameter.ndim != 2:
        raise ValueError("matrix-aware features require a 2-D parameter tensor")

    parameter = parameter.detach()
    grad = grad.detach()
    momentum = momentum.detach()
    second_moment = second_moment.detach()

    rms = second_moment.clamp_min(0).add(eps).sqrt()
    row_grad_rms = grad.square().mean(dim=1, keepdim=True).add(eps).sqrt().expand_as(grad)
    col_grad_rms = grad.square().mean(dim=0, keepdim=True).add(eps).sqrt().expand_as(grad)
    parameter_rms = parameter.square().mean().add(eps).sqrt()
    progress = min(max(step / total_steps, 0.0), 1.0)

    def flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1)

    flat_grad = flat(grad)
    return torch.stack(
        (
            flat_grad,
            flat(momentum),
            flat(rms),
            flat(parameter),
            flat(row_grad_rms),
            flat(col_grad_rms),
            parameter_rms.expand_as(flat_grad),
            torch.full_like(flat_grad, progress),
        ),
        dim=-1,
    )
