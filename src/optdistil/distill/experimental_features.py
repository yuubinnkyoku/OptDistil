from __future__ import annotations

import torch

from optdistil.distill.features import build_matrix_aware_features


def _validate(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    total_steps: int,
) -> None:
    if parameter.ndim != 2:
        raise ValueError("hybrid matrix features require a 2-D parameter tensor")
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape")
    if momentum.shape != parameter.shape or second_moment.shape != parameter.shape:
        raise ValueError("student state tensors must match parameter shape")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")


def _normalized_gram_cubic(matrix: torch.Tensor, *, eps: float) -> torch.Tensor:
    rows, cols = matrix.shape
    if rows <= cols:
        cubic = (matrix @ matrix.mT) @ matrix
    else:
        cubic = matrix @ (matrix.mT @ matrix)
    return cubic / matrix.square().sum().clamp_min(eps)


def build_hybrid_gram_features(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Combine cheap row/column scale summaries with one cross-coordinate Gram direction.

    The eight features are

    ``grad, momentum, rms, row_grad_rms, col_grad_rms, grad_gram, parameter_rms, progress``.

    Raw parameter is intentionally omitted. On translated quadratic tasks the gradient
    already contains the error signal while the absolute parameter value is not invariant
    to the randomly sampled optimum. This makes room for a cross-coordinate basis without
    increasing the 153-parameter student's input dimension.
    """
    _validate(parameter, grad, momentum, second_moment, total_steps)
    parameter = parameter.detach()
    grad = grad.detach()
    momentum = momentum.detach()
    second_moment = second_moment.detach()

    rms = second_moment.clamp_min(0).add(eps).sqrt()
    row_grad_rms = grad.square().mean(dim=1, keepdim=True).add(eps).sqrt().expand_as(grad)
    col_grad_rms = grad.square().mean(dim=0, keepdim=True).add(eps).sqrt().expand_as(grad)
    grad_gram = _normalized_gram_cubic(grad, eps=eps)
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
            flat(row_grad_rms),
            flat(col_grad_rms),
            flat(grad_gram),
            parameter_rms.expand_as(flat_grad),
            torch.full_like(flat_grad, progress),
        ),
        dim=-1,
    )


def build_matrix_aware_no_ema(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Matrix-aware features with momentum/RMS channels zeroed, still 8-wide."""
    features = build_matrix_aware_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=step,
        total_steps=total_steps,
        eps=eps,
    ).clone()
    features[:, 1] = 0.0
    features[:, 2] = 0.0
    return features


def build_matrix_aware_no_progress(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Matrix-aware features with the training-progress channel zeroed, still 8-wide."""
    features = build_matrix_aware_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=step,
        total_steps=total_steps,
        eps=eps,
    ).clone()
    features[:, 7] = 0.0
    return features
