from __future__ import annotations

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
    """Build the initial teacher-independent feature set used by tiny students.

    The returned tensor is flattened across parameters and has shape ``[numel, 8]``.
    Keeping this representation teacher-independent is deliberate: teacher capacity can
    change without silently giving the student more information.
    """
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape")
    if momentum.shape != parameter.shape or second_moment.shape != parameter.shape:
        raise ValueError("student state tensors must match parameter shape")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

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
