from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

MULTITENSOR_FEATURE_NAMES = (
    "grad",
    "momentum",
    "rms",
    "parameter",
    "tensor_grad_rms",
    "tensor_param_rms",
    "global_grad_rms",
    "progress",
)

LOCAL_ONLY_FEATURE_NAMES = (
    "grad",
    "momentum",
    "rms",
    "parameter",
    "tensor_grad_rms",
    "tensor_param_rms",
    "local_grad_rms_repeat",
    "progress",
)


def build_multitensor_features(
    parameters: Sequence[Tensor],
    grads: Sequence[Tensor],
    momentums: Sequence[Tensor],
    second_moments: Sequence[Tensor],
    *,
    step: int,
    total_steps: int,
    include_global: bool = True,
    eps: float = 1e-8,
) -> list[Tensor]:
    """Build per-tensor [numel_i, 8] features for a shared 153-param student.

    Default feature family (include_global=True):
      1. grad
      2. momentum
      3. RMS second moment
      4. parameter
      5. tensor-local grad RMS (broadcast)
      6. tensor-local parameter RMS (broadcast)
      7. global grad RMS across all tensors (broadcast)  -- or local repeat if ablated
      8. training progress
    """
    if not parameters:
        raise ValueError("parameters must be non-empty")
    if not (len(parameters) == len(grads) == len(momentums) == len(second_moments)):
        raise ValueError("all multi-tensor sequences must have equal length")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    parameters = [p.detach() for p in parameters]
    grads = [g.detach() for g in grads]
    momentums = [m.detach() for m in momentums]
    second_moments = [s.detach() for s in second_moments]

    for p, g, m, v in zip(parameters, grads, momentums, second_moments, strict=True):
        if p.shape != g.shape or p.shape != m.shape or p.shape != v.shape:
            raise ValueError("parameter/grad/state shapes must match per tensor")

    global_sq = torch.zeros((), dtype=torch.float32)
    for grad in grads:
        global_sq = global_sq + grad.reshape(-1).float().square().sum()
    global_grad_rms = (global_sq / max(sum(g.numel() for g in grads), 1)).sqrt().clamp_min(eps)
    progress = min(max(step / total_steps, 0.0), 1.0)

    features: list[Tensor] = []
    for parameter, grad, momentum, second_moment in zip(
        parameters, grads, momentums, second_moments, strict=True
    ):
        flat_grad = grad.reshape(-1)
        rms = second_moment.clamp_min(0).add(eps).sqrt().reshape(-1)
        tensor_grad_rms = grad.square().mean().add(eps).sqrt()
        tensor_param_rms = parameter.square().mean().add(eps).sqrt()
        global_channel = global_grad_rms if include_global else tensor_grad_rms
        feature = torch.stack(
            (
                flat_grad,
                momentum.reshape(-1),
                rms,
                parameter.reshape(-1),
                tensor_grad_rms.expand_as(flat_grad),
                tensor_param_rms.expand_as(flat_grad),
                global_channel.expand_as(flat_grad),
                torch.full_like(flat_grad, progress),
            ),
            dim=-1,
        )
        features.append(feature)
    return features


def concatenate_features(per_tensor: Sequence[Tensor]) -> Tensor:
    if not per_tensor:
        raise ValueError("per_tensor features must be non-empty")
    return torch.cat(per_tensor, dim=0)


def split_update(
    flat_update: Tensor, shapes: Sequence[torch.Size]
) -> list[Tensor]:
    pieces: list[Tensor] = []
    offset = 0
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        pieces.append(flat_update[offset : offset + size].reshape(shape))
        offset += size
    if offset != flat_update.numel():
        raise ValueError("flat update length does not match shapes")
    return pieces
