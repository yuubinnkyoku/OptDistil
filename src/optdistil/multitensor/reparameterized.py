from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.tasks import MultiTensorTask, make_task


class ReparameterizedTask:
    """Exact positive diagonal reparameterization of a multi-tensor task.

    The optimizer sees coordinates ``theta_i`` while the wrapped task is evaluated at
    ``p_i = scale_i * theta_i``. Initial function values are preserved by dividing the
    wrapped initial parameters by the same scales. Only optimizer geometry changes.
    """

    def __init__(self, base_task: MultiTensorTask, parameter_scales: Sequence[float]) -> None:
        scales = tuple(float(scale) for scale in parameter_scales)
        if len(scales) != len(base_task.parameter_shapes):
            raise ValueError("parameter_scales must match the base parameter count")
        if any(scale <= 0.0 or not math.isfinite(scale) for scale in scales):
            raise ValueError("parameter_scales must be positive and finite")
        self.base_task = base_task
        self.parameter_scales = scales
        self.parameter_names = tuple(
            getattr(base_task, "parameter_names", tuple(f"t{i}" for i in range(len(scales))))
        )
        self.parameter_roles = tuple(
            getattr(base_task, "parameter_roles", tuple("tensor" for _ in scales))
        )

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return list(self.base_task.parameter_shapes)

    @property
    def sample_count(self) -> int:
        return self.base_task.sample_count

    def to_base_parameters(self, params: ParamCollection) -> ParamCollection:
        if len(params) != len(self.parameter_scales):
            raise ValueError("parameter collection size mismatch")
        return ParamCollection(
            [
                tensor * scale
                for tensor, scale in zip(params, self.parameter_scales, strict=True)
            ]
        )

    def loss(self, params: ParamCollection) -> Tensor:
        return self.base_task.loss(self.to_base_parameters(params))

    def loss_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        return self.base_task.loss_on_samples(self.to_base_parameters(params), sample_indices)

    def grad(self, params: ParamCollection) -> list[Tensor]:
        base_grads = self.base_task.grad(self.to_base_parameters(params))
        return [
            grad * scale
            for grad, scale in zip(base_grads, self.parameter_scales, strict=True)
        ]

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        base_grads = self.base_task.grad_on_samples(
            self.to_base_parameters(params), sample_indices
        )
        return [
            grad * scale
            for grad, scale in zip(base_grads, self.parameter_scales, strict=True)
        ]


def reparameterize_initial(
    base_initial: ParamCollection,
    parameter_scales: Sequence[float],
) -> ParamCollection:
    scales = tuple(float(scale) for scale in parameter_scales)
    if len(base_initial) != len(scales):
        raise ValueError("parameter_scales must match the base parameter count")
    if any(scale <= 0.0 or not math.isfinite(scale) for scale in scales):
        raise ValueError("parameter_scales must be positive and finite")
    return ParamCollection(
        [tensor / scale for tensor, scale in zip(base_initial, scales, strict=True)]
    )


def sample_log_uniform_scales(
    seed: int,
    count: int,
    *,
    min_scale: float,
    max_scale: float,
) -> tuple[float, ...]:
    """Draw deterministic independent positive scales uniformly in log space."""
    if count <= 0:
        raise ValueError("count must be positive")
    if min_scale <= 0.0 or max_scale < min_scale:
        raise ValueError("require 0 < min_scale <= max_scale")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low = math.log(min_scale)
    high = math.log(max_scale)
    draws = torch.rand((count,), generator=generator)
    return tuple(math.exp(low + (high - low) * float(value)) for value in draws)


def make_reparameterized_task(
    architecture: str,
    seed: int,
    *,
    width: int = 8,
    samples: int = 64,
    input_condition: float = 30.0,
    scale_seed: int | None = None,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    device: torch.device | str = "cpu",
) -> tuple[ParamCollection, ReparameterizedTask]:
    base_initial, base_task = make_task(
        architecture,
        seed,
        width=width,
        samples=samples,
        input_condition=input_condition,
        device=device,
    )
    scales = sample_log_uniform_scales(
        seed + 7_000_000 if scale_seed is None else scale_seed,
        len(base_initial),
        min_scale=min_scale,
        max_scale=max_scale,
    )
    initial = reparameterize_initial(base_initial, scales)
    return initial, ReparameterizedTask(base_task, scales)


class FunctionSpaceNormGradTeacher:
    """Privileged NormGrad oracle invariant to the exact reparameterization above.

    For ``p_i = s_i theta_i``, a theta-space update of norm ``lr / s_i`` produces a
    function-space parameter update of norm ``lr`` along the corresponding negative
    gradient direction. The oracle sees ``s_i``; the distilled Student does not.
    """

    def __init__(self, lr: float, parameter_scales: Sequence[float], eps: float = 1e-8) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        self.lr = float(lr)
        self.parameter_scales = tuple(float(scale) for scale in parameter_scales)
        self.eps = float(eps)

    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        if len(params) != len(grads) or len(grads) != len(self.parameter_scales):
            raise ValueError("parameter, gradient, and scale counts must match")
        updates: list[Tensor] = []
        for grad, scale in zip(grads, self.parameter_scales, strict=True):
            norm_dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
            norm = grad.reshape(-1).to(norm_dtype).norm().clamp_min(self.eps)
            updates.append(-self.lr * grad / (scale * norm.to(grad.dtype)))
        return updates
