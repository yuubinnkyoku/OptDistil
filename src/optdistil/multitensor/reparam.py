"""Positive diagonal tensor-wise reparameterization: p_i = s_i * theta_i.

The task objective is always defined on function-space parameters ``p``.
Optimizers step in ``theta``-space. Gradients transform as ``g_theta = s * g_p``.
Students observe only ``theta``-space quantities; privileged controls may use ``s``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.stochastic import (
    MultiTensorCase,
    RolloutMetrics,
    _finalize,
    batch_sequence,
)
from optdistil.multitensor.teachers import MultiTensorTeacher, NormGradTensorWise

TRAIN_SCALE_RANGE = (0.5, 2.0)
IID_SCALE_RANGE = (0.5, 2.0)
OOD_SCALE_RANGE = (0.25, 4.0)
STRONG_OOD_SCALE_RANGE = (0.1, 10.0)


def sample_log_uniform_scales(
    count: int,
    *,
    low: float,
    high: float,
    seed: int,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Sample ``count`` positive scales log-uniformly from [low, high]."""
    if count <= 0:
        raise ValueError("count must be positive")
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("require 0 < low < high")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    log_low = math.log(low)
    log_high = math.log(high)
    log_scales = log_low + (log_high - log_low) * torch.rand(count, generator=generator)
    scales = log_scales.exp().tolist()
    device = torch.device(device)
    return [float(s.to(device)) for s in torch.tensor(scales, device=device)]


def validate_scales(scales: Sequence[float], expected_count: int) -> list[float]:
    values = [float(s) for s in scales]
    if len(values) != expected_count:
        raise ValueError(f"expected {expected_count} scales, got {len(values)}")
    if any((not math.isfinite(s)) or s <= 0.0 for s in values):
        raise ValueError("all scales must be positive and finite")
    return values


class ReparameterizedTask:
    """Wrap a function-space task so optimizers step in theta with p = s * theta."""

    def __init__(self, inner: Any, scales: Sequence[float]) -> None:
        self.inner = inner
        expected = len(inner.parameter_shapes)
        self.scales = validate_scales(scales, expected)
        self._scale_tensors = None

    def _scale_list(self, reference: Tensor) -> list[Tensor]:
        if self._scale_tensors is None or self._scale_tensors[0].device != reference.device:
            self._scale_tensors = [
                torch.tensor(s, dtype=torch.float32, device=reference.device) for s in self.scales
            ]
        return self._scale_tensors

    def _to_p(self, theta: ParamCollection) -> ParamCollection:
        scales = self._scale_list(theta[0])
        return ParamCollection(
            [tensor.detach() * scale for tensor, scale in zip(theta, scales, strict=True)]
        )

    def _to_g_theta(self, g_p: Sequence[Tensor]) -> list[Tensor]:
        return [grad.detach() * scale for grad, scale in zip(g_p, self.scales, strict=True)]

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return list(self.inner.parameter_shapes)

    @property
    def sample_count(self) -> int:
        return self.inner.sample_count

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.inner.parameter_names)

    @property
    def parameter_roles(self) -> tuple[str, ...]:
        return tuple(self.inner.parameter_roles)

    def loss(self, params: ParamCollection) -> Tensor:
        return self.inner.loss(self._to_p(params))

    def loss_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        return self.inner.loss_on_samples(self._to_p(params), sample_indices)

    def grad(self, params: ParamCollection) -> list[Tensor]:
        return self._to_g_theta(self.inner.grad(self._to_p(params)))

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        return self._to_g_theta(self.inner.grad_on_samples(self._to_p(params), sample_indices))


def reparameterize_initial(
    initial: ParamCollection,
    scales: Sequence[float],
) -> ParamCollection:
    values = validate_scales(scales, len(initial))
    return ParamCollection(
        [tensor.detach() / scale for tensor, scale in zip(initial, values, strict=True)]
    )


def make_reparameterized_case(
    case: MultiTensorCase,
    *,
    scales: Sequence[float],
) -> tuple[MultiTensorCase, list[float]]:
    """Return a theta-space case equivalent in function/objective to ``case``."""
    values = validate_scales(scales, len(case.initial))
    theta0 = reparameterize_initial(case.initial, values)
    task = ReparameterizedTask(case.task, values)
    reparam_case = MultiTensorCase(
        initial=theta0,
        task=task,
        batch_seed=case.batch_seed,
        architecture=case.architecture,
        width=case.width,
        condition=case.condition,
    )
    return reparam_case, values


class PrivilegedFunctionSpaceNormGrad(MultiTensorTeacher):
    """Function-space local NormGrad using privileged reparameterization scales.

    Ordinary local NormGrad on theta: u_theta = -lr * g_theta / ||g_theta||.
    Function-space step in p is then s * u_theta, so larger-s tensors take larger
    function-space steps. Privileged control removes that distortion:
    u_theta_priv = u_theta / s, which equals -lr * g_p / (s ||g_p||).
    """

    def __init__(self, *, lr: float, scales: Sequence[float], eps: float = 1e-8) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.scales = validate_scales(scales, len(scales))
        self.eps = eps
        self._inner = NormGradTensorWise(lr=lr, eps=eps)

    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        updates = self._inner.step(params, grads)
        return [
            update / scale for update, scale in zip(updates, self.scales, strict=True)
        ]


class ProjectedNormGradDirection(MultiTensorTeacher):
    """Project an arbitrary update onto per-tensor local NormGrad directions.

    Keeps only the per-tensor scale along d_l = -g_l / ||g_l||, discarding
    within-tensor direction adaptation.
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        self.eps = eps

    def project(
        self,
        updates: Sequence[Tensor],
        grads: Sequence[Tensor],
    ) -> list[Tensor]:
        projected: list[Tensor] = []
        for update, grad in zip(updates, grads, strict=True):
            if update.shape != grad.shape:
                raise ValueError("update and grad shapes must match")
            g = grad.reshape(-1).float()
            u = update.reshape(-1).float()
            norm = g.norm().clamp_min(self.eps)
            direction = -g / norm
            scale = torch.dot(u, direction)
            projected.append((scale * direction).reshape(update.shape).to(update.dtype))
        return projected

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        """Unit local NormGrad directions (lr=1)."""
        unit: list[Tensor] = []
        for grad in grads:
            g = grad.reshape(-1).float()
            norm = g.norm().clamp_min(self.eps)
            unit.append((-g / norm).reshape(grad.shape).to(grad.dtype))
        return unit


@dataclass(frozen=True, slots=True)
class ReparamRolloutResult:
    metrics: RolloutMetrics
    scales: tuple[float, ...]
    architecture: str
    condition: float


@torch.no_grad()
def rollout_reparameterized(
    teacher: MultiTensorTeacher,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> RolloutMetrics:
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
            return _finalize(losses, initial_loss)
        losses.append(float(case.task.loss(params)))
    return _finalize(losses, initial_loss)
