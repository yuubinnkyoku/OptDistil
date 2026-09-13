from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from optdistil.multitensor.params import ParamCollection
from optdistil.teachers.muon import zeropower_via_newton_schulz5


class MultiTensorTeacher:
    """Base interface: step returns per-tensor updates without mutating parameters."""

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        raise NotImplementedError


def _check_shapes(params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
    if len(grads) != len(params):
        raise ValueError("grad count must match parameter count")
    grad_list = list(grads)
    for parameter, grad in zip(params, grad_list, strict=True):
        if parameter.shape != grad.shape:
            raise ValueError("gradient shape must match parameter shape")
    return grad_list


class SGDMultiTensor(MultiTensorTeacher):
    def __init__(self, *, lr: float = 0.1) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        grad_list = _check_shapes(params, grads)
        return [-self.lr * grad for grad in grad_list]


class MomentumSGDMultiTensor(MultiTensorTeacher):
    def __init__(self, *, lr: float = 0.1, momentum: float = 0.9) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.momentum = momentum
        self.buffers: list[Tensor] | None = None

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        grad_list = _check_shapes(params, grads)
        if self.buffers is None:
            self.buffers = [torch.zeros_like(grad) for grad in grad_list]
        updates: list[Tensor] = []
        for buffer, grad in zip(self.buffers, grad_list, strict=True):
            buffer.mul_(self.momentum).add_(grad)
            updates.append(-self.lr * buffer)
        return updates


class NormGradTensorWise(MultiTensorTeacher):
    """u_l = -lr * g_l / ||g_l||"""

    def __init__(self, *, lr: float = 0.1, eps: float = 1e-8) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.eps = eps

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        grad_list = _check_shapes(params, grads)
        updates: list[Tensor] = []
        for grad in grad_list:
            norm = grad.reshape(-1).float().norm().clamp_min(self.eps)
            updates.append(-self.lr * grad / norm.to(grad.dtype))
        return updates


class NormGradGlobal(MultiTensorTeacher):
    """u_l = -lr * g_l / sqrt(sum_j ||g_j||^2)"""

    def __init__(self, *, lr: float = 0.1, eps: float = 1e-8) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.eps = eps

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        grad_list = _check_shapes(params, grads)
        total = torch.zeros((), dtype=torch.float32)
        for grad in grad_list:
            total = total + grad.reshape(-1).float().square().sum()
        global_norm = total.sqrt().clamp_min(self.eps)
        return [-self.lr * grad / global_norm.to(grad.dtype) for grad in grad_list]


class AdamWMultiTensor(MultiTensorTeacher):
    """Per-tensor AdamW state (independent m/v per tensor)."""

    def __init__(
        self,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_number = 0
        self.m: list[Tensor] | None = None
        self.v: list[Tensor] | None = None

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        grad_list = _check_shapes(params, grads)
        if self.m is None or self.v is None:
            self.m = [torch.zeros_like(grad) for grad in grad_list]
            self.v = [torch.zeros_like(grad) for grad in grad_list]
        self.step_number += 1
        bias1 = 1.0 - self.beta1**self.step_number
        bias2 = 1.0 - self.beta2**self.step_number
        updates: list[Tensor] = []
        for m, v, grad, parameter in zip(self.m, self.v, grad_list, params, strict=True):
            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            m_hat = m / bias1
            v_hat = v / bias2
            update = -self.lr * m_hat / (v_hat.sqrt() + self.eps)
            if self.weight_decay:
                update = update.add(parameter, alpha=-self.lr * self.weight_decay)
            updates.append(update)
        return updates


class MuonHybridTeacher(MultiTensorTeacher):
    """Muon on 2-D tensors; momentum SGD on 1-D tensors (biases)."""

    def __init__(
        self,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        bias_lr: float | None = None,
    ) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.bias_lr = bias_lr if bias_lr is not None else lr
        self.buffers: list[Tensor] | None = None

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        grad_list = _check_shapes(params, grads)
        if self.buffers is None:
            self.buffers = [torch.zeros_like(grad) for grad in grad_list]
        updates: list[Tensor] = []
        for buffer, grad, parameter in zip(self.buffers, grad_list, params, strict=True):
            buffer.mul_(self.momentum).add_(grad)
            if grad.ndim == 2:
                direction = grad.add(buffer, alpha=self.momentum) if self.nesterov else buffer
                updates.append(-self.lr * zeropower_via_newton_schulz5(direction, self.ns_steps))
            else:
                updates.append(-self.bias_lr * buffer)
        return updates


class LBFGSMultiTensor(MultiTensorTeacher):
    """Lightweight L-BFGS over the global flattened parameter vector.

    History uses global inner products. Optional two-scale bootstrap mirrors the
    single-tensor stochastic suite: first steps use normalized gradient, later steps
    use the L-BFGS direction.
    """

    def __init__(
        self,
        *,
        lr: float = 0.1,
        history_size: int = 10,
        bootstrap_steps: int = 0,
        bootstrap_lr: float | None = None,
        damping: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = lr
        self.history_size = history_size
        self.bootstrap_steps = bootstrap_steps
        self.bootstrap_lr = bootstrap_lr if bootstrap_lr is not None else lr
        self.damping = damping
        self.s_history: list[Tensor] = []
        self.y_history: list[Tensor] = []
        self.rho_history: list[float] = []
        self.prev_flat: Tensor | None = None
        self.prev_grad: Tensor | None = None
        self.step_number = 0

    @torch.no_grad()
    def step(self, params: ParamCollection, grads: Sequence[Tensor]) -> list[Tensor]:
        _check_shapes(params, grads)
        flat_param = params.flat_copy()
        flat_grad = torch.cat([grad.reshape(-1).float() for grad in grads])
        self.step_number += 1

        if self.step_number <= self.bootstrap_steps:
            norm = flat_grad.norm().clamp_min(1e-8)
            flat_update = -self.bootstrap_lr * flat_grad / norm
            self.prev_flat = flat_param
            self.prev_grad = flat_grad
            updates = params.apply_flat(flat_update.to(dtype=params[0].dtype))
            return list(updates)

        if self.prev_flat is not None and self.prev_grad is not None:
            s = flat_param - self.prev_flat
            y = flat_grad - self.prev_grad
            sy = float(torch.dot(s, y))
            if sy > 1e-12:
                rho = 1.0 / sy
                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(rho)
                if len(self.s_history) > self.history_size:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                    self.rho_history.pop(0)

        direction = self._two_loop(flat_grad)
        self.prev_flat = flat_param
        self.prev_grad = flat_grad
        flat_update = -self.lr * direction
        updates = params.apply_flat(flat_update.to(dtype=params[0].dtype))
        return list(updates)

    def _two_loop(self, grad: Tensor) -> Tensor:
        q = grad.clone()
        alphas: list[float] = []
        for s, y, rho in zip(
            reversed(self.s_history), reversed(self.y_history), reversed(self.rho_history), strict=True
        ):
            alpha = rho * float(torch.dot(s, q))
            alphas.append(alpha)
            q = q - alpha * y
        if self.s_history:
            last_s = self.s_history[-1]
            last_y = self.y_history[-1]
            yy = max(float(torch.dot(last_y, last_y)), 1e-12)
            gamma = max(float(torch.dot(last_s, last_y)) / yy, 1e-8)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y, rho), alpha in zip(
            zip(self.s_history, self.y_history, self.rho_history, strict=True),
            reversed(alphas),
            strict=True,
        ):
            beta = rho * float(torch.dot(y, r))
            r = r + s * (alpha - beta)
        return r


def make_teacher(method: str, lr: float) -> MultiTensorTeacher:
    if method == "sgd":
        return SGDMultiTensor(lr=lr)
    if method == "momentum":
        return MomentumSGDMultiTensor(lr=lr)
    if method == "norm_grad_local":
        return NormGradTensorWise(lr=lr)
    if method == "norm_grad_global":
        return NormGradGlobal(lr=lr)
    if method == "adamw":
        return AdamWMultiTensor(lr=lr)
    if method == "muon":
        return MuonHybridTeacher(lr=lr)
    if method == "lbfgs":
        return LBFGSMultiTensor(lr=lr, bootstrap_steps=2)
    raise ValueError(f"unknown multi-tensor teacher method: {method}")


TEACHER_METHODS = (
    "sgd",
    "momentum",
    "norm_grad_local",
    "norm_grad_global",
    "adamw",
    "muon",
    "lbfgs",
)
