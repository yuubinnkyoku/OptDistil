from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newton_schulz5(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the polar factor with Muon's common quintic Newton-Schulz iteration."""
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization requires a matrix")
    if steps <= 0:
        raise ValueError("steps must be positive")

    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm() + eps)
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.mT
    return x.to(matrix.dtype)


class MuonTeacher:
    """Reference Muon-style teacher for 2-D parameter tensors.

    This intentionally exposes a simple, stable research baseline rather than mirroring
    every implementation-specific learning-rate scaling rule used by production Muon.
    """

    def __init__(
        self,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.buffer: Tensor | None = None

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if parameter.ndim != 2 or grad.ndim != 2:
            raise ValueError("MuonTeacher currently supports 2-D tensors only")
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        if self.buffer is None:
            self.buffer = torch.zeros_like(grad)
        if self.buffer.shape != grad.shape:
            raise ValueError("teacher state shape changed; create a separate teacher per tensor")

        self.buffer.mul_(self.momentum).add_(grad)
        direction = (
            grad.add(self.buffer, alpha=self.momentum) if self.nesterov else self.buffer
        )
        update = -self.lr * zeropower_via_newton_schulz5(direction, self.ns_steps)
        if self.weight_decay:
            update = update.add(parameter, alpha=-self.lr * self.weight_decay)
        return update
