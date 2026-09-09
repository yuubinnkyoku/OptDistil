from __future__ import annotations

import torch
from torch import Tensor


class MomentumDirectionTeacher:
    """Muon momentum/Nesterov direction without Newton-Schulz orthogonalization."""

    def __init__(
        self,
        *,
        lr: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.eps = eps
        self.buffer: Tensor | None = None

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
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
        norm = direction.float().norm().clamp_min(self.eps)
        return -self.lr * direction / norm.to(direction.dtype)
