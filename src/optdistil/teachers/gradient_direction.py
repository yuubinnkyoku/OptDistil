from __future__ import annotations

import torch
from torch import Tensor


class GradientDirectionTeacher:
    """Scale-invariant negative-gradient teacher for mechanistic ablations."""

    def __init__(self, *, lr: float = 0.1, eps: float = 1e-8) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lr = lr
        self.eps = eps

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        norm = grad.float().norm().clamp_min(self.eps)
        return -self.lr * grad / norm.to(grad.dtype)
