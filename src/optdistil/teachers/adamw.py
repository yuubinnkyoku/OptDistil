from __future__ import annotations

import torch
from torch import Tensor


class AdamWTeacher:
    """Small functional AdamW teacher that returns updates without mutating parameters."""

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
        self.m: Tensor | None = None
        self.v: Tensor | None = None

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)
        if self.m.shape != grad.shape:
            raise ValueError("teacher state shape changed; create a separate teacher per tensor")

        assert self.v is not None
        self.step_number += 1
        self.m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

        bias1 = 1.0 - self.beta1**self.step_number
        bias2 = 1.0 - self.beta2**self.step_number
        m_hat = self.m / bias1
        v_hat = self.v / bias2
        update = -self.lr * m_hat / (v_hat.sqrt() + self.eps)
        if self.weight_decay:
            update = update.add(parameter, alpha=-self.lr * self.weight_decay)
        return update
