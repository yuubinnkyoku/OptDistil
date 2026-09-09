from __future__ import annotations

import torch
from torch import Tensor


class RandomDirectionTeacher:
    """Stateful random-direction teacher used as a negative distillation control."""

    def __init__(self, *, lr: float = 0.1, seed: int = 0, eps: float = 1e-8) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lr = lr
        self.seed = seed
        self.eps = eps
        self.generator: torch.Generator | None = None
        self.generator_device: torch.device | None = None

    def _generator_for(self, device: torch.device) -> torch.Generator:
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
            self.generator_device = device
        elif self.generator_device != device:
            raise ValueError("teacher device changed; create a separate teacher per device")
        return self.generator

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        generator = self._generator_for(grad.device)
        direction = torch.randn(
            grad.shape,
            generator=generator,
            device=grad.device,
            dtype=grad.dtype,
        )
        norm = direction.float().norm().clamp_min(self.eps)
        return -self.lr * direction / norm.to(direction.dtype)
