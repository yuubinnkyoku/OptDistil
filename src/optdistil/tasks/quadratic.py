from __future__ import annotations

import torch
from torch import Tensor


class QuadraticTask:
    """A cheap deterministic task for optimizer and distillation smoke tests."""

    def __init__(self, target: Tensor, curvature: Tensor | float = 1.0) -> None:
        self.target = target.detach().clone()
        self.curvature = torch.as_tensor(curvature, dtype=target.dtype, device=target.device)

    def loss(self, parameter: Tensor) -> Tensor:
        delta = parameter - self.target
        return 0.5 * (self.curvature * delta.square()).sum()

    def grad(self, parameter: Tensor) -> Tensor:
        return self.curvature * (parameter - self.target)
