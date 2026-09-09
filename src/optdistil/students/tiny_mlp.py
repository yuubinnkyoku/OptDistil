from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyMLPOptimizer(nn.Module):
    """Celo2-base-like elementwise student with two tiny hidden layers by default."""

    def __init__(
        self,
        *,
        feature_dim: int = 8,
        hidden_dim: int = 8,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")

        layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.Tanh()))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [numel, feature_dim]")
        return self.network(features).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class StudentState:
    """Teacher-independent EMA state used to construct student observations."""

    def __init__(self, shape: torch.Size | tuple[int, ...], *, device=None, dtype=None) -> None:
        self.momentum = torch.zeros(shape, device=device, dtype=dtype)
        self.second_moment = torch.zeros(shape, device=device, dtype=dtype)
        self.step_number = 0

    @torch.no_grad()
    def observe(
        self,
        grad: Tensor,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> tuple[Tensor, Tensor]:
        if grad.shape != self.momentum.shape:
            raise ValueError("gradient shape does not match student state")
        self.step_number += 1
        self.momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        self.second_moment.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        return self.momentum, self.second_moment
