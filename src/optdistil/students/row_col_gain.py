from __future__ import annotations

import math

import torch
from torch import Tensor, nn

ROW_COL_GAIN_FEATURE_NAMES = (
    "base_update",
    "log_row_grad_rms",
    "log_row_momentum_rms",
    "log_row_parameter_rms",
    "log_col_grad_rms",
    "log_col_momentum_rms",
    "log_col_parameter_rms",
    "progress",
)


def _rms(values: Tensor, *, dim: int, eps: float) -> Tensor:
    return values.square().mean(dim=dim, keepdim=True).add(eps).sqrt()


def build_row_col_gain_features(
    parameter: Tensor,
    grad: Tensor,
    momentum: Tensor,
    second_moment: Tensor,
    *,
    step: int,
    total_steps: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tensor:
    """Build separable row/column statistics around a bias-corrected RMS-momentum base."""
    if parameter.ndim != 2:
        raise ValueError("row/column gain features require a 2-D parameter tensor")
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape")
    if momentum.shape != parameter.shape or second_moment.shape != parameter.shape:
        raise ValueError("student state tensors must match parameter shape")
    if parameter.numel() == 0:
        raise ValueError("parameter tensor must be non-empty")
    if step <= 0:
        raise ValueError("step must be positive for EMA bias correction")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("EMA betas must lie in [0, 1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    parameter = parameter.detach()
    grad = grad.detach()
    momentum = momentum.detach()
    second_moment = second_moment.detach()

    momentum_hat = momentum / (1.0 - beta1**step)
    second_moment_hat = second_moment / (1.0 - beta2**step)

    row_grad_rms = _rms(grad, dim=1, eps=eps).expand_as(grad)
    row_momentum_rms = _rms(momentum_hat, dim=1, eps=eps).expand_as(momentum_hat)
    row_parameter_rms = _rms(parameter, dim=1, eps=eps).expand_as(parameter)

    col_grad_rms = _rms(grad, dim=0, eps=eps).expand_as(grad)
    col_momentum_rms = _rms(momentum_hat, dim=0, eps=eps).expand_as(momentum_hat)
    col_parameter_rms = _rms(parameter, dim=0, eps=eps).expand_as(parameter)

    global_denom = second_moment_hat.clamp_min(0).mean().add(eps).sqrt()
    base_update = -momentum_hat / global_denom
    progress = min(max(step / total_steps, 0.0), 1.0)

    def flat(values: Tensor) -> Tensor:
        return values.reshape(-1)

    return torch.stack(
        (
            flat(base_update),
            flat(row_grad_rms).add(eps).log(),
            flat(row_momentum_rms).add(eps).log(),
            flat(row_parameter_rms).add(eps).log(),
            flat(col_grad_rms).add(eps).log(),
            flat(col_momentum_rms).add(eps).log(),
            flat(col_parameter_rms).add(eps).log(),
            torch.full_like(flat(base_update), progress),
        ),
        dim=-1,
    )


class RowColGainOptimizer(nn.Module):
    """Tiny shared-MLP separable row/column gain optimizer.

    The default shared ``4 -> 8 -> 8 -> 1`` network has 121 trainable parameters, the
    same learned-parameter budget as ``BlockGainOptimizer``. The final layer starts at
    zero, making the initial policy exactly the bias-corrected global-RMS momentum base.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 8,
        hidden_layers: int = 2,
        log_gain_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if not math.isfinite(log_gain_limit) or log_gain_limit <= 0.0:
            raise ValueError("log_gain_limit must be positive and finite")

        layers: list[nn.Module] = []
        in_dim = 4
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.Tanh()))
            in_dim = hidden_dim
        output = nn.Linear(in_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)
        self.log_gain_limit = float(log_gain_limit)
        self.register_buffer("output_scale", torch.tensor(1.0))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(ROW_COL_GAIN_FEATURE_NAMES):
            raise ValueError("features must have shape [numel, 8]")

        base_update = features[:, 0]
        progress = features[:, 7:8]
        row_features = torch.cat((features[:, 1:4], progress), dim=-1)
        col_features = torch.cat((features[:, 4:7], progress), dim=-1)

        row_raw = self.network(row_features).squeeze(-1)
        col_raw = self.network(col_features).squeeze(-1)
        log_gain = 0.5 * self.log_gain_limit * (
            torch.tanh(row_raw) + torch.tanh(col_raw)
        )
        gain = torch.exp(log_gain)
        return base_update * gain * self.output_scale

    @torch.no_grad()
    def set_output_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("output scale must be positive and finite")
        self.output_scale.fill_(scale)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
