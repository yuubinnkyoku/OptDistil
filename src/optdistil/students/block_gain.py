from __future__ import annotations

import math

import torch
from torch import Tensor, nn

BLOCK_GAIN_FEATURE_NAMES = (
    "base_update",
    "log_grad_rms",
    "log_momentum_rms",
    "log_parameter_rms",
    "progress",
)


def _block_rms(values: Tensor, *, block_size: int, eps: float) -> Tensor:
    flat = values.reshape(-1)
    numel = flat.numel()
    blocks = math.ceil(numel / block_size)
    padded_numel = blocks * block_size

    if padded_numel != numel:
        flat = torch.nn.functional.pad(flat, (0, padded_numel - numel))

    squared = flat.square().reshape(blocks, block_size)
    counts = torch.full(
        (blocks,),
        block_size,
        device=values.device,
        dtype=values.dtype,
    )
    if numel % block_size:
        counts[-1] = numel % block_size
    return (squared.sum(dim=1) / counts).add(eps).sqrt()


def _expand_blocks(values: Tensor, *, numel: int, block_size: int) -> Tensor:
    return values.repeat_interleave(block_size)[:numel]


def build_block_gain_features(
    parameter: Tensor,
    grad: Tensor,
    momentum: Tensor,
    second_moment: Tensor,
    *,
    step: int,
    total_steps: int,
    block_size: int = 64,
    eps: float = 1e-8,
) -> Tensor:
    """Build a blockwise observation for a cheap multiplicative learned correction.

    The first column is a stable block-RMS momentum update. The remaining four columns
    are constant within each block, so ``BlockGainOptimizer`` necessarily predicts one
    shared gain per block without needing an explicit block id.
    """
    if parameter.shape != grad.shape:
        raise ValueError("parameter and grad must have the same shape")
    if momentum.shape != parameter.shape or second_moment.shape != parameter.shape:
        raise ValueError("student state tensors must match parameter shape")
    if parameter.numel() == 0:
        raise ValueError("parameter tensor must be non-empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    parameter = parameter.detach()
    grad = grad.detach()
    momentum = momentum.detach()
    second_moment = second_moment.detach()

    numel = parameter.numel()
    grad_rms = _block_rms(grad, block_size=block_size, eps=eps)
    momentum_rms = _block_rms(momentum, block_size=block_size, eps=eps)
    parameter_rms = _block_rms(parameter, block_size=block_size, eps=eps)

    flat_second_moment = second_moment.reshape(-1).clamp_min(0)
    blocks = math.ceil(numel / block_size)
    padded_numel = blocks * block_size
    if padded_numel != numel:
        flat_second_moment = torch.nn.functional.pad(
            flat_second_moment,
            (0, padded_numel - numel),
        )
    second_moment_blocks = flat_second_moment.reshape(blocks, block_size)
    counts = torch.full(
        (blocks,),
        block_size,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    if numel % block_size:
        counts[-1] = numel % block_size
    block_rms_denom = (second_moment_blocks.sum(dim=1) / counts).add(eps).sqrt()
    denom = _expand_blocks(block_rms_denom, numel=numel, block_size=block_size)
    base_update = -momentum.reshape(-1) / denom

    progress = min(max(step / total_steps, 0.0), 1.0)
    block_observations = torch.stack(
        (
            grad_rms.add(eps).log(),
            momentum_rms.add(eps).log(),
            parameter_rms.add(eps).log(),
            torch.full_like(grad_rms, progress),
        ),
        dim=-1,
    )
    element_observations = block_observations.repeat_interleave(block_size, dim=0)[:numel]
    return torch.cat((base_update.unsqueeze(-1), element_observations), dim=-1)


class BlockGainOptimizer(nn.Module):
    """Predict one bounded multiplicative gain for each block-RMS base update.

    With the default 4 -> 8 -> 8 -> 1 network this student has only 121 trainable
    parameters. Its expensive operations are block reductions; deployment then needs
    only a tiny shared MLP and an elementwise multiply on top of the stable base update.
    The final layer is zero-initialized, so the untrained model is exactly the base
    optimizer instead of a random optimizer policy.
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
        if features.ndim != 2 or features.shape[1] != len(BLOCK_GAIN_FEATURE_NAMES):
            raise ValueError("features must have shape [numel, 5]")
        base_update = features[:, 0]
        raw_log_gain = self.network(features[:, 1:]).squeeze(-1)
        gain = torch.exp(self.log_gain_limit * torch.tanh(raw_log_gain))
        return base_update * gain * self.output_scale

    @torch.no_grad()
    def set_output_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("output scale must be positive and finite")
        self.output_scale.fill_(scale)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
