from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from optdistil.teachers.meta_attention import _ContextualBlock
from optdistil.teachers.meta_mlp import MetaMLPTeacher, _rms


@dataclass(frozen=True, slots=True)
class MetaRecurrentAttentionState:
    momentum: Tensor
    second_moment: Tensor
    memory: Tensor
    step_number: int


class _RecurrentContextNetwork(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        d_model: int,
        num_heads: int,
        depth: int,
        ff_multiplier: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_projection = nn.Linear(feature_dim + 2, d_model)
        self.recurrence = nn.GRUCell(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                _ContextualBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coordinates(
        shape: torch.Size,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if len(shape) != 2:
            raise ValueError("recurrent contextual teacher requires a 2-D parameter tensor")
        rows, cols = shape
        row = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
        col = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
        row_grid, col_grid = torch.meshgrid(row, col, indexing="ij")
        return torch.stack((row_grid, col_grid), dim=-1).reshape(rows * cols, 2)

    def forward(
        self,
        features: Tensor,
        memory: Tensor,
        *,
        matrix_shape: torch.Size,
    ) -> tuple[Tensor, Tensor]:
        coordinates = self._coordinates(
            matrix_shape,
            device=features.device,
            dtype=features.dtype,
        )
        inputs = self.input_projection(torch.cat((features, coordinates), dim=-1))
        if memory.shape != inputs.shape:
            raise ValueError("recurrent memory shape must match the matrix token shape")
        tokens = self.recurrence(inputs, memory).unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        next_memory = tokens.squeeze(0)
        raw = self.output(self.output_norm(tokens)).squeeze(0)
        return raw, next_memory


class MetaRecurrentAttentionTeacher(MetaMLPTeacher):
    """Heavy learned-optimizer teacher with persistent cross-coordinate memory.

    The teacher keeps a hidden vector for every matrix element. Current optimizer features
    are fused with the previous hidden state through a GRU, then self-attention exchanges
    information across all coordinates. The resulting contextual representation is carried
    into the next optimization step.

    As with the other meta teachers, the final policy head is zero-initialized. The first
    policy therefore exactly matches the same bias-corrected Adam-like base update, while
    meta-training can learn to exploit privileged temporal and spatial context later.
    """

    def __init__(
        self,
        *,
        d_model: int = 28,
        num_heads: int = 4,
        depth: int = 2,
        ff_multiplier: int = 2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        horizon: int = 16,
        initial_step_scale: float = 0.1,
        max_step_scale: float = 1.0,
        log_gain_limit: float = 1.0,
        residual_limit: float = 1.5,
    ) -> None:
        if d_model <= 0 or num_heads <= 0 or depth <= 0 or ff_multiplier <= 0:
            raise ValueError("recurrent attention dimensions and depth must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        super().__init__(
            hidden_dim=d_model,
            hidden_layers=1,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            horizon=horizon,
            initial_step_scale=initial_step_scale,
            max_step_scale=max_step_scale,
            log_gain_limit=log_gain_limit,
            residual_limit=residual_limit,
        )
        feature_dim = self.network[0].in_features
        self.network = _RecurrentContextNetwork(
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            depth=depth,
            ff_multiplier=ff_multiplier,
        )
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self._state: MetaRecurrentAttentionState | None = None

    def initial_state(self, parameter: Tensor) -> MetaRecurrentAttentionState:
        return MetaRecurrentAttentionState(
            momentum=torch.zeros_like(parameter),
            second_moment=torch.zeros_like(parameter),
            memory=torch.zeros(
                (parameter.numel(), self.d_model),
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            step_number=0,
        )

    def reset(self) -> None:
        self._state = None

    def functional_step(
        self,
        parameter: Tensor,
        grad: Tensor,
        state: MetaRecurrentAttentionState,
    ) -> tuple[Tensor, MetaRecurrentAttentionState]:
        if state.momentum.shape != parameter.shape or state.second_moment.shape != parameter.shape:
            raise ValueError("optimizer state shape must match parameter")
        expected_memory_shape = (parameter.numel(), self.d_model)
        if tuple(state.memory.shape) != expected_memory_shape:
            raise ValueError("optimizer memory shape must match parameter tokens")

        step_number = state.step_number + 1
        momentum = self.beta1 * state.momentum + (1.0 - self.beta1) * grad
        second_moment = self.beta2 * state.second_moment + (1.0 - self.beta2) * grad.square()
        features, base_update = self._features(
            parameter,
            grad,
            momentum,
            second_moment,
            step_number=step_number,
        )
        raw, memory = self.network(
            features,
            state.memory,
            matrix_shape=parameter.shape,
        )
        raw = raw.reshape(*parameter.shape, 2)
        log_gain = self.log_gain_limit * torch.tanh(raw[..., 0])
        residual = self.residual_limit * torch.tanh(raw[..., 1])
        base_rms = _rms(base_update, eps=self.eps)
        direction = base_update * torch.exp(log_gain) + base_rms * residual
        update = self.step_scale * direction
        return update, MetaRecurrentAttentionState(
            momentum=momentum,
            second_moment=second_moment,
            memory=memory,
            step_number=step_number,
        )

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if self._state is None:
            self._state = self.initial_state(parameter)
        update, state = self.functional_step(parameter, grad, self._state)
        self._state = MetaRecurrentAttentionState(
            momentum=state.momentum.detach(),
            second_moment=state.second_moment.detach(),
            memory=state.memory.detach(),
            step_number=state.step_number,
        )
        return update.detach()
