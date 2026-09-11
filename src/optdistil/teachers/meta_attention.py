from __future__ import annotations

import torch
from torch import Tensor, nn

from optdistil.teachers.meta_mlp import MetaMLPTeacher, MetaTeacherState, _rms


class _ContextualBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, ff_multiplier: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(d_model)
        hidden_dim = ff_multiplier * d_model
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feed_forward(self.ff_norm(tokens))


class _ContextualUpdateNetwork(nn.Module):
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
        # Two continuous coordinates expose matrix position without fixing a maximum shape.
        self.input_projection = nn.Linear(feature_dim + 2, d_model)
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
        # The complete contextual branch initially contributes exactly zero, so the
        # optimizer starts from the same Adam-like base as MetaMLPTeacher.
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
            raise ValueError("contextual teacher requires a 2-D parameter tensor")
        rows, cols = shape
        row = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
        col = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
        row_grid, col_grid = torch.meshgrid(row, col, indexing="ij")
        return torch.stack((row_grid, col_grid), dim=-1).reshape(rows * cols, 2)

    def forward(self, features: Tensor, *, matrix_shape: torch.Size) -> Tensor:
        coordinates = self._coordinates(
            matrix_shape,
            device=features.device,
            dtype=features.dtype,
        )
        tokens = self.input_projection(torch.cat((features, coordinates), dim=-1)).unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output(self.output_norm(tokens)).squeeze(0)


class MetaAttentionTeacher(MetaMLPTeacher):
    """Meta-trained optimizer whose expensive teacher policy sees the whole matrix.

    Every matrix element is represented as one token. The token starts from the same
    optimizer features used by ``MetaMLPTeacher`` plus continuous row/column coordinates,
    then self-attention allows update decisions to depend on all other coordinates.

    The deployment student never sees these attention activations. This intentionally
    creates a privileged-compute teacher for testing whether global optimizer reasoning can
    be compressed into the fixed tiny elementwise student.
    """

    def __init__(
        self,
        *,
        d_model: int = 32,
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
            raise ValueError("attention dimensions, depth, and FF multiplier must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        # Initialize the shared state/features/base-update machinery, then replace the
        # elementwise MLP with the contextual network. Replaced modules are deregistered by
        # nn.Module, so parameter_count includes only the attention policy and step scale.
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
        self.network = _ContextualUpdateNetwork(
            feature_dim=feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            depth=depth,
            ff_multiplier=ff_multiplier,
        )
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.depth = int(depth)

    def functional_step(
        self,
        parameter: Tensor,
        grad: Tensor,
        state: MetaTeacherState,
    ) -> tuple[Tensor, MetaTeacherState]:
        if state.momentum.shape != parameter.shape or state.second_moment.shape != parameter.shape:
            raise ValueError("optimizer state shape must match parameter")

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
        raw = self.network(features, matrix_shape=parameter.shape).reshape(*parameter.shape, 2)
        log_gain = self.log_gain_limit * torch.tanh(raw[..., 0])
        residual = self.residual_limit * torch.tanh(raw[..., 1])
        base_rms = _rms(base_update, eps=self.eps)
        direction = base_update * torch.exp(log_gain) + base_rms * residual
        update = self.step_scale * direction
        return update, MetaTeacherState(momentum, second_moment, step_number)
