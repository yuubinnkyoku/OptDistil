from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

META_TEACHER_FEATURE_NAMES = (
    "base_update",
    "normalized_grad",
    "normalized_momentum",
    "normalized_rms",
    "normalized_parameter",
    "row_grad_rms",
    "col_grad_rms",
    "row_momentum_rms",
    "col_momentum_rms",
    "log_global_grad_rms",
    "log_parameter_rms",
    "progress",
)


@dataclass(frozen=True, slots=True)
class MetaTeacherState:
    momentum: Tensor
    second_moment: Tensor
    step_number: int


def _rms(values: Tensor, *, eps: float) -> Tensor:
    return values.square().mean().add(eps).sqrt()


def _inverse_sigmoid(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class MetaMLPTeacher(nn.Module):
    """Differentiable learned optimizer used as a scalable distillation teacher.

    The teacher starts exactly from a bias-corrected Adam-like update. A shared MLP then
    predicts a bounded multiplicative gain and an additive residual for every matrix
    element. The residual makes the teacher capable of changing update direction rather
    than merely learning a learning-rate schedule.

    ``functional_step`` is differentiable through both optimizer state and inner-task
    parameters, so the teacher can be outer-trained directly on rollout loss. ``step`` is
    the stateful no-grad adapter used by the normal trajectory/evaluation pipeline.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        hidden_layers: int = 2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        horizon: int = 16,
        initial_step_scale: float = 0.1,
        max_step_scale: float = 1.0,
        log_gain_limit: float = 1.0,
        residual_limit: float = 1.5,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("EMA betas must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 < initial_step_scale < max_step_scale:
            raise ValueError("initial_step_scale must lie inside (0, max_step_scale)")
        if log_gain_limit <= 0.0 or residual_limit <= 0.0:
            raise ValueError("gain and residual limits must be positive")

        layers: list[nn.Module] = []
        in_dim = len(META_TEACHER_FEATURE_NAMES)
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.Tanh()))
            in_dim = hidden_dim
        output = nn.Linear(in_dim, 2)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

        probability = initial_step_scale / max_step_scale
        self.step_scale_logit = nn.Parameter(torch.tensor(_inverse_sigmoid(probability)))
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.horizon = int(horizon)
        self.max_step_scale = float(max_step_scale)
        self.log_gain_limit = float(log_gain_limit)
        self.residual_limit = float(residual_limit)

        self._state: MetaTeacherState | None = None

    @property
    def step_scale(self) -> Tensor:
        return self.max_step_scale * torch.sigmoid(self.step_scale_logit)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(self, parameter: Tensor) -> MetaTeacherState:
        return MetaTeacherState(
            momentum=torch.zeros_like(parameter),
            second_moment=torch.zeros_like(parameter),
            step_number=0,
        )

    def reset(self) -> None:
        self._state = None

    def _features(
        self,
        parameter: Tensor,
        grad: Tensor,
        momentum: Tensor,
        second_moment: Tensor,
        *,
        step_number: int,
    ) -> tuple[Tensor, Tensor]:
        if parameter.ndim != 2:
            raise ValueError("MetaMLPTeacher currently requires a 2-D parameter tensor")
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")

        bias1 = 1.0 - self.beta1**step_number
        bias2 = 1.0 - self.beta2**step_number
        m_hat = momentum / bias1
        v_hat = second_moment / bias2
        rms = v_hat.clamp_min(0).add(self.eps).sqrt()
        base_update = -m_hat / (rms + self.eps)

        grad_rms = _rms(grad, eps=self.eps)
        momentum_rms = _rms(m_hat, eps=self.eps)
        rms_rms = _rms(rms, eps=self.eps)
        parameter_rms = _rms(parameter, eps=self.eps)
        row_grad_rms = grad.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        col_grad_rms = grad.square().mean(dim=0, keepdim=True).add(self.eps).sqrt()
        row_momentum_rms = m_hat.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        col_momentum_rms = m_hat.square().mean(dim=0, keepdim=True).add(self.eps).sqrt()
        progress = min(max(step_number / self.horizon, 0.0), 1.0)

        features = torch.stack(
            (
                base_update,
                grad / grad_rms,
                m_hat / momentum_rms,
                rms / rms_rms,
                parameter / parameter_rms,
                row_grad_rms.expand_as(grad) / grad_rms,
                col_grad_rms.expand_as(grad) / grad_rms,
                row_momentum_rms.expand_as(m_hat) / momentum_rms,
                col_momentum_rms.expand_as(m_hat) / momentum_rms,
                torch.log(grad_rms).expand_as(grad),
                torch.log(parameter_rms).expand_as(parameter),
                torch.full_like(parameter, progress),
            ),
            dim=-1,
        )
        return features.reshape(-1, len(META_TEACHER_FEATURE_NAMES)), base_update

    def functional_step(
        self,
        parameter: Tensor,
        grad: Tensor,
        state: MetaTeacherState,
    ) -> tuple[Tensor, MetaTeacherState]:
        """Return a differentiable update and next optimizer state."""
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
        raw = self.network(features).reshape(*parameter.shape, 2)
        log_gain = self.log_gain_limit * torch.tanh(raw[..., 0])
        residual = self.residual_limit * torch.tanh(raw[..., 1])
        base_rms = _rms(base_update, eps=self.eps)
        direction = base_update * torch.exp(log_gain) + base_rms * residual
        update = self.step_scale * direction
        next_state = MetaTeacherState(momentum, second_moment, step_number)
        return update, next_state

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        """Stateful adapter used by the standard teacher rollout API."""
        if self._state is None:
            self._state = self.initial_state(parameter)
        update, state = self.functional_step(parameter, grad, self._state)
        self._state = MetaTeacherState(
            state.momentum.detach(),
            state.second_moment.detach(),
            state.step_number,
        )
        return update.detach()
