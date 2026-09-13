from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor

from optdistil.multitensor.params import ParamCollection


class MultiTensorTask(Protocol):
    @property
    def parameter_shapes(self) -> list[torch.Size]: ...

    @property
    def sample_count(self) -> int: ...

    def loss(self, params: ParamCollection) -> Tensor: ...

    def loss_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor: ...

    def grad(self, params: ParamCollection) -> list[Tensor]: ...

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]: ...


def _validate_indices(sample_indices: Tensor, sample_count: int) -> Tensor:
    if sample_indices.ndim != 1 or sample_indices.numel() <= 0:
        raise ValueError("sample_indices must be a non-empty 1-D tensor")
    if sample_indices.dtype != torch.long:
        raise ValueError("sample_indices must use torch.long dtype")
    if int(sample_indices.min()) < 0 or int(sample_indices.max()) >= sample_count:
        raise ValueError("sample index out of range")
    return sample_indices


class TwoLayerMLPRegressionTask:
    """Tiny two-layer MLP regression with planted targets.

    parameters: W1 [hidden, input], b1 [hidden], W2 [output, hidden], b2 [output]
    prediction: W2 @ tanh(W1 @ x + b1) + b2
    loss: 0.5 / n * ||prediction - target||_F^2
    """

    def __init__(self, inputs: Tensor, target: Tensor, *, hidden_dim: int) -> None:
        if inputs.ndim != 2 or target.ndim != 2:
            raise ValueError("inputs and target must be matrices")
        if inputs.shape[1] != target.shape[1]:
            raise ValueError("inputs and target must share sample count")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.inputs = inputs.detach().clone()
        self.target = target.detach().clone()
        self.hidden_dim = hidden_dim
        self.input_dim = inputs.shape[0]
        self.output_dim = target.shape[0]
        self._parameter_shapes = [
            torch.Size((hidden_dim, self.input_dim)),
            torch.Size((hidden_dim,)),
            torch.Size((self.output_dim, hidden_dim)),
            torch.Size((self.output_dim,)),
        ]
        self.parameter_names = ("W1", "b1", "W2", "b2")
        self.parameter_roles = ("matrix", "vector", "matrix", "vector")

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return list(self._parameter_shapes)

    @property
    def sample_count(self) -> int:
        return self.inputs.shape[1]

    def _validate(self, params: ParamCollection) -> list[Tensor]:
        if len(params) != len(self._parameter_shapes):
            raise ValueError("parameter collection size mismatch")
        for tensor, shape in zip(params, self._parameter_shapes, strict=True):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"parameter shape mismatch: expected {shape}, got {tensor.shape}")
        return list(params)

    def _forward(
        self,
        params: Sequence[Tensor],
        inputs: Tensor,
    ) -> Tensor:
        w1, b1, w2, b2 = params
        hidden = torch.tanh(w1 @ inputs + b1.unsqueeze(1))
        return w2 @ hidden + b2.unsqueeze(1)

    def prediction(self, params: ParamCollection) -> Tensor:
        tensors = self._validate(params)
        return self._forward(tensors, self.inputs)

    def prediction_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        return self._forward(tensors, self.inputs.index_select(1, indices))

    def loss(self, params: ParamCollection) -> Tensor:
        residual = self.prediction(params) - self.target
        return 0.5 * residual.square().sum() / self.sample_count

    def loss_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        inputs = self.inputs.index_select(1, indices)
        target = self.target.index_select(1, indices)
        residual = self._forward(tensors, inputs) - target
        return 0.5 * residual.square().sum() / indices.numel()

    def _analytic_grads(
        self,
        params: Sequence[Tensor],
        inputs: Tensor,
        target: Tensor,
    ) -> list[Tensor]:
        w1, b1, w2, b2 = params
        n = inputs.shape[1]
        preact = w1 @ inputs + b1.unsqueeze(1)
        hidden = torch.tanh(preact)
        pred = w2 @ hidden + b2.unsqueeze(1)
        residual = pred - target
        # dL/dpred = residual / n
        g_w2 = residual @ hidden.mT / n
        g_b2 = residual.sum(dim=1) / n
        hidden_grad = (w2.mT @ residual) * (1.0 - hidden.square())
        g_w1 = hidden_grad @ inputs.mT / n
        g_b1 = hidden_grad.sum(dim=1) / n
        return [g_w1, g_b1, g_w2, g_b2]

    def grad(self, params: ParamCollection) -> list[Tensor]:
        tensors = self._validate(params)
        return self._analytic_grads(tensors, self.inputs, self.target)

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        inputs = self.inputs.index_select(1, indices)
        target = self.target.index_select(1, indices)
        return self._analytic_grads(tensors, inputs, target)


class ResidualMLPRegressionTask:
    """Tiny residual MLP regression with planted targets.

    parameters: W1, b1, W2, b2, W_skip
    prediction: W2 @ tanh(W1 @ x + b1) + b2 + W_skip @ x
    The skip path changes gradient geometry relative to the plain two-layer MLP.
    """

    def __init__(self, inputs: Tensor, target: Tensor, *, hidden_dim: int) -> None:
        if inputs.ndim != 2 or target.ndim != 2:
            raise ValueError("inputs and target must be matrices")
        if inputs.shape[1] != target.shape[1]:
            raise ValueError("inputs and target must share sample count")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.inputs = inputs.detach().clone()
        self.target = target.detach().clone()
        self.hidden_dim = hidden_dim
        self.input_dim = inputs.shape[0]
        self.output_dim = target.shape[0]
        self._parameter_shapes = [
            torch.Size((hidden_dim, self.input_dim)),
            torch.Size((hidden_dim,)),
            torch.Size((self.output_dim, hidden_dim)),
            torch.Size((self.output_dim,)),
            torch.Size((self.output_dim, self.input_dim)),
        ]
        self.parameter_names = ("W1", "b1", "W2", "b2", "W_skip")
        self.parameter_roles = ("matrix", "vector", "matrix", "vector", "matrix")

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return list(self._parameter_shapes)

    @property
    def sample_count(self) -> int:
        return self.inputs.shape[1]

    def _validate(self, params: ParamCollection) -> list[Tensor]:
        if len(params) != len(self._parameter_shapes):
            raise ValueError("parameter collection size mismatch")
        for tensor, shape in zip(params, self._parameter_shapes, strict=True):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"parameter shape mismatch: expected {shape}, got {tensor.shape}")
        return list(params)

    def _forward(self, params: Sequence[Tensor], inputs: Tensor) -> Tensor:
        w1, b1, w2, b2, w_skip = params
        hidden = torch.tanh(w1 @ inputs + b1.unsqueeze(1))
        return w2 @ hidden + b2.unsqueeze(1) + w_skip @ inputs

    def prediction(self, params: ParamCollection) -> Tensor:
        tensors = self._validate(params)
        return self._forward(tensors, self.inputs)

    def prediction_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        return self._forward(tensors, self.inputs.index_select(1, indices))

    def loss(self, params: ParamCollection) -> Tensor:
        residual = self.prediction(params) - self.target
        return 0.5 * residual.square().sum() / self.sample_count

    def loss_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        inputs = self.inputs.index_select(1, indices)
        target = self.target.index_select(1, indices)
        residual = self._forward(tensors, inputs) - target
        return 0.5 * residual.square().sum() / indices.numel()

    def _analytic_grads(
        self,
        params: Sequence[Tensor],
        inputs: Tensor,
        target: Tensor,
    ) -> list[Tensor]:
        w1, b1, w2, b2, w_skip = params
        n = inputs.shape[1]
        preact = w1 @ inputs + b1.unsqueeze(1)
        hidden = torch.tanh(preact)
        pred = w2 @ hidden + b2.unsqueeze(1) + w_skip @ inputs
        residual = pred - target
        g_w2 = residual @ hidden.mT / n
        g_b2 = residual.sum(dim=1) / n
        g_w_skip = residual @ inputs.mT / n
        hidden_grad = (w2.mT @ residual) * (1.0 - hidden.square())
        g_w1 = hidden_grad @ inputs.mT / n
        g_b1 = hidden_grad.sum(dim=1) / n
        return [g_w1, g_b1, g_w2, g_b2, g_w_skip]

    def grad(self, params: ParamCollection) -> list[Tensor]:
        tensors = self._validate(params)
        return self._analytic_grads(tensors, self.inputs, self.target)

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        inputs = self.inputs.index_select(1, indices)
        target = self.target.index_select(1, indices)
        return self._analytic_grads(tensors, inputs, target)


def _make_conditioned_inputs(
    generator: torch.Generator,
    *,
    input_dim: int,
    samples: int,
    input_condition: float,
) -> Tensor:
    rotation_raw = torch.randn((input_dim, input_dim), generator=generator)
    rotation, _ = torch.linalg.qr(rotation_raw)
    singular_values = torch.logspace(0.0, 0.5 * math.log10(input_condition), input_dim)
    transform = rotation @ torch.diag(singular_values) @ rotation.mT
    inputs = transform @ torch.randn((input_dim, samples), generator=generator)
    return inputs / inputs.square().mean().sqrt().clamp_min(1e-8)


def make_two_layer_mlp(
    seed: int,
    *,
    input_dim: int = 8,
    hidden_dim: int = 8,
    output_dim: int = 4,
    samples: int = 64,
    input_condition: float = 30.0,
    planted_scale: float = 0.7,
    initial_scale: float = 0.2,
    target_noise: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ParamCollection, TwoLayerMLPRegressionTask]:
    if min(input_dim, hidden_dim, output_dim, samples) <= 0:
        raise ValueError("all dimensions and sample count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _make_conditioned_inputs(
        generator,
        input_dim=input_dim,
        samples=samples,
        input_condition=input_condition,
    )

    planted_w1 = (
        torch.randn((hidden_dim, input_dim), generator=generator)
        * planted_scale
        / math.sqrt(input_dim)
    )
    planted_b1 = torch.randn((hidden_dim,), generator=generator) * planted_scale * 0.1
    planted_w2 = (
        torch.randn((output_dim, hidden_dim), generator=generator)
        * planted_scale
        / math.sqrt(hidden_dim)
    )
    planted_b2 = torch.randn((output_dim,), generator=generator) * planted_scale * 0.1

    hidden = torch.tanh(planted_w1 @ inputs + planted_b1.unsqueeze(1))
    target = planted_w2 @ hidden + planted_b2.unsqueeze(1)
    if target_noise > 0.0:
        target = target + target_noise * torch.randn(target.shape, generator=generator)

    initial = ParamCollection(
        [
            torch.randn((hidden_dim, input_dim), generator=generator)
            * initial_scale
            / math.sqrt(input_dim),
            torch.randn((hidden_dim,), generator=generator) * initial_scale * 0.1,
            torch.randn((output_dim, hidden_dim), generator=generator)
            * initial_scale
            / math.sqrt(hidden_dim),
            torch.randn((output_dim,), generator=generator) * initial_scale * 0.1,
        ]
    )
    device = torch.device(device)
    initial = initial.to(device=device, dtype=dtype)
    task = TwoLayerMLPRegressionTask(
        inputs.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
        hidden_dim=hidden_dim,
    )
    return initial, task


def make_residual_mlp(
    seed: int,
    *,
    input_dim: int = 8,
    hidden_dim: int = 8,
    output_dim: int = 4,
    samples: int = 64,
    input_condition: float = 30.0,
    planted_scale: float = 0.7,
    initial_scale: float = 0.2,
    skip_scale: float = 0.3,
    target_noise: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ParamCollection, ResidualMLPRegressionTask]:
    if min(input_dim, hidden_dim, output_dim, samples) <= 0:
        raise ValueError("all dimensions and sample count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _make_conditioned_inputs(
        generator,
        input_dim=input_dim,
        samples=samples,
        input_condition=input_condition,
    )

    planted_w1 = (
        torch.randn((hidden_dim, input_dim), generator=generator)
        * planted_scale
        / math.sqrt(input_dim)
    )
    planted_b1 = torch.randn((hidden_dim,), generator=generator) * planted_scale * 0.1
    planted_w2 = (
        torch.randn((output_dim, hidden_dim), generator=generator)
        * planted_scale
        / math.sqrt(hidden_dim)
    )
    planted_b2 = torch.randn((output_dim,), generator=generator) * planted_scale * 0.1
    planted_skip = (
        torch.randn((output_dim, input_dim), generator=generator)
        * skip_scale
        / math.sqrt(input_dim)
    )

    hidden = torch.tanh(planted_w1 @ inputs + planted_b1.unsqueeze(1))
    target = planted_w2 @ hidden + planted_b2.unsqueeze(1) + planted_skip @ inputs
    if target_noise > 0.0:
        target = target + target_noise * torch.randn(target.shape, generator=generator)

    initial = ParamCollection(
        [
            torch.randn((hidden_dim, input_dim), generator=generator)
            * initial_scale
            / math.sqrt(input_dim),
            torch.randn((hidden_dim,), generator=generator) * initial_scale * 0.1,
            torch.randn((output_dim, hidden_dim), generator=generator)
            * initial_scale
            / math.sqrt(hidden_dim),
            torch.randn((output_dim,), generator=generator) * initial_scale * 0.1,
            torch.randn((output_dim, input_dim), generator=generator)
            * initial_scale
            / math.sqrt(input_dim),
        ]
    )
    device = torch.device(device)
    initial = initial.to(device=device, dtype=dtype)
    task = ResidualMLPRegressionTask(
        inputs.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
        hidden_dim=hidden_dim,
    )
    return initial, task


def make_task(
    architecture: str,
    seed: int,
    *,
    width: int = 8,
    samples: int = 64,
    input_condition: float = 30.0,
    device: torch.device | str = "cpu",
) -> tuple[ParamCollection, MultiTensorTask]:
    """Create a multi-tensor task by architecture name.

    ``width`` maps to input/hidden dimensions. Output dim is max(2, width // 2).
    """
    output_dim = max(2, width // 2)
    if architecture == "two_layer":
        return make_two_layer_mlp(
            seed,
            input_dim=width,
            hidden_dim=width,
            output_dim=output_dim,
            samples=samples,
            input_condition=input_condition,
            device=device,
        )
    if architecture == "residual":
        return make_residual_mlp(
            seed,
            input_dim=width,
            hidden_dim=width,
            output_dim=output_dim,
            samples=samples,
            input_condition=input_condition,
            device=device,
        )
    raise ValueError(f"unknown architecture: {architecture}")
