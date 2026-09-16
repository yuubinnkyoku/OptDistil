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


class ThreeLayerMLPRegressionTask:
    """Three-layer tanh MLP regression with planted targets.

    parameters: W1, b1, W2, b2, W3, b3
    prediction: W3 @ tanh(W2 @ tanh(W1 @ x + b1) + b2) + b3
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
            torch.Size((hidden_dim, hidden_dim)),
            torch.Size((hidden_dim,)),
            torch.Size((self.output_dim, hidden_dim)),
            torch.Size((self.output_dim,)),
        ]
        self.parameter_names = ("W1", "b1", "W2", "b2", "W3", "b3")
        self.parameter_roles = ("matrix", "vector", "matrix", "vector", "matrix", "vector")

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
        w1, b1, w2, b2, w3, b3 = params
        h1 = torch.tanh(w1 @ inputs + b1.unsqueeze(1))
        h2 = torch.tanh(w2 @ h1 + b2.unsqueeze(1))
        return w3 @ h2 + b3.unsqueeze(1)

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
        w1, b1, w2, b2, w3, b3 = params
        n = inputs.shape[1]
        preact1 = w1 @ inputs + b1.unsqueeze(1)
        h1 = torch.tanh(preact1)
        preact2 = w2 @ h1 + b2.unsqueeze(1)
        h2 = torch.tanh(preact2)
        pred = w3 @ h2 + b3.unsqueeze(1)
        residual = pred - target
        g_w3 = residual @ h2.mT / n
        g_b3 = residual.sum(dim=1) / n
        h2_grad = (w3.mT @ residual) * (1.0 - h2.square())
        g_w2 = h2_grad @ h1.mT / n
        g_b2 = h2_grad.sum(dim=1) / n
        h1_grad = (w2.mT @ h2_grad) * (1.0 - h1.square())
        g_w1 = h1_grad @ inputs.mT / n
        g_b1 = h1_grad.sum(dim=1) / n
        return [g_w1, g_b1, g_w2, g_b2, g_w3, g_b3]

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


def make_three_layer_mlp(
    seed: int,
    *,
    input_dim: int = 8,
    hidden_dim: int = 8,
    output_dim: int = 4,
    samples: int = 64,
    input_condition: float = 30.0,
    planted_scale: float = 0.7,
    initial_scale: float = 0.2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ParamCollection, ThreeLayerMLPRegressionTask]:
    if min(input_dim, hidden_dim, output_dim, samples) <= 0:
        raise ValueError("all dimensions and sample count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _make_conditioned_inputs(
        generator,
        input_dim=input_dim,
        samples=samples,
        input_condition=input_condition,
    )

    def _mat(out_dim: int, in_dim: int, scale: float) -> Tensor:
        return torch.randn((out_dim, in_dim), generator=generator) * scale / math.sqrt(in_dim)

    def _vec(dim: int, scale: float) -> Tensor:
        return torch.randn((dim,), generator=generator) * scale * 0.1

    planted = [
        _mat(hidden_dim, input_dim, planted_scale),
        _vec(hidden_dim, planted_scale),
        _mat(hidden_dim, hidden_dim, planted_scale),
        _vec(hidden_dim, planted_scale),
        _mat(output_dim, hidden_dim, planted_scale),
        _vec(output_dim, planted_scale),
    ]
    h1 = torch.tanh(planted[0] @ inputs + planted[1].unsqueeze(1))
    h2 = torch.tanh(planted[2] @ h1 + planted[3].unsqueeze(1))
    target = planted[4] @ h2 + planted[5].unsqueeze(1)

    initial = ParamCollection(
        [
            _mat(hidden_dim, input_dim, initial_scale),
            _vec(hidden_dim, initial_scale),
            _mat(hidden_dim, hidden_dim, initial_scale),
            _vec(hidden_dim, initial_scale),
            _mat(output_dim, hidden_dim, initial_scale),
            _vec(output_dim, initial_scale),
        ]
    )
    device = torch.device(device)
    initial = initial.to(device=device, dtype=dtype)
    task = ThreeLayerMLPRegressionTask(
        inputs.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
        hidden_dim=hidden_dim,
    )
    return initial, task


class AllMatrixRegressionTask:
    """Chain of matrices only: y = W3 @ tanh(W2 @ tanh(W1 @ x)).

    No bias tensors, so matrix/vector role structure cannot explain LR gains.
    Shapes differ by construction so numel/fan-in partitions remain testable.
    """

    def __init__(self, inputs: Tensor, target: Tensor, shapes: list[torch.Size]) -> None:
        if len(shapes) != 3:
            raise ValueError("expected three matrix shapes")
        self.inputs = inputs.detach().clone()
        self.target = target.detach().clone()
        self._parameter_shapes = list(shapes)
        self.parameter_names = ("W1", "W2", "W3")
        self.parameter_roles = ("matrix", "matrix", "matrix")

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return list(self._parameter_shapes)

    @property
    def sample_count(self) -> int:
        return self.inputs.shape[1]

    def _validate(self, params: ParamCollection) -> list[Tensor]:
        if len(params) != 3:
            raise ValueError("parameter collection size mismatch")
        for tensor, shape in zip(params, self._parameter_shapes, strict=True):
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"parameter shape mismatch: expected {shape}, got {tensor.shape}")
        return list(params)

    def _forward(self, params: Sequence[Tensor], inputs: Tensor) -> Tensor:
        w1, w2, w3 = params
        h1 = torch.tanh(w1 @ inputs)
        h2 = torch.tanh(w2 @ h1)
        return w3 @ h2

    def prediction(self, params: ParamCollection) -> Tensor:
        return self._forward(self._validate(params), self.inputs)

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
        self, params: Sequence[Tensor], inputs: Tensor, target: Tensor
    ) -> list[Tensor]:
        w1, w2, w3 = params
        n = inputs.shape[1]
        h1 = torch.tanh(w1 @ inputs)
        h2 = torch.tanh(w2 @ h1)
        pred = w3 @ h2
        residual = pred - target
        g_w3 = residual @ h2.mT / n
        h2_grad = (w3.mT @ residual) * (1.0 - h2.square())
        g_w2 = h2_grad @ h1.mT / n
        h1_grad = (w2.mT @ h2_grad) * (1.0 - h1.square())
        g_w1 = h1_grad @ inputs.mT / n
        return [g_w1, g_w2, g_w3]

    def grad(self, params: ParamCollection) -> list[Tensor]:
        return self._analytic_grads(self._validate(params), self.inputs, self.target)

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        return self._analytic_grads(
            tensors, self.inputs.index_select(1, indices), self.target.index_select(1, indices)
        )


class AllVectorRegressionTask:
    """Linear model on concatenated affine features with vector-only parameters.

    parameters: three vectors of different lengths, used as
    y = W(v3) @ x + diag-ish path through tanh(W(v1)@x) mixed by W(v2).
    Simpler: y = v3 outer feature map of tanh(diag(v1) @ x) + v2 term.
    Implemented as three matrix-free vector interactions via broadcasting.
    """

    def __init__(self, inputs: Tensor, target: Tensor, dims: Sequence[int]) -> None:
        if len(dims) != 3:
            raise ValueError("expected three vector dims")
        self.inputs = inputs.detach().clone()
        self.target = target.detach().clone()
        self._parameter_shapes = [torch.Size((d,)) for d in dims]
        self.parameter_names = ("v1", "v2", "v3")
        self.parameter_roles = ("vector", "vector", "vector")

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return list(self._parameter_shapes)

    @property
    def sample_count(self) -> int:
        return self.inputs.shape[1]

    def _validate(self, params: ParamCollection) -> list[Tensor]:
        if len(params) != 3:
            raise ValueError("parameter collection size mismatch")
        for tensor, shape in zip(params, self._parameter_shapes, strict=True):
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"parameter shape mismatch: expected {shape}, got {tensor.shape}")
        return list(params)

    def _forward(self, params: Sequence[Tensor], inputs: Tensor) -> Tensor:
        v1, v2, v3 = params
        # v1 scales input channels (length = input dim)
        scaled = v1.unsqueeze(1) * inputs
        h = torch.tanh(scaled)
        # v2 scales hidden channels (length = input dim)
        mixed = v2.unsqueeze(1) * h
        # v3 maps to output via fixed random-free reduction: mean over a partition
        # Use first output_dim features mixed by v3.
        out_dim = v3.shape[0]
        if mixed.shape[0] < out_dim:
            raise ValueError("input dim must be >= output dim")
        return mixed[:out_dim] + v3.unsqueeze(1)

    def prediction(self, params: ParamCollection) -> Tensor:
        return self._forward(self._validate(params), self.inputs)

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
        self, params: Sequence[Tensor], inputs: Tensor, target: Tensor
    ) -> list[Tensor]:
        v1, v2, v3 = params
        n = inputs.shape[1]
        scaled = v1.unsqueeze(1) * inputs
        h = torch.tanh(scaled)
        mixed = v2.unsqueeze(1) * h
        out_dim = v3.shape[0]
        pred = mixed[:out_dim] + v3.unsqueeze(1)
        residual = pred - target
        g_v3 = residual.sum(dim=1) / n
        g_v2 = torch.zeros_like(v2)
        g_v2[:out_dim] = (residual * h[:out_dim]).sum(dim=1) / n
        g_v1 = torch.zeros_like(v1)
        factor = residual * v2[:out_dim].unsqueeze(1) * (1.0 - h[:out_dim].square())
        g_v1[:out_dim] = (factor * inputs[:out_dim]).sum(dim=1) / n
        return [g_v1, g_v2, g_v3]

    def grad(self, params: ParamCollection) -> list[Tensor]:
        return self._analytic_grads(self._validate(params), self.inputs, self.target)

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        tensors = self._validate(params)
        indices = _validate_indices(sample_indices, self.sample_count).to(self.inputs.device)
        return self._analytic_grads(
            tensors, self.inputs.index_select(1, indices), self.target.index_select(1, indices)
        )


def make_all_matrix(
    seed: int,
    *,
    input_dim: int = 8,
    samples: int = 64,
    input_condition: float = 30.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ParamCollection, AllMatrixRegressionTask]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _make_conditioned_inputs(
        generator, input_dim=input_dim, samples=samples, input_condition=input_condition
    )
    # Deliberately unequal shapes.
    h1, h2, out_dim = max(2, input_dim // 2), max(2, input_dim // 4), max(2, input_dim // 2)
    shapes = [
        torch.Size((h1, input_dim)),
        torch.Size((h2, h1)),
        torch.Size((out_dim, h2)),
    ]
    planted = [
        torch.randn(tuple(shape), generator=generator) * 0.7 / math.sqrt(shape[1])
        for shape in shapes
    ]
    h = torch.tanh(planted[0] @ inputs)
    h = torch.tanh(planted[1] @ h)
    target = planted[2] @ h
    initial = ParamCollection(
        [
            torch.randn(tuple(shape), generator=generator) * 0.2 / math.sqrt(shape[1])
            for shape in shapes
        ]
    )
    device = torch.device(device)
    task = AllMatrixRegressionTask(
        inputs.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
        shapes=shapes,
    )
    return initial.to(device=device, dtype=dtype), task


def make_all_vector(
    seed: int,
    *,
    input_dim: int = 8,
    samples: int = 64,
    input_condition: float = 30.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ParamCollection, AllVectorRegressionTask]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = _make_conditioned_inputs(
        generator, input_dim=input_dim, samples=samples, input_condition=input_condition
    )
    out_dim = max(2, input_dim // 2)
    dims = [input_dim, input_dim, out_dim]
    planted_v1 = torch.randn((input_dim,), generator=generator) * 0.7
    planted_v2 = torch.randn((input_dim,), generator=generator) * 0.7
    planted_v3 = torch.randn((out_dim,), generator=generator) * 0.1
    scaled = planted_v1.unsqueeze(1) * inputs
    h = torch.tanh(scaled)
    mixed = planted_v2.unsqueeze(1) * h
    target = mixed[:out_dim] + planted_v3.unsqueeze(1)
    initial = ParamCollection(
        [
            torch.randn((input_dim,), generator=generator) * 0.2,
            torch.randn((input_dim,), generator=generator) * 0.2,
            torch.randn((out_dim,), generator=generator) * 0.05,
        ]
    )
    device = torch.device(device)
    task = AllVectorRegressionTask(
        inputs.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
        dims=dims,
    )
    return initial.to(device=device, dtype=dtype), task


class IsoShapeSpectrumTask:
    """Four same-shape matrices with distinct least-squares curvatures.

    Loss: (1/2n) sum_i ||A_i @ vec(θ_i) - b_i||² with A_i having prescribed
    condition numbers. Kills rank/numel confounds so only curvature can explain
    heterogeneous LRs.
    """

    def __init__(
        self,
        a_ops: list[Tensor],
        targets: list[Tensor],
        shape: torch.Size,
    ) -> None:
        if len(a_ops) != len(targets):
            raise ValueError("a_ops and targets must align")
        self.a_ops = [a.detach().clone() for a in a_ops]
        self.targets = [t.detach().clone() for t in targets]
        self._shape = shape
        self._n = self.a_ops[0].shape[0]
        self.parameter_names = tuple(f"P{i}" for i in range(len(a_ops)))
        self.parameter_roles = tuple("matrix" for _ in a_ops)

    @property
    def parameter_shapes(self) -> list[torch.Size]:
        return [self._shape for _ in self.a_ops]

    @property
    def sample_count(self) -> int:
        return self._n

    def _validate(self, params: ParamCollection) -> list[Tensor]:
        if len(params) != len(self.a_ops):
            raise ValueError("parameter collection size mismatch")
        for tensor in params:
            if tuple(tensor.shape) != tuple(self._shape):
                raise ValueError(f"parameter shape mismatch: expected {self._shape}")
        return list(params)

    def _loss_and_grads(
        self, params: Sequence[Tensor], sample_indices: Tensor | None
    ) -> tuple[Tensor, list[Tensor]]:
        grads: list[Tensor] = []
        total = torch.zeros((), dtype=torch.float32)
        for param, a_op, target in zip(params, self.a_ops, self.targets, strict=True):
            theta = param.reshape(-1)
            if sample_indices is None:
                a = a_op
                b = target
            else:
                a = a_op.index_select(0, sample_indices)
                b = target.index_select(0, sample_indices)
            residual = a @ theta - b
            n = a.shape[0]
            total = total + 0.5 * residual.square().sum() / n
            g = (a.mT @ residual) / n
            grads.append(g.reshape(param.shape).to(param.dtype))
        return total, grads

    def loss(self, params: ParamCollection) -> Tensor:
        value, _ = self._loss_and_grads(self._validate(params), None)
        return value

    def loss_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> Tensor:
        indices = _validate_indices(sample_indices, self.sample_count).to(
            self.a_ops[0].device
        )
        value, _ = self._loss_and_grads(self._validate(params), indices)
        return value

    def grad(self, params: ParamCollection) -> list[Tensor]:
        _, grads = self._loss_and_grads(self._validate(params), None)
        return grads

    def grad_on_samples(self, params: ParamCollection, sample_indices: Tensor) -> list[Tensor]:
        indices = _validate_indices(sample_indices, self.sample_count).to(
            self.a_ops[0].device
        )
        _, grads = self._loss_and_grads(self._validate(params), indices)
        return grads


def make_iso_shape_spectrum(
    seed: int,
    *,
    width: int = 8,
    samples: int = 48,
    input_condition: float = 100.0,
    conditions: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ParamCollection, IsoShapeSpectrumTask]:
    """Same-shape matrices, staircase condition numbers.

    ``input_condition`` sets the maximum condition in the staircase when
    ``conditions`` is omitted.
    """
    if width < 2:
        raise ValueError("width must be >= 2")
    if conditions is None:
        max_kappa = max(float(input_condition), 1.0)
        conditions = (1.0, max_kappa**0.33, max_kappa**0.66, max_kappa)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = torch.Size((width, width))
    dim = width * width
    a_ops: list[Tensor] = []
    targets: list[Tensor] = []
    initials: list[Tensor] = []
    for condition in conditions:
        rows = min(samples, dim)
        v, _ = torch.linalg.qr(torch.randn((dim, rows), generator=generator))
        u2, _ = torch.linalg.qr(torch.randn((samples, rows), generator=generator))
        log_s = torch.linspace(0.0, math.log10(max(float(condition), 1.0)), rows)
        sigma = 10.0**log_s
        a = (u2 * sigma.unsqueeze(0)) @ v.mT
        a_ops.append(a)
        theta_star = torch.randn((dim,), generator=generator) * 0.3
        targets.append(a @ theta_star)
        initials.append(torch.randn((dim,), generator=generator) * 0.1)
    device_t = torch.device(device)
    task = IsoShapeSpectrumTask(
        [a.to(device=device_t, dtype=dtype) for a in a_ops],
        [t.to(device=device_t, dtype=dtype) for t in targets],
        shape=shape,
    )
    initial = ParamCollection([t.reshape(shape) for t in initials]).to(
        device=device_t, dtype=dtype
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
    if architecture == "three_layer":
        return make_three_layer_mlp(
            seed,
            input_dim=width,
            hidden_dim=width,
            output_dim=output_dim,
            samples=samples,
            input_condition=input_condition,
            device=device,
        )
    if architecture == "all_matrix":
        return make_all_matrix(
            seed,
            input_dim=width,
            samples=samples,
            input_condition=input_condition,
            device=device,
        )
    if architecture == "all_vector":
        return make_all_vector(
            seed,
            input_dim=width,
            samples=samples,
            input_condition=input_condition,
            device=device,
        )
    if architecture == "iso_shape":
        return make_iso_shape_spectrum(
            seed,
            width=width,
            samples=samples,
            input_condition=input_condition,
            device=device,
        )
    raise ValueError(f"unknown architecture: {architecture}")
