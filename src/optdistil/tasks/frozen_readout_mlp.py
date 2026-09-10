from __future__ import annotations

import math

import torch
from torch import Tensor


class FrozenReadoutMLPTask:
    """Nonlinear single-matrix optimization task with fixed inputs and readout.

    Only the hidden weight matrix ``W`` is optimized::

        prediction = readout @ tanh(W @ inputs)
        loss       = 0.5 / samples * ||prediction - target||_F^2

    Keeping the parameter as one 2-D tensor lets the existing optimizer/trajectory API
    move beyond quadratics without introducing parameter pytrees at the same time.
    """

    def __init__(self, inputs: Tensor, readout: Tensor, target: Tensor) -> None:
        if inputs.ndim != 2 or readout.ndim != 2 or target.ndim != 2:
            raise ValueError("inputs, readout, and target must all be matrices")
        hidden_dim = readout.shape[1]
        if inputs.shape[1] != target.shape[1]:
            raise ValueError("inputs and target must contain the same number of samples")
        if readout.shape[0] != target.shape[0]:
            raise ValueError("readout output dimension must match target")
        if hidden_dim <= 0 or inputs.shape[0] <= 0 or inputs.shape[1] <= 0:
            raise ValueError("task dimensions must be positive")

        device = target.device
        dtype = target.dtype
        self.inputs = inputs.detach().clone().to(device=device, dtype=dtype)
        self.readout = readout.detach().clone().to(device=device, dtype=dtype)
        self.target = target.detach().clone()

    @property
    def parameter_shape(self) -> tuple[int, int]:
        return self.readout.shape[1], self.inputs.shape[0]

    def _validate_parameter(self, parameter: Tensor) -> None:
        if tuple(parameter.shape) != self.parameter_shape:
            raise ValueError(f"parameter must have shape {self.parameter_shape}")

    def prediction(self, parameter: Tensor) -> Tensor:
        self._validate_parameter(parameter)
        hidden = torch.tanh(parameter @ self.inputs)
        return self.readout @ hidden

    def loss(self, parameter: Tensor) -> Tensor:
        residual = self.prediction(parameter) - self.target
        return 0.5 * residual.square().sum() / self.inputs.shape[1]

    def grad(self, parameter: Tensor) -> Tensor:
        """Analytic gradient of the nonlinear objective with respect to ``W``."""
        self._validate_parameter(parameter)
        preactivation = parameter @ self.inputs
        hidden = torch.tanh(preactivation)
        residual = self.readout @ hidden - self.target
        hidden_grad = (self.readout.mT @ residual) * (1.0 - hidden.square())
        return (hidden_grad @ self.inputs.mT) / self.inputs.shape[1]


def make_frozen_readout_mlp(
    seed: int,
    *,
    hidden_dim: int = 8,
    input_dim: int = 8,
    output_dim: int = 4,
    samples: int = 64,
    input_condition: float = 30.0,
    planted_scale: float = 0.7,
    initial_scale: float = 0.2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, FrozenReadoutMLPTask]:
    """Create a deterministic realizable nonlinear matrix-regression task.

    ``input_condition`` controls the condition number of the population covariance
    induced by the linear input transform before finite-sample noise. The transformed
    inputs are RMS-normalized afterwards, preserving anisotropy while keeping activation
    scales comparable across conditions.
    """
    if min(hidden_dim, input_dim, output_dim, samples) <= 0:
        raise ValueError("all dimensions and sample count must be positive")
    if input_condition < 1.0 or not math.isfinite(input_condition):
        raise ValueError("input_condition must be finite and at least 1")
    if planted_scale <= 0.0 or initial_scale <= 0.0:
        raise ValueError("weight scales must be positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    rotation_raw = torch.randn((input_dim, input_dim), generator=generator)
    rotation, _ = torch.linalg.qr(rotation_raw)
    # A transform with singular-value condition sqrt(kappa) induces covariance
    # condition approximately kappa before finite-sample effects.
    singular_values = torch.logspace(
        0.0,
        0.5 * math.log10(input_condition),
        input_dim,
    )
    transform = rotation @ torch.diag(singular_values) @ rotation.mT
    inputs = transform @ torch.randn((input_dim, samples), generator=generator)
    inputs = inputs / inputs.square().mean().sqrt().clamp_min(1e-8)

    readout = torch.randn((output_dim, hidden_dim), generator=generator) / math.sqrt(hidden_dim)
    planted = (
        torch.randn((hidden_dim, input_dim), generator=generator)
        * planted_scale
        / math.sqrt(input_dim)
    )
    initial = (
        torch.randn((hidden_dim, input_dim), generator=generator)
        * initial_scale
        / math.sqrt(input_dim)
    )
    target = readout @ torch.tanh(planted @ inputs)

    device = torch.device(device)
    initial = initial.to(device=device, dtype=dtype)
    task = FrozenReadoutMLPTask(
        inputs.to(device=device, dtype=dtype),
        readout.to(device=device, dtype=dtype),
        target.to(device=device, dtype=dtype),
    )
    return initial, task
