from __future__ import annotations

import torch

from optdistil.distill.secant_meta import secant_meta_objective
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.coupled_quadratic import CoupledMatrixQuadraticTask


def make_case():
    initial = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float64)
    target = torch.zeros_like(initial)
    left = torch.tensor([[1.4, 0.2], [0.2, 0.9]], dtype=torch.float64)
    right = torch.tensor([[1.1, -0.1], [-0.1, 0.8]], dtype=torch.float64)
    return initial, CoupledMatrixQuadraticTask(target, left, right)


def assert_finite_meta_gradients(
    mode: str,
    gain_normalization: float,
    *,
    gain_bound: float = 4.0,
) -> None:
    torch.manual_seed(7)
    student = TinyMLPOptimizer().to(dtype=torch.float64)
    objective = secant_meta_objective(
        student,
        [make_case()],
        steps=3,
        history_size=2,
        mode=mode,
        gain_normalization=gain_normalization,
        gain_bound=gain_bound,
    )
    assert torch.isfinite(objective)
    objective.backward()
    gradients = [parameter.grad for parameter in student.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_full_update_secant_meta_gradient_is_finite() -> None:
    assert_finite_meta_gradients("full_update", 1.0)


def test_scalar_gain_secant_meta_gradient_is_finite() -> None:
    assert_finite_meta_gradients("scalar_gain", 0.01)


def test_bounded_gain_secant_meta_gradient_is_finite() -> None:
    assert_finite_meta_gradients("bounded_gain", 0.01, gain_bound=4.0)
