from __future__ import annotations

import math

import torch

from optdistil.distill.direct_meta import (
    direct_meta_objective,
    train_direct_student,
    zero_initialize_student_output,
)
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp


def make_cases(seed: int):
    return [
        make_frozen_readout_mlp(
            seed + index,
            hidden_dim=3,
            input_dim=3,
            output_dim=2,
            samples=12,
            input_condition=5.0,
        )
        for index in range(2)
    ]


def test_zero_initialized_direct_student_outputs_zero() -> None:
    student = TinyMLPOptimizer()
    zero_initialize_student_output(student)
    features = torch.randn(7, 8)
    assert torch.equal(student(features), torch.zeros(7))
    assert student.parameter_count == 153


def test_direct_meta_objective_has_finite_gradient() -> None:
    student = TinyMLPOptimizer()
    zero_initialize_student_output(student)
    objective = direct_meta_objective(student, make_cases(10)[:1], steps=3)
    objective.backward()

    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_short_direct_meta_training_is_finite() -> None:
    student = TinyMLPOptimizer()
    zero_initialize_student_output(student)
    history = train_direct_student(
        student,
        make_cases(20),
        make_cases(40),
        steps=3,
        iterations=2,
        outer_lr=1e-3,
    )

    assert len(history) == 2
    assert all(math.isfinite(row.train_objective) for row in history)
    assert all(math.isfinite(row.validation_loss_ratio) for row in history)
    assert student.parameter_count == 153
