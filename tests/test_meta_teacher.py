from __future__ import annotations

import math

import torch

from optdistil.distill.meta_train import meta_objective, train_meta_teacher
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.meta_mlp import MetaMLPTeacher


def test_zero_initialized_meta_teacher_matches_adam_like_base() -> None:
    teacher = MetaMLPTeacher(
        hidden_dim=8,
        horizon=4,
        initial_step_scale=0.2,
        max_step_scale=1.0,
    )
    parameter = torch.tensor([[0.5, -0.3], [0.1, 0.7]])
    grad = torch.tensor([[2.0, -0.5], [-1.0, 0.25]])
    state = teacher.initial_state(parameter)

    update, next_state = teacher.functional_step(parameter, grad, state)
    expected_base = -grad / (grad.abs() + teacher.eps)
    expected = teacher.step_scale.detach() * expected_base

    assert next_state.step_number == 1
    assert torch.allclose(update.detach(), expected, atol=1e-6, rtol=1e-6)


def test_meta_objective_backpropagates_finite_gradients() -> None:
    initial, task = make_frozen_readout_mlp(
        7,
        hidden_dim=3,
        input_dim=3,
        output_dim=2,
        samples=12,
        input_condition=5.0,
    )
    teacher = MetaMLPTeacher(hidden_dim=8, horizon=3, initial_step_scale=0.05)

    objective = meta_objective(teacher, [(initial, task)], steps=3)
    objective.backward()

    gradients = [parameter.grad for parameter in teacher.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_short_meta_training_restores_finite_validation_checkpoint() -> None:
    train_cases = [
        make_frozen_readout_mlp(
            20 + index,
            hidden_dim=3,
            input_dim=3,
            output_dim=2,
            samples=12,
            input_condition=5.0,
        )
        for index in range(2)
    ]
    validation_cases = [
        make_frozen_readout_mlp(
            40 + index,
            hidden_dim=3,
            input_dim=3,
            output_dim=2,
            samples=12,
            input_condition=5.0,
        )
        for index in range(2)
    ]
    teacher = MetaMLPTeacher(hidden_dim=8, horizon=3, initial_step_scale=0.05)

    history = train_meta_teacher(
        teacher,
        train_cases,
        validation_cases,
        steps=3,
        iterations=2,
        outer_lr=1e-3,
    )

    assert len(history) == 2
    assert all(math.isfinite(step.train_objective) for step in history)
    assert all(math.isfinite(step.validation_loss_ratio) for step in history)
    assert 0.0 < float(teacher.step_scale) < teacher.max_step_scale
