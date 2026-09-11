from __future__ import annotations

import torch

from optdistil.distill.meta_train import meta_objective
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.meta_attention import MetaAttentionTeacher


def test_zero_initialized_attention_teacher_matches_adam_like_base() -> None:
    teacher = MetaAttentionTeacher(
        d_model=16,
        num_heads=4,
        depth=1,
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


def test_attention_meta_objective_backpropagates_finite_gradients() -> None:
    initial, task = make_frozen_readout_mlp(
        17,
        hidden_dim=3,
        input_dim=3,
        output_dim=2,
        samples=12,
        input_condition=5.0,
    )
    teacher = MetaAttentionTeacher(
        d_model=16,
        num_heads=4,
        depth=1,
        horizon=3,
        initial_step_scale=0.05,
    )

    objective = meta_objective(teacher, [(initial, task)], steps=3)
    objective.backward()

    gradients = [parameter.grad for parameter in teacher.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_attention_teacher_accepts_rectangular_matrices() -> None:
    teacher = MetaAttentionTeacher(d_model=16, num_heads=4, depth=1, horizon=2)
    parameter = torch.randn(3, 5)
    grad = torch.randn_like(parameter)

    update, state = teacher.functional_step(
        parameter,
        grad,
        teacher.initial_state(parameter),
    )

    assert update.shape == parameter.shape
    assert state.step_number == 1
    assert torch.isfinite(update).all()
