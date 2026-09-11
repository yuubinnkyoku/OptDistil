from __future__ import annotations

import torch

from optdistil.distill.meta_train import meta_objective
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.meta_recurrent_attention import MetaRecurrentAttentionTeacher


def test_recurrent_teacher_starts_from_same_adam_like_base() -> None:
    teacher = MetaRecurrentAttentionTeacher(
        d_model=12,
        num_heads=3,
        depth=1,
        horizon=4,
        initial_step_scale=0.2,
    )
    parameter = torch.tensor([[0.5, -0.3], [0.1, 0.7]])
    grad = torch.tensor([[2.0, -0.5], [-1.0, 0.25]])
    state = teacher.initial_state(parameter)

    update, next_state = teacher.functional_step(parameter, grad, state)
    expected_base = -grad / (grad.abs() + teacher.eps)
    expected = teacher.step_scale.detach() * expected_base

    assert torch.allclose(update.detach(), expected, atol=1e-6, rtol=1e-6)
    assert next_state.step_number == 1
    assert next_state.memory.shape == (parameter.numel(), teacher.d_model)
    assert torch.isfinite(next_state.memory).all()


def test_recurrent_teacher_carries_memory_across_steps() -> None:
    teacher = MetaRecurrentAttentionTeacher(d_model=12, num_heads=3, depth=1, horizon=3)
    parameter = torch.randn(3, 3)
    grad = torch.randn_like(parameter)
    state0 = teacher.initial_state(parameter)

    update1, state1 = teacher.functional_step(parameter, grad, state0)
    update2, state2 = teacher.functional_step(parameter + update1, grad * 0.7, state1)

    assert state1.step_number == 1
    assert state2.step_number == 2
    assert not torch.allclose(state1.memory, state0.memory)
    assert not torch.allclose(state2.memory, state1.memory)
    assert torch.isfinite(update2).all()


def test_recurrent_attention_meta_objective_has_finite_gradients() -> None:
    initial, task = make_frozen_readout_mlp(
        27,
        hidden_dim=3,
        input_dim=3,
        output_dim=2,
        samples=12,
        input_condition=5.0,
    )
    teacher = MetaRecurrentAttentionTeacher(
        d_model=12,
        num_heads=3,
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
