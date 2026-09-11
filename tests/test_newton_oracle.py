from __future__ import annotations

import torch

from optdistil.tasks.coupled_quadratic import CoupledMatrixQuadraticTask
from optdistil.teachers.newton_oracle import CoupledNewtonOracleTeacher


def make_task() -> tuple[torch.Tensor, CoupledMatrixQuadraticTask]:
    initial = torch.tensor([[0.8, -0.4], [0.2, 0.7]], dtype=torch.float64)
    target = torch.tensor([[0.1, 0.3], [-0.5, 0.2]], dtype=torch.float64)
    left = torch.tensor([[1.4, 0.2], [0.2, 0.9]], dtype=torch.float64)
    right = torch.tensor([[1.1, -0.15], [-0.15, 0.8]], dtype=torch.float64)
    return initial, CoupledMatrixQuadraticTask(target, left, right)


def test_full_newton_step_reaches_quadratic_target() -> None:
    initial, task = make_task()
    teacher = CoupledNewtonOracleTeacher(task.left, task.right, lr=1.0)

    update = teacher.step(initial, task.grad(initial))
    final = initial + update

    assert torch.allclose(final, task.target, atol=1e-10, rtol=1e-10)
    assert float(task.loss(final)) < 1e-18
    assert torch.allclose(task.grad(final), torch.zeros_like(final), atol=1e-10, rtol=1e-10)


def test_damped_newton_step_contracts_parameter_error_exactly() -> None:
    initial, task = make_task()
    teacher = CoupledNewtonOracleTeacher(task.left, task.right, lr=0.5)

    update = teacher.step(initial, task.grad(initial))
    final = initial + update

    expected = task.target + 0.5 * (initial - task.target)
    assert torch.allclose(final, expected, atol=1e-10, rtol=1e-10)


def test_newton_teacher_rejects_incompatible_shape() -> None:
    initial, task = make_task()
    teacher = CoupledNewtonOracleTeacher(task.left, task.right, lr=0.5)

    bad_parameter = torch.zeros(3, 2, dtype=initial.dtype)
    bad_grad = torch.zeros_like(bad_parameter)
    try:
        teacher.step(bad_parameter, bad_grad)
    except ValueError as exc:
        assert "left factor" in str(exc)
    else:
        raise AssertionError("expected incompatible parameter rows to be rejected")
