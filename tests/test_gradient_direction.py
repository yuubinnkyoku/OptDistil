import torch

from optdistil.teachers.gradient_direction import GradientDirectionTeacher


def test_gradient_direction_teacher_has_fixed_norm_and_negative_direction() -> None:
    parameter = torch.zeros((2, 2))
    grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    teacher = GradientDirectionTeacher(lr=0.25)

    update = teacher.step(parameter, grad)

    assert torch.allclose(update.norm(), torch.tensor(0.25))
    assert torch.dot(update.reshape(-1), grad.reshape(-1)) < 0
    assert torch.allclose(update, -0.25 * grad / grad.norm())
