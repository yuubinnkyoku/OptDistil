import torch

from optdistil.teachers.momentum_direction import MomentumDirectionTeacher


def test_momentum_direction_matches_first_nesterov_direction() -> None:
    parameter = torch.zeros((2, 2))
    grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    teacher = MomentumDirectionTeacher(lr=0.5, momentum=0.9, nesterov=True)

    update = teacher.step(parameter, grad)

    expected_direction = grad + 0.9 * grad
    expected = -0.5 * expected_direction / expected_direction.norm()
    assert torch.allclose(update, expected)
    assert torch.allclose(update.norm(), torch.tensor(0.5))


def test_momentum_direction_accumulates_state() -> None:
    parameter = torch.zeros((2, 2))
    teacher = MomentumDirectionTeacher(lr=0.2, momentum=0.5, nesterov=False)

    teacher.step(parameter, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    update = teacher.step(parameter, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))

    expected_direction = torch.tensor([[0.5, 1.0], [0.0, 0.0]])
    expected = -0.2 * expected_direction / expected_direction.norm()
    assert torch.allclose(update, expected)
