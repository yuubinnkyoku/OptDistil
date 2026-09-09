import torch

from optdistil.teachers.random_direction import RandomDirectionTeacher


def test_random_direction_has_fixed_norm_and_is_reproducible() -> None:
    parameter = torch.zeros((3, 3))
    grad = torch.ones((3, 3))
    first = RandomDirectionTeacher(lr=0.25, seed=7)
    second = RandomDirectionTeacher(lr=0.25, seed=7)

    first_update = first.step(parameter, grad)
    second_update = second.step(parameter, grad)

    assert torch.allclose(first_update.norm(), torch.tensor(0.25))
    assert torch.allclose(first_update, second_update)


def test_random_direction_advances_generator_state() -> None:
    parameter = torch.zeros((3, 3))
    grad = torch.ones((3, 3))
    teacher = RandomDirectionTeacher(lr=0.1, seed=11)

    first_update = teacher.step(parameter, grad)
    second_update = teacher.step(parameter, grad)

    assert not torch.allclose(first_update, second_update)
