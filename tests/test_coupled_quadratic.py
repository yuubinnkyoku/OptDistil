import torch

from optdistil.tasks.coupled_quadratic import CoupledMatrixQuadraticTask


def test_coupled_quadratic_gradient_matches_autograd() -> None:
    target = torch.tensor([[0.2, -0.1], [0.3, 0.4]])
    left = torch.tensor([[1.5, 0.4], [0.4, 0.8]])
    right = torch.tensor([[1.2, -0.3], [-0.3, 0.9]])
    task = CoupledMatrixQuadraticTask(target, left, right)

    parameter = torch.tensor([[0.7, -0.2], [0.1, 0.9]], requires_grad=True)
    expected = torch.autograd.grad(task.loss(parameter), parameter)[0]
    actual = task.grad(parameter.detach())

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_coupled_quadratic_has_cross_coordinate_coupling() -> None:
    target = torch.zeros((2, 2))
    left = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    right = torch.tensor([[1.0, 0.4], [0.4, 1.0]])
    task = CoupledMatrixQuadraticTask(target, left, right)

    parameter = torch.zeros((2, 2))
    parameter[0, 0] = 1.0
    grad = task.grad(parameter)

    assert torch.count_nonzero(grad).item() == 4
