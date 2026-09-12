import torch

from optdistil.tasks.frozen_readout_mlp import FrozenReadoutMLPTask, make_frozen_readout_mlp


def test_analytic_gradient_matches_autograd() -> None:
    initial, task = make_frozen_readout_mlp(
        17,
        hidden_dim=4,
        input_dim=3,
        output_dim=2,
        samples=11,
        input_condition=20.0,
        dtype=torch.float64,
    )
    parameter = initial.detach().clone().requires_grad_(True)
    autograd_gradient = torch.autograd.grad(task.loss(parameter), parameter)[0]

    torch.testing.assert_close(task.grad(parameter.detach()), autograd_gradient, rtol=1e-9, atol=1e-10)


def test_minibatch_analytic_gradient_matches_autograd() -> None:
    initial, task = make_frozen_readout_mlp(
        19,
        hidden_dim=4,
        input_dim=3,
        output_dim=2,
        samples=13,
        input_condition=50.0,
        dtype=torch.float64,
    )
    indices = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    parameter = initial.detach().clone().requires_grad_(True)
    autograd_gradient = torch.autograd.grad(task.loss_on_samples(parameter, indices), parameter)[0]

    torch.testing.assert_close(
        task.grad_on_samples(parameter.detach(), indices),
        autograd_gradient,
        rtol=1e-9,
        atol=1e-10,
    )


def test_full_sample_minibatch_matches_full_objective() -> None:
    initial, task = make_frozen_readout_mlp(21, samples=16, dtype=torch.float64)
    indices = torch.arange(task.sample_count, dtype=torch.long)

    torch.testing.assert_close(task.loss_on_samples(initial, indices), task.loss(initial))
    torch.testing.assert_close(task.grad_on_samples(initial, indices), task.grad(initial))


def test_factory_is_deterministic_and_condition_changes_task() -> None:
    initial_a, task_a = make_frozen_readout_mlp(23, input_condition=10.0)
    initial_b, task_b = make_frozen_readout_mlp(23, input_condition=10.0)
    initial_c, task_c = make_frozen_readout_mlp(23, input_condition=100.0)

    torch.testing.assert_close(initial_a, initial_b)
    torch.testing.assert_close(task_a.inputs, task_b.inputs)
    torch.testing.assert_close(task_a.target, task_b.target)
    assert not torch.allclose(task_a.inputs, task_c.inputs)
    # The random initial parameter is deliberately independent of conditioning.
    torch.testing.assert_close(initial_a, initial_c)


def test_loss_and_gradient_are_finite() -> None:
    initial, task = make_frozen_readout_mlp(5, input_condition=300.0)

    assert torch.isfinite(task.loss(initial))
    assert torch.isfinite(task.grad(initial)).all()
    assert task.grad(initial).shape == initial.shape


def test_task_rejects_wrong_parameter_shape() -> None:
    inputs = torch.ones((3, 5))
    readout = torch.ones((2, 4))
    target = torch.ones((2, 5))
    task = FrozenReadoutMLPTask(inputs, readout, target)

    try:
        task.loss(torch.ones((3, 4)))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("expected ValueError")
