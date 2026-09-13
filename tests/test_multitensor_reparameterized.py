import pytest
import torch

from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.reparameterized import (
    FunctionSpaceNormGradTeacher,
    ReparameterizedTask,
    reparameterize_initial,
    sample_log_uniform_scales,
)
from optdistil.multitensor.tasks import make_residual_mlp, make_two_layer_mlp


def test_reparameterized_initial_preserves_objective_exactly() -> None:
    base_initial, base_task = make_residual_mlp(
        17,
        input_dim=4,
        hidden_dim=4,
        output_dim=2,
        samples=11,
        input_condition=20.0,
        dtype=torch.float64,
    )
    scales = (0.5, 1.7, 2.0, 0.8, 1.25)
    initial = reparameterize_initial(base_initial, scales)
    task = ReparameterizedTask(base_task, scales)

    torch.testing.assert_close(task.loss(initial), base_task.loss(base_initial), rtol=0.0, atol=0.0)
    converted = task.to_base_parameters(initial)
    for actual, expected in zip(converted, base_initial, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_reparameterized_gradient_matches_autograd() -> None:
    base_initial, base_task = make_two_layer_mlp(
        23,
        input_dim=3,
        hidden_dim=4,
        output_dim=2,
        samples=13,
        input_condition=15.0,
        dtype=torch.float64,
    )
    scales = (0.4, 1.3, 2.2, 0.7)
    task = ReparameterizedTask(base_task, scales)
    initial = reparameterize_initial(base_initial, scales)
    theta = ParamCollection(
        [tensor.detach().clone().requires_grad_(True) for tensor in initial]
    )

    autograd = torch.autograd.grad(task.loss(theta), theta.tensors)
    analytic = task.grad(theta)
    for actual, expected in zip(analytic, autograd, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_sample_log_uniform_scales_is_deterministic_and_bounded() -> None:
    first = sample_log_uniform_scales(101, 5, min_scale=0.25, max_scale=4.0)
    second = sample_log_uniform_scales(101, 5, min_scale=0.25, max_scale=4.0)
    other = sample_log_uniform_scales(102, 5, min_scale=0.25, max_scale=4.0)

    assert first == second
    assert first != other
    assert all(0.25 <= value <= 4.0 for value in first)


def test_function_space_normgrad_has_equal_base_update_norms() -> None:
    base_initial, base_task = make_two_layer_mlp(
        31,
        input_dim=3,
        hidden_dim=4,
        output_dim=2,
        samples=17,
        dtype=torch.float64,
    )
    scales = (0.5, 2.0, 1.25, 0.8)
    task = ReparameterizedTask(base_task, scales)
    initial = reparameterize_initial(base_initial, scales)
    grads = task.grad(initial)
    teacher = FunctionSpaceNormGradTeacher(0.07, scales)

    updates = teacher.step(initial, grads)
    base_grads = base_task.grad(base_initial)
    for scale, update, base_grad in zip(scales, updates, base_grads, strict=True):
        function_update = update * scale
        assert float(function_update.norm()) == pytest.approx(0.07, rel=1e-9, abs=1e-10)
        cosine = torch.dot(
            function_update.reshape(-1), -base_grad.reshape(-1)
        ) / (function_update.norm() * base_grad.norm())
        assert float(cosine) == pytest.approx(1.0, rel=1e-9, abs=1e-10)


def test_reparameterization_changes_coordinate_gradient_scale() -> None:
    base_initial, base_task = make_two_layer_mlp(
        41,
        input_dim=3,
        hidden_dim=3,
        output_dim=2,
        samples=11,
        dtype=torch.float64,
    )
    scales = (0.5, 2.0, 1.5, 0.25)
    task = ReparameterizedTask(base_task, scales)
    initial = reparameterize_initial(base_initial, scales)

    transformed_grads = task.grad(initial)
    base_grads = base_task.grad(base_initial)
    for scale, actual, base_grad in zip(scales, transformed_grads, base_grads, strict=True):
        torch.testing.assert_close(actual, base_grad * scale, rtol=1e-12, atol=1e-12)
