from __future__ import annotations

import torch

from optdistil.distill.secant_features import SecantFeatureState
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def test_secant_features_keep_student_at_153_parameters() -> None:
    state = SecantFeatureState()
    parameter = torch.randn(4, 6)
    grad = torch.randn_like(parameter)
    momentum = torch.randn_like(parameter)
    second_moment = torch.rand_like(parameter)

    features = state.build(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=8,
    )

    assert features.shape == (24, 8)
    assert TinyMLPOptimizer(feature_dim=features.shape[1]).parameter_count == 153


def test_first_secant_direction_is_negative_gradient_with_matched_rms() -> None:
    state = SecantFeatureState()
    parameter = torch.randn(2, 3, dtype=torch.float64)
    grad = torch.tensor([[1.0, -2.0, 3.0], [-0.5, 1.5, -1.0]], dtype=torch.float64)
    momentum = torch.zeros_like(parameter)
    second_moment = grad.square()

    features = state.build(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=4,
    )

    torch.testing.assert_close(features[:, 5], (-grad).reshape(-1), rtol=1e-7, atol=1e-7)


def test_one_pair_lbfgs_recovers_isotropic_inverse_curvature_direction() -> None:
    state = SecantFeatureState()
    hessian_scale = 4.0
    parameter0 = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    grad0 = hessian_scale * parameter0
    zeros = torch.zeros_like(parameter0)
    state.build(parameter0, grad0, zeros, grad0.square(), step=1, total_steps=4)

    parameter1 = parameter0 - 0.1 * grad0
    grad1 = hessian_scale * parameter1
    features = state.build(parameter1, grad1, zeros, grad1.square(), step=2, total_steps=4)

    direction = features[:, 5].reshape_as(grad1)
    cosine = torch.nn.functional.cosine_similarity(
        direction.reshape(1, -1), (-grad1).reshape(1, -1)
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-7, atol=1e-7)


def test_raw_secant_direction_preserves_inverse_curvature_scale() -> None:
    state = SecantFeatureState(normalize_direction=False)
    hessian_scale = 4.0
    parameter0 = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    grad0 = hessian_scale * parameter0
    zeros = torch.zeros_like(parameter0)
    state.build(parameter0, grad0, zeros, grad0.square(), step=1, total_steps=4)

    parameter1 = parameter0 - 0.1 * grad0
    grad1 = hessian_scale * parameter1
    features = state.build(parameter1, grad1, zeros, grad1.square(), step=2, total_steps=4)

    direction = features[:, 5].reshape_as(grad1)
    torch.testing.assert_close(direction, -grad1 / hessian_scale, rtol=1e-6, atol=1e-6)


def test_multi_pair_history_stays_bounded_and_preserves_isotropic_direction() -> None:
    state = SecantFeatureState(history_size=2)
    hessian_scale = 3.0
    parameter = torch.tensor([[1.0, -0.5], [2.0, -1.0]], dtype=torch.float64)
    zeros = torch.zeros_like(parameter)

    for step in range(1, 6):
        grad = hessian_scale * parameter
        features = state.build(
            parameter,
            grad,
            zeros,
            grad.square(),
            step=step,
            total_steps=5,
        )
        direction = features[:, 5].reshape_as(grad)
        cosine = torch.nn.functional.cosine_similarity(
            direction.reshape(1, -1), (-grad).reshape(1, -1)
        )
        torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-7, atol=1e-7)
        parameter = parameter - 0.1 * grad

    assert state.stored_pairs == 2
    assert state.max_additional_state_tensors == 6


def test_reset_clears_secant_history() -> None:
    state = SecantFeatureState(history_size=4)
    parameter = torch.ones((2, 2))
    grad = 2.0 * parameter
    zeros = torch.zeros_like(parameter)
    state.build(parameter, grad, zeros, grad.square(), step=1, total_steps=3)
    parameter = parameter - 0.1 * grad
    grad = 2.0 * parameter
    state.build(parameter, grad, zeros, grad.square(), step=2, total_steps=3)
    assert state.stored_pairs == 1

    state.reset()
    assert state.stored_pairs == 0
    assert state.previous_parameter is None
    assert state.previous_grad is None
