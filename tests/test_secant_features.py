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

    # The raw one-pair L-BFGS direction is -g / h for H=hI.  The feature rescales it to
    # the gradient RMS, so it must remain exactly parallel to -g.
    direction = features[:, 5].reshape_as(grad1)
    cosine = torch.nn.functional.cosine_similarity(direction.reshape(1, -1), (-grad1).reshape(1, -1))
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-7, atol=1e-7)
