from __future__ import annotations

import torch

from optdistil.distill.features import build_gram_matrix_features
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def test_gram_features_keep_fixed_eight_dimensional_student_input() -> None:
    parameter = torch.randn(4, 6)
    grad = torch.randn_like(parameter)
    momentum = torch.randn_like(parameter)
    second_moment = torch.rand_like(parameter)

    features = build_gram_matrix_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=3,
        total_steps=8,
    )

    assert features.shape == (parameter.numel(), 8)
    assert TinyMLPOptimizer(feature_dim=features.shape[1]).parameter_count == 153


def test_grad_gram_feature_matches_normalized_cubic_matrix_product() -> None:
    parameter = torch.zeros(2, 3, dtype=torch.float64)
    grad = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=torch.float64)
    momentum = torch.tensor([[0.2, -0.4, 1.0], [2.0, 0.3, -0.7]], dtype=torch.float64)
    second_moment = torch.ones_like(parameter)

    features = build_gram_matrix_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=8,
    )

    expected_grad = ((grad @ grad.mT) @ grad) / grad.square().sum()
    expected_momentum = ((momentum @ momentum.mT) @ momentum) / momentum.square().sum()
    torch.testing.assert_close(features[:, 4], expected_grad.reshape(-1))
    torch.testing.assert_close(features[:, 5], expected_momentum.reshape(-1))


def test_normalized_gram_feature_scales_linearly_with_gradient() -> None:
    parameter = torch.zeros(3, 2)
    grad = torch.randn_like(parameter)
    momentum = torch.randn_like(parameter)
    second_moment = torch.ones_like(parameter)

    original = build_gram_matrix_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=4,
    )
    scaled = build_gram_matrix_features(
        parameter,
        5.0 * grad,
        momentum,
        second_moment,
        step=1,
        total_steps=4,
    )

    torch.testing.assert_close(scaled[:, 4], 5.0 * original[:, 4], rtol=1e-5, atol=1e-6)
