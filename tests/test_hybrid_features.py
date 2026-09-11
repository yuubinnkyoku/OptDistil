from __future__ import annotations

import torch

from optdistil.distill.experimental_features import build_hybrid_gram_features
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def test_hybrid_features_keep_student_at_153_parameters() -> None:
    parameter = torch.randn(4, 6)
    grad = torch.randn_like(parameter)
    momentum = torch.randn_like(parameter)
    second_moment = torch.rand_like(parameter)

    features = build_hybrid_gram_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=2,
        total_steps=8,
    )

    assert features.shape == (24, 8)
    assert TinyMLPOptimizer(feature_dim=features.shape[1]).parameter_count == 153


def test_hybrid_features_contain_row_col_and_grad_gram_terms() -> None:
    parameter = torch.randn(2, 3, dtype=torch.float64)
    grad = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]], dtype=torch.float64)
    momentum = torch.randn_like(parameter)
    second_moment = torch.rand_like(parameter)

    features = build_hybrid_gram_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=4,
    )

    row_rms = grad.square().mean(dim=1, keepdim=True).add(1e-8).sqrt().expand_as(grad)
    col_rms = grad.square().mean(dim=0, keepdim=True).add(1e-8).sqrt().expand_as(grad)
    gram = ((grad @ grad.mT) @ grad) / grad.square().sum()

    torch.testing.assert_close(features[:, 3], row_rms.reshape(-1))
    torch.testing.assert_close(features[:, 4], col_rms.reshape(-1))
    torch.testing.assert_close(features[:, 5], gram.reshape(-1))
