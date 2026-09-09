from __future__ import annotations

import pytest
import torch
from torch import nn

from optdistil.distill.alignment import (
    flattened_cosine,
    probe_reference_gradient_geometry,
    probe_same_state_alignment,
)
from optdistil.tasks.quadratic import QuadraticTask


class NegativeGradientStudent(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return -features[:, 0]


class ScaledGradientReference:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    @torch.no_grad()
    def step(self, parameter: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        del parameter
        return -self.scale * grad


class OrthogonalGradientReference:
    @torch.no_grad()
    def step(self, parameter: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        del parameter
        flat = grad.reshape(-1)
        if flat.numel() % 2:
            raise ValueError("test reference requires an even number of parameters")
        rotated = torch.empty_like(flat)
        rotated[0::2] = -flat[1::2]
        rotated[1::2] = flat[0::2]
        return rotated.reshape_as(grad)


def test_flattened_cosine_handles_basic_directions() -> None:
    x = torch.tensor([[1.0, 0.0]])
    same = torch.tensor([[2.0, 0.0]])
    opposite = torch.tensor([[-3.0, 0.0]])
    orthogonal = torch.tensor([[0.0, 4.0]])

    assert flattened_cosine(x, same) == pytest.approx(1.0)
    assert flattened_cosine(x, opposite) == pytest.approx(-1.0)
    assert flattened_cosine(x, orthogonal) == pytest.approx(0.0)
    assert flattened_cosine(x, torch.zeros_like(x)) == pytest.approx(0.0)


def test_reference_gradient_geometry_detects_matching_direction() -> None:
    initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    task = QuadraticTask(torch.zeros_like(initial))

    result = probe_reference_gradient_geometry(
        initial,
        task,
        reference=ScaledGradientReference(scale=0.25),
        steps=1,
    )

    assert result.cosine_mean == pytest.approx(1.0)
    assert result.disagreement_mean == pytest.approx(0.0)
    assert result.rollout.finite


def test_reference_gradient_geometry_detects_orthogonal_direction() -> None:
    initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    task = QuadraticTask(torch.zeros_like(initial))

    result = probe_reference_gradient_geometry(
        initial,
        task,
        reference=OrthogonalGradientReference(),
        steps=1,
    )

    assert result.cosine_mean == pytest.approx(0.0, abs=1e-7)
    assert result.disagreement_mean == pytest.approx(1.0, abs=1e-7)


def test_same_state_alignment_matches_gradient_reference() -> None:
    initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    task = QuadraticTask(torch.zeros_like(initial))

    result = probe_same_state_alignment(
        NegativeGradientStudent(),
        initial,
        task,
        reference=ScaledGradientReference(),
        steps=1,
        trajectory="student",
    )

    assert len(result.steps) == 1
    assert result.student_reference_cosine_mean == pytest.approx(1.0)
    assert result.student_negative_gradient_cosine_mean == pytest.approx(1.0)
    assert result.reference_negative_gradient_cosine_mean == pytest.approx(1.0)
    assert result.alignment_delta_mean == pytest.approx(0.0)
    assert result.student_reference_norm_ratio_mean == pytest.approx(1.0)
    assert result.rollout.loss_ratio == pytest.approx(0.0)


def test_same_state_alignment_preserves_direction_but_exposes_scale() -> None:
    initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    task = QuadraticTask(torch.zeros_like(initial))

    result = probe_same_state_alignment(
        NegativeGradientStudent(),
        initial,
        task,
        reference=ScaledGradientReference(scale=2.0),
        steps=1,
        trajectory="reference",
    )

    assert result.student_reference_cosine_mean == pytest.approx(1.0)
    assert result.student_reference_norm_ratio_mean == pytest.approx(0.5)


def test_same_state_alignment_rejects_unknown_trajectory() -> None:
    initial = torch.ones((2, 2))
    task = QuadraticTask(torch.zeros_like(initial))

    with pytest.raises(ValueError, match="trajectory"):
        probe_same_state_alignment(
            NegativeGradientStudent(),
            initial,
            task,
            reference=ScaledGradientReference(),
            steps=1,
            trajectory="unknown",  # type: ignore[arg-type]
        )
