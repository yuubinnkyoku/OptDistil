from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def direction_loss(student_update: Tensor, teacher_update: Tensor, eps: float = 1e-8) -> Tensor:
    """Cosine-direction loss over one flattened parameter tensor."""
    student = student_update.reshape(-1)
    teacher = teacher_update.reshape(-1)
    if student.shape != teacher.shape:
        raise ValueError("student and teacher updates must have the same number of elements")
    denominator = student.norm() * teacher.norm()
    cosine = torch.dot(student, teacher) / denominator.clamp_min(eps)
    return 1.0 - cosine


def magnitude_loss(student_update: Tensor, teacher_update: Tensor, eps: float = 1e-8) -> Tensor:
    """Squared error between log update norms."""
    student_norm = student_update.reshape(-1).norm().clamp_min(eps)
    teacher_norm = teacher_update.reshape(-1).norm().clamp_min(eps)
    return (student_norm.log() - teacher_norm.log()).square()


@dataclass(frozen=True, slots=True)
class DistillationLossWeights:
    direction: float = 0.7
    magnitude: float = 0.3


def distillation_loss(
    student_update: Tensor,
    teacher_update: Tensor,
    *,
    weights: DistillationLossWeights = DistillationLossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    direction = direction_loss(student_update, teacher_update)
    magnitude = magnitude_loss(student_update, teacher_update)
    total = weights.direction * direction + weights.magnitude * magnitude
    return total, {"total": total, "direction": direction, "magnitude": magnitude}
