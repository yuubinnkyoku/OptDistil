from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from optdistil.distill.losses import DistillationLossWeights, distillation_loss
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def train_student(
    student: nn.Module,
    records: Iterable[TrajectoryRecord],
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weights: DistillationLossWeights | None = None,
) -> list[float]:
    """Distill a student on a small in-memory trajectory collection.

    This intentionally uses AdamW only as the *outer* optimizer for the first research
    milestone. Replacing the outer optimizer is orthogonal to the distillation API.
    """
    records = list(records)
    if not records:
        raise ValueError("at least one trajectory record is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    outer = torch.optim.AdamW(student.parameters(), lr=lr)
    history: list[float] = []
    student.train()

    for _ in range(epochs):
        total = 0.0
        for record in records:
            outer.zero_grad(set_to_none=True)
            predicted = student(record.features)
            loss, _ = distillation_loss(predicted, record.teacher_update, weights=weights)
            loss.backward()
            outer.step()
            total += float(loss.detach())
        history.append(total / len(records))

    return history


@torch.no_grad()
def magnitude_calibration_scale(
    student: nn.Module,
    records: Iterable[TrajectoryRecord],
    *,
    eps: float = 1e-8,
) -> float:
    """Return the global scale minimizing average squared log-norm error.

    For records ``k``, the current magnitude objective is

    ``(log ||s * u_k|| - log ||t_k||)^2``.

    Its optimum over one positive global multiplier is the geometric mean of the
    teacher/student norm ratios. This calibration changes no update directions.
    """
    records = list(records)
    if not records:
        raise ValueError("at least one trajectory record is required")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    log_ratios = []
    student.eval()
    for record in records:
        predicted_norm = student(record.features).reshape(-1).norm().clamp_min(eps)
        teacher_norm = record.teacher_update.reshape(-1).norm().clamp_min(eps)
        log_ratios.append(teacher_norm.log() - predicted_norm.log())

    scale = torch.stack(log_ratios).mean().exp()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("computed calibration scale is not positive and finite")
    return float(scale)


@torch.no_grad()
def calibrate_student_magnitude(
    student: TinyMLPOptimizer,
    records: Iterable[TrajectoryRecord],
    *,
    eps: float = 1e-8,
) -> float:
    """Apply the closed-form global magnitude calibration and return the multiplier."""
    records = list(records)
    multiplier = magnitude_calibration_scale(student, records, eps=eps)
    current_scale = float(student.output_scale)
    student.set_output_scale(current_scale * multiplier)
    return multiplier
