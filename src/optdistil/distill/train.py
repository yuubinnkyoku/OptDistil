from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from optdistil.distill.losses import DistillationLossWeights, distillation_loss
from optdistil.distill.trajectory import TrajectoryRecord


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
