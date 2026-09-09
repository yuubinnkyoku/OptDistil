from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from optdistil.distill.collect import TeacherOptimizer, collect_teacher_step
from optdistil.distill.features import build_elementwise_features
from optdistil.distill.losses import DistillationLossWeights, distillation_loss
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState


class OptimizationTask(Protocol):
    def loss(self, parameter: Tensor) -> Tensor: ...

    def grad(self, parameter: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Loss trace and final parameter from an optimizer rollout."""

    losses: tuple[float, ...]
    final_parameter: Tensor

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    @property
    def loss_ratio(self) -> float:
        denominator = max(abs(self.initial_loss), 1e-12)
        return self.final_loss / denominator

    @property
    def finite(self) -> bool:
        return all(torch.isfinite(torch.tensor(value)).item() for value in self.losses)


@torch.no_grad()
def collect_teacher_trajectory(
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    teacher: TeacherOptimizer,
    steps: int,
    teacher_name: str | None = None,
) -> tuple[list[TrajectoryRecord], RolloutResult]:
    """Roll a teacher forward while recording only student-visible observations."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    parameter = initial_parameter.detach().clone()
    student_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    losses = [float(task.loss(parameter))]
    records: list[TrajectoryRecord] = []

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        record = collect_teacher_step(
            parameter,
            grad,
            teacher=teacher,
            student_state=student_state,
            step=step,
            total_steps=steps,
            teacher_name=teacher_name,
        )
        loss_before = losses[-1]
        parameter = parameter + record.teacher_update.reshape_as(parameter)
        loss_after = float(task.loss(parameter))
        record.metadata.update(
            {
                "loss_before": loss_before,
                "loss_after": loss_after,
            }
        )
        records.append(record)
        losses.append(loss_after)

    return records, RolloutResult(tuple(losses), parameter.detach().clone())


@torch.no_grad()
def rollout_student(
    student: nn.Module,
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    steps: int,
) -> RolloutResult:
    """Use a distilled elementwise student as an optimizer on a fresh task."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    parameter = initial_parameter.detach().clone()
    state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    losses = [float(task.loss(parameter))]
    student.eval()

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        momentum, second_moment = state.observe(grad)
        features = build_elementwise_features(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        update = student(features).reshape_as(parameter)
        parameter = parameter + update
        losses.append(float(task.loss(parameter)))
        if not torch.isfinite(parameter).all():
            break

    return RolloutResult(tuple(losses), parameter.detach().clone())


@torch.no_grad()
def evaluate_imitation(
    student: nn.Module,
    records: list[TrajectoryRecord],
    *,
    weights: DistillationLossWeights | None = None,
) -> dict[str, float]:
    """Average distillation losses over held-out teacher trajectory records."""
    if not records:
        raise ValueError("at least one trajectory record is required")

    totals = {"total": 0.0, "direction": 0.0, "magnitude": 0.0}
    student.eval()
    for record in records:
        predicted = student(record.features)
        _, parts = distillation_loss(predicted, record.teacher_update, weights=weights)
        for name in totals:
            totals[name] += float(parts[name])

    return {name: value / len(records) for name, value in totals.items()}
