from __future__ import annotations

from typing import Protocol

from torch import Tensor

from optdistil.distill.features import build_elementwise_features
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState


class TeacherOptimizer(Protocol):
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor: ...


def collect_teacher_step(
    parameter: Tensor,
    grad: Tensor,
    *,
    teacher: TeacherOptimizer,
    student_state: StudentState,
    step: int,
    total_steps: int,
    teacher_name: str | None = None,
) -> TrajectoryRecord:
    """Record one teacher update using observations available to the tiny student."""
    momentum, second_moment = student_state.observe(grad)
    features = build_elementwise_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=step,
        total_steps=total_steps,
    )
    teacher_update = teacher.step(parameter, grad).detach().reshape(-1)
    return TrajectoryRecord(
        features=features.detach(),
        teacher_update=teacher_update,
        metadata={"step": step, "teacher": teacher_name or type(teacher).__name__},
    )
