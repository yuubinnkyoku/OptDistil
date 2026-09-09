from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from optdistil.distill.collect import TeacherOptimizer, collect_teacher_step
from optdistil.distill.features import FeatureBuilder, build_elementwise_features
from optdistil.distill.losses import DistillationLossWeights, distillation_loss
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer


class OptimizationTask(Protocol):
    def loss(self, parameter: Tensor) -> Tensor: ...

    def grad(self, parameter: Tensor) -> Tensor: ...


class HessianVectorTask(OptimizationTask, Protocol):
    def hessian_vector(self, vector: Tensor) -> Tensor: ...


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
    def normalized_aulc(self) -> float:
        """Trapezoidal area under the loss curve normalized by the initial loss and horizon."""
        if len(self.losses) <= 1:
            return 1.0
        denominator = max(abs(self.initial_loss), 1e-12)
        area = sum(
            0.5 * (left + right)
            for left, right in zip(self.losses[:-1], self.losses[1:], strict=True)
        )
        return area / ((len(self.losses) - 1) * denominator)

    @property
    def finite(self) -> bool:
        return all(torch.isfinite(torch.tensor(value)).item() for value in self.losses)


@dataclass(frozen=True, slots=True)
class ScaleSelectionResult:
    """Best non-trainable output scale selected on validation rollouts."""

    scale: float
    validation_loss_ratio: float
    scores: tuple[tuple[float, float], ...]


@torch.no_grad()
def collect_teacher_trajectory(
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    teacher: TeacherOptimizer,
    steps: int,
    teacher_name: str | None = None,
    feature_builder: FeatureBuilder = build_elementwise_features,
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
            feature_builder=feature_builder,
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
def rollout_teacher(
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    teacher: TeacherOptimizer,
    steps: int,
) -> RolloutResult:
    """Evaluate a fresh teacher instance without constructing student features."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    parameter = initial_parameter.detach().clone()
    losses = [float(task.loss(parameter))]
    for _ in range(steps):
        grad = task.grad(parameter)
        update = teacher.step(parameter, grad).detach()
        parameter = parameter + update
        losses.append(float(task.loss(parameter)))
        if not torch.isfinite(parameter).all():
            break

    return RolloutResult(tuple(losses), parameter.detach().clone())


@torch.no_grad()
def rollout_exact_line_search_gradient(
    initial_parameter: Tensor,
    task: HessianVectorTask,
    *,
    steps: int,
    eps: float = 1e-12,
) -> RolloutResult:
    """Steepest descent with the exact per-step line search for a quadratic objective.

    For a quadratic with Hessian H and gradient g, the minimizing step along -g is

        alpha = <g, g> / <g, H g>.

    This is a deliberately strong gradient-only oracle: it removes learning-rate tuning as
    an explanation for an optimizer's advantage while preserving the raw gradient direction.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    parameter = initial_parameter.detach().clone()
    losses = [float(task.loss(parameter))]
    for _ in range(steps):
        grad = task.grad(parameter)
        grad_float = grad.float()
        numerator = grad_float.square().sum()
        if float(numerator) <= eps:
            losses.append(float(task.loss(parameter)))
            continue

        h_grad = task.hessian_vector(grad).float()
        denominator = (grad_float * h_grad).sum()
        if not torch.isfinite(denominator) or float(denominator) <= eps:
            break

        alpha = numerator / denominator
        parameter = parameter - grad * alpha.to(device=grad.device, dtype=grad.dtype)
        losses.append(float(task.loss(parameter)))
        if not torch.isfinite(parameter).all():
            break

    return RolloutResult(tuple(losses), parameter.detach().clone())


@torch.no_grad()
def rollout_student(
    student: nn.Module,
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    steps: int,
    feature_builder: FeatureBuilder = build_elementwise_features,
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
        features = feature_builder(
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
def select_student_output_scale(
    student: TinyMLPOptimizer,
    validation_cases: Iterable[tuple[Tensor, OptimizationTask]],
    *,
    candidates: Iterable[float],
    steps: int,
    feature_builder: FeatureBuilder = build_elementwise_features,
) -> ScaleSelectionResult:
    """Select one deployment scale using downstream validation loss, not teacher norms.

    This is a tiny post-distillation hyperparameter search. It changes only the student's
    non-trainable scalar buffer, so the learned optimizer remains the same size and the
    deployment-time cost remains one scalar multiply per update tensor.
    """
    validation_cases = list(validation_cases)
    candidates = list(candidates)
    if not validation_cases:
        raise ValueError("at least one validation case is required")
    if not candidates:
        raise ValueError("at least one scale candidate is required")
    if steps <= 0:
        raise ValueError("steps must be positive")

    scores: list[tuple[float, float]] = []
    for scale in candidates:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale candidates must be positive and finite")
        student.set_output_scale(scale)
        ratios = []
        for initial_parameter, task in validation_cases:
            result = rollout_student(
                student,
                initial_parameter,
                task,
                steps=steps,
                feature_builder=feature_builder,
            )
            ratios.append(result.loss_ratio if result.finite else math.inf)
        score = sum(ratios) / len(ratios)
        scores.append((scale, score))

    best_scale, best_score = min(scores, key=lambda item: item[1])
    if not math.isfinite(best_score):
        raise ValueError("all scale candidates produced non-finite validation rollouts")
    student.set_output_scale(best_scale)
    return ScaleSelectionResult(best_scale, best_score, tuple(scores))


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
