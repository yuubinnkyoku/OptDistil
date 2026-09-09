from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from optdistil.distill.collect import TeacherOptimizer
from optdistil.distill.features import FeatureBuilder, build_elementwise_features
from optdistil.distill.rollout import OptimizationTask, RolloutResult
from optdistil.students.tiny_mlp import StudentState

TrajectoryDriver = Literal["student", "reference"]


@dataclass(frozen=True, slots=True)
class AlignmentStep:
    """Direction and magnitude agreement measured on one shared optimizer state."""

    step: int
    student_reference_cosine: float
    student_negative_gradient_cosine: float
    reference_negative_gradient_cosine: float
    student_reference_norm_ratio: float


@dataclass(frozen=True, slots=True)
class SameStateAlignmentResult:
    """Alignment trace while either the student or reference drives the trajectory."""

    trajectory: TrajectoryDriver
    steps: tuple[AlignmentStep, ...]
    rollout: RolloutResult

    @property
    def student_reference_cosine_mean(self) -> float:
        return _mean(step.student_reference_cosine for step in self.steps)

    @property
    def student_negative_gradient_cosine_mean(self) -> float:
        return _mean(step.student_negative_gradient_cosine for step in self.steps)

    @property
    def reference_negative_gradient_cosine_mean(self) -> float:
        return _mean(step.reference_negative_gradient_cosine for step in self.steps)

    @property
    def alignment_delta_mean(self) -> float:
        """Positive means the student points closer to the reference than to -gradient."""
        return _mean(
            step.student_reference_cosine - step.student_negative_gradient_cosine
            for step in self.steps
        )

    @property
    def student_reference_norm_ratio_mean(self) -> float:
        return _mean(step.student_reference_norm_ratio for step in self.steps)


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def flattened_cosine(left: Tensor, right: Tensor, *, eps: float = 1e-12) -> float:
    """Cosine similarity after flattening, returning zero for a near-zero direction."""
    if left.shape != right.shape:
        raise ValueError("alignment tensors must have the same shape")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator) <= eps:
        return 0.0
    cosine = (left_flat @ right_flat) / denominator
    return float(cosine.clamp(-1.0, 1.0))


@torch.no_grad()
def probe_same_state_alignment(
    student: nn.Module,
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    reference: TeacherOptimizer,
    steps: int,
    trajectory: TrajectoryDriver,
    feature_builder: FeatureBuilder = build_elementwise_features,
    eps: float = 1e-12,
) -> SameStateAlignmentResult:
    """Compare student, reference, and -gradient directions on exactly the same states.

    The selected ``trajectory`` chooses only which update advances the parameter. Both the
    student state and the stateful reference optimizer observe the same gradient sequence,
    so every reported cosine compares updates computed from an identical parameter/gradient
    history. Running both trajectory modes checks that a conclusion is not an artifact of
    evaluating only on states visited by one optimizer.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if trajectory not in {"student", "reference"}:
        raise ValueError("trajectory must be 'student' or 'reference'")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    parameter = initial_parameter.detach().clone()
    state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    losses = [float(task.loss(parameter))]
    alignment_steps: list[AlignmentStep] = []
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
        student_update = student(features).reshape_as(parameter)
        reference_update = reference.step(parameter, grad).detach().reshape_as(parameter)
        negative_gradient = -grad

        if not (
            torch.isfinite(student_update).all()
            and torch.isfinite(reference_update).all()
            and torch.isfinite(negative_gradient).all()
        ):
            break

        reference_norm = reference_update.float().norm().clamp_min(eps)
        norm_ratio = student_update.float().norm() / reference_norm
        alignment_steps.append(
            AlignmentStep(
                step=step,
                student_reference_cosine=flattened_cosine(
                    student_update, reference_update, eps=eps
                ),
                student_negative_gradient_cosine=flattened_cosine(
                    student_update, negative_gradient, eps=eps
                ),
                reference_negative_gradient_cosine=flattened_cosine(
                    reference_update, negative_gradient, eps=eps
                ),
                student_reference_norm_ratio=float(norm_ratio),
            )
        )

        driver_update = student_update if trajectory == "student" else reference_update
        parameter = parameter + driver_update
        losses.append(float(task.loss(parameter)))
        if not torch.isfinite(parameter).all():
            break

    return SameStateAlignmentResult(
        trajectory=trajectory,
        steps=tuple(alignment_steps),
        rollout=RolloutResult(tuple(losses), parameter.detach().clone()),
    )
