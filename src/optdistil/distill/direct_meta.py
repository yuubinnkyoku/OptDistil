from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from optdistil.distill.features import FeatureBuilder, build_matrix_aware_features
from optdistil.distill.rollout import OptimizationTask, rollout_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer

Case = tuple[Tensor, OptimizationTask]


@dataclass(frozen=True, slots=True)
class DirectMetaTrainStep:
    iteration: int
    train_objective: float
    validation_loss_ratio: float
    grad_norm: float


def zero_initialize_student_output(student: TinyMLPOptimizer) -> None:
    """Make a TinyMLP start from the no-update policy without changing its size."""
    final_linear = next(
        module for module in reversed(student.network) if isinstance(module, nn.Linear)
    )
    nn.init.zeros_(final_linear.weight)
    if final_linear.bias is not None:
        nn.init.zeros_(final_linear.bias)


def differentiable_student_rollout(
    student: TinyMLPOptimizer,
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """First-order differentiable rollout of the deployment student.

    Student-visible observations are intentionally stop-gradient, matching the stored
    trajectory interface used during distillation. Meta-gradients still flow through
    every predicted update and through the accumulated inner parameter. This provides a
    strong direct-meta baseline without granting the tiny student richer information than
    it receives after distillation.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("EMA betas must lie in [0, 1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    parameter = initial_parameter.detach().clone()
    momentum = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    initial_loss = task.loss(parameter).detach().clamp_min(eps)
    ratios: list[Tensor] = []

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        detached_grad = grad.detach()
        momentum = beta1 * momentum + (1.0 - beta1) * detached_grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * detached_grad.square()
        features = feature_builder(
            parameter,
            detached_grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        update = student(features).reshape_as(parameter)
        parameter = parameter + update
        ratios.append(task.loss(parameter) / initial_loss)

    return ratios[-1], torch.stack(ratios).mean()


def direct_meta_objective(
    student: TinyMLPOptimizer,
    cases: list[Case],
    *,
    steps: int,
    final_weight: float = 0.7,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
) -> Tensor:
    if not cases:
        raise ValueError("at least one meta-training case is required")
    if not 0.0 <= final_weight <= 1.0:
        raise ValueError("final_weight must lie in [0, 1]")

    objectives: list[Tensor] = []
    for initial, task in cases:
        final_ratio, mean_ratio = differentiable_student_rollout(
            student,
            initial,
            task,
            steps=steps,
            feature_builder=feature_builder,
        )
        objectives.append(final_weight * final_ratio + (1.0 - final_weight) * mean_ratio)
    return torch.stack(objectives).mean()


@torch.no_grad()
def evaluate_direct_student(
    student: TinyMLPOptimizer,
    cases: list[Case],
    *,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
) -> float:
    if not cases:
        raise ValueError("at least one evaluation case is required")
    ratios: list[float] = []
    for initial, task in cases:
        result = rollout_student(
            student,
            initial,
            task,
            steps=steps,
            feature_builder=feature_builder,
        )
        ratios.append(result.loss_ratio if result.finite else math.inf)
    return sum(ratios) / len(ratios)


def _clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def train_direct_student(
    student: TinyMLPOptimizer,
    train_cases: list[Case],
    validation_cases: list[Case],
    *,
    steps: int,
    iterations: int = 50,
    outer_lr: float = 3e-3,
    grad_clip: float = 1.0,
    final_weight: float = 0.7,
    validation_interval: int = 1,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
) -> list[DirectMetaTrainStep]:
    """Directly meta-train the fixed deployment student and restore best validation state."""
    if not train_cases or not validation_cases:
        raise ValueError("train and validation cases must both be non-empty")
    if steps <= 0 or iterations <= 0:
        raise ValueError("steps and iterations must be positive")
    if outer_lr <= 0.0 or grad_clip <= 0.0:
        raise ValueError("outer_lr and grad_clip must be positive")
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive")

    outer = torch.optim.Adam(student.parameters(), lr=outer_lr)
    best_validation = evaluate_direct_student(
        student,
        validation_cases,
        steps=steps,
        feature_builder=feature_builder,
    )
    best_state = _clone_state_dict(student)
    history: list[DirectMetaTrainStep] = []

    for iteration in range(1, iterations + 1):
        student.train()
        outer.zero_grad(set_to_none=True)
        objective = direct_meta_objective(
            student,
            train_cases,
            steps=steps,
            final_weight=final_weight,
            feature_builder=feature_builder,
        )
        if not torch.isfinite(objective):
            raise RuntimeError(f"non-finite direct meta-objective at iteration {iteration}")
        objective.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        grad_norm = float(grad_norm_tensor.detach())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite direct meta-gradient at iteration {iteration}")
        outer.step()

        if iteration % validation_interval == 0 or iteration == iterations:
            validation_ratio = evaluate_direct_student(
                student,
                validation_cases,
                steps=steps,
                feature_builder=feature_builder,
            )
            if validation_ratio < best_validation:
                best_validation = validation_ratio
                best_state = _clone_state_dict(student)
        else:
            validation_ratio = history[-1].validation_loss_ratio if history else best_validation

        history.append(
            DirectMetaTrainStep(
                iteration=iteration,
                train_objective=float(objective.detach()),
                validation_loss_ratio=validation_ratio,
                grad_norm=grad_norm,
            )
        )

    student.load_state_dict(best_state)
    student.eval()
    return history
