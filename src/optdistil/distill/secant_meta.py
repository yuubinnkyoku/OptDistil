from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from optdistil.distill.rollout import OptimizationTask
from optdistil.distill.secant_features import SecantFeatureState
from optdistil.students.tiny_mlp import TinyMLPOptimizer

Case = tuple[Tensor, OptimizationTask]
SecantUpdateMode = Literal["full_update", "scalar_gain"]


@dataclass(frozen=True, slots=True)
class SecantMetaTrainStep:
    iteration: int
    train_objective: float
    validation_loss_ratio: float
    grad_norm: float


def _student_update(
    student: TinyMLPOptimizer,
    features: Tensor,
    *,
    mode: SecantUpdateMode,
    gain_normalization: float,
) -> Tensor:
    if mode == "full_update":
        return student(features)
    if mode == "scalar_gain":
        gain = gain_normalization * student(features).mean()
        return gain * features[:, 5]
    raise ValueError(f"unknown secant update mode: {mode}")


def differentiable_secant_student_rollout(
    student: TinyMLPOptimizer,
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    steps: int,
    history_size: int = 4,
    mode: SecantUpdateMode = "full_update",
    gain_normalization: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Differentiable closed-loop rollout with stop-gradient L-BFGS observations.

    The deployment observations are treated as constants, matching supervised trajectory
    distillation. Gradients still flow through every predicted update and accumulated model
    parameter, so task loss can repair teacher-forcing distribution shift without granting
    the Student privileged differentiable curvature information.
    """
    if steps <= 0 or history_size <= 0:
        raise ValueError("steps and history_size must be positive")
    if gain_normalization <= 0.0 or not math.isfinite(gain_normalization):
        raise ValueError("gain_normalization must be positive and finite")

    parameter = initial_parameter.detach().clone()
    momentum = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    secant_state = SecantFeatureState(history_size=history_size)
    initial_loss = task.loss(parameter).detach().clamp_min(eps)
    ratios: list[Tensor] = []

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        detached_grad = grad.detach()
        momentum = beta1 * momentum + (1.0 - beta1) * detached_grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * detached_grad.square()
        features = secant_state.build(
            parameter.detach(),
            detached_grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        update = _student_update(
            student,
            features,
            mode=mode,
            gain_normalization=gain_normalization,
        ).reshape_as(parameter)
        parameter = parameter + update
        ratios.append(task.loss(parameter) / initial_loss)

    return ratios[-1], torch.stack(ratios).mean()


def secant_meta_objective(
    student: TinyMLPOptimizer,
    cases: list[Case],
    *,
    steps: int,
    history_size: int = 4,
    mode: SecantUpdateMode = "full_update",
    gain_normalization: float = 1.0,
    final_weight: float = 0.7,
) -> Tensor:
    if not cases:
        raise ValueError("at least one meta-training case is required")
    if not 0.0 <= final_weight <= 1.0:
        raise ValueError("final_weight must lie in [0, 1]")

    objectives = []
    for initial, task in cases:
        final_ratio, mean_ratio = differentiable_secant_student_rollout(
            student,
            initial,
            task,
            steps=steps,
            history_size=history_size,
            mode=mode,
            gain_normalization=gain_normalization,
        )
        objectives.append(final_weight * final_ratio + (1.0 - final_weight) * mean_ratio)
    return torch.stack(objectives).mean()


@torch.no_grad()
def evaluate_secant_meta_student(
    student: TinyMLPOptimizer,
    cases: list[Case],
    *,
    steps: int,
    history_size: int = 4,
    mode: SecantUpdateMode = "full_update",
    gain_normalization: float = 1.0,
) -> float:
    if not cases:
        raise ValueError("at least one evaluation case is required")

    ratios: list[float] = []
    student.eval()
    for initial, task in cases:
        parameter = initial.detach().clone()
        momentum = torch.zeros_like(parameter)
        second_moment = torch.zeros_like(parameter)
        secant_state = SecantFeatureState(history_size=history_size)
        initial_loss = float(task.loss(parameter))
        final_loss = initial_loss

        for step in range(1, steps + 1):
            grad = task.grad(parameter)
            momentum = 0.9 * momentum + 0.1 * grad
            second_moment = 0.999 * second_moment + 0.001 * grad.square()
            features = secant_state.build(
                parameter,
                grad,
                momentum,
                second_moment,
                step=step,
                total_steps=steps,
            )
            update = _student_update(
                student,
                features,
                mode=mode,
                gain_normalization=gain_normalization,
            ).reshape_as(parameter)
            parameter = parameter + update
            final_loss = float(task.loss(parameter))
            if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
                final_loss = math.inf
                break

        ratios.append(final_loss / max(abs(initial_loss), 1e-12))
    return sum(ratios) / len(ratios)


def _clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def train_secant_meta_student(
    student: TinyMLPOptimizer,
    train_cases: list[Case],
    validation_cases: list[Case],
    *,
    steps: int,
    history_size: int = 4,
    mode: SecantUpdateMode = "full_update",
    gain_normalization: float = 1.0,
    iterations: int = 20,
    outer_lr: float = 1e-3,
    grad_clip: float = 1.0,
    final_weight: float = 0.7,
    validation_interval: int = 1,
) -> list[SecantMetaTrainStep]:
    """Meta-finetune a distilled secant-aware Student and restore best validation state."""
    if not train_cases or not validation_cases:
        raise ValueError("train and validation cases must both be non-empty")
    if iterations <= 0 or outer_lr <= 0.0 or grad_clip <= 0.0:
        raise ValueError("iterations, outer_lr, and grad_clip must be positive")
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive")

    outer = torch.optim.Adam(student.parameters(), lr=outer_lr)
    best_validation = evaluate_secant_meta_student(
        student,
        validation_cases,
        steps=steps,
        history_size=history_size,
        mode=mode,
        gain_normalization=gain_normalization,
    )
    best_state = _clone_state_dict(student)
    history: list[SecantMetaTrainStep] = []

    for iteration in range(1, iterations + 1):
        student.train()
        outer.zero_grad(set_to_none=True)
        objective = secant_meta_objective(
            student,
            train_cases,
            steps=steps,
            history_size=history_size,
            mode=mode,
            gain_normalization=gain_normalization,
            final_weight=final_weight,
        )
        if not torch.isfinite(objective):
            raise RuntimeError(f"non-finite secant meta-objective at iteration {iteration}")
        objective.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        grad_norm = float(grad_norm_tensor.detach())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite secant meta-gradient at iteration {iteration}")
        outer.step()

        if iteration % validation_interval == 0 or iteration == iterations:
            validation_ratio = evaluate_secant_meta_student(
                student,
                validation_cases,
                steps=steps,
                history_size=history_size,
                mode=mode,
                gain_normalization=gain_normalization,
            )
            if validation_ratio < best_validation:
                best_validation = validation_ratio
                best_state = _clone_state_dict(student)
        else:
            validation_ratio = history[-1].validation_loss_ratio if history else best_validation

        history.append(
            SecantMetaTrainStep(
                iteration=iteration,
                train_objective=float(objective.detach()),
                validation_loss_ratio=validation_ratio,
                grad_norm=grad_norm,
            )
        )

    student.load_state_dict(best_state)
    student.eval()
    return history
