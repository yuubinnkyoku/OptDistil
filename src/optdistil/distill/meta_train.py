from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from optdistil.distill.rollout import OptimizationTask, rollout_teacher
from optdistil.teachers.meta_mlp import MetaMLPTeacher

Case = tuple[Tensor, OptimizationTask]


@dataclass(frozen=True, slots=True)
class MetaTrainStep:
    iteration: int
    train_objective: float
    validation_loss_ratio: float
    step_scale: float
    grad_norm: float


@dataclass(frozen=True, slots=True)
class DifferentiableRolloutResult:
    final_loss_ratio: Tensor
    mean_loss_ratio: Tensor
    final_parameter: Tensor


def differentiable_rollout(
    teacher: MetaMLPTeacher,
    initial_parameter: Tensor,
    task: OptimizationTask,
    *,
    steps: int,
    eps: float = 1e-12,
) -> DifferentiableRolloutResult:
    """Unroll a learned optimizer while preserving meta-gradients."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    parameter = initial_parameter.detach().clone()
    state = teacher.initial_state(parameter)
    initial_loss = task.loss(parameter).detach().clamp_min(eps)
    ratios: list[Tensor] = []

    for _ in range(steps):
        grad = task.grad(parameter)
        update, state = teacher.functional_step(parameter, grad, state)
        parameter = parameter + update
        ratio = task.loss(parameter) / initial_loss
        ratios.append(ratio)

    return DifferentiableRolloutResult(
        final_loss_ratio=ratios[-1],
        mean_loss_ratio=torch.stack(ratios).mean(),
        final_parameter=parameter,
    )


def meta_objective(
    teacher: MetaMLPTeacher,
    cases: list[Case],
    *,
    steps: int,
    final_weight: float = 0.7,
) -> Tensor:
    """Average normalized long-horizon objective across an inner-task distribution."""
    if not cases:
        raise ValueError("at least one meta-training case is required")
    if not 0.0 <= final_weight <= 1.0:
        raise ValueError("final_weight must lie in [0, 1]")

    objectives: list[Tensor] = []
    for initial_parameter, task in cases:
        result = differentiable_rollout(teacher, initial_parameter, task, steps=steps)
        objectives.append(
            final_weight * result.final_loss_ratio
            + (1.0 - final_weight) * result.mean_loss_ratio
        )
    return torch.stack(objectives).mean()


@torch.no_grad()
def evaluate_meta_teacher(
    teacher: MetaMLPTeacher,
    cases: list[Case],
    *,
    steps: int,
) -> float:
    """Evaluate a learned teacher with fresh optimizer state on every task."""
    if not cases:
        raise ValueError("at least one evaluation case is required")

    ratios: list[float] = []
    for initial_parameter, task in cases:
        fresh_teacher = copy.deepcopy(teacher)
        fresh_teacher.reset()
        result = rollout_teacher(
            initial_parameter,
            task,
            teacher=fresh_teacher,
            steps=steps,
        )
        ratios.append(result.loss_ratio if result.finite else math.inf)
    return sum(ratios) / len(ratios)


def _clone_state_dict(module: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def train_meta_teacher(
    teacher: MetaMLPTeacher,
    train_cases: list[Case],
    validation_cases: list[Case],
    *,
    steps: int,
    iterations: int = 50,
    outer_lr: float = 3e-3,
    grad_clip: float = 1.0,
    final_weight: float = 0.7,
    validation_interval: int = 1,
) -> list[MetaTrainStep]:
    """Outer-train a learned optimizer and restore its best validation checkpoint."""
    if not train_cases or not validation_cases:
        raise ValueError("train and validation cases must both be non-empty")
    if iterations <= 0 or steps <= 0:
        raise ValueError("iterations and steps must be positive")
    if outer_lr <= 0.0 or grad_clip <= 0.0:
        raise ValueError("outer_lr and grad_clip must be positive")
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive")

    optimizer = torch.optim.Adam(teacher.parameters(), lr=outer_lr)
    best_validation = evaluate_meta_teacher(teacher, validation_cases, steps=steps)
    best_state = _clone_state_dict(teacher)
    history: list[MetaTrainStep] = []

    for iteration in range(1, iterations + 1):
        teacher.train()
        optimizer.zero_grad(set_to_none=True)
        objective = meta_objective(
            teacher,
            train_cases,
            steps=steps,
            final_weight=final_weight,
        )
        if not torch.isfinite(objective):
            raise RuntimeError(f"non-finite meta-objective at iteration {iteration}")
        objective.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(teacher.parameters(), grad_clip)
        grad_norm = float(grad_norm_tensor.detach())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite meta-gradient at iteration {iteration}")
        optimizer.step()

        if iteration % validation_interval == 0 or iteration == iterations:
            teacher.eval()
            validation_ratio = evaluate_meta_teacher(
                teacher,
                validation_cases,
                steps=steps,
            )
            if validation_ratio < best_validation:
                best_validation = validation_ratio
                best_state = _clone_state_dict(teacher)
        else:
            validation_ratio = history[-1].validation_loss_ratio if history else best_validation

        history.append(
            MetaTrainStep(
                iteration=iteration,
                train_objective=float(objective.detach()),
                validation_loss_ratio=validation_ratio,
                step_scale=float(teacher.step_scale.detach()),
                grad_norm=grad_norm,
            )
        )

    teacher.load_state_dict(best_state)
    teacher.eval()
    teacher.reset()
    return history
