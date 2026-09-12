from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import (
    DISTILL_WEIGHTS,
    flatten,
    make_split,
    make_teacher,
    select_lr,
)
from probe_secant_history_pareto import collect_secant_records, select_secant_scale

from optdistil.distill.secant_features import SecantFeatureState
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer

SOURCES = ("norm_gradient", "muon", "newton_050")
HISTORY_SIZE = 4


@dataclass(frozen=True, slots=True)
class DecompositionRun:
    source: str
    seed: int
    student_parameters: int
    validation_scale: float
    test_loss_ratio: float
    cosine_student_secant_mean: float
    cosine_teacher_secant_mean: float
    cosine_student_teacher_mean: float
    residual_fraction_mean: float
    projection_gain_mean: float
    projection_gain_std: float
    projection_gain_min: float
    projection_gain_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a secant-aware distilled Student update into a global projection onto "
            "the normalized L-BFGS direction plus an orthogonal residual. This tests whether "
            "distillation mainly learns adaptive gain or a genuinely new direction."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=201000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.student_validation_tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 10
    args.student_seeds = 3


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    denominator = left.norm() * right.norm()
    if float(denominator) <= eps:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def projection_stats(update: torch.Tensor, direction: torch.Tensor, eps: float = 1e-12):
    update_flat = update.reshape(-1).float()
    direction_flat = direction.reshape(-1).float()
    denominator = torch.dot(direction_flat, direction_flat)
    if float(denominator) <= eps or float(update_flat.norm()) <= eps:
        return 0.0, 1.0
    gain = torch.dot(update_flat, direction_flat) / denominator
    residual = update_flat - gain * direction_flat
    residual_fraction = residual.norm() / update_flat.norm()
    return float(gain), float(residual_fraction)


@torch.no_grad()
def evaluate_decomposition(student, split, *, source: str, lrs, steps: int):
    ratios: list[float] = []
    student_secant: list[float] = []
    teacher_secant: list[float] = []
    student_teacher: list[float] = []
    residual_fractions: list[float] = []
    gains: list[float] = []

    student.eval()
    for cases in split.values():
        for initial, task in cases:
            parameter = initial.detach().clone()
            ema_state = StudentState(
                parameter.shape,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            secant_state = SecantFeatureState(history_size=HISTORY_SIZE)
            teacher = make_teacher(source, task, lrs)
            initial_loss = float(task.loss(parameter))
            final_loss = initial_loss

            for step in range(1, steps + 1):
                grad = task.grad(parameter)
                momentum, second_moment = ema_state.observe(grad)
                features = secant_state.build(
                    parameter,
                    grad,
                    momentum,
                    second_moment,
                    step=step,
                    total_steps=steps,
                )
                direction = features[:, 5].reshape_as(parameter)
                student_update = student(features).reshape_as(parameter)
                teacher_update = teacher.step(parameter, grad).detach()

                student_secant.append(cosine(student_update, direction))
                teacher_secant.append(cosine(teacher_update, direction))
                student_teacher.append(cosine(student_update, teacher_update))
                gain, residual_fraction = projection_stats(student_update, direction)
                gains.append(gain)
                residual_fractions.append(residual_fraction)

                parameter = parameter + student_update
                final_loss = float(task.loss(parameter))
                if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
                    final_loss = math.inf
                    break

            ratios.append(final_loss / max(abs(initial_loss), 1e-12))

    return {
        "test_loss_ratio": statistics.fmean(ratios),
        "cosine_student_secant_mean": statistics.fmean(student_secant),
        "cosine_teacher_secant_mean": statistics.fmean(teacher_secant),
        "cosine_student_teacher_mean": statistics.fmean(student_teacher),
        "residual_fraction_mean": statistics.fmean(residual_fractions),
        "projection_gain_mean": statistics.fmean(gains),
        "projection_gain_std": statistics.pstdev(gains),
        "projection_gain_min": min(gains),
        "projection_gain_max": max(gains),
    }


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, task counts, epochs, and seed counts must be positive")

    device = torch.device(args.device)
    lr_validation_split = make_split(
        seed_base=141000,
        count=args.lr_validation_tasks,
        size=args.size,
        device=device,
    )
    distill_split = make_split(
        seed_base=146000,
        count=args.distill_train_tasks,
        size=args.size,
        device=device,
    )
    student_validation_split = make_split(
        seed_base=151000,
        count=args.student_validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(
        seed_base=156000,
        count=args.test_tasks,
        size=args.size,
        device=device,
    )
    lr_cases = flatten(lr_validation_split)
    distill_cases = flatten(distill_split)
    student_validation_cases = flatten(student_validation_split)

    lrs = {
        source: select_lr(source, lr_cases, steps=args.steps)
        for source in ("muon", "norm_gradient")
    }
    records = {
        source: collect_secant_records(
            source,
            distill_cases,
            lrs=lrs,
            steps=args.steps,
            history_size=HISTORY_SIZE,
        )
        for source in SOURCES
    }

    runs: list[DecompositionRun] = []
    for source_index, source in enumerate(SOURCES):
        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 1000 * source_index + seed_index
            torch.manual_seed(seed)
            student = TinyMLPOptimizer(feature_dim=8, hidden_dim=8, hidden_layers=2).to(device)
            train_student(
                student,
                records[source],
                epochs=args.distill_epochs,
                lr=3e-3,
                weights=DISTILL_WEIGHTS,
            )
            scale, _ = select_secant_scale(
                student,
                student_validation_cases,
                steps=args.steps,
                history_size=HISTORY_SIZE,
            )
            metrics = evaluate_decomposition(
                student,
                test_split,
                source=source,
                lrs=lrs,
                steps=args.steps,
            )
            runs.append(
                DecompositionRun(
                    source=source,
                    seed=seed,
                    student_parameters=student.parameter_count,
                    validation_scale=scale,
                    **metrics,
                )
            )

    summary = {}
    for source in SOURCES:
        group = [run for run in runs if run.source == source]
        summary[source] = {
            field: statistics.fmean(getattr(run, field) for run in group)
            for field in (
                "test_loss_ratio",
                "cosine_student_secant_mean",
                "cosine_teacher_secant_mean",
                "cosine_student_teacher_mean",
                "residual_fraction_mean",
                "projection_gain_mean",
                "projection_gain_std",
            )
        }
        summary[source]["student_parameters"] = group[0].student_parameters
        summary[source]["student_seed_count"] = len(group)

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_size": HISTORY_SIZE,
        "tuned_lrs": lrs,
        "runs": [asdict(run) for run in runs],
        "summary": summary,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
