from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.meta_train import evaluate_meta_teacher, train_meta_teacher
from optdistil.distill.rollout import (
    OptimizationTask,
    collect_teacher_trajectory,
    rollout_student,
    rollout_teacher,
    select_student_output_scale,
)
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.meta_attention import MetaAttentionTeacher
from optdistil.teachers.meta_mlp import MetaMLPTeacher

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (10.0, 100.0)
SCALE_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
DISTILL_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)


@dataclass(frozen=True, slots=True)
class TeacherResult:
    name: str
    parameters: int
    meta_validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]
    step_scale: float


@dataclass(frozen=True, slots=True)
class StudentResult:
    teacher_name: str
    seed: int
    parameters: int
    validation_loss_ratio: float
    validation_scale: float
    final_distillation_loss: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare elementwise and contextual learned-optimizer teachers."
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--meta-iterations", type=int, default=50)
    parser.add_argument("--train-tasks", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=4)
    parser.add_argument("--distill-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--teacher-seed", type=int, default=61000)
    parser.add_argument("--student-seed", type=int, default=71000)
    parser.add_argument("--outer-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick_config(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 8
    args.meta_iterations = 12
    args.train_tasks = 2
    args.validation_tasks = 2
    args.distill_tasks = 2
    args.student_validation_tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 8
    args.student_seeds = 2


def make_split(
    *,
    seed_base: int,
    count_per_condition: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[float, list[Case]]:
    return {
        condition: [
            make_frozen_readout_mlp(
                seed_base + 1000 * condition_index + index,
                hidden_dim=args.size,
                input_dim=args.size,
                output_dim=args.output_dim,
                samples=args.samples,
                input_condition=condition,
                device=device,
            )
            for index in range(count_per_condition)
        ]
        for condition_index, condition in enumerate(CONDITIONS)
    }


def flatten_split(split: dict[float, list[Case]]) -> list[Case]:
    return [case for condition in CONDITIONS for case in split[condition]]


def evaluate_teacher_split(
    teacher,
    split: dict[float, list[Case]],
    *,
    steps: int,
) -> tuple[float, dict[str, float]]:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        ratios: list[float] = []
        for initial, task in split[condition]:
            fresh = copy.deepcopy(teacher)
            fresh.reset()
            result = rollout_teacher(initial, task, teacher=fresh, steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def collect_records(teacher, cases: list[Case], *, steps: int, name: str):
    records = []
    for initial, task in cases:
        fresh = copy.deepcopy(teacher)
        fresh.reset()
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=fresh,
            steps=steps,
            teacher_name=name,
            feature_builder=build_matrix_aware_features,
        )
        records.extend(trajectory)
    return records


def evaluate_student_split(
    student: TinyMLPOptimizer,
    split: dict[float, list[Case]],
    *,
    steps: int,
) -> tuple[float, dict[str, float]]:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        ratios: list[float] = []
        for initial, task in split[condition]:
            result = rollout_student(
                student,
                initial,
                task,
                steps=steps,
                feature_builder=build_matrix_aware_features,
            )
            ratios.append(result.loss_ratio if result.finite else math.inf)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def main() -> None:
    args = parse_args()
    apply_quick_config(args)
    if min(
        args.size,
        args.output_dim,
        args.samples,
        args.steps,
        args.meta_iterations,
        args.train_tasks,
        args.validation_tasks,
        args.distill_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("dimensions, task counts, iterations, epochs, and seeds must be positive")

    device = torch.device(args.device)
    train_split = make_split(seed_base=41000, count_per_condition=args.train_tasks, args=args, device=device)
    validation_split = make_split(
        seed_base=46000,
        count_per_condition=args.validation_tasks,
        args=args,
        device=device,
    )
    distill_split = make_split(
        seed_base=51000,
        count_per_condition=args.distill_tasks,
        args=args,
        device=device,
    )
    student_validation_split = make_split(
        seed_base=56000,
        count_per_condition=args.student_validation_tasks,
        args=args,
        device=device,
    )
    test_split = make_split(seed_base=61000, count_per_condition=args.test_tasks, args=args, device=device)

    train_cases = flatten_split(train_split)
    validation_cases = flatten_split(validation_split)
    distill_cases = flatten_split(distill_split)
    student_validation_cases = flatten_split(student_validation_split)

    teacher_factories = {
        "mlp128": lambda: MetaMLPTeacher(
            hidden_dim=128,
            horizon=args.steps,
            beta1=0.9,
            beta2=0.99,
            initial_step_scale=0.1,
        ),
        "attention32x2": lambda: MetaAttentionTeacher(
            d_model=32,
            num_heads=4,
            depth=2,
            horizon=args.steps,
            beta1=0.9,
            beta2=0.99,
            initial_step_scale=0.1,
        ),
        "attention64x2": lambda: MetaAttentionTeacher(
            d_model=64,
            num_heads=4,
            depth=2,
            horizon=args.steps,
            beta1=0.9,
            beta2=0.99,
            initial_step_scale=0.1,
        ),
    }

    teacher_results: list[TeacherResult] = []
    student_results: list[StudentResult] = []
    histories: dict[str, list[dict]] = {}

    for teacher_index, (name, factory) in enumerate(teacher_factories.items()):
        torch.manual_seed(args.teacher_seed + teacher_index)
        teacher = factory().to(device)
        history = train_meta_teacher(
            teacher,
            train_cases,
            validation_cases,
            steps=args.steps,
            iterations=args.meta_iterations,
            outer_lr=args.outer_lr,
            grad_clip=1.0,
            validation_interval=2 if args.quick else 5,
        )
        histories[name] = [asdict(row) for row in history]
        validation_ratio = evaluate_meta_teacher(teacher, validation_cases, steps=args.steps)
        test_ratio, test_by_condition = evaluate_teacher_split(
            teacher,
            test_split,
            steps=args.steps,
        )
        teacher_results.append(
            TeacherResult(
                name=name,
                parameters=teacher.parameter_count,
                meta_validation_loss_ratio=validation_ratio,
                test_loss_ratio=test_ratio,
                test_loss_ratio_by_condition=test_by_condition,
                step_scale=float(teacher.step_scale.detach()),
            )
        )

        records = collect_records(teacher, distill_cases, steps=args.steps, name=name)
        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 100 * teacher_index + seed_index
            torch.manual_seed(seed)
            student = TinyMLPOptimizer(feature_dim=8, hidden_dim=8, hidden_layers=2).to(device)
            distill_history = train_student(
                student,
                records,
                epochs=args.distill_epochs,
                lr=3e-3,
                weights=DISTILL_WEIGHTS,
            )
            scale_result = select_student_output_scale(
                student,
                student_validation_cases,
                candidates=SCALE_CANDIDATES,
                steps=args.steps,
                feature_builder=build_matrix_aware_features,
            )
            student_test_ratio, student_by_condition = evaluate_student_split(
                student,
                test_split,
                steps=args.steps,
            )
            student_results.append(
                StudentResult(
                    teacher_name=name,
                    seed=seed,
                    parameters=student.parameter_count,
                    validation_loss_ratio=scale_result.validation_loss_ratio,
                    validation_scale=scale_result.scale,
                    final_distillation_loss=distill_history[-1],
                    test_loss_ratio=student_test_ratio,
                    test_loss_ratio_by_condition=student_by_condition,
                )
            )

    student_summary = {}
    for name in teacher_factories:
        group = [row for row in student_results if row.teacher_name == name]
        ratios = [row.test_loss_ratio for row in group]
        student_summary[name] = {
            "parameters": group[0].parameters,
            "test_loss_ratio_mean": statistics.fmean(ratios),
            "test_loss_ratio_seed_std": statistics.pstdev(ratios) if len(ratios) > 1 else 0.0,
            "validation_loss_ratio_mean": statistics.fmean(row.validation_loss_ratio for row in group),
            "final_distillation_loss_mean": statistics.fmean(
                row.final_distillation_loss for row in group
            ),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "teachers": [asdict(row) for row in teacher_results],
        "students": [asdict(row) for row in student_results],
        "student_summary": student_summary,
        "meta_histories": histories,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
