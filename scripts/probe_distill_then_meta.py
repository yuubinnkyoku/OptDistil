from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.direct_meta import evaluate_direct_student, train_direct_student
from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.meta_train import train_meta_teacher
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
from optdistil.teachers.meta_mlp import MetaMLPTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (10.0, 100.0)
NORMGRAD_LR_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2)
STUDENT_SCALE_CANDIDATES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
OBJECTIVES = (
    ("direction_only", DistillationLossWeights(direction=1.0, magnitude=0.0)),
    ("joint", DistillationLossWeights(direction=0.7, magnitude=0.3)),
)


@dataclass(frozen=True, slots=True)
class InitializationResult:
    source: str
    teacher_parameters: int | None
    distillation_objective: str | None
    student_seed: int
    student_parameters: int
    pre_meta_validation_loss_ratio: float
    post_meta_validation_loss_ratio: float
    output_scale: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare analytic and learned-teacher distillation as initialization for the "
            "same fixed 153-parameter student's downstream meta-training."
        )
    )
    parser.add_argument("--teacher-widths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--teacher-meta-iterations", type=int, default=20)
    parser.add_argument("--student-meta-iterations", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=12)
    parser.add_argument("--teacher-seed", type=int, default=46000)
    parser.add_argument("--student-seed", type=int, default=52000)
    parser.add_argument("--student-seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.teacher_widths = [32, 128]
    args.teacher_meta_iterations = 12
    args.student_meta_iterations = 12
    args.tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 8
    args.student_seeds = 2


def make_split(
    seed_base: int,
    count: int,
    *,
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
            for index in range(count)
        ]
        for condition_index, condition in enumerate(CONDITIONS)
    }


def flatten(split: dict[float, list[Case]]) -> list[Case]:
    return [case for condition in CONDITIONS for case in split[condition]]


def evaluate_by_condition(
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
            ratios.append(result.loss_ratio)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def collect_records(teacher, cases: list[Case], *, steps: int, name: str):
    records = []
    for initial, task in cases:
        fresh = copy.deepcopy(teacher)
        if hasattr(fresh, "reset"):
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


def select_normgrad_lr(validation_cases: list[Case], *, steps: int) -> float:
    scores: list[tuple[float, float]] = []
    for lr in NORMGRAD_LR_CANDIDATES:
        ratios = []
        for initial, task in validation_cases:
            result = rollout_teacher(
                initial,
                task,
                teacher=MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5),
                steps=steps,
            )
            ratios.append(result.loss_ratio)
        scores.append((lr, statistics.fmean(ratios)))
    return min(scores, key=lambda item: item[1])[0]


def select_distilled_initialization(
    records,
    selection_cases: list[Case],
    *,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TinyMLPOptimizer, str, float]:
    candidates = []
    for objective_name, weights in OBJECTIVES:
        torch.manual_seed(seed)
        student = TinyMLPOptimizer().to(device)
        history = train_student(
            student,
            records,
            epochs=args.distill_epochs,
            lr=3e-3,
            weights=weights,
        )
        student.set_output_scale(1.0)
        scale = select_student_output_scale(
            student,
            selection_cases,
            candidates=STUDENT_SCALE_CANDIDATES,
            steps=args.steps,
            feature_builder=build_matrix_aware_features,
        )
        candidates.append(
            (
                scale.validation_loss_ratio,
                objective_name,
                copy.deepcopy(student),
                history[-1],
            )
        )
    validation, objective_name, student, _ = min(candidates, key=lambda item: item[0])
    return student, objective_name, validation


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if not args.teacher_widths or min(args.teacher_widths) <= 0:
        raise ValueError("teacher widths must be positive")
    if min(
        args.size,
        args.output_dim,
        args.samples,
        args.steps,
        args.teacher_meta_iterations,
        args.student_meta_iterations,
        args.tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("dimensions and experiment counts must be positive")

    device = torch.device(args.device)
    teacher_train = make_split(11000, args.tasks, args=args, device=device)
    teacher_validation = make_split(15000, args.tasks, args=args, device=device)
    distill_train = make_split(19000, args.tasks, args=args, device=device)
    distill_selection = make_split(23000, args.tasks, args=args, device=device)
    student_meta_train = make_split(26000, args.tasks, args=args, device=device)
    student_meta_validation = make_split(28000, args.tasks, args=args, device=device)
    scale_validation = make_split(30000, args.tasks, args=args, device=device)
    test_split = make_split(33000, args.test_tasks, args=args, device=device)

    teacher_train_cases = flatten(teacher_train)
    teacher_validation_cases = flatten(teacher_validation)
    distill_train_cases = flatten(distill_train)
    distill_selection_cases = flatten(distill_selection)
    student_meta_train_cases = flatten(student_meta_train)
    student_meta_validation_cases = flatten(student_meta_validation)
    scale_validation_cases = flatten(scale_validation)

    normgrad_lr = select_normgrad_lr(teacher_validation_cases, steps=args.steps)
    normgrad = MuonNormGradientTeacher(lr=normgrad_lr, momentum=0.95, ns_steps=5)
    sources: list[tuple[str, object, int | None]] = [
        ("normgrad", normgrad, None),
    ]
    teacher_metrics: dict[str, dict] = {
        "normgrad": {"lr": normgrad_lr},
    }

    for width in sorted(set(args.teacher_widths)):
        torch.manual_seed(args.teacher_seed + width)
        teacher = MetaMLPTeacher(
            hidden_dim=width,
            horizon=args.steps,
            beta1=0.9,
            beta2=0.99,
            initial_step_scale=0.1,
        ).to(device)
        history = train_meta_teacher(
            teacher,
            teacher_train_cases,
            teacher_validation_cases,
            steps=args.steps,
            iterations=args.teacher_meta_iterations,
            outer_lr=3e-3,
            grad_clip=1.0,
            validation_interval=2 if args.quick else 5,
        )
        sources.append((f"learned_width_{width}", teacher, teacher.parameter_count))
        teacher_metrics[f"learned_width_{width}"] = {
            "parameters": teacher.parameter_count,
            "final_meta_train_objective": history[-1].train_objective,
            "validation_loss_ratio": evaluate_direct_teacher(teacher, teacher_validation_cases, args.steps),
        }

    results: list[InitializationResult] = []
    for source_name, teacher, teacher_parameters in sources:
        records = collect_records(
            teacher,
            distill_train_cases,
            steps=args.steps,
            name=source_name,
        )
        for seed_offset in range(args.student_seeds):
            seed = args.student_seed + seed_offset
            student, objective_name, pre_meta_validation = select_distilled_initialization(
                records,
                distill_selection_cases,
                seed=seed,
                args=args,
                device=device,
            )
            student.set_output_scale(1.0)
            train_direct_student(
                student,
                student_meta_train_cases,
                student_meta_validation_cases,
                steps=args.steps,
                iterations=args.student_meta_iterations,
                outer_lr=3e-3,
                grad_clip=1.0,
                validation_interval=2 if args.quick else 5,
                feature_builder=build_matrix_aware_features,
            )
            post_meta_validation = evaluate_direct_student(
                student,
                student_meta_validation_cases,
                steps=args.steps,
                feature_builder=build_matrix_aware_features,
            )
            student.set_output_scale(1.0)
            scale = select_student_output_scale(
                student,
                scale_validation_cases,
                candidates=STUDENT_SCALE_CANDIDATES,
                steps=args.steps,
                feature_builder=build_matrix_aware_features,
            )
            test_ratio, test_by_condition = evaluate_by_condition(
                student,
                test_split,
                steps=args.steps,
            )
            results.append(
                InitializationResult(
                    source=source_name,
                    teacher_parameters=teacher_parameters,
                    distillation_objective=objective_name,
                    student_seed=seed,
                    student_parameters=student.parameter_count,
                    pre_meta_validation_loss_ratio=pre_meta_validation,
                    post_meta_validation_loss_ratio=post_meta_validation,
                    output_scale=scale.scale,
                    test_loss_ratio=test_ratio,
                    test_loss_ratio_by_condition=test_by_condition,
                )
            )

    grouped: dict[str, list[InitializationResult]] = {}
    for row in results:
        grouped.setdefault(row.source, []).append(row)
    summaries = {
        source: {
            "teacher_parameters": rows[0].teacher_parameters,
            "student_parameters": rows[0].student_parameters,
            "test_loss_ratio_mean": statistics.fmean(row.test_loss_ratio for row in rows),
            "test_loss_ratio_seed_std": (
                statistics.pstdev(row.test_loss_ratio for row in rows) if len(rows) > 1 else 0.0
            ),
            "post_meta_validation_mean": statistics.fmean(
                row.post_meta_validation_loss_ratio for row in rows
            ),
            "selected_objectives": [row.distillation_objective for row in rows],
        }
        for source, rows in grouped.items()
    }

    payload = {
        "experiment": "distillation_as_meta_initialization",
        "quick": args.quick,
        "conditions": CONDITIONS,
        "steps": args.steps,
        "teacher_widths": sorted(set(args.teacher_widths)),
        "teacher_meta_iterations": args.teacher_meta_iterations,
        "student_meta_iterations": args.student_meta_iterations,
        "student_parameters": 153,
        "teacher_metrics": teacher_metrics,
        "results": [asdict(row) for row in results],
        "summaries": summaries,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    print("DISTILL_THEN_META_SUMMARY_BEGIN")
    for source, summary in summaries.items():
        print(
            f"source={source} test={summary['test_loss_ratio_mean']:.6f} "
            f"seed_std={summary['test_loss_ratio_seed_std']:.6f} "
            f"post_meta_val={summary['post_meta_validation_mean']:.6f}"
        )
    print("DISTILL_THEN_META_SUMMARY_END")


def evaluate_direct_teacher(teacher: MetaMLPTeacher, cases: list[Case], steps: int) -> float:
    ratios: list[float] = []
    for initial, task in cases:
        fresh = copy.deepcopy(teacher)
        fresh.reset()
        result = rollout_teacher(initial, task, teacher=fresh, steps=steps)
        ratios.append(result.loss_ratio)
    return statistics.fmean(ratios)


if __name__ == "__main__":
    main()
