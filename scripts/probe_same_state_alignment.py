from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from compare_teachers import (
    MUON_LR_CANDIDATES,
    OBJECTIVES,
    STUDENT_SCALE_CANDIDATES,
    make_coupled_quadratic,
    select_teacher_lr,
)

from optdistil.distill.alignment import probe_same_state_alignment
from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.rollout import collect_teacher_trajectory, select_student_output_scale
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

CONDITIONS = (30.0, 300.0)
TeacherFactory = Callable[[float], object]


@dataclass(frozen=True, slots=True)
class SeedAlignmentResult:
    condition: float
    student_teacher: str
    objective: str
    student_seed: int
    trajectory: str
    source_teacher_lr: float
    reference_muon_lr: float
    validation_scale: float
    train_distillation_loss: float
    student_reference_cosine: float
    student_negative_gradient_cosine: float
    reference_negative_gradient_cosine: float
    alignment_delta: float
    student_reference_norm_ratio: float
    driver_loss_ratio: float
    all_finite: bool


@dataclass(frozen=True, slots=True)
class AlignmentSummary:
    condition: float
    student_teacher: str
    objective: str
    trajectory: str
    student_seeds: int
    source_teacher_lr: float
    reference_muon_lr: float
    validation_scale_mean: float
    student_reference_cosine_mean: float
    student_reference_cosine_seed_std: float
    student_negative_gradient_cosine_mean: float
    reference_negative_gradient_cosine_mean: float
    alignment_delta_mean: float
    alignment_delta_seed_std: float
    student_reference_norm_ratio_mean: float
    driver_loss_ratio_mean: float
    driver_loss_ratio_seed_std: float
    all_finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a distilled student's alignment with Muon and -gradient on identical "
            "states visited by either the student or Muon."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--train-tasks", type=int, default=2)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-seed", type=int, default=1234)
    parser.add_argument("--student-seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def train_probe_student(
    teacher_name: str,
    teacher_factory: TeacherFactory,
    *,
    teacher_lr: float,
    task_factory,
    student_validation_cases,
    train_tasks: int,
    steps: int,
    epochs: int,
    student_seed: int,
    objective_weights,
    device: torch.device,
) -> tuple[TinyMLPOptimizer, float, float]:
    train_records = []
    for task_index in range(train_tasks):
        initial, task = task_factory(1000 + task_index)
        records, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
            teacher_name=teacher_name,
            feature_builder=build_matrix_aware_features,
        )
        train_records.extend(records)

    torch.manual_seed(student_seed)
    student = TinyMLPOptimizer().to(device)
    history = train_student(
        student,
        train_records,
        epochs=epochs,
        lr=3e-3,
        weights=objective_weights,
    )

    student.set_output_scale(1.0)
    scale_selection = select_student_output_scale(
        student,
        student_validation_cases,
        candidates=STUDENT_SCALE_CANDIDATES,
        steps=steps,
        feature_builder=build_matrix_aware_features,
    )
    return student, history[-1], scale_selection.scale


def evaluate_student_seed(
    student: TinyMLPOptimizer,
    *,
    condition: float,
    student_teacher: str,
    objective: str,
    student_seed: int,
    source_teacher_lr: float,
    reference_muon_lr: float,
    validation_scale: float,
    train_distillation_loss: float,
    test_cases,
    steps: int,
) -> list[SeedAlignmentResult]:
    results: list[SeedAlignmentResult] = []
    for trajectory in ("student", "reference"):
        case_results = [
            probe_same_state_alignment(
                student,
                initial,
                task,
                reference=MuonTeacher(
                    lr=reference_muon_lr,
                    momentum=0.95,
                    ns_steps=5,
                ),
                steps=steps,
                trajectory=trajectory,
                feature_builder=build_matrix_aware_features,
            )
            for initial, task in test_cases
        ]
        driver_ratios = [result.rollout.loss_ratio for result in case_results]
        results.append(
            SeedAlignmentResult(
                condition=condition,
                student_teacher=student_teacher,
                objective=objective,
                student_seed=student_seed,
                trajectory=trajectory,
                source_teacher_lr=source_teacher_lr,
                reference_muon_lr=reference_muon_lr,
                validation_scale=validation_scale,
                train_distillation_loss=train_distillation_loss,
                student_reference_cosine=statistics.fmean(
                    result.student_reference_cosine_mean for result in case_results
                ),
                student_negative_gradient_cosine=statistics.fmean(
                    result.student_negative_gradient_cosine_mean for result in case_results
                ),
                reference_negative_gradient_cosine=statistics.fmean(
                    result.reference_negative_gradient_cosine_mean for result in case_results
                ),
                alignment_delta=statistics.fmean(
                    result.alignment_delta_mean for result in case_results
                ),
                student_reference_norm_ratio=statistics.fmean(
                    result.student_reference_norm_ratio_mean for result in case_results
                ),
                driver_loss_ratio=statistics.fmean(driver_ratios),
                all_finite=all(
                    result.rollout.finite and len(result.steps) == steps
                    for result in case_results
                ),
            )
        )
    return results


def summarize(results: list[SeedAlignmentResult]) -> list[AlignmentSummary]:
    groups: dict[tuple[float, str, str, str], list[SeedAlignmentResult]] = {}
    for result in results:
        key = (
            result.condition,
            result.student_teacher,
            result.objective,
            result.trajectory,
        )
        groups.setdefault(key, []).append(result)

    summaries: list[AlignmentSummary] = []
    for group in groups.values():
        first = group[0]
        student_reference = [row.student_reference_cosine for row in group]
        deltas = [row.alignment_delta for row in group]
        driver_ratios = [row.driver_loss_ratio for row in group]
        student_reference_mean, student_reference_std = mean_and_std(student_reference)
        delta_mean, delta_std = mean_and_std(deltas)
        driver_mean, driver_std = mean_and_std(driver_ratios)
        summaries.append(
            AlignmentSummary(
                condition=first.condition,
                student_teacher=first.student_teacher,
                objective=first.objective,
                trajectory=first.trajectory,
                student_seeds=len(group),
                source_teacher_lr=first.source_teacher_lr,
                reference_muon_lr=first.reference_muon_lr,
                validation_scale_mean=statistics.fmean(row.validation_scale for row in group),
                student_reference_cosine_mean=student_reference_mean,
                student_reference_cosine_seed_std=student_reference_std,
                student_negative_gradient_cosine_mean=statistics.fmean(
                    row.student_negative_gradient_cosine for row in group
                ),
                reference_negative_gradient_cosine_mean=statistics.fmean(
                    row.reference_negative_gradient_cosine for row in group
                ),
                alignment_delta_mean=delta_mean,
                alignment_delta_seed_std=delta_std,
                student_reference_norm_ratio_mean=statistics.fmean(
                    row.student_reference_norm_ratio for row in group
                ),
                driver_loss_ratio_mean=driver_mean,
                driver_loss_ratio_seed_std=driver_std,
                all_finite=all(row.all_finite for row in group),
            )
        )
    return summaries


def print_compact_summaries(summaries: list[AlignmentSummary]) -> None:
    print("SAME_STATE_ALIGNMENT_BEGIN")
    for row in sorted(
        summaries,
        key=lambda item: (
            item.condition,
            item.student_teacher,
            item.objective,
            item.trajectory,
        ),
    ):
        print(
            "ALIGN "
            f"condition={row.condition:g} "
            f"student_teacher={row.student_teacher} "
            f"objective={row.objective} "
            f"trajectory={row.trajectory} "
            f"source_lr={row.source_teacher_lr:.6g} "
            f"muon_lr={row.reference_muon_lr:.6g} "
            f"seeds={row.student_seeds} "
            f"scale={row.validation_scale_mean:.3f} "
            f"cos_s_muon={row.student_reference_cosine_mean:.6f}"
            f"+/-{row.student_reference_cosine_seed_std:.6f} "
            f"cos_s_grad={row.student_negative_gradient_cosine_mean:.6f} "
            f"delta={row.alignment_delta_mean:.6f}"
            f"+/-{row.alignment_delta_seed_std:.6f} "
            f"cos_muon_grad={row.reference_negative_gradient_cosine_mean:.6f} "
            f"norm_s_muon={row.student_reference_norm_ratio_mean:.6f} "
            f"driver_ratio={row.driver_loss_ratio_mean:.6f}"
            f"+/-{row.driver_loss_ratio_seed_std:.6f} "
            f"finite={int(row.all_finite)}"
        )
    print("SAME_STATE_ALIGNMENT_END")


def main() -> None:
    args = parse_args()
    if min(
        args.size,
        args.steps,
        args.train_tasks,
        args.validation_tasks,
        args.test_tasks,
        args.epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, task counts, epochs, and student seeds must be positive")

    device = torch.device(args.device)
    student_seeds = [args.student_seed + offset for offset in range(args.student_seeds)]
    source_specs: tuple[tuple[str, TeacherFactory], ...] = (
        ("muon", lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)),
        (
            "muon_norm_gradient",
            lambda lr: MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5),
        ),
    )

    all_results: list[SeedAlignmentResult] = []
    tuned_lrs: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        task_factory = lambda seed, condition=condition: make_coupled_quadratic(
            seed,
            size=args.size,
            condition=condition,
            device=device,
        )
        teacher_validation_cases = [
            task_factory(7001 + task_index) for task_index in range(args.validation_tasks)
        ]
        student_validation_cases = [
            task_factory(8001 + task_index) for task_index in range(args.validation_tasks)
        ]
        test_cases = [
            task_factory(9001 + task_index) for task_index in range(args.test_tasks)
        ]

        condition_lrs: dict[str, float] = {}
        for teacher_name, teacher_factory in source_specs:
            best_lr, _ = select_teacher_lr(
                teacher_factory,
                teacher_validation_cases,
                candidates=MUON_LR_CANDIDATES,
                steps=args.steps,
            )
            condition_lrs[teacher_name] = best_lr
        tuned_lrs[f"{condition:g}"] = condition_lrs
        reference_muon_lr = condition_lrs["muon"]

        for teacher_name, teacher_factory in source_specs:
            source_teacher_lr = condition_lrs[teacher_name]
            for objective_name, objective_weights in OBJECTIVES:
                for student_seed in student_seeds:
                    student, train_loss, validation_scale = train_probe_student(
                        teacher_name,
                        teacher_factory,
                        teacher_lr=source_teacher_lr,
                        task_factory=task_factory,
                        student_validation_cases=student_validation_cases,
                        train_tasks=args.train_tasks,
                        steps=args.steps,
                        epochs=args.epochs,
                        student_seed=student_seed,
                        objective_weights=objective_weights,
                        device=device,
                    )
                    all_results.extend(
                        evaluate_student_seed(
                            student,
                            condition=condition,
                            student_teacher=teacher_name,
                            objective=objective_name,
                            student_seed=student_seed,
                            source_teacher_lr=source_teacher_lr,
                            reference_muon_lr=reference_muon_lr,
                            validation_scale=validation_scale,
                            train_distillation_loss=train_loss,
                            test_cases=test_cases,
                            steps=args.steps,
                        )
                    )

    summaries = summarize(all_results)
    payload = {
        "experiment": "same_state_student_muon_alignment",
        "conditions": CONDITIONS,
        "feature_set": "matrix_aware",
        "objectives": [name for name, _ in OBJECTIVES],
        "source_teachers": [name for name, _ in source_specs],
        "size": args.size,
        "steps": args.steps,
        "train_tasks": args.train_tasks,
        "validation_tasks": args.validation_tasks,
        "test_tasks": args.test_tasks,
        "epochs": args.epochs,
        "student_seeds": args.student_seeds,
        "device": str(device),
        "tuned_lrs": tuned_lrs,
        "summaries": [asdict(summary) for summary in summaries],
        "results": [asdict(result) for result in all_results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print_compact_summaries(summaries)


if __name__ == "__main__":
    main()
