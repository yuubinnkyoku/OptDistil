from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from optdistil.distill.features import (
    FeatureBuilder,
    build_elementwise_features,
    build_matrix_aware_features,
)
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.rollout import (
    collect_teacher_trajectory,
    evaluate_imitation,
    rollout_student,
    rollout_teacher,
    select_student_output_scale,
)
from optdistil.distill.train import calibrate_student_magnitude, train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.quadratic import QuadraticTask
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.muon import MuonTeacher

TeacherFactory = Callable[[float], object]
ADAMW_LR_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2)
MUON_LR_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)
STUDENT_SCALE_CANDIDATES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
OBJECTIVES: tuple[tuple[str, DistillationLossWeights], ...] = (
    ("joint", DistillationLossWeights(direction=0.7, magnitude=0.3)),
    ("direction_only", DistillationLossWeights(direction=1.0, magnitude=0.0)),
)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    teacher: str
    teacher_lr: float
    teacher_lr_at_boundary: bool
    teacher_validation_loss_ratio: float
    feature_set: str
    objective: str
    student_seed: int
    student_parameters: int
    train_records: int
    train_distillation_loss: float
    teacher_magnitude_scale: float
    teacher_calibrated_magnitude_loss: float
    teacher_calibrated_student_loss_ratio: float
    teacher_calibrated_student_loss_ratio_std: float
    validation_scale: float
    validation_scale_at_boundary: bool
    validation_loss_ratio: float
    heldout_imitation_loss_uncalibrated: float
    heldout_direction_loss_uncalibrated: float
    heldout_magnitude_loss_uncalibrated: float
    student_loss_ratio_uncalibrated: float
    student_loss_ratio_uncalibrated_std: float
    student_final_loss_uncalibrated: float
    heldout_imitation_loss: float
    heldout_direction_loss: float
    heldout_magnitude_loss: float
    teacher_loss_ratio: float
    teacher_loss_ratio_std: float
    student_loss_ratio: float
    student_loss_ratio_std: float
    student_final_loss: float
    student_finite: bool


@dataclass(frozen=True, slots=True)
class SeedSummary:
    teacher: str
    teacher_lr: float
    teacher_lr_at_boundary: bool
    feature_set: str
    objective: str
    student_seeds: int
    teacher_loss_ratio: float
    teacher_loss_ratio_std: float
    student_loss_ratio_mean: float
    student_loss_ratio_seed_std: float
    student_loss_ratio_best: float
    student_loss_ratio_worst: float
    student_task_std_mean: float
    heldout_direction_loss_mean: float
    validation_scale_mean: float
    validation_scale_boundary_fraction: float
    all_finite: bool


def make_quadratic(seed: int, *, size: int, device: torch.device) -> tuple[torch.Tensor, QuadraticTask]:
    """Construct a reproducible matrix-shaped quadratic with mild anisotropy."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn((size, size), generator=generator) * 0.6
    target = torch.randn((size, size), generator=generator) * 0.3

    row = torch.linspace(0.6, 1.8, size).square().unsqueeze(1)
    col = torch.linspace(0.8, 1.4, size).unsqueeze(0)
    jitter = 0.9 + 0.2 * torch.rand((size, size), generator=generator)
    curvature = row * col * jitter

    initial = initial.to(device)
    target = target.to(device)
    curvature = curvature.to(device)
    return initial, QuadraticTask(target, curvature)


def mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def select_teacher_lr(
    teacher_factory: TeacherFactory,
    validation_cases: list[tuple[torch.Tensor, QuadraticTask]],
    *,
    candidates: tuple[float, ...],
    steps: int,
) -> tuple[float, float]:
    """Tune teacher learning rate on the validation task distribution."""
    if not candidates:
        raise ValueError("at least one teacher learning-rate candidate is required")

    scores: list[tuple[float, float]] = []
    for lr in candidates:
        ratios = []
        for initial_parameter, task in validation_cases:
            result = rollout_teacher(
                initial_parameter,
                task,
                teacher=teacher_factory(lr),
                steps=steps,
            )
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scores.append((lr, statistics.fmean(ratios)))

    best_lr, best_score = min(scores, key=lambda item: item[1])
    if not math.isfinite(best_score):
        raise ValueError("all teacher learning-rate candidates produced non-finite rollouts")
    return best_lr, best_score


def evaluate_student_rollouts(
    student: TinyMLPOptimizer,
    test_cases: list[tuple[torch.Tensor, QuadraticTask]],
    *,
    steps: int,
    feature_builder: FeatureBuilder,
) -> tuple[float, float, float, bool]:
    ratios: list[float] = []
    final_losses: list[float] = []
    finite = True
    for initial_parameter, task in test_cases:
        result = rollout_student(
            student,
            initial_parameter,
            task,
            steps=steps,
            feature_builder=feature_builder,
        )
        finite = finite and result.finite
        ratios.append(result.loss_ratio if result.finite else math.inf)
        final_losses.append(result.final_loss)

    ratio_mean, ratio_std = mean_and_std(ratios)
    return ratio_mean, ratio_std, statistics.fmean(final_losses), finite


def run_one_teacher(
    name: str,
    teacher_factory: TeacherFactory,
    *,
    teacher_lr: float,
    teacher_lr_candidates: tuple[float, ...],
    teacher_validation_loss_ratio: float,
    feature_set: str,
    feature_builder: FeatureBuilder,
    objective_name: str,
    objective_weights: DistillationLossWeights,
    validation_cases: list[tuple[torch.Tensor, QuadraticTask]],
    test_cases: list[tuple[torch.Tensor, QuadraticTask]],
    train_tasks: int,
    steps: int,
    epochs: int,
    size: int,
    student_seed: int,
    device: torch.device,
) -> ComparisonResult:
    train_records = []
    for task_index in range(train_tasks):
        initial, task = make_quadratic(1000 + task_index, size=size, device=device)
        records, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
            teacher_name=name,
            feature_builder=feature_builder,
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

    heldout_records = []
    teacher_ratios: list[float] = []
    for initial_parameter, task in test_cases:
        records, teacher_rollout = collect_teacher_trajectory(
            initial_parameter,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
            teacher_name=name,
            feature_builder=feature_builder,
        )
        heldout_records.extend(records)
        teacher_ratios.append(teacher_rollout.loss_ratio if teacher_rollout.finite else math.inf)
    teacher_loss_ratio, teacher_loss_ratio_std = mean_and_std(teacher_ratios)

    imitation_uncalibrated = evaluate_imitation(
        student,
        heldout_records,
        weights=objective_weights,
    )
    (
        student_loss_ratio_uncalibrated,
        student_loss_ratio_uncalibrated_std,
        student_final_loss_uncalibrated,
        _,
    ) = evaluate_student_rollouts(
        student,
        test_cases,
        steps=steps,
        feature_builder=feature_builder,
    )

    teacher_magnitude_scale = calibrate_student_magnitude(student, train_records)
    teacher_calibrated_imitation = evaluate_imitation(
        student,
        heldout_records,
        weights=objective_weights,
    )
    (
        teacher_calibrated_student_loss_ratio,
        teacher_calibrated_student_loss_ratio_std,
        _,
        _,
    ) = evaluate_student_rollouts(
        student,
        test_cases,
        steps=steps,
        feature_builder=feature_builder,
    )

    student.set_output_scale(1.0)
    scale_selection = select_student_output_scale(
        student,
        validation_cases,
        candidates=STUDENT_SCALE_CANDIDATES,
        steps=steps,
        feature_builder=feature_builder,
    )

    imitation = evaluate_imitation(student, heldout_records, weights=objective_weights)
    student_loss_ratio, student_loss_ratio_std, student_final_loss, student_finite = (
        evaluate_student_rollouts(
            student,
            test_cases,
            steps=steps,
            feature_builder=feature_builder,
        )
    )

    return ComparisonResult(
        teacher=name,
        teacher_lr=teacher_lr,
        teacher_lr_at_boundary=teacher_lr in {
            teacher_lr_candidates[0],
            teacher_lr_candidates[-1],
        },
        teacher_validation_loss_ratio=teacher_validation_loss_ratio,
        feature_set=feature_set,
        objective=objective_name,
        student_seed=student_seed,
        student_parameters=student.parameter_count,
        train_records=len(train_records),
        train_distillation_loss=history[-1],
        teacher_magnitude_scale=teacher_magnitude_scale,
        teacher_calibrated_magnitude_loss=teacher_calibrated_imitation["magnitude"],
        teacher_calibrated_student_loss_ratio=teacher_calibrated_student_loss_ratio,
        teacher_calibrated_student_loss_ratio_std=teacher_calibrated_student_loss_ratio_std,
        validation_scale=scale_selection.scale,
        validation_scale_at_boundary=scale_selection.scale in {
            STUDENT_SCALE_CANDIDATES[0],
            STUDENT_SCALE_CANDIDATES[-1],
        },
        validation_loss_ratio=scale_selection.validation_loss_ratio,
        heldout_imitation_loss_uncalibrated=imitation_uncalibrated["total"],
        heldout_direction_loss_uncalibrated=imitation_uncalibrated["direction"],
        heldout_magnitude_loss_uncalibrated=imitation_uncalibrated["magnitude"],
        student_loss_ratio_uncalibrated=student_loss_ratio_uncalibrated,
        student_loss_ratio_uncalibrated_std=student_loss_ratio_uncalibrated_std,
        student_final_loss_uncalibrated=student_final_loss_uncalibrated,
        heldout_imitation_loss=imitation["total"],
        heldout_direction_loss=imitation["direction"],
        heldout_magnitude_loss=imitation["magnitude"],
        teacher_loss_ratio=teacher_loss_ratio,
        teacher_loss_ratio_std=teacher_loss_ratio_std,
        student_loss_ratio=student_loss_ratio,
        student_loss_ratio_std=student_loss_ratio_std,
        student_final_loss=student_final_loss,
        student_finite=student_finite,
    )


def summarize_seed_results(results: list[ComparisonResult]) -> list[SeedSummary]:
    groups: dict[tuple[str, str, str], list[ComparisonResult]] = {}
    for result in results:
        key = (result.teacher, result.feature_set, result.objective)
        groups.setdefault(key, []).append(result)

    summaries = []
    for group in groups.values():
        seed_ratios = [result.student_loss_ratio for result in group]
        ratio_mean, ratio_seed_std = mean_and_std(seed_ratios)
        first = group[0]
        summaries.append(
            SeedSummary(
                teacher=first.teacher,
                teacher_lr=first.teacher_lr,
                teacher_lr_at_boundary=first.teacher_lr_at_boundary,
                feature_set=first.feature_set,
                objective=first.objective,
                student_seeds=len(group),
                teacher_loss_ratio=first.teacher_loss_ratio,
                teacher_loss_ratio_std=first.teacher_loss_ratio_std,
                student_loss_ratio_mean=ratio_mean,
                student_loss_ratio_seed_std=ratio_seed_std,
                student_loss_ratio_best=min(seed_ratios),
                student_loss_ratio_worst=max(seed_ratios),
                student_task_std_mean=statistics.fmean(
                    result.student_loss_ratio_std for result in group
                ),
                heldout_direction_loss_mean=statistics.fmean(
                    result.heldout_direction_loss for result in group
                ),
                validation_scale_mean=statistics.fmean(
                    result.validation_scale for result in group
                ),
                validation_scale_boundary_fraction=statistics.fmean(
                    float(result.validation_scale_at_boundary) for result in group
                ),
                all_finite=all(result.student_finite for result in group),
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tiny students distilled from tuned AdamW and Muon teachers."
    )
    parser.add_argument("--train-tasks", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--student-seed", type=int, default=1234, help="First student seed.")
    parser.add_argument("--student-seeds", type=int, default=5, help="Number of student seeds.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny configuration suitable for CI and smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_tasks = 2
        args.validation_tasks = 2
        args.test_tasks = 4
        args.steps = 8
        args.epochs = 10
        args.size = 6
        args.student_seeds = 3

    if min(
        args.train_tasks,
        args.validation_tasks,
        args.test_tasks,
        args.steps,
        args.epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("task counts, steps, epochs, and student seeds must all be positive")

    device = torch.device(args.device)
    validation_cases = [
        make_quadratic(8001 + task_index, size=args.size, device=device)
        for task_index in range(args.validation_tasks)
    ]
    test_cases = [
        make_quadratic(9001 + task_index, size=args.size, device=device)
        for task_index in range(args.test_tasks)
    ]
    teacher_specs: tuple[tuple[str, TeacherFactory, tuple[float, ...]], ...] = (
        (
            "adamw",
            lambda lr: AdamWTeacher(lr=lr, betas=(0.9, 0.99)),
            ADAMW_LR_CANDIDATES,
        ),
        (
            "muon",
            lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5),
            MUON_LR_CANDIDATES,
        ),
    )
    tuned_teachers = [
        (
            name,
            factory,
            lr_candidates,
            *select_teacher_lr(
                factory,
                validation_cases,
                candidates=lr_candidates,
                steps=args.steps,
            ),
        )
        for name, factory, lr_candidates in teacher_specs
    ]
    feature_sets: tuple[tuple[str, FeatureBuilder], ...] = (
        ("elementwise", build_elementwise_features),
        ("matrix_aware", build_matrix_aware_features),
    )
    student_seeds = [args.student_seed + offset for offset in range(args.student_seeds)]

    results = [
        run_one_teacher(
            teacher_name,
            factory,
            teacher_lr=teacher_lr,
            teacher_lr_candidates=teacher_lr_candidates,
            teacher_validation_loss_ratio=teacher_validation_loss_ratio,
            feature_set=feature_name,
            feature_builder=feature_builder,
            objective_name=objective_name,
            objective_weights=objective_weights,
            validation_cases=validation_cases,
            test_cases=test_cases,
            train_tasks=args.train_tasks,
            steps=args.steps,
            epochs=args.epochs,
            size=args.size,
            student_seed=student_seed,
            device=device,
        )
        for (
            teacher_name,
            factory,
            teacher_lr_candidates,
            teacher_lr,
            teacher_validation_loss_ratio,
        ) in tuned_teachers
        for feature_name, feature_builder in feature_sets
        for objective_name, objective_weights in OBJECTIVES
        for student_seed in student_seeds
    ]
    summaries = summarize_seed_results(results)

    payload = {
        "experiment": "expanded_sweep_multi_seed_objective_ablation",
        "train_tasks": args.train_tasks,
        "validation_tasks": args.validation_tasks,
        "test_tasks": args.test_tasks,
        "teacher_lr_candidates": {
            "adamw": ADAMW_LR_CANDIDATES,
            "muon": MUON_LR_CANDIDATES,
        },
        "student_scale_candidates": STUDENT_SCALE_CANDIDATES,
        "objectives": [name for name, _ in OBJECTIVES],
        "student_seed_base": args.student_seed,
        "student_seeds": args.student_seeds,
        "steps": args.steps,
        "epochs": args.epochs,
        "size": args.size,
        "device": str(device),
        "summaries": [asdict(summary) for summary in summaries],
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
