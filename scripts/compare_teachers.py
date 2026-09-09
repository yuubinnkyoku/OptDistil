from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from optdistil.distill.features import (
    FeatureBuilder,
    build_elementwise_features,
    build_matrix_aware_features,
)
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
TEACHER_LR_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2)
STUDENT_SCALE_CANDIDATES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    teacher: str
    teacher_lr: float
    teacher_validation_loss_ratio: float
    feature_set: str
    student_parameters: int
    train_records: int
    train_distillation_loss: float
    teacher_magnitude_scale: float
    teacher_calibrated_magnitude_loss: float
    teacher_calibrated_student_loss_ratio: float
    validation_scale: float
    validation_loss_ratio: float
    heldout_imitation_loss_uncalibrated: float
    heldout_direction_loss_uncalibrated: float
    heldout_magnitude_loss_uncalibrated: float
    student_loss_ratio_uncalibrated: float
    student_final_loss_uncalibrated: float
    heldout_imitation_loss: float
    heldout_direction_loss: float
    heldout_magnitude_loss: float
    teacher_loss_ratio: float
    student_loss_ratio: float
    student_final_loss: float
    student_finite: bool


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


def select_teacher_lr(
    teacher_factory: TeacherFactory,
    validation_cases: list[tuple[torch.Tensor, QuadraticTask]],
    *,
    steps: int,
) -> tuple[float, float]:
    """Tune teacher learning rate on the same validation distribution used by students."""
    scores: list[tuple[float, float]] = []
    for lr in TEACHER_LR_CANDIDATES:
        ratios = []
        for initial_parameter, task in validation_cases:
            result = rollout_teacher(
                initial_parameter,
                task,
                teacher=teacher_factory(lr),
                steps=steps,
            )
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scores.append((lr, sum(ratios) / len(ratios)))

    best_lr, best_score = min(scores, key=lambda item: item[1])
    if not math.isfinite(best_score):
        raise ValueError("all teacher learning-rate candidates produced non-finite rollouts")
    return best_lr, best_score


def run_one_teacher(
    name: str,
    teacher_factory: TeacherFactory,
    *,
    teacher_lr: float,
    teacher_validation_loss_ratio: float,
    feature_set: str,
    feature_builder: FeatureBuilder,
    validation_cases: list[tuple[torch.Tensor, QuadraticTask]],
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
    history = train_student(student, train_records, epochs=epochs, lr=3e-3)

    heldout_initial, heldout_task = make_quadratic(9001, size=size, device=device)
    heldout_records, teacher_rollout = collect_teacher_trajectory(
        heldout_initial,
        heldout_task,
        teacher=teacher_factory(teacher_lr),
        steps=steps,
        teacher_name=name,
        feature_builder=feature_builder,
    )

    imitation_uncalibrated = evaluate_imitation(student, heldout_records)
    student_rollout_uncalibrated = rollout_student(
        student,
        heldout_initial,
        heldout_task,
        steps=steps,
        feature_builder=feature_builder,
    )

    teacher_magnitude_scale = calibrate_student_magnitude(student, train_records)
    teacher_calibrated_imitation = evaluate_imitation(student, heldout_records)
    teacher_calibrated_rollout = rollout_student(
        student,
        heldout_initial,
        heldout_task,
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

    imitation = evaluate_imitation(student, heldout_records)
    student_rollout = rollout_student(
        student,
        heldout_initial,
        heldout_task,
        steps=steps,
        feature_builder=feature_builder,
    )

    return ComparisonResult(
        teacher=name,
        teacher_lr=teacher_lr,
        teacher_validation_loss_ratio=teacher_validation_loss_ratio,
        feature_set=feature_set,
        student_parameters=student.parameter_count,
        train_records=len(train_records),
        train_distillation_loss=history[-1],
        teacher_magnitude_scale=teacher_magnitude_scale,
        teacher_calibrated_magnitude_loss=teacher_calibrated_imitation["magnitude"],
        teacher_calibrated_student_loss_ratio=teacher_calibrated_rollout.loss_ratio,
        validation_scale=scale_selection.scale,
        validation_loss_ratio=scale_selection.validation_loss_ratio,
        heldout_imitation_loss_uncalibrated=imitation_uncalibrated["total"],
        heldout_direction_loss_uncalibrated=imitation_uncalibrated["direction"],
        heldout_magnitude_loss_uncalibrated=imitation_uncalibrated["magnitude"],
        student_loss_ratio_uncalibrated=student_rollout_uncalibrated.loss_ratio,
        student_final_loss_uncalibrated=student_rollout_uncalibrated.final_loss,
        heldout_imitation_loss=imitation["total"],
        heldout_direction_loss=imitation["direction"],
        heldout_magnitude_loss=imitation["magnitude"],
        teacher_loss_ratio=teacher_rollout.loss_ratio,
        student_loss_ratio=student_rollout.loss_ratio,
        student_final_loss=student_rollout.final_loss,
        student_finite=student_rollout.finite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tiny students distilled from tuned AdamW and Muon teachers."
    )
    parser.add_argument("--train-tasks", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--student-seed", type=int, default=1234)
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
        args.steps = 8
        args.epochs = 10
        args.size = 6

    device = torch.device(args.device)
    validation_cases = [
        make_quadratic(8001 + task_index, size=args.size, device=device)
        for task_index in range(args.validation_tasks)
    ]
    teacher_specs: tuple[tuple[str, TeacherFactory], ...] = (
        (
            "adamw",
            lambda lr: AdamWTeacher(lr=lr, betas=(0.9, 0.99)),
        ),
        (
            "muon",
            lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5),
        ),
    )
    tuned_teachers = [
        (name, factory, *select_teacher_lr(factory, validation_cases, steps=args.steps))
        for name, factory in teacher_specs
    ]
    feature_sets: tuple[tuple[str, FeatureBuilder], ...] = (
        ("elementwise", build_elementwise_features),
        ("matrix_aware", build_matrix_aware_features),
    )

    results = [
        run_one_teacher(
            teacher_name,
            factory,
            teacher_lr=teacher_lr,
            teacher_validation_loss_ratio=teacher_validation_loss_ratio,
            feature_set=feature_name,
            feature_builder=feature_builder,
            validation_cases=validation_cases,
            train_tasks=args.train_tasks,
            steps=args.steps,
            epochs=args.epochs,
            size=args.size,
            student_seed=args.student_seed,
            device=device,
        )
        for teacher_name, factory, teacher_lr, teacher_validation_loss_ratio in tuned_teachers
        for feature_name, feature_builder in feature_sets
    ]

    payload = {
        "experiment": "tuned_teacher_feature_scale_comparison",
        "train_tasks": args.train_tasks,
        "validation_tasks": args.validation_tasks,
        "teacher_lr_candidates": TEACHER_LR_CANDIDATES,
        "student_scale_candidates": STUDENT_SCALE_CANDIDATES,
        "steps": args.steps,
        "epochs": args.epochs,
        "size": args.size,
        "device": str(device),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
