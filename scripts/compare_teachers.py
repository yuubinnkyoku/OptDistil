from __future__ import annotations

import argparse
import json
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
)
from optdistil.distill.train import calibrate_student_magnitude, train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.quadratic import QuadraticTask
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.muon import MuonTeacher

TeacherFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    teacher: str
    feature_set: str
    student_parameters: int
    train_records: int
    train_distillation_loss: float
    calibration_scale: float
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


def run_one_teacher(
    name: str,
    teacher_factory: TeacherFactory,
    *,
    feature_set: str,
    feature_builder: FeatureBuilder,
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
            teacher=teacher_factory(),
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
        teacher=teacher_factory(),
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

    calibration_scale = calibrate_student_magnitude(student, train_records)
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
        feature_set=feature_set,
        student_parameters=student.parameter_count,
        train_records=len(train_records),
        train_distillation_loss=history[-1],
        calibration_scale=calibration_scale,
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
        description="Compare tiny students distilled from AdamW and Muon teachers."
    )
    parser.add_argument("--train-tasks", type=int, default=4)
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
        args.steps = 8
        args.epochs = 10
        args.size = 6

    device = torch.device(args.device)
    teachers: tuple[tuple[str, TeacherFactory], ...] = (
        (
            "adamw",
            lambda: AdamWTeacher(lr=0.03, betas=(0.9, 0.99)),
        ),
        (
            "muon",
            lambda: MuonTeacher(lr=0.03, momentum=0.95, ns_steps=5),
        ),
    )
    feature_sets: tuple[tuple[str, FeatureBuilder], ...] = (
        ("elementwise", build_elementwise_features),
        ("matrix_aware", build_matrix_aware_features),
    )

    results = [
        run_one_teacher(
            teacher_name,
            factory,
            feature_set=feature_name,
            feature_builder=feature_builder,
            train_tasks=args.train_tasks,
            steps=args.steps,
            epochs=args.epochs,
            size=args.size,
            student_seed=args.student_seed,
            device=device,
        )
        for teacher_name, factory in teachers
        for feature_name, feature_builder in feature_sets
    ]

    payload = {
        "experiment": "teacher_feature_comparison",
        "train_tasks": args.train_tasks,
        "steps": args.steps,
        "epochs": args.epochs,
        "size": args.size,
        "device": str(device),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
