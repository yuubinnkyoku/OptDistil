from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
from compare_teachers import (
    OBJECTIVES,
    make_coupled_quadratic,
    run_one_teacher,
    select_teacher_lr,
    summarize_seed_results,
)

from optdistil.distill.features import build_matrix_aware_features
from optdistil.teachers.gradient_direction import GradientDirectionTeacher
from optdistil.teachers.momentum_direction import MomentumDirectionTeacher
from optdistil.teachers.muon import MuonTeacher

LR_CANDIDATES = (0.005, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2)
CONDITIONS = (30.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tiny students distilled from gradient, momentum, and Muon directions."
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--train-tasks", type=int, default=2)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-seed", type=int, default=1234)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def print_compact_summaries(summaries: list[dict[str, object]]) -> None:
    """Emit stable one-line summaries so CI logs remain easy to inspect."""
    print("MECHANISM_SUMMARY_BEGIN")
    for row in sorted(
        summaries,
        key=lambda item: (float(item["condition"]), str(item["teacher"]), str(item["objective"])),
    ):
        print(
            "MECH "
            f"condition={float(row['condition']):g} "
            f"teacher={row['teacher']} "
            f"objective={row['objective']} "
            f"teacher_lr={float(row['teacher_lr']):.6g} "
            f"teacher_ratio={float(row['teacher_loss_ratio']):.6f}"
            f"+/-{float(row['teacher_loss_ratio_std']):.6f} "
            f"student_ratio={float(row['student_loss_ratio_mean']):.6f}"
            f"+/-{float(row['student_loss_ratio_seed_std']):.6f} "
            f"scale={float(row['validation_scale_mean']):.3f} "
            f"dir_loss={float(row['heldout_direction_loss_mean']):.6f} "
            f"scale_boundary={float(row['validation_scale_boundary_fraction']):.3f}"
        )
    print("MECHANISM_SUMMARY_END")


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
    teacher_specs = (
        ("gradient_direction", lambda lr: GradientDirectionTeacher(lr=lr)),
        (
            "momentum_direction",
            lambda lr: MomentumDirectionTeacher(lr=lr, momentum=0.95, nesterov=True),
        ),
        ("muon", lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)),
    )
    student_seeds = [args.student_seed + offset for offset in range(args.student_seeds)]

    all_results = []
    all_summaries = []
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

        condition_results = []
        for teacher_name, factory in teacher_specs:
            teacher_lr, teacher_validation_ratio = select_teacher_lr(
                factory,
                teacher_validation_cases,
                candidates=LR_CANDIDATES,
                steps=args.steps,
            )
            for objective_name, objective_weights in OBJECTIVES:
                for student_seed in student_seeds:
                    result = run_one_teacher(
                        teacher_name,
                        factory,
                        teacher_lr=teacher_lr,
                        teacher_lr_candidates=LR_CANDIDATES,
                        teacher_validation_loss_ratio=teacher_validation_ratio,
                        feature_set="matrix_aware",
                        feature_builder=build_matrix_aware_features,
                        objective_name=objective_name,
                        objective_weights=objective_weights,
                        student_validation_cases=student_validation_cases,
                        test_cases=test_cases,
                        task_factory=task_factory,
                        train_tasks=args.train_tasks,
                        steps=args.steps,
                        epochs=args.epochs,
                        student_seed=student_seed,
                        device=device,
                    )
                    row = asdict(result)
                    row["condition"] = condition
                    all_results.append(row)
                    condition_results.append(result)

        for summary in summarize_seed_results(condition_results):
            row = asdict(summary)
            row["condition"] = condition
            all_summaries.append(row)

    payload = {
        "experiment": "student_teacher_mechanism_probe",
        "conditions": CONDITIONS,
        "lr_candidates": LR_CANDIDATES,
        "feature_set": "matrix_aware",
        "objectives": [name for name, _ in OBJECTIVES],
        "size": args.size,
        "steps": args.steps,
        "train_tasks": args.train_tasks,
        "validation_tasks": args.validation_tasks,
        "test_tasks": args.test_tasks,
        "epochs": args.epochs,
        "student_seeds": args.student_seeds,
        "device": str(device),
        "summaries": all_summaries,
        "results": all_results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print_compact_summaries(all_summaries)


if __name__ == "__main__":
    main()
