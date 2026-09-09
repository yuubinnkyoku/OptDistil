from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable

import torch

from compare_teachers import make_coupled_quadratic, mean_and_std
from optdistil.distill.rollout import rollout_teacher
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.gradient_direction import GradientDirectionTeacher
from optdistil.teachers.momentum_direction import MomentumDirectionTeacher
from optdistil.teachers.muon import MuonTeacher

TeacherFactory = Callable[[float], object]
LR_CANDIDATES = (0.005, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2)
CONDITIONS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def evaluate_teacher(
    factory: TeacherFactory,
    *,
    lr: float,
    condition: float,
    size: int,
    steps: int,
    seeds: range,
    device: torch.device,
) -> tuple[float, float]:
    ratios: list[float] = []
    for seed in seeds:
        initial, task = make_coupled_quadratic(
            seed,
            size=size,
            condition=condition,
            device=device,
        )
        rollout = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
        ratios.append(rollout.loss_ratio if rollout.finite else math.inf)
    return mean_and_std(ratios)


def tune_teacher(
    factory: TeacherFactory,
    *,
    condition: float,
    size: int,
    steps: int,
    validation_seeds: range,
    device: torch.device,
) -> tuple[float, float]:
    scores = []
    for lr in LR_CANDIDATES:
        score, _ = evaluate_teacher(
            factory,
            lr=lr,
            condition=condition,
            size=size,
            steps=steps,
            seeds=validation_seeds,
            device=device,
        )
        scores.append((lr, score))
    best_lr, best_score = min(scores, key=lambda item: item[1])
    return best_lr, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe optimizer teachers across coupled-quadratic condition numbers."
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.size, args.steps, args.validation_tasks, args.test_tasks) <= 0:
        raise ValueError("size, steps, and task counts must be positive")

    device = torch.device(args.device)
    teacher_specs: tuple[tuple[str, TeacherFactory], ...] = (
        ("adamw", lambda lr: AdamWTeacher(lr=lr, betas=(0.9, 0.99))),
        ("gradient_direction", lambda lr: GradientDirectionTeacher(lr=lr)),
        (
            "momentum_direction",
            lambda lr: MomentumDirectionTeacher(lr=lr, momentum=0.95, nesterov=True),
        ),
        ("muon", lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)),
    )

    results = []
    for condition in CONDITIONS:
        for name, factory in teacher_specs:
            best_lr, validation_ratio = tune_teacher(
                factory,
                condition=condition,
                size=args.size,
                steps=args.steps,
                validation_seeds=range(7001, 7001 + args.validation_tasks),
                device=device,
            )
            test_ratio, test_std = evaluate_teacher(
                factory,
                lr=best_lr,
                condition=condition,
                size=args.size,
                steps=args.steps,
                seeds=range(9001, 9001 + args.test_tasks),
                device=device,
            )
            results.append(
                {
                    "condition": condition,
                    "teacher": name,
                    "lr": best_lr,
                    "lr_at_boundary": best_lr in {LR_CANDIDATES[0], LR_CANDIDATES[-1]},
                    "validation_loss_ratio": validation_ratio,
                    "test_loss_ratio": test_ratio,
                    "test_loss_ratio_std": test_std,
                }
            )

    payload = {
        "experiment": "teacher_condition_probe",
        "conditions": CONDITIONS,
        "lr_candidates": LR_CANDIDATES,
        "size": args.size,
        "steps": args.steps,
        "validation_tasks": args.validation_tasks,
        "test_tasks": args.test_tasks,
        "device": str(device),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
