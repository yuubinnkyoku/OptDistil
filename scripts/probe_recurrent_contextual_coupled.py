from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.meta_train import evaluate_meta_teacher, train_meta_teacher
from optdistil.distill.rollout import OptimizationTask, rollout_teacher
from optdistil.tasks.coupled_quadratic import CoupledMatrixQuadraticTask
from optdistil.teachers.meta_attention import MetaAttentionTeacher
from optdistil.teachers.meta_mlp import MetaMLPTeacher
from optdistil.teachers.meta_recurrent_attention import MetaRecurrentAttentionTeacher

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (30.0, 300.0)


@dataclass(frozen=True, slots=True)
class Run:
    teacher: str
    seed: int
    parameters: int
    validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]
    step_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local, static-context, and recurrent-context learned teachers."
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--meta-iterations", type=int, default=60)
    parser.add_argument("--train-tasks", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--teacher-seeds", type=int, default=3)
    parser.add_argument("--teacher-seed", type=int, default=91000)
    parser.add_argument("--outer-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.meta_iterations = 30
    args.train_tasks = 2
    args.validation_tasks = 2
    args.test_tasks = 4
    args.teacher_seeds = 3


def random_spd_factor(
    generator: torch.Generator,
    *,
    size: int,
    hessian_condition: float,
    device: torch.device,
) -> torch.Tensor:
    raw = torch.randn((size, size), generator=generator)
    q, _ = torch.linalg.qr(raw)
    spectrum = torch.logspace(0.0, 0.25 * math.log10(hessian_condition), size)
    return (q @ torch.diag(spectrum) @ q.mT).to(device)


def make_case(
    seed: int,
    *,
    size: int,
    condition: float,
    device: torch.device,
) -> Case:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn((size, size), generator=generator) * 0.6
    target = torch.randn((size, size), generator=generator) * 0.3
    left = random_spd_factor(
        generator,
        size=size,
        hessian_condition=condition,
        device=device,
    )
    right = random_spd_factor(
        generator,
        size=size,
        hessian_condition=condition,
        device=device,
    )
    return initial.to(device), CoupledMatrixQuadraticTask(target.to(device), left, right)


def make_split(
    *,
    seed_base: int,
    count: int,
    size: int,
    device: torch.device,
) -> dict[float, list[Case]]:
    return {
        condition: [
            make_case(
                seed_base + 1000 * condition_index + index,
                size=size,
                condition=condition,
                device=device,
            )
            for index in range(count)
        ]
        for condition_index, condition in enumerate(CONDITIONS)
    }


def flatten(split: dict[float, list[Case]]) -> list[Case]:
    return [case for condition in CONDITIONS for case in split[condition]]


def evaluate_by_condition(
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


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.steps,
        args.meta_iterations,
        args.train_tasks,
        args.validation_tasks,
        args.test_tasks,
        args.teacher_seeds,
    ) <= 0:
        raise ValueError("sizes, iterations, task counts, and seed count must be positive")

    device = torch.device(args.device)
    train_split = make_split(seed_base=91000, count=args.train_tasks, size=args.size, device=device)
    validation_split = make_split(
        seed_base=96000,
        count=args.validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(seed_base=101000, count=args.test_tasks, size=args.size, device=device)
    train_cases = flatten(train_split)
    validation_cases = flatten(validation_split)

    factories = {
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
        "recurrent_attention28x2": lambda: MetaRecurrentAttentionTeacher(
            d_model=28,
            num_heads=4,
            depth=2,
            horizon=args.steps,
            beta1=0.9,
            beta2=0.99,
            initial_step_scale=0.1,
        ),
    }

    runs: list[Run] = []
    histories: dict[str, list[dict]] = {}
    for teacher_index, (name, factory) in enumerate(factories.items()):
        for seed_index in range(args.teacher_seeds):
            seed = args.teacher_seed + 100 * teacher_index + seed_index
            torch.manual_seed(seed)
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
            histories[f"{name}:{seed}"] = [asdict(row) for row in history]
            validation_ratio = evaluate_meta_teacher(
                teacher,
                validation_cases,
                steps=args.steps,
            )
            test_ratio, test_by_condition = evaluate_by_condition(
                teacher,
                test_split,
                steps=args.steps,
            )
            runs.append(
                Run(
                    teacher=name,
                    seed=seed,
                    parameters=teacher.parameter_count,
                    validation_loss_ratio=validation_ratio,
                    test_loss_ratio=test_ratio,
                    test_loss_ratio_by_condition=test_by_condition,
                    step_scale=float(teacher.step_scale.detach()),
                )
            )

    summary = {}
    for name in factories:
        group = [run for run in runs if run.teacher == name]
        test_values = [run.test_loss_ratio for run in group]
        validation_values = [run.validation_loss_ratio for run in group]
        summary[name] = {
            "parameters": group[0].parameters,
            "test_loss_ratio_mean": statistics.fmean(test_values),
            "test_loss_ratio_seed_std": statistics.pstdev(test_values),
            "validation_loss_ratio_mean": statistics.fmean(validation_values),
            "validation_loss_ratio_seed_std": statistics.pstdev(validation_values),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "runs": [asdict(run) for run in runs],
        "summary": summary,
        "meta_histories": histories,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
