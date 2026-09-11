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
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (30.0, 300.0)
LR_CANDIDATES = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8)


@dataclass(frozen=True, slots=True)
class LearnedRun:
    teacher: str
    seed: int
    parameters: int
    validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]
    step_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether contextual learned teachers help on dense coupled quadratics."
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--meta-iterations", type=int, default=40)
    parser.add_argument("--train-tasks", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--teacher-seeds", type=int, default=3)
    parser.add_argument("--teacher-seed", type=int, default=81000)
    parser.add_argument("--outer-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.meta_iterations = 20
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
    factor_condition = hessian_condition**0.25
    spectrum = torch.logspace(0.0, math.log10(factor_condition), size)
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
            if hasattr(fresh, "reset"):
                fresh.reset()
            result = rollout_teacher(initial, task, teacher=fresh, steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def select_analytic_lr(factory, validation_cases: list[Case], *, steps: int) -> tuple[float, float]:
    scores = []
    for lr in LR_CANDIDATES:
        ratios = []
        for initial, task in validation_cases:
            result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scores.append((lr, statistics.fmean(ratios)))
    return min(scores, key=lambda item: item[1])


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
    train_split = make_split(seed_base=71000, count=args.train_tasks, size=args.size, device=device)
    validation_split = make_split(
        seed_base=76000,
        count=args.validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(seed_base=81000, count=args.test_tasks, size=args.size, device=device)
    train_cases = flatten(train_split)
    validation_cases = flatten(validation_split)

    analytic_factories = {
        "muon": lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5),
        "norm_gradient": lambda lr: MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5),
    }
    analytics = {}
    for name, factory in analytic_factories.items():
        lr, validation_ratio = select_analytic_lr(factory, validation_cases, steps=args.steps)
        test_ratio, test_by_condition = evaluate_by_condition(
            factory(lr),
            test_split,
            steps=args.steps,
        )
        analytics[name] = {
            "lr": lr,
            "validation_loss_ratio": validation_ratio,
            "test_loss_ratio": test_ratio,
            "test_loss_ratio_by_condition": test_by_condition,
        }

    learned_factories = {
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
    }

    runs: list[LearnedRun] = []
    histories: dict[str, list[dict]] = {}
    for teacher_index, (name, factory) in enumerate(learned_factories.items()):
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
                LearnedRun(
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
    for name in learned_factories:
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
        "analytic_baselines": analytics,
        "learned_runs": [asdict(run) for run in runs],
        "learned_summary": summary,
        "meta_histories": histories,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
