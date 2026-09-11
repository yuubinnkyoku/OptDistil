from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.features import (
    FeatureBuilder,
    build_gram_matrix_features,
    build_matrix_aware_features,
)
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.rollout import (
    OptimizationTask,
    collect_teacher_trajectory,
    rollout_student,
    rollout_teacher,
    select_student_output_scale,
)
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.coupled_quadratic import CoupledMatrixQuadraticTask
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher
from optdistil.teachers.newton_oracle import CoupledNewtonOracleTeacher

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (30.0, 300.0)
ANALYTIC_LR_CANDIDATES = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8)
STUDENT_SCALE_CANDIDATES = (
    0.003,
    0.01,
    0.03,
    0.06,
    0.1,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
)
DISTILL_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)


@dataclass(frozen=True, slots=True)
class Run:
    source: str
    features: str
    seed: int
    student_parameters: int
    train_records: int
    final_distillation_loss: float
    validation_loss_ratio: float
    validation_scale: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether one-step Gram matrix basis features make privileged matrix "
            "teacher updates more distillable without increasing student parameters."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--student-seed", type=int, default=151000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.student_validation_tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 10
    args.student_seeds = 3


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


def make_teacher(source: str, task: OptimizationTask, lrs: dict[str, float]):
    if source == "muon":
        return MuonTeacher(lr=lrs[source], momentum=0.95, ns_steps=5)
    if source == "norm_gradient":
        return MuonNormGradientTeacher(lr=lrs[source], momentum=0.95, ns_steps=5)
    if not isinstance(task, CoupledMatrixQuadraticTask):
        raise TypeError("Newton sources require CoupledMatrixQuadraticTask")
    if source == "newton_025":
        return CoupledNewtonOracleTeacher(task.left, task.right, lr=0.25)
    if source == "newton_050":
        return CoupledNewtonOracleTeacher(task.left, task.right, lr=0.5)
    raise KeyError(source)


def select_lr(source: str, cases: list[Case], *, steps: int) -> float:
    if source == "muon":
        factory = lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)
    elif source == "norm_gradient":
        factory = lambda lr: MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5)
    else:
        raise KeyError(source)

    scores: list[tuple[float, float]] = []
    for lr in ANALYTIC_LR_CANDIDATES:
        ratios = []
        for initial, task in cases:
            result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scores.append((lr, statistics.fmean(ratios)))
    return min(scores, key=lambda item: item[1])[0]


def collect_records(
    source: str,
    cases: list[Case],
    *,
    lrs: dict[str, float],
    steps: int,
    feature_builder: FeatureBuilder,
):
    records = []
    for initial, task in cases:
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=make_teacher(source, task, lrs),
            steps=steps,
            teacher_name=source,
            feature_builder=feature_builder,
        )
        records.extend(trajectory)
    return records


def evaluate_student(
    student: TinyMLPOptimizer,
    split: dict[float, list[Case]],
    *,
    steps: int,
    feature_builder: FeatureBuilder,
) -> tuple[float, dict[str, float]]:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        ratios = []
        for initial, task in split[condition]:
            result = rollout_student(
                student,
                initial,
                task,
                steps=steps,
                feature_builder=feature_builder,
            )
            ratios.append(result.loss_ratio if result.finite else math.inf)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, task counts, epochs, and seeds must be positive")

    device = torch.device(args.device)
    lr_validation_split = make_split(
        seed_base=141000,
        count=args.lr_validation_tasks,
        size=args.size,
        device=device,
    )
    distill_split = make_split(
        seed_base=146000,
        count=args.distill_train_tasks,
        size=args.size,
        device=device,
    )
    student_validation_split = make_split(
        seed_base=151000,
        count=args.student_validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(
        seed_base=156000,
        count=args.test_tasks,
        size=args.size,
        device=device,
    )

    lrs = {
        source: select_lr(source, flatten(lr_validation_split), steps=args.steps)
        for source in ("muon", "norm_gradient")
    }
    feature_specs: tuple[tuple[str, FeatureBuilder], ...] = (
        ("row_col_rms", build_matrix_aware_features),
        ("gram_cubic", build_gram_matrix_features),
    )
    sources = ("muon", "norm_gradient", "newton_025", "newton_050")
    distill_cases = flatten(distill_split)
    student_validation_cases = flatten(student_validation_split)

    runs: list[Run] = []
    for source_index, source in enumerate(sources):
        for feature_index, (feature_name, feature_builder) in enumerate(feature_specs):
            records = collect_records(
                source,
                distill_cases,
                lrs=lrs,
                steps=args.steps,
                feature_builder=feature_builder,
            )
            for seed_index in range(args.student_seeds):
                seed = args.student_seed + 1000 * source_index + 100 * feature_index + seed_index
                torch.manual_seed(seed)
                student = TinyMLPOptimizer(feature_dim=8, hidden_dim=8, hidden_layers=2).to(device)
                history = train_student(
                    student,
                    records,
                    epochs=args.distill_epochs,
                    lr=3e-3,
                    weights=DISTILL_WEIGHTS,
                )
                scale = select_student_output_scale(
                    student,
                    student_validation_cases,
                    candidates=STUDENT_SCALE_CANDIDATES,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                test_ratio, by_condition = evaluate_student(
                    student,
                    test_split,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                runs.append(
                    Run(
                        source=source,
                        features=feature_name,
                        seed=seed,
                        student_parameters=student.parameter_count,
                        train_records=len(records),
                        final_distillation_loss=history[-1],
                        validation_loss_ratio=scale.validation_loss_ratio,
                        validation_scale=scale.scale,
                        test_loss_ratio=test_ratio,
                        test_loss_ratio_by_condition=by_condition,
                    )
                )

    summary = {}
    for source in sources:
        summary[source] = {}
        for feature_name, _ in feature_specs:
            group = [
                row for row in runs if row.source == source and row.features == feature_name
            ]
            tests = [row.test_loss_ratio for row in group]
            summary[source][feature_name] = {
                "student_parameters": group[0].student_parameters,
                "test_loss_ratio_mean": statistics.fmean(tests),
                "test_loss_ratio_seed_std": statistics.pstdev(tests),
                "validation_loss_ratio_mean": statistics.fmean(
                    row.validation_loss_ratio for row in group
                ),
                "final_distillation_loss_mean": statistics.fmean(
                    row.final_distillation_loss for row in group
                ),
                "validation_scale_mean": statistics.fmean(row.validation_scale for row in group),
            }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "tuned_lrs": lrs,
        "runs": [asdict(row) for row in runs],
        "summary": summary,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
