from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.meta_train import evaluate_meta_teacher, train_meta_teacher
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
from optdistil.teachers.meta_mlp import MetaMLPTeacher
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
class TeacherResult:
    name: str
    parameters: int | None
    validation_loss_ratio: float | None
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]
    note: str


@dataclass(frozen=True, slots=True)
class StudentRun:
    teacher_name: str
    seed: int
    parameters: int
    train_records: int
    final_distillation_loss: float
    validation_loss_ratio: float
    validation_scale: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether increasingly strong teachers produce better fixed-size distilled "
            "students on coupled quadratics."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--meta-iterations", type=int, default=60)
    parser.add_argument("--meta-train-tasks", type=int, default=4)
    parser.add_argument("--meta-validation-tasks", type=int, default=4)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--teacher-seed", type=int, default=121000)
    parser.add_argument("--student-seed", type=int, default=131000)
    parser.add_argument("--outer-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.meta_iterations = 30
    args.meta_train_tasks = 2
    args.meta_validation_tasks = 2
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


def make_task_aware_teacher(name: str, task: OptimizationTask, learned: MetaMLPTeacher, lrs: dict[str, float]):
    if name == "muon":
        return MuonTeacher(lr=lrs[name], momentum=0.95, ns_steps=5)
    if name == "norm_gradient":
        return MuonNormGradientTeacher(lr=lrs[name], momentum=0.95, ns_steps=5)
    if name == "learned_mlp128":
        teacher = copy.deepcopy(learned)
        teacher.reset()
        return teacher
    if not isinstance(task, CoupledMatrixQuadraticTask):
        raise TypeError("Newton sources require CoupledMatrixQuadraticTask")
    if name == "newton_025":
        return CoupledNewtonOracleTeacher(task.left, task.right, lr=0.25)
    if name == "newton_050":
        return CoupledNewtonOracleTeacher(task.left, task.right, lr=0.5)
    if name == "newton_exact_ceiling":
        return CoupledNewtonOracleTeacher(task.left, task.right, lr=1.0)
    raise KeyError(name)


def select_lr(name: str, cases: list[Case], *, steps: int) -> tuple[float, float]:
    if name == "muon":
        factory = lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)
    elif name == "norm_gradient":
        factory = lambda lr: MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5)
    else:
        raise KeyError(name)

    scores: list[tuple[float, float]] = []
    for lr in ANALYTIC_LR_CANDIDATES:
        ratios = []
        for initial, task in cases:
            result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scores.append((lr, statistics.fmean(ratios)))
    return min(scores, key=lambda item: item[1])


def evaluate_source(
    name: str,
    split: dict[float, list[Case]],
    *,
    learned: MetaMLPTeacher,
    lrs: dict[str, float],
    steps: int,
) -> tuple[float, dict[str, float]]:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        ratios = []
        for initial, task in split[condition]:
            teacher = make_task_aware_teacher(name, task, learned, lrs)
            result = rollout_teacher(initial, task, teacher=teacher, steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def collect_records(
    name: str,
    cases: list[Case],
    *,
    learned: MetaMLPTeacher,
    lrs: dict[str, float],
    steps: int,
):
    records = []
    for initial, task in cases:
        teacher = make_task_aware_teacher(name, task, learned, lrs)
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=teacher,
            steps=steps,
            teacher_name=name,
            feature_builder=build_matrix_aware_features,
        )
        records.extend(trajectory)
    return records


def evaluate_student(
    student: TinyMLPOptimizer,
    split: dict[float, list[Case]],
    *,
    steps: int,
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
                feature_builder=build_matrix_aware_features,
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
        args.meta_iterations,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, iteration counts, task counts, epochs, and seeds must be positive")

    device = torch.device(args.device)
    meta_train_split = make_split(
        seed_base=111000,
        count=args.meta_train_tasks,
        size=args.size,
        device=device,
    )
    meta_validation_split = make_split(
        seed_base=116000,
        count=args.meta_validation_tasks,
        size=args.size,
        device=device,
    )
    lr_validation_split = make_split(
        seed_base=121000,
        count=args.lr_validation_tasks,
        size=args.size,
        device=device,
    )
    distill_split = make_split(
        seed_base=126000,
        count=args.distill_train_tasks,
        size=args.size,
        device=device,
    )
    student_validation_split = make_split(
        seed_base=131000,
        count=args.student_validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(
        seed_base=136000,
        count=args.test_tasks,
        size=args.size,
        device=device,
    )

    meta_train_cases = flatten(meta_train_split)
    meta_validation_cases = flatten(meta_validation_split)
    lr_validation_cases = flatten(lr_validation_split)
    distill_cases = flatten(distill_split)
    student_validation_cases = flatten(student_validation_split)

    tuned_lrs = {}
    tuned_lr_validation = {}
    for name in ("muon", "norm_gradient"):
        lr, ratio = select_lr(name, lr_validation_cases, steps=args.steps)
        tuned_lrs[name] = lr
        tuned_lr_validation[name] = ratio

    torch.manual_seed(args.teacher_seed)
    learned = MetaMLPTeacher(
        hidden_dim=128,
        horizon=args.steps,
        beta1=0.9,
        beta2=0.99,
        initial_step_scale=0.1,
    ).to(device)
    meta_history = train_meta_teacher(
        learned,
        meta_train_cases,
        meta_validation_cases,
        steps=args.steps,
        iterations=args.meta_iterations,
        outer_lr=args.outer_lr,
        grad_clip=1.0,
        validation_interval=2 if args.quick else 5,
    )
    learned_validation = evaluate_meta_teacher(
        learned,
        meta_validation_cases,
        steps=args.steps,
    )

    distill_sources = (
        "muon",
        "norm_gradient",
        "learned_mlp128",
        "newton_025",
        "newton_050",
    )
    teacher_results: list[TeacherResult] = []
    for name in (*distill_sources, "newton_exact_ceiling"):
        test_ratio, by_condition = evaluate_source(
            name,
            test_split,
            learned=learned,
            lrs=tuned_lrs,
            steps=args.steps,
        )
        if name in tuned_lrs:
            validation_ratio: float | None = tuned_lr_validation[name]
        elif name == "learned_mlp128":
            validation_ratio = learned_validation
        else:
            validation_ratio = None
        teacher_results.append(
            TeacherResult(
                name=name,
                parameters=learned.parameter_count if name == "learned_mlp128" else None,
                validation_loss_ratio=validation_ratio,
                test_loss_ratio=test_ratio,
                test_loss_ratio_by_condition=by_condition,
                note=(
                    "teacher-only ceiling; not distilled because one-step convergence makes "
                    "later trajectory targets degenerate"
                    if name == "newton_exact_ceiling"
                    else "distillation source"
                ),
            )
        )

    student_runs: list[StudentRun] = []
    for source_index, name in enumerate(distill_sources):
        records = collect_records(
            name,
            distill_cases,
            learned=learned,
            lrs=tuned_lrs,
            steps=args.steps,
        )
        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 100 * source_index + seed_index
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
                feature_builder=build_matrix_aware_features,
            )
            test_ratio, by_condition = evaluate_student(student, test_split, steps=args.steps)
            student_runs.append(
                StudentRun(
                    teacher_name=name,
                    seed=seed,
                    parameters=student.parameter_count,
                    train_records=len(records),
                    final_distillation_loss=history[-1],
                    validation_loss_ratio=scale.validation_loss_ratio,
                    validation_scale=scale.scale,
                    test_loss_ratio=test_ratio,
                    test_loss_ratio_by_condition=by_condition,
                )
            )

    student_summary = {}
    for name in distill_sources:
        group = [row for row in student_runs if row.teacher_name == name]
        tests = [row.test_loss_ratio for row in group]
        student_summary[name] = {
            "parameters": group[0].parameters,
            "test_loss_ratio_mean": statistics.fmean(tests),
            "test_loss_ratio_seed_std": statistics.pstdev(tests) if len(tests) > 1 else 0.0,
            "validation_loss_ratio_mean": statistics.fmean(row.validation_loss_ratio for row in group),
            "final_distillation_loss_mean": statistics.fmean(
                row.final_distillation_loss for row in group
            ),
            "validation_scale_mean": statistics.fmean(row.validation_scale for row in group),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "tuned_lrs": tuned_lrs,
        "teachers": [asdict(row) for row in teacher_results],
        "students": [asdict(row) for row in student_runs],
        "student_summary": student_summary,
        "meta_history": [asdict(row) for row in meta_history],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
