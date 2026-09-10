from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter
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
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.meta_mlp import MetaMLPTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (10.0, 100.0)
ANALYTIC_LR_CANDIDATES = (0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8)
STUDENT_SCALE_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
OBJECTIVES = (
    ("direction_only", DistillationLossWeights(direction=1.0, magnitude=0.0)),
    ("joint", DistillationLossWeights(direction=0.7, magnitude=0.3)),
)


@dataclass(frozen=True, slots=True)
class TeacherSummary:
    width: int
    parameters: int
    meta_validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]
    step_scale: float
    final_meta_train_objective: float


@dataclass(frozen=True, slots=True)
class StudentRun:
    width: int
    teacher_parameters: int
    objective: str
    seed: int
    student_parameters: int
    validation_loss_ratio: float
    validation_scale: float
    final_distillation_loss: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


@dataclass(frozen=True, slots=True)
class StudentSummary:
    width: int
    teacher_parameters: int
    objective: str
    seeds: int
    student_parameters: int
    test_loss_ratio_mean: float
    test_loss_ratio_seed_std: float
    validation_loss_ratio_mean: float
    validation_scale_mean: float


@dataclass(frozen=True, slots=True)
class SelectedStudentSummary:
    width: int
    teacher_parameters: int
    seeds: int
    student_parameters: int
    selected_objectives: dict[str, int]
    test_loss_ratio_mean: float
    test_loss_ratio_seed_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Meta-train increasingly large learned optimizer teachers, then distill every "
            "teacher into the same 153-parameter student."
        )
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--meta-iterations", type=int, default=50)
    parser.add_argument("--meta-train-tasks", type=int, default=4)
    parser.add_argument("--meta-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--teacher-seed", type=int, default=41000)
    parser.add_argument("--student-seed", type=int, default=51000)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--outer-lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick_config(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.widths = [8, 32, 128]
    args.steps = 8
    args.meta_iterations = 12
    args.meta_train_tasks = 2
    args.meta_validation_tasks = 2
    args.distill_train_tasks = 2
    args.student_validation_tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 8
    args.student_seeds = 2


def make_split(
    *,
    seed_base: int,
    count_per_condition: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[float, list[Case]]:
    split: dict[float, list[Case]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        split[condition] = [
            make_frozen_readout_mlp(
                seed_base + 1000 * condition_index + index,
                hidden_dim=args.size,
                input_dim=args.size,
                output_dim=args.output_dim,
                samples=args.samples,
                input_condition=condition,
                device=device,
            )
            for index in range(count_per_condition)
        ]
    return split


def flatten_split(split: dict[float, list[Case]]) -> list[Case]:
    return [case for condition in CONDITIONS for case in split[condition]]


def mean_loss_ratio_teacher(teacher, cases: list[Case], *, steps: int) -> float:
    ratios: list[float] = []
    for initial, task in cases:
        fresh = copy.deepcopy(teacher)
        if hasattr(fresh, "reset"):
            fresh.reset()
        result = rollout_teacher(initial, task, teacher=fresh, steps=steps)
        ratios.append(result.loss_ratio if result.finite else math.inf)
    return statistics.fmean(ratios)


def evaluate_teacher_by_condition(
    teacher: MetaMLPTeacher,
    split: dict[float, list[Case]],
    *,
    steps: int,
) -> tuple[float, dict[str, float]]:
    by_condition = {
        f"{condition:g}": mean_loss_ratio_teacher(teacher, split[condition], steps=steps)
        for condition in CONDITIONS
    }
    return statistics.fmean(by_condition.values()), by_condition


def select_analytic_lr(factory, validation_cases: list[Case], *, steps: int) -> tuple[float, float]:
    scored: list[tuple[float, float]] = []
    for lr in ANALYTIC_LR_CANDIDATES:
        ratios: list[float] = []
        for initial, task in validation_cases:
            result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scored.append((lr, statistics.fmean(ratios)))
    best_lr, best_ratio = min(scored, key=lambda item: item[1])
    return best_lr, best_ratio


def evaluate_analytic(factory, lr: float, split: dict[float, list[Case]], *, steps: int) -> dict:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        ratios: list[float] = []
        for initial, task in split[condition]:
            result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return {
        "lr": lr,
        "test_loss_ratio": statistics.fmean(by_condition.values()),
        "test_loss_ratio_by_condition": by_condition,
    }


def collect_distillation_records(
    teacher: MetaMLPTeacher,
    cases: list[Case],
    *,
    steps: int,
    teacher_name: str,
):
    records = []
    for initial, task in cases:
        fresh = copy.deepcopy(teacher)
        fresh.reset()
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=fresh,
            steps=steps,
            teacher_name=teacher_name,
            feature_builder=build_matrix_aware_features,
        )
        records.extend(trajectory)
    return records


def evaluate_student_split(
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


def summarize_students(runs: list[StudentRun]) -> list[StudentSummary]:
    groups: dict[tuple[int, str], list[StudentRun]] = {}
    for run in runs:
        groups.setdefault((run.width, run.objective), []).append(run)

    summaries: list[StudentSummary] = []
    for (width, objective), group in groups.items():
        first = group[0]
        test_ratios = [row.test_loss_ratio for row in group]
        summaries.append(
            StudentSummary(
                width=width,
                teacher_parameters=first.teacher_parameters,
                objective=objective,
                seeds=len(group),
                student_parameters=first.student_parameters,
                test_loss_ratio_mean=statistics.fmean(test_ratios),
                test_loss_ratio_seed_std=(
                    statistics.pstdev(test_ratios) if len(test_ratios) > 1 else 0.0
                ),
                validation_loss_ratio_mean=statistics.fmean(
                    row.validation_loss_ratio for row in group
                ),
                validation_scale_mean=statistics.fmean(row.validation_scale for row in group),
            )
        )
    return sorted(summaries, key=lambda row: (row.width, row.objective))


def select_objective_per_seed(runs: list[StudentRun]) -> list[SelectedStudentSummary]:
    by_width_seed: dict[tuple[int, int], list[StudentRun]] = {}
    for run in runs:
        by_width_seed.setdefault((run.width, run.seed), []).append(run)

    selected: dict[int, list[StudentRun]] = {}
    for (width, _), candidates in by_width_seed.items():
        winner = min(candidates, key=lambda row: row.validation_loss_ratio)
        selected.setdefault(width, []).append(winner)

    summaries: list[SelectedStudentSummary] = []
    for width, group in selected.items():
        first = group[0]
        ratios = [row.test_loss_ratio for row in group]
        summaries.append(
            SelectedStudentSummary(
                width=width,
                teacher_parameters=first.teacher_parameters,
                seeds=len(group),
                student_parameters=first.student_parameters,
                selected_objectives=dict(Counter(row.objective for row in group)),
                test_loss_ratio_mean=statistics.fmean(ratios),
                test_loss_ratio_seed_std=(
                    statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
                ),
            )
        )
    return sorted(summaries, key=lambda row: row.width)


def nonincreasing(values: list[float], *, tolerance: float = 1e-9) -> bool:
    return all(right <= left + tolerance for left, right in zip(values[:-1], values[1:], strict=True))


def main() -> None:
    args = parse_args()
    apply_quick_config(args)
    if not args.widths or min(args.widths) <= 0:
        raise ValueError("all teacher widths must be positive")
    if min(
        args.size,
        args.output_dim,
        args.samples,
        args.steps,
        args.meta_iterations,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.distill_train_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("dimensions, task counts, iterations, epochs, and seeds must be positive")

    device = torch.device(args.device)
    meta_train_split = make_split(
        seed_base=11000,
        count_per_condition=args.meta_train_tasks,
        args=args,
        device=device,
    )
    meta_validation_split = make_split(
        seed_base=16000,
        count_per_condition=args.meta_validation_tasks,
        args=args,
        device=device,
    )
    distill_train_split = make_split(
        seed_base=21000,
        count_per_condition=args.distill_train_tasks,
        args=args,
        device=device,
    )
    student_validation_split = make_split(
        seed_base=26000,
        count_per_condition=args.student_validation_tasks,
        args=args,
        device=device,
    )
    test_split = make_split(
        seed_base=31000,
        count_per_condition=args.test_tasks,
        args=args,
        device=device,
    )

    meta_train_cases = flatten_split(meta_train_split)
    meta_validation_cases = flatten_split(meta_validation_split)
    distill_train_cases = flatten_split(distill_train_split)
    student_validation_cases = flatten_split(student_validation_split)

    analytic_specs = {
        "adamw": lambda lr: AdamWTeacher(lr=lr, betas=(0.9, 0.99)),
        "muon_norm_gradient": lambda lr: MuonNormGradientTeacher(
            lr=lr,
            momentum=0.95,
            ns_steps=5,
        ),
    }
    analytic_baselines: dict[str, dict] = {}
    for name, factory in analytic_specs.items():
        lr, validation_ratio = select_analytic_lr(
            factory,
            meta_validation_cases,
            steps=args.steps,
        )
        result = evaluate_analytic(factory, lr, test_split, steps=args.steps)
        result["validation_loss_ratio"] = validation_ratio
        analytic_baselines[name] = result

    teacher_summaries: list[TeacherSummary] = []
    student_runs: list[StudentRun] = []
    meta_histories: dict[str, list[dict]] = {}

    for width in sorted(set(args.widths)):
        torch.manual_seed(args.teacher_seed + width)
        teacher = MetaMLPTeacher(
            hidden_dim=width,
            horizon=args.steps,
            beta1=0.9,
            beta2=0.99,
            initial_step_scale=0.1,
        ).to(device)
        history = train_meta_teacher(
            teacher,
            meta_train_cases,
            meta_validation_cases,
            steps=args.steps,
            iterations=args.meta_iterations,
            outer_lr=args.outer_lr,
            grad_clip=1.0,
            validation_interval=2 if args.quick else 5,
        )
        meta_histories[str(width)] = [asdict(row) for row in history]
        meta_validation_ratio = evaluate_meta_teacher(
            teacher,
            meta_validation_cases,
            steps=args.steps,
        )
        teacher_test_ratio, teacher_by_condition = evaluate_teacher_by_condition(
            teacher,
            test_split,
            steps=args.steps,
        )
        teacher_summaries.append(
            TeacherSummary(
                width=width,
                parameters=teacher.parameter_count,
                meta_validation_loss_ratio=meta_validation_ratio,
                test_loss_ratio=teacher_test_ratio,
                test_loss_ratio_by_condition=teacher_by_condition,
                step_scale=float(teacher.step_scale.detach()),
                final_meta_train_objective=history[-1].train_objective,
            )
        )

        records = collect_distillation_records(
            teacher,
            distill_train_cases,
            steps=args.steps,
            teacher_name=f"meta_mlp_width_{width}",
        )
        for objective_name, objective in OBJECTIVES:
            for seed_offset in range(args.student_seeds):
                seed = args.student_seed + seed_offset
                torch.manual_seed(seed)
                student = TinyMLPOptimizer().to(device)
                distill_history = train_student(
                    student,
                    records,
                    epochs=args.distill_epochs,
                    lr=3e-3,
                    weights=objective,
                )
                student.set_output_scale(1.0)
                scale_selection = select_student_output_scale(
                    student,
                    student_validation_cases,
                    candidates=STUDENT_SCALE_CANDIDATES,
                    steps=args.steps,
                    feature_builder=build_matrix_aware_features,
                )
                test_ratio, test_by_condition = evaluate_student_split(
                    student,
                    test_split,
                    steps=args.steps,
                )
                student_runs.append(
                    StudentRun(
                        width=width,
                        teacher_parameters=teacher.parameter_count,
                        objective=objective_name,
                        seed=seed,
                        student_parameters=student.parameter_count,
                        validation_loss_ratio=scale_selection.validation_loss_ratio,
                        validation_scale=scale_selection.scale,
                        final_distillation_loss=distill_history[-1],
                        test_loss_ratio=test_ratio,
                        test_loss_ratio_by_condition=test_by_condition,
                    )
                )

    student_summaries = summarize_students(student_runs)
    selected_summaries = select_objective_per_seed(student_runs)
    ordered_teachers = sorted(teacher_summaries, key=lambda row: row.parameters)
    ordered_selected = sorted(selected_summaries, key=lambda row: row.teacher_parameters)

    payload = {
        "experiment": "meta_teacher_capacity_scaling",
        "quick": args.quick,
        "conditions": CONDITIONS,
        "widths": sorted(set(args.widths)),
        "steps": args.steps,
        "meta_iterations": args.meta_iterations,
        "meta_train_tasks_per_condition": args.meta_train_tasks,
        "meta_validation_tasks_per_condition": args.meta_validation_tasks,
        "distill_train_tasks_per_condition": args.distill_train_tasks,
        "student_validation_tasks_per_condition": args.student_validation_tasks,
        "test_tasks_per_condition": args.test_tasks,
        "distill_epochs": args.distill_epochs,
        "student_seeds": args.student_seeds,
        "analytic_baselines": analytic_baselines,
        "teacher_summaries": [asdict(row) for row in teacher_summaries],
        "student_summaries": [asdict(row) for row in student_summaries],
        "selected_student_summaries": [asdict(row) for row in selected_summaries],
        "student_runs": [asdict(row) for row in student_runs],
        "meta_histories": meta_histories,
        "scaling_checks": {
            "teacher_test_loss_nonincreasing_with_parameters": nonincreasing(
                [row.test_loss_ratio for row in ordered_teachers]
            ),
            "selected_student_test_loss_nonincreasing_with_teacher_parameters": nonincreasing(
                [row.test_loss_ratio_mean for row in ordered_selected]
            ),
        },
    }

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    print("META_TEACHER_SCALING_SUMMARY_BEGIN")
    for row in teacher_summaries:
        print(
            "TEACHER "
            f"width={row.width} params={row.parameters} "
            f"val={row.meta_validation_loss_ratio:.6f} test={row.test_loss_ratio:.6f} "
            f"step_scale={row.step_scale:.6f}"
        )
    for row in selected_summaries:
        print(
            "STUDENT "
            f"teacher_width={row.width} teacher_params={row.teacher_parameters} "
            f"student_params={row.student_parameters} test={row.test_loss_ratio_mean:.6f} "
            f"seed_std={row.test_loss_ratio_seed_std:.6f} selected={row.selected_objectives}"
        )
    print("META_TEACHER_SCALING_SUMMARY_END")


if __name__ == "__main__":
    main()
