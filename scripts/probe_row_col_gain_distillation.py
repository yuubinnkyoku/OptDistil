from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from compare_teachers import (
    MUON_LR_CANDIDATES,
    OBJECTIVES,
    Case,
    make_coupled_quadratic,
    select_teacher_lr,
)
from probe_muon_disagreement_tasks import CONDITIONS, select_disagreement_cases
from torch import nn

from optdistil.distill.rollout import (
    collect_teacher_trajectory,
    rollout_student,
    rollout_teacher,
    select_student_output_scale,
)
from optdistil.distill.train import train_student
from optdistil.students.row_col_gain import RowColGainOptimizer, build_row_col_gain_features
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

TeacherFactory = Callable[[float], object]
SCALE_CANDIDATES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


class IdentityRowColGain(nn.Module):
    """Global-RMS momentum baseline with no learned row/column correction."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("output_scale", torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0] * self.output_scale

    @torch.no_grad()
    def set_output_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("output scale must be positive and finite")
        self.output_scale.fill_(scale)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    condition: float
    validation_scale: float
    test_loss_ratio: float
    test_aulc: float
    finite: bool


@dataclass(frozen=True, slots=True)
class StudentResult:
    condition: float
    teacher: str
    objective: str
    student_seed: int
    student_parameters: int
    teacher_lr: float
    validation_scale: float
    train_distillation_loss: float
    teacher_loss_ratio: float
    student_loss_ratio: float
    student_aulc: float
    finite: bool


@dataclass(frozen=True, slots=True)
class StudentSummary:
    condition: float
    teacher: str
    objective: str
    student_seeds: int
    student_parameters: int
    teacher_lr: float
    teacher_loss_ratio: float
    student_loss_ratio_mean: float
    student_loss_ratio_seed_std: float
    student_aulc_mean: float
    validation_scale_mean: float
    all_finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Muon and norm-gradient distillation into a fixed-budget separable "
            "row/column gain student on disagreement-selected coupled quadratics."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--train-tasks", type=int, default=2)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-seed", type=int, default=10100)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate_student(
    student: nn.Module,
    cases: list[Case],
    *,
    steps: int,
) -> tuple[float, float, bool]:
    ratios: list[float] = []
    aulcs: list[float] = []
    finite = True
    for initial, task in cases:
        result = rollout_student(
            student,
            initial,
            task,
            steps=steps,
            feature_builder=build_row_col_gain_features,
        )
        finite = finite and result.finite
        ratios.append(result.loss_ratio if result.finite else math.inf)
        aulcs.append(result.normalized_aulc if result.finite else math.inf)
    return statistics.fmean(ratios), statistics.fmean(aulcs), finite


def evaluate_teacher(
    teacher_factory: TeacherFactory,
    teacher_lr: float,
    cases: list[Case],
    *,
    steps: int,
) -> float:
    ratios = []
    for initial, task in cases:
        result = rollout_teacher(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
        )
        ratios.append(result.loss_ratio if result.finite else math.inf)
    return statistics.fmean(ratios)


def train_student_for_teacher(
    teacher_name: str,
    teacher_factory: TeacherFactory,
    *,
    teacher_lr: float,
    train_cases: list[Case],
    validation_cases: list[Case],
    steps: int,
    epochs: int,
    student_seed: int,
    objective_weights,
    device: torch.device,
) -> tuple[RowColGainOptimizer, float, float]:
    records = []
    for initial, task in train_cases:
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
            teacher_name=teacher_name,
            feature_builder=build_row_col_gain_features,
        )
        records.extend(trajectory)

    torch.manual_seed(student_seed)
    student = RowColGainOptimizer().to(device)
    history = train_student(student, records, epochs=epochs, lr=3e-3, weights=objective_weights)
    student.set_output_scale(1.0)
    scale = select_student_output_scale(
        student,
        validation_cases,
        candidates=SCALE_CANDIDATES,
        steps=steps,
        feature_builder=build_row_col_gain_features,
    ).scale
    return student, history[-1], scale


def summarize(results: list[StudentResult]) -> list[StudentSummary]:
    groups: dict[tuple[float, str, str], list[StudentResult]] = {}
    for result in results:
        groups.setdefault((result.condition, result.teacher, result.objective), []).append(result)

    summaries: list[StudentSummary] = []
    for group in groups.values():
        first = group[0]
        ratios = [row.student_loss_ratio for row in group]
        summaries.append(
            StudentSummary(
                condition=first.condition,
                teacher=first.teacher,
                objective=first.objective,
                student_seeds=len(group),
                student_parameters=first.student_parameters,
                teacher_lr=first.teacher_lr,
                teacher_loss_ratio=first.teacher_loss_ratio,
                student_loss_ratio_mean=statistics.fmean(ratios),
                student_loss_ratio_seed_std=(
                    statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
                ),
                student_aulc_mean=statistics.fmean(row.student_aulc for row in group),
                validation_scale_mean=statistics.fmean(row.validation_scale for row in group),
                all_finite=all(row.finite for row in group),
            )
        )
    return summaries


def main() -> None:
    args = parse_args()
    if min(
        args.size,
        args.steps,
        args.pool_size,
        args.train_tasks,
        args.validation_tasks,
        args.test_tasks,
        args.epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, task counts, epochs, and student seeds must be positive")
    if max(args.train_tasks, args.validation_tasks, args.test_tasks) > args.pool_size:
        raise ValueError("pool size must cover every selected split size")

    device = torch.device(args.device)
    student_seeds = [args.student_seed + offset for offset in range(args.student_seeds)]
    sources: tuple[tuple[str, TeacherFactory], ...] = (
        ("muon", lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)),
        (
            "muon_norm_gradient",
            lambda lr: MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5),
        ),
    )
    split_specs = (
        ("train", 11000, args.train_tasks),
        ("validation", 21000, args.validation_tasks),
        ("test", 31000, args.test_tasks),
    )

    baselines: list[BaselineResult] = []
    results: list[StudentResult] = []
    selections = []
    tuned_lrs: dict[str, dict[str, float]] = {}

    for condition_index, condition in enumerate(CONDITIONS):
        task_factory = lambda seed, condition=condition: make_coupled_quadratic(
            seed,
            size=args.size,
            condition=condition,
            device=device,
        )
        lr_validation_cases = [
            task_factory(1000 + 100 * condition_index + index)
            for index in range(max(4, args.validation_tasks))
        ]
        condition_lrs: dict[str, float] = {}
        for teacher_name, teacher_factory in sources:
            best_lr, _ = select_teacher_lr(
                teacher_factory,
                lr_validation_cases,
                candidates=MUON_LR_CANDIDATES,
                steps=args.steps,
            )
            condition_lrs[teacher_name] = best_lr
        tuned_lrs[f"{condition:g}"] = condition_lrs

        selected_by_split: dict[str, list[Case]] = {}
        for split, base_seed, count in split_specs:
            cases, selection = select_disagreement_cases(
                task_factory,
                seed_start=base_seed + 1000 * condition_index,
                pool_size=args.pool_size,
                select_count=count,
                muon_lr=condition_lrs["muon"],
                steps=args.steps,
                condition=condition,
                split=split,
            )
            selected_by_split[split] = cases
            selections.append(asdict(selection))

        baseline = IdentityRowColGain().to(device)
        baseline_scale = select_student_output_scale(
            baseline,
            selected_by_split["validation"],
            candidates=SCALE_CANDIDATES,
            steps=args.steps,
            feature_builder=build_row_col_gain_features,
        ).scale
        baseline_ratio, baseline_aulc, baseline_finite = evaluate_student(
            baseline,
            selected_by_split["test"],
            steps=args.steps,
        )
        baselines.append(
            BaselineResult(
                condition=condition,
                validation_scale=baseline_scale,
                test_loss_ratio=baseline_ratio,
                test_aulc=baseline_aulc,
                finite=baseline_finite,
            )
        )

        for teacher_name, teacher_factory in sources:
            teacher_lr = condition_lrs[teacher_name]
            teacher_ratio = evaluate_teacher(
                teacher_factory,
                teacher_lr,
                selected_by_split["test"],
                steps=args.steps,
            )
            for objective_name, objective_weights in OBJECTIVES:
                for student_seed in student_seeds:
                    student, train_loss, validation_scale = train_student_for_teacher(
                        teacher_name,
                        teacher_factory,
                        teacher_lr=teacher_lr,
                        train_cases=selected_by_split["train"],
                        validation_cases=selected_by_split["validation"],
                        steps=args.steps,
                        epochs=args.epochs,
                        student_seed=student_seed,
                        objective_weights=objective_weights,
                        device=device,
                    )
                    student_ratio, student_aulc, finite = evaluate_student(
                        student,
                        selected_by_split["test"],
                        steps=args.steps,
                    )
                    results.append(
                        StudentResult(
                            condition=condition,
                            teacher=teacher_name,
                            objective=objective_name,
                            student_seed=student_seed,
                            student_parameters=student.parameter_count,
                            teacher_lr=teacher_lr,
                            validation_scale=validation_scale,
                            train_distillation_loss=train_loss,
                            teacher_loss_ratio=teacher_ratio,
                            student_loss_ratio=student_ratio,
                            student_aulc=student_aulc,
                            finite=finite,
                        )
                    )

    summaries = summarize(results)
    payload = {
        "experiment": "row_col_gain_on_muon_disagreement_selected_tasks",
        "student": "RowColGainOptimizer",
        "student_parameters": RowColGainOptimizer().parameter_count,
        "structure": "shared transpose-symmetric row/column gain MLP",
        "conditions": CONDITIONS,
        "size": args.size,
        "steps": args.steps,
        "pool_size": args.pool_size,
        "train_tasks": args.train_tasks,
        "validation_tasks": args.validation_tasks,
        "test_tasks": args.test_tasks,
        "epochs": args.epochs,
        "student_seeds": args.student_seeds,
        "device": str(device),
        "tuned_lrs": tuned_lrs,
        "selection_summaries": selections,
        "baselines": [asdict(row) for row in baselines],
        "student_summaries": [asdict(row) for row in summaries],
        "student_results": [asdict(row) for row in results],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    print("ROW_COL_GAIN_SUMMARY_BEGIN")
    for row in baselines:
        print(
            "BASE "
            f"condition={row.condition:g} scale={row.validation_scale:g} "
            f"loss_ratio={row.test_loss_ratio:.6f} aulc={row.test_aulc:.6f}"
        )
    for row in summaries:
        print(
            "STUDENT "
            f"condition={row.condition:g} teacher={row.teacher} objective={row.objective} "
            f"loss_ratio={row.student_loss_ratio_mean:.6f} "
            f"seed_std={row.student_loss_ratio_seed_std:.6f} "
            f"aulc={row.student_aulc_mean:.6f} scale={row.validation_scale_mean:.6f}"
        )
    print("ROW_COL_GAIN_SUMMARY_END")


if __name__ == "__main__":
    main()
