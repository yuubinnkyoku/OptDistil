from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch
from torch import nn

from optdistil.distill.alignment import probe_reference_gradient_geometry
from optdistil.distill.features import FeatureBuilder, build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.rollout import (
    OptimizationTask,
    collect_teacher_trajectory,
    rollout_student,
    rollout_teacher,
    select_student_output_scale,
)
from optdistil.distill.train import train_student
from optdistil.students.block_gain import BlockGainOptimizer, build_block_gain_features
from optdistil.students.row_col_gain import RowColGainOptimizer, build_row_col_gain_features
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

Case = tuple[torch.Tensor, OptimizationTask]
TeacherFactory = Callable[[float], object]
StudentFactory = Callable[[], nn.Module]
CONDITIONS = (10.0, 100.0)
TEACHER_LR_CANDIDATES = (0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8)
STUDENT_SCALE_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
OBJECTIVES = (
    ("joint", DistillationLossWeights(direction=0.7, magnitude=0.3)),
    ("direction_only", DistillationLossWeights(direction=1.0, magnitude=0.0)),
)


@dataclass(frozen=True, slots=True)
class TeacherResult:
    condition: float
    teacher: str
    lr: float
    lr_at_boundary: bool
    validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_std: float
    test_aulc: float
    finite: bool


@dataclass(frozen=True, slots=True)
class StudentResult:
    condition: float
    teacher: str
    student: str
    objective: str
    seed: int
    parameters: int
    teacher_lr: float
    validation_scale: float
    scale_at_boundary: bool
    train_distillation_loss: float
    test_loss_ratio: float
    test_aulc: float
    finite: bool


@dataclass(frozen=True, slots=True)
class StudentSummary:
    condition: float
    teacher: str
    student: str
    objective: str
    seeds: int
    parameters: int
    teacher_lr: float
    validation_scale_mean: float
    scale_boundary_fraction: float
    test_loss_ratio_mean: float
    test_loss_ratio_seed_std: float
    test_aulc_mean: float
    all_finite: bool


@dataclass(frozen=True, slots=True)
class BaselineResult:
    condition: float
    student: str
    parameters: int
    validation_scale: float
    test_loss_ratio: float
    test_aulc: float
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate optimizer distillation on a nonlinear frozen-readout tanh MLP "
            "without disagreement-based task selection."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--train-tasks", type=int, default=3)
    parser.add_argument("--teacher-validation-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--student-seed", type=int, default=12000)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def select_teacher_lr(
    factory: TeacherFactory,
    cases: list[Case],
    *,
    steps: int,
) -> tuple[float, float]:
    scores: list[tuple[float, float]] = []
    for lr in TEACHER_LR_CANDIDATES:
        ratios = []
        for initial, task in cases:
            result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
            ratios.append(result.loss_ratio if result.finite else math.inf)
        scores.append((lr, statistics.fmean(ratios)))
    best_lr, best_score = min(scores, key=lambda item: item[1])
    if not math.isfinite(best_score):
        raise RuntimeError("all teacher learning rates were non-finite")
    return best_lr, best_score


def evaluate_teacher(
    factory: TeacherFactory,
    lr: float,
    cases: list[Case],
    *,
    steps: int,
) -> tuple[float, float, float, bool]:
    ratios: list[float] = []
    aulcs: list[float] = []
    finite = True
    for initial, task in cases:
        result = rollout_teacher(initial, task, teacher=factory(lr), steps=steps)
        finite = finite and result.finite
        ratios.append(result.loss_ratio if result.finite else math.inf)
        aulcs.append(result.normalized_aulc if result.finite else math.inf)
    ratio_mean, ratio_std = mean_std(ratios)
    return ratio_mean, ratio_std, statistics.fmean(aulcs), finite


def evaluate_student(
    student: nn.Module,
    cases: list[Case],
    *,
    steps: int,
    feature_builder: FeatureBuilder,
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
            feature_builder=feature_builder,
        )
        finite = finite and result.finite
        ratios.append(result.loss_ratio if result.finite else math.inf)
        aulcs.append(result.normalized_aulc if result.finite else math.inf)
    return statistics.fmean(ratios), statistics.fmean(aulcs), finite


def train_one_student(
    teacher_name: str,
    teacher_factory: TeacherFactory,
    teacher_lr: float,
    student_factory: StudentFactory,
    feature_builder: FeatureBuilder,
    *,
    train_cases: list[Case],
    validation_cases: list[Case],
    steps: int,
    epochs: int,
    student_seed: int,
    objective: DistillationLossWeights,
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    records = []
    for initial, task in train_cases:
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
            teacher_name=teacher_name,
            feature_builder=feature_builder,
        )
        records.extend(trajectory)

    torch.manual_seed(student_seed)
    student = student_factory().to(device)
    history = train_student(student, records, epochs=epochs, lr=3e-3, weights=objective)
    student.set_output_scale(1.0)
    scale = select_student_output_scale(
        student,
        validation_cases,
        candidates=STUDENT_SCALE_CANDIDATES,
        steps=steps,
        feature_builder=feature_builder,
    ).scale
    return student, history[-1], scale


def summarize(results: list[StudentResult]) -> list[StudentSummary]:
    groups: dict[tuple[float, str, str, str], list[StudentResult]] = {}
    for result in results:
        key = (result.condition, result.teacher, result.student, result.objective)
        groups.setdefault(key, []).append(result)

    summaries: list[StudentSummary] = []
    for group in groups.values():
        first = group[0]
        ratios = [row.test_loss_ratio for row in group]
        summaries.append(
            StudentSummary(
                condition=first.condition,
                teacher=first.teacher,
                student=first.student,
                objective=first.objective,
                seeds=len(group),
                parameters=first.parameters,
                teacher_lr=first.teacher_lr,
                validation_scale_mean=statistics.fmean(row.validation_scale for row in group),
                scale_boundary_fraction=statistics.fmean(
                    float(row.scale_at_boundary) for row in group
                ),
                test_loss_ratio_mean=statistics.fmean(ratios),
                test_loss_ratio_seed_std=(
                    statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
                ),
                test_aulc_mean=statistics.fmean(row.test_aulc for row in group),
                all_finite=all(row.finite for row in group),
            )
        )
    return summaries


def main() -> None:
    args = parse_args()
    if min(
        args.size,
        args.output_dim,
        args.samples,
        args.steps,
        args.train_tasks,
        args.teacher_validation_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("dimensions, counts, steps, epochs, and seeds must be positive")

    device = torch.device(args.device)
    student_seeds = [args.student_seed + offset for offset in range(args.student_seeds)]
    teacher_specs: tuple[tuple[str, TeacherFactory], ...] = (
        ("adamw", lambda lr: AdamWTeacher(lr=lr, betas=(0.9, 0.99))),
        ("muon", lambda lr: MuonTeacher(lr=lr, momentum=0.95, ns_steps=5)),
        (
            "muon_norm_gradient",
            lambda lr: MuonNormGradientTeacher(lr=lr, momentum=0.95, ns_steps=5),
        ),
    )
    student_specs: tuple[tuple[str, StudentFactory, FeatureBuilder], ...] = (
        ("tiny_matrix", TinyMLPOptimizer, build_matrix_aware_features),
        (
            "block_row",
            BlockGainOptimizer,
            partial(build_block_gain_features, block_size=args.size),
        ),
        ("row_col", RowColGainOptimizer, build_row_col_gain_features),
    )

    teacher_results: list[TeacherResult] = []
    student_results: list[StudentResult] = []
    baselines: list[BaselineResult] = []
    geometry: dict[str, dict[str, float]] = {}

    for condition_index, condition in enumerate(CONDITIONS):
        def make_case(seed: int, condition: float = condition) -> Case:
            return make_frozen_readout_mlp(
                seed,
                hidden_dim=args.size,
                input_dim=args.size,
                output_dim=args.output_dim,
                samples=args.samples,
                input_condition=condition,
                device=device,
            )

        train_cases = [
            make_case(11000 + 1000 * condition_index + index)
            for index in range(args.train_tasks)
        ]
        teacher_validation_cases = [
            make_case(21000 + 1000 * condition_index + index)
            for index in range(args.teacher_validation_tasks)
        ]
        student_validation_cases = [
            make_case(26000 + 1000 * condition_index + index)
            for index in range(args.student_validation_tasks)
        ]
        test_cases = [
            make_case(31000 + 1000 * condition_index + index)
            for index in range(args.test_tasks)
        ]

        tuned: dict[str, tuple[TeacherFactory, float]] = {}
        for teacher_name, teacher_factory in teacher_specs:
            lr, validation_ratio = select_teacher_lr(
                teacher_factory,
                teacher_validation_cases,
                steps=args.steps,
            )
            tuned[teacher_name] = (teacher_factory, lr)
            ratio, ratio_std, aulc, finite = evaluate_teacher(
                teacher_factory,
                lr,
                test_cases,
                steps=args.steps,
            )
            teacher_results.append(
                TeacherResult(
                    condition=condition,
                    teacher=teacher_name,
                    lr=lr,
                    lr_at_boundary=lr in {
                        TEACHER_LR_CANDIDATES[0],
                        TEACHER_LR_CANDIDATES[-1],
                    },
                    validation_loss_ratio=validation_ratio,
                    test_loss_ratio=ratio,
                    test_loss_ratio_std=ratio_std,
                    test_aulc=aulc,
                    finite=finite,
                )
            )

        muon_lr = tuned["muon"][1]
        cosines: list[float] = []
        disagreements: list[float] = []
        for initial, task in test_cases:
            result = probe_reference_gradient_geometry(
                initial,
                task,
                reference=MuonTeacher(lr=muon_lr, momentum=0.95, ns_steps=5),
                steps=args.steps,
            )
            cosines.append(result.cosine_mean)
            disagreements.append(result.disagreement_mean)
        geometry[f"{condition:g}"] = {
            "muon_negative_gradient_cosine_mean": statistics.fmean(cosines),
            "muon_negative_gradient_disagreement_mean": statistics.fmean(disagreements),
        }

        # Zero-initialized gain students are meaningful analytic baselines after scale tuning.
        for student_name, student_factory, feature_builder in student_specs[1:]:
            baseline = student_factory().to(device)
            baseline.set_output_scale(1.0)
            scale = select_student_output_scale(
                baseline,
                student_validation_cases,
                candidates=STUDENT_SCALE_CANDIDATES,
                steps=args.steps,
                feature_builder=feature_builder,
            ).scale
            ratio, aulc, finite = evaluate_student(
                baseline,
                test_cases,
                steps=args.steps,
                feature_builder=feature_builder,
            )
            baselines.append(
                BaselineResult(
                    condition=condition,
                    student=student_name,
                    parameters=sum(parameter.numel() for parameter in baseline.parameters()),
                    validation_scale=scale,
                    test_loss_ratio=ratio,
                    test_aulc=aulc,
                    finite=finite,
                )
            )

        for teacher_name, (teacher_factory, teacher_lr) in tuned.items():
            for student_name, student_factory, feature_builder in student_specs:
                for objective_name, objective in OBJECTIVES:
                    for student_seed in student_seeds:
                        student, train_loss, validation_scale = train_one_student(
                            teacher_name,
                            teacher_factory,
                            teacher_lr,
                            student_factory,
                            feature_builder,
                            train_cases=train_cases,
                            validation_cases=student_validation_cases,
                            steps=args.steps,
                            epochs=args.epochs,
                            student_seed=student_seed,
                            objective=objective,
                            device=device,
                        )
                        ratio, aulc, finite = evaluate_student(
                            student,
                            test_cases,
                            steps=args.steps,
                            feature_builder=feature_builder,
                        )
                        parameters = sum(parameter.numel() for parameter in student.parameters())
                        student_results.append(
                            StudentResult(
                                condition=condition,
                                teacher=teacher_name,
                                student=student_name,
                                objective=objective_name,
                                seed=student_seed,
                                parameters=parameters,
                                teacher_lr=teacher_lr,
                                validation_scale=validation_scale,
                                scale_at_boundary=validation_scale in {
                                    STUDENT_SCALE_CANDIDATES[0],
                                    STUDENT_SCALE_CANDIDATES[-1],
                                },
                                train_distillation_loss=train_loss,
                                test_loss_ratio=ratio,
                                test_aulc=aulc,
                                finite=finite,
                            )
                        )

    summaries = summarize(student_results)
    payload = {
        "experiment": "nonlinear_frozen_readout_mlp_distillation",
        "task_selection": "none; fixed disjoint random seed ranges",
        "conditions": CONDITIONS,
        "size": args.size,
        "output_dim": args.output_dim,
        "samples": args.samples,
        "steps": args.steps,
        "train_tasks": args.train_tasks,
        "teacher_validation_tasks": args.teacher_validation_tasks,
        "student_validation_tasks": args.student_validation_tasks,
        "test_tasks": args.test_tasks,
        "epochs": args.epochs,
        "student_seeds": args.student_seeds,
        "device": str(device),
        "teacher_results": [asdict(row) for row in teacher_results],
        "student_baselines": [asdict(row) for row in baselines],
        "student_summaries": [asdict(row) for row in summaries],
        "student_results": [asdict(row) for row in student_results],
        "muon_gradient_geometry": geometry,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    print("NONLINEAR_MATRIX_SUMMARY_BEGIN")
    for row in teacher_results:
        print(
            "TEACHER "
            f"condition={row.condition:g} name={row.teacher} lr={row.lr:g} "
            f"ratio={row.test_loss_ratio:.6f} aulc={row.test_aulc:.6f}"
        )
    for row in baselines:
        print(
            "BASE "
            f"condition={row.condition:g} student={row.student} "
            f"ratio={row.test_loss_ratio:.6f} aulc={row.test_aulc:.6f} "
            f"scale={row.validation_scale:g}"
        )
    for row in summaries:
        print(
            "STUDENT "
            f"condition={row.condition:g} teacher={row.teacher} student={row.student} "
            f"objective={row.objective} ratio={row.test_loss_ratio_mean:.6f} "
            f"seed_std={row.test_loss_ratio_seed_std:.6f} "
            f"aulc={row.test_aulc_mean:.6f} scale={row.validation_scale_mean:.6f}"
        )
    for condition, values in geometry.items():
        print(
            "GEOMETRY "
            f"condition={condition} "
            f"cos_muon_neg_grad={values['muon_negative_gradient_cosine_mean']:.6f} "
            f"disagreement={values['muon_negative_gradient_disagreement_mean']:.6f}"
        )
    print("NONLINEAR_MATRIX_SUMMARY_END")


if __name__ == "__main__":
    main()
