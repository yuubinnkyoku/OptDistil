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
from optdistil.students.block_gain import BlockGainOptimizer, build_block_gain_features
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

TeacherFactory = Callable[[float], object]
BLOCK_GAIN_SCALE_CANDIDATES = (
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
    4.0,
)


class IdentityBlockGain(nn.Module):
    """The block-RMS momentum base optimizer with no learned correction."""

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
    block_size: int
    validation_scale: float
    test_loss_ratio: float
    test_aulc: float
    finite: bool


@dataclass(frozen=True, slots=True)
class StudentResult:
    condition: float
    block_size: int
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
    block_size: int
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
            "Distill Muon and its norm-gradient control into a row-block gain student "
            "on teacher-disagreement-selected coupled quadratics."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=0, help="0 means one matrix row per block")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--train-tasks", type=int, default=2)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-seed", type=int, default=9100)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate_student(
    student: nn.Module,
    cases: list[Case],
    *,
    steps: int,
    feature_builder,
) -> tuple[float, float, bool]:
    loss_ratios: list[float] = []
    aulcs: list[float] = []
    finite = True
    for initial, task in cases:
        rollout = rollout_student(
            student,
            initial,
            task,
            steps=steps,
            feature_builder=feature_builder,
        )
        finite = finite and rollout.finite
        loss_ratios.append(rollout.loss_ratio if rollout.finite else math.inf)
        aulcs.append(rollout.normalized_aulc if rollout.finite else math.inf)
    return statistics.fmean(loss_ratios), statistics.fmean(aulcs), finite


def evaluate_teacher(
    teacher_factory: TeacherFactory,
    teacher_lr: float,
    cases: list[Case],
    *,
    steps: int,
) -> float:
    ratios = []
    for initial, task in cases:
        rollout = rollout_teacher(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
        )
        ratios.append(rollout.loss_ratio if rollout.finite else math.inf)
    return statistics.fmean(ratios)


def train_block_gain_student(
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
    feature_builder,
    device: torch.device,
) -> tuple[BlockGainOptimizer, float, float]:
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
    student = BlockGainOptimizer().to(device)
    history = train_student(
        student,
        records,
        epochs=epochs,
        lr=3e-3,
        weights=objective_weights,
    )
    student.set_output_scale(1.0)
    scale = select_student_output_scale(
        student,
        validation_cases,
        candidates=BLOCK_GAIN_SCALE_CANDIDATES,
        steps=steps,
        feature_builder=feature_builder,
    ).scale
    return student, history[-1], scale


def summarize(results: list[StudentResult]) -> list[StudentSummary]:
    groups: dict[tuple[float, int, str, str], list[StudentResult]] = {}
    for result in results:
        key = (result.condition, result.block_size, result.teacher, result.objective)
        groups.setdefault(key, []).append(result)

    summaries: list[StudentSummary] = []
    for group in groups.values():
        first = group[0]
        loss_ratios = [row.student_loss_ratio for row in group]
        summaries.append(
            StudentSummary(
                condition=first.condition,
                block_size=first.block_size,
                teacher=first.teacher,
                objective=first.objective,
                student_seeds=len(group),
                student_parameters=first.student_parameters,
                teacher_lr=first.teacher_lr,
                teacher_loss_ratio=first.teacher_loss_ratio,
                student_loss_ratio_mean=statistics.fmean(loss_ratios),
                student_loss_ratio_seed_std=(
                    statistics.pstdev(loss_ratios) if len(loss_ratios) > 1 else 0.0
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

    block_size = args.block_size or args.size
    if block_size <= 0:
        raise ValueError("block size must be positive")

    device = torch.device(args.device)
    feature_builder = partial(build_block_gain_features, block_size=block_size)
    student_seeds = [args.student_seed + offset for offset in range(args.student_seeds)]
    source_specs: tuple[tuple[str, TeacherFactory], ...] = (
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
        for teacher_name, teacher_factory in source_specs:
            best_lr, _ = select_teacher_lr(
                teacher_factory,
                lr_validation_cases,
                candidates=MUON_LR_CANDIDATES,
                steps=args.steps,
            )
            condition_lrs[teacher_name] = best_lr
        tuned_lrs[f"{condition:g}"] = condition_lrs
        reference_muon_lr = condition_lrs["muon"]

        selected_by_split: dict[str, list[Case]] = {}
        for split, base_seed, count in split_specs:
            cases, selection = select_disagreement_cases(
                task_factory,
                seed_start=base_seed + 1000 * condition_index,
                pool_size=args.pool_size,
                select_count=count,
                muon_lr=reference_muon_lr,
                steps=args.steps,
                condition=condition,
                split=split,
            )
            selected_by_split[split] = cases
            selections.append(asdict(selection))

        base = IdentityBlockGain().to(device)
        base_scale = select_student_output_scale(
            base,
            selected_by_split["validation"],
            candidates=BLOCK_GAIN_SCALE_CANDIDATES,
            steps=args.steps,
            feature_builder=feature_builder,
        ).scale
        base_ratio, base_aulc, base_finite = evaluate_student(
            base,
            selected_by_split["test"],
            steps=args.steps,
            feature_builder=feature_builder,
        )
        baselines.append(
            BaselineResult(
                condition=condition,
                block_size=block_size,
                validation_scale=base_scale,
                test_loss_ratio=base_ratio,
                test_aulc=base_aulc,
                finite=base_finite,
            )
        )

        for teacher_name, teacher_factory in source_specs:
            teacher_lr = condition_lrs[teacher_name]
            teacher_ratio = evaluate_teacher(
                teacher_factory,
                teacher_lr,
                selected_by_split["test"],
                steps=args.steps,
            )
            for objective_name, objective_weights in OBJECTIVES:
                for student_seed in student_seeds:
                    student, train_loss, validation_scale = train_block_gain_student(
                        teacher_name,
                        teacher_factory,
                        teacher_lr=teacher_lr,
                        train_cases=selected_by_split["train"],
                        validation_cases=selected_by_split["validation"],
                        steps=args.steps,
                        epochs=args.epochs,
                        student_seed=student_seed,
                        objective_weights=objective_weights,
                        feature_builder=feature_builder,
                        device=device,
                    )
                    student_ratio, student_aulc, finite = evaluate_student(
                        student,
                        selected_by_split["test"],
                        steps=args.steps,
                        feature_builder=feature_builder,
                    )
                    results.append(
                        StudentResult(
                            condition=condition,
                            block_size=block_size,
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
        "experiment": "block_gain_on_muon_disagreement_selected_tasks",
        "student": "BlockGainOptimizer",
        "student_parameters": BlockGainOptimizer().parameter_count,
        "block_size": block_size,
        "block_interpretation": "contiguous row blocks when block_size == matrix size",
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
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    print("BLOCK_GAIN_SUMMARY_BEGIN")
    for row in baselines:
        print(
            "BASE "
            f"condition={row.condition:g} block_size={row.block_size} "
            f"scale={row.validation_scale:g} loss_ratio={row.test_loss_ratio:.6f} "
            f"aulc={row.test_aulc:.6f}"
        )
    for row in summaries:
        print(
            "STUDENT "
            f"condition={row.condition:g} teacher={row.teacher} objective={row.objective} "
            f"loss_ratio={row.student_loss_ratio_mean:.6f} "
            f"seed_std={row.student_loss_ratio_seed_std:.6f} "
            f"aulc={row.student_aulc_mean:.6f} scale={row.validation_scale_mean:.6f}"
        )
    print("BLOCK_GAIN_SUMMARY_END")


if __name__ == "__main__":
    main()
