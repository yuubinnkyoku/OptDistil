from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
from compare_teachers import (
    MUON_LR_CANDIDATES,
    OBJECTIVES,
    STUDENT_SCALE_CANDIDATES,
    Case,
    make_coupled_quadratic,
    select_teacher_lr,
)
from probe_same_state_alignment import (
    SeedAlignmentResult,
    evaluate_student_seed,
    print_compact_summaries,
    summarize,
)

from optdistil.distill.alignment import probe_reference_gradient_geometry
from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.rollout import collect_teacher_trajectory, select_student_output_scale
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher

CONDITIONS = (30.0, 300.0)
TeacherFactory = Callable[[float], object]


@dataclass(frozen=True, slots=True)
class RankedCase:
    seed: int
    disagreement: float
    cosine: float
    loss_ratio: float
    case: Case


@dataclass(frozen=True, slots=True)
class SelectionSummary:
    condition: float
    split: str
    pool_size: int
    selected_size: int
    pool_cosine_mean: float
    selected_cosine_mean: float
    pool_disagreement_mean: float
    selected_disagreement_mean: float
    selected_disagreement_min: float
    selected_disagreement_max: float
    selected_seeds: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select coupled quadratics where tuned Muon disagrees most with steepest descent, "
            "then compare Muon- and norm-gradient-distilled students on held-out selected tasks."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--train-tasks", type=int, default=2)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-seed", type=int, default=4321)
    parser.add_argument("--student-seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def select_disagreement_cases(
    task_factory,
    *,
    seed_start: int,
    pool_size: int,
    select_count: int,
    muon_lr: float,
    steps: int,
    condition: float,
    split: str,
) -> tuple[list[Case], SelectionSummary]:
    if select_count > pool_size:
        raise ValueError("selected task count cannot exceed pool size")

    ranked: list[RankedCase] = []
    for offset in range(pool_size):
        seed = seed_start + offset
        case = task_factory(seed)
        initial, task = case
        geometry = probe_reference_gradient_geometry(
            initial,
            task,
            reference=MuonTeacher(lr=muon_lr, momentum=0.95, ns_steps=5),
            steps=steps,
        )
        if geometry.rollout.finite and len(geometry.cosines) == steps:
            ranked.append(
                RankedCase(
                    seed=seed,
                    disagreement=geometry.disagreement_mean,
                    cosine=geometry.cosine_mean,
                    loss_ratio=geometry.rollout.loss_ratio,
                    case=case,
                )
            )

    if len(ranked) < select_count:
        raise RuntimeError(
            f"only {len(ranked)} finite candidates available for {select_count} selections"
        )
    ranked.sort(key=lambda row: row.disagreement, reverse=True)
    selected = ranked[:select_count]
    summary = SelectionSummary(
        condition=condition,
        split=split,
        pool_size=len(ranked),
        selected_size=len(selected),
        pool_cosine_mean=statistics.fmean(row.cosine for row in ranked),
        selected_cosine_mean=statistics.fmean(row.cosine for row in selected),
        pool_disagreement_mean=statistics.fmean(row.disagreement for row in ranked),
        selected_disagreement_mean=statistics.fmean(row.disagreement for row in selected),
        selected_disagreement_min=min(row.disagreement for row in selected),
        selected_disagreement_max=max(row.disagreement for row in selected),
        selected_seeds=tuple(row.seed for row in selected),
    )
    return [row.case for row in selected], summary


def train_selected_student(
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
) -> tuple[TinyMLPOptimizer, float, float]:
    records = []
    for initial, task in train_cases:
        trajectory, _ = collect_teacher_trajectory(
            initial,
            task,
            teacher=teacher_factory(teacher_lr),
            steps=steps,
            teacher_name=teacher_name,
            feature_builder=build_matrix_aware_features,
        )
        records.extend(trajectory)

    torch.manual_seed(student_seed)
    student = TinyMLPOptimizer().to(device)
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
        candidates=STUDENT_SCALE_CANDIDATES,
        steps=steps,
        feature_builder=build_matrix_aware_features,
    ).scale
    return student, history[-1], scale


def print_selection_summaries(summaries: list[SelectionSummary]) -> None:
    print("DISAGREEMENT_SELECTION_BEGIN")
    for row in summaries:
        seeds = ",".join(str(seed) for seed in row.selected_seeds)
        print(
            "SELECT "
            f"condition={row.condition:g} split={row.split} "
            f"pool={row.pool_size} selected={row.selected_size} "
            f"pool_cos={row.pool_cosine_mean:.6f} "
            f"selected_cos={row.selected_cosine_mean:.6f} "
            f"pool_disagreement={row.pool_disagreement_mean:.6f} "
            f"selected_disagreement={row.selected_disagreement_mean:.6f} "
            f"range=[{row.selected_disagreement_min:.6f},{row.selected_disagreement_max:.6f}] "
            f"seeds={seeds}"
        )
    print("DISAGREEMENT_SELECTION_END")


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

    selection_summaries: list[SelectionSummary] = []
    alignment_results: list[SeedAlignmentResult] = []
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
            selection_summaries.append(selection)

        for teacher_name, teacher_factory in source_specs:
            teacher_lr = condition_lrs[teacher_name]
            for objective_name, objective_weights in OBJECTIVES:
                for student_seed in student_seeds:
                    student, train_loss, validation_scale = train_selected_student(
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
                    alignment_results.extend(
                        evaluate_student_seed(
                            student,
                            condition=condition,
                            student_teacher=teacher_name,
                            objective=objective_name,
                            student_seed=student_seed,
                            source_teacher_lr=teacher_lr,
                            reference_muon_lr=reference_muon_lr,
                            validation_scale=validation_scale,
                            train_distillation_loss=train_loss,
                            test_cases=selected_by_split["test"],
                            steps=args.steps,
                        )
                    )

    alignment_summaries = summarize(alignment_results)
    payload = {
        "experiment": "muon_gradient_disagreement_selected_tasks",
        "selection_rule": "top teacher-only mean(1-cos(muon,-gradient)) per disjoint split",
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
        "selection_summaries": [asdict(row) for row in selection_summaries],
        "alignment_summaries": [asdict(row) for row in alignment_summaries],
        "alignment_results": [asdict(row) for row in alignment_results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    print_selection_summaries(selection_summaries)
    print_compact_summaries(alignment_summaries)


if __name__ == "__main__":
    main()
