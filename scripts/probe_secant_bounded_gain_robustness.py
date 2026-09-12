from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import make_case, make_teacher, select_lr
from probe_secant_bounded_gain_control import (
    FULL_DISTILL_WEIGHTS,
    GAIN_BOUND,
    HISTORY_SIZE,
    OUTER_LR_CANDIDATES,
    clone_state,
    gain_normalization,
    select_bounded_scale,
    target_clip_fraction,
    train_bounded_gain_student,
)
from probe_secant_history_pareto import collect_secant_records, select_secant_scale

from optdistil.distill.secant_meta import (
    evaluate_secant_meta_student,
    train_secant_meta_student,
)
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer

TRAIN_CONDITIONS = (30.0, 300.0)
OOD_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
SOURCES = ("norm_gradient", "muon", "newton_050")


@dataclass(frozen=True, slots=True)
class RobustnessRun:
    source: str
    seed: int
    mode: str
    student_parameters: int
    validation_scale: float
    selected_outer_lr: float
    meta_validation_loss_ratio: float
    iid_test_loss_ratio: float
    ood_test_loss_ratio: float
    iid_by_condition: dict[str, float]
    ood_by_condition: dict[str, float]
    iid_finite_fraction: float
    ood_finite_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test full-update and bounded scalar-gain secant Students with larger "
            "validation/test splits and condition-number OOD evaluation. Hyperparameters "
            "are selected only on conditions 30 and 300; OOD conditions are evaluation-only."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--scale-validation-tasks", type=int, default=4)
    parser.add_argument("--meta-train-tasks", type=int, default=4)
    parser.add_argument("--meta-validation-tasks", type=int, default=6)
    parser.add_argument("--iid-test-tasks", type=int, default=12)
    parser.add_argument("--ood-test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--meta-iterations", type=int, default=15)
    parser.add_argument("--validation-interval", type=int, default=3)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=251000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.scale_validation_tasks = 2
    args.meta_train_tasks = 2
    args.meta_validation_tasks = 3
    args.iid_test_tasks = 4
    args.ood_test_tasks = 3
    args.distill_epochs = 8
    args.meta_iterations = 6
    args.validation_interval = 2
    args.student_seeds = 2


def make_condition_split(
    conditions: tuple[float, ...],
    *,
    seed_base: int,
    count: int,
    size: int,
    device: torch.device,
) -> dict[float, list[tuple[torch.Tensor, object]]]:
    return {
        condition: [
            make_case(
                seed_base + 10000 * condition_index + index,
                size=size,
                condition=condition,
                device=device,
            )
            for index in range(count)
        ]
        for condition_index, condition in enumerate(conditions)
    }


def flatten(split) -> list:
    return [case for cases in split.values() for case in cases]


def clone_student_from_state(state, *, device: torch.device) -> TinyMLPOptimizer:
    student = TinyMLPOptimizer().to(device)
    student.load_state_dict(state)
    return student


def select_meta_lr(
    initial_state,
    *,
    device: torch.device,
    train_cases,
    validation_cases,
    steps: int,
    iterations: int,
    validation_interval: int,
    mode: str,
    normalization: float,
) -> tuple[TinyMLPOptimizer, float, float]:
    candidates = []
    for outer_lr in OUTER_LR_CANDIDATES:
        student = clone_student_from_state(initial_state, device=device)
        train_secant_meta_student(
            student,
            train_cases,
            validation_cases,
            steps=steps,
            history_size=HISTORY_SIZE,
            mode=mode,
            gain_normalization=normalization,
            gain_bound=GAIN_BOUND,
            iterations=iterations,
            outer_lr=outer_lr,
            grad_clip=1.0,
            final_weight=0.7,
            validation_interval=validation_interval,
        )
        validation_ratio = evaluate_secant_meta_student(
            student,
            validation_cases,
            steps=steps,
            history_size=HISTORY_SIZE,
            mode=mode,
            gain_normalization=normalization,
            gain_bound=GAIN_BOUND,
        )
        candidates.append((validation_ratio, outer_lr, student))
    best_validation, best_lr, best_student = min(candidates, key=lambda item: item[0])
    return best_student, best_lr, best_validation


def evaluate_by_condition(
    student: TinyMLPOptimizer,
    split,
    *,
    steps: int,
    mode: str,
    normalization: float,
) -> tuple[float, dict[str, float], float]:
    by_condition: dict[str, float] = {}
    ratios: list[float] = []
    finite = 0
    total = 0
    for condition, cases in split.items():
        condition_ratios = []
        for case in cases:
            ratio = evaluate_secant_meta_student(
                student,
                [case],
                steps=steps,
                history_size=HISTORY_SIZE,
                mode=mode,
                gain_normalization=normalization,
                gain_bound=GAIN_BOUND,
            )
            condition_ratios.append(ratio)
            ratios.append(ratio)
            finite += int(math.isfinite(ratio))
            total += 1
        by_condition[str(condition)] = statistics.fmean(condition_ratios)
    return statistics.fmean(ratios), by_condition, finite / max(total, 1)


@torch.no_grad()
def teacher_ratio(source: str, initial, task, *, lrs: dict[str, float], steps: int) -> float:
    parameter = initial.detach().clone()
    teacher = make_teacher(source, task, lrs)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss
    for _ in range(steps):
        grad = task.grad(parameter)
        parameter = parameter + teacher.step(parameter, grad)
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def teacher_summary(source: str, split, *, lrs: dict[str, float], steps: int) -> dict:
    by_condition = {}
    all_ratios = []
    for condition, cases in split.items():
        ratios = [teacher_ratio(source, initial, task, lrs=lrs, steps=steps) for initial, task in cases]
        by_condition[str(condition)] = statistics.fmean(ratios)
        all_ratios.extend(ratios)
    return {
        "mean_loss_ratio": statistics.fmean(all_ratios),
        "median_loss_ratio": statistics.median(all_ratios),
        "finite_fraction": sum(math.isfinite(value) for value in all_ratios) / len(all_ratios),
        "by_condition": by_condition,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "seed_std": statistics.pstdev(values),
        "finite_seed_fraction": len(finite_values) / len(values),
    }


def main() -> None:
    args = parse_args()
    apply_quick(args)
    numeric_args = (
        args.size,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.scale_validation_tasks,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.iid_test_tasks,
        args.ood_test_tasks,
        args.distill_epochs,
        args.meta_iterations,
        args.validation_interval,
        args.student_seeds,
    )
    if min(numeric_args) <= 0:
        raise ValueError("all sizes, task counts, iterations, intervals, and seed counts must be positive")

    device = torch.device(args.device)
    train_specs = {
        "lr_validation": (181000, args.lr_validation_tasks),
        "distill": (191000, args.distill_train_tasks),
        "scale_validation": (201000, args.scale_validation_tasks),
        "meta_train": (211000, args.meta_train_tasks),
        "meta_validation": (221000, args.meta_validation_tasks),
        "iid_test": (231000, args.iid_test_tasks),
    }
    splits = {
        name: make_condition_split(
            TRAIN_CONDITIONS,
            seed_base=seed_base,
            count=count,
            size=args.size,
            device=device,
        )
        for name, (seed_base, count) in train_specs.items()
    }
    ood_split = make_condition_split(
        OOD_CONDITIONS,
        seed_base=241000,
        count=args.ood_test_tasks,
        size=args.size,
        device=device,
    )

    lr_cases = flatten(splits["lr_validation"])
    distill_cases = flatten(splits["distill"])
    scale_validation_cases = flatten(splits["scale_validation"])
    meta_train_cases = flatten(splits["meta_train"])
    meta_validation_cases = flatten(splits["meta_validation"])

    lrs = {
        source: select_lr(source, lr_cases, steps=args.steps)
        for source in ("norm_gradient", "muon")
    }
    records = {
        source: collect_secant_records(
            source,
            distill_cases,
            lrs=lrs,
            steps=args.steps,
            history_size=HISTORY_SIZE,
        )
        for source in SOURCES
    }
    normalizations = {source: gain_normalization(records[source]) for source in SOURCES}
    clip_fractions = {
        source: target_clip_fraction(records[source], normalization=normalizations[source])
        for source in SOURCES
    }

    teachers = {
        source: {
            "iid": teacher_summary(source, splits["iid_test"], lrs=lrs, steps=args.steps),
            "ood": teacher_summary(source, ood_split, lrs=lrs, steps=args.steps),
        }
        for source in SOURCES
    }

    runs: list[RobustnessRun] = []
    for source_index, source in enumerate(SOURCES):
        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 1000 * source_index + seed_index

            torch.manual_seed(seed)
            full_student = TinyMLPOptimizer().to(device)
            train_student(
                full_student,
                records[source],
                epochs=args.distill_epochs,
                lr=3e-3,
                weights=FULL_DISTILL_WEIGHTS,
            )
            full_scale, _ = select_secant_scale(
                full_student,
                scale_validation_cases,
                steps=args.steps,
                history_size=HISTORY_SIZE,
            )
            full_best, full_lr, full_meta_validation = select_meta_lr(
                clone_state(full_student),
                device=device,
                train_cases=meta_train_cases,
                validation_cases=meta_validation_cases,
                steps=args.steps,
                iterations=args.meta_iterations,
                validation_interval=args.validation_interval,
                mode="full_update",
                normalization=1.0,
            )
            full_iid, full_iid_by_condition, full_iid_finite = evaluate_by_condition(
                full_best,
                splits["iid_test"],
                steps=args.steps,
                mode="full_update",
                normalization=1.0,
            )
            full_ood, full_ood_by_condition, full_ood_finite = evaluate_by_condition(
                full_best,
                ood_split,
                steps=args.steps,
                mode="full_update",
                normalization=1.0,
            )
            runs.append(
                RobustnessRun(
                    source=source,
                    seed=seed,
                    mode="full_update",
                    student_parameters=full_student.parameter_count,
                    validation_scale=full_scale,
                    selected_outer_lr=full_lr,
                    meta_validation_loss_ratio=full_meta_validation,
                    iid_test_loss_ratio=full_iid,
                    ood_test_loss_ratio=full_ood,
                    iid_by_condition=full_iid_by_condition,
                    ood_by_condition=full_ood_by_condition,
                    iid_finite_fraction=full_iid_finite,
                    ood_finite_fraction=full_ood_finite,
                )
            )

            torch.manual_seed(seed)
            bounded_student = TinyMLPOptimizer().to(device)
            normalization = normalizations[source]
            train_bounded_gain_student(
                bounded_student,
                records[source],
                normalization=normalization,
                epochs=args.distill_epochs,
                lr=3e-3,
            )
            bounded_scale, _, _ = select_bounded_scale(
                bounded_student,
                scale_validation_cases,
                steps=args.steps,
                normalization=normalization,
            )
            bounded_best, bounded_lr, bounded_meta_validation = select_meta_lr(
                clone_state(bounded_student),
                device=device,
                train_cases=meta_train_cases,
                validation_cases=meta_validation_cases,
                steps=args.steps,
                iterations=args.meta_iterations,
                validation_interval=args.validation_interval,
                mode="bounded_gain",
                normalization=normalization,
            )
            bounded_iid, bounded_iid_by_condition, bounded_iid_finite = evaluate_by_condition(
                bounded_best,
                splits["iid_test"],
                steps=args.steps,
                mode="bounded_gain",
                normalization=normalization,
            )
            bounded_ood, bounded_ood_by_condition, bounded_ood_finite = evaluate_by_condition(
                bounded_best,
                ood_split,
                steps=args.steps,
                mode="bounded_gain",
                normalization=normalization,
            )
            runs.append(
                RobustnessRun(
                    source=source,
                    seed=seed,
                    mode="bounded_gain",
                    student_parameters=bounded_student.parameter_count,
                    validation_scale=bounded_scale,
                    selected_outer_lr=bounded_lr,
                    meta_validation_loss_ratio=bounded_meta_validation,
                    iid_test_loss_ratio=bounded_iid,
                    ood_test_loss_ratio=bounded_ood,
                    iid_by_condition=bounded_iid_by_condition,
                    ood_by_condition=bounded_ood_by_condition,
                    iid_finite_fraction=bounded_iid_finite,
                    ood_finite_fraction=bounded_ood_finite,
                )
            )

    summary = {}
    paired = {}
    for source in SOURCES:
        source_runs = [run for run in runs if run.source == source]
        summary[source] = {}
        for mode in ("full_update", "bounded_gain"):
            group = [run for run in source_runs if run.mode == mode]
            summary[source][mode] = {
                "student_parameters": group[0].student_parameters,
                "iid": summarize_values([run.iid_test_loss_ratio for run in group]),
                "ood": summarize_values([run.ood_test_loss_ratio for run in group]),
                "meta_validation_mean": statistics.fmean(
                    run.meta_validation_loss_ratio for run in group
                ),
                "validation_scale_mean": statistics.fmean(run.validation_scale for run in group),
                "selected_outer_lr_mean": statistics.fmean(run.selected_outer_lr for run in group),
                "iid_case_finite_fraction_mean": statistics.fmean(
                    run.iid_finite_fraction for run in group
                ),
                "ood_case_finite_fraction_mean": statistics.fmean(
                    run.ood_finite_fraction for run in group
                ),
                "iid_by_condition_mean": {
                    str(condition): statistics.fmean(
                        run.iid_by_condition[str(condition)] for run in group
                    )
                    for condition in TRAIN_CONDITIONS
                },
                "ood_by_condition_mean": {
                    str(condition): statistics.fmean(
                        run.ood_by_condition[str(condition)] for run in group
                    )
                    for condition in OOD_CONDITIONS
                },
            }
        summary[source]["bounded_gain"]["gain_normalization"] = normalizations[source]
        summary[source]["bounded_gain"]["target_clip_fraction"] = clip_fractions[source]

        full = {run.seed: run for run in source_runs if run.mode == "full_update"}
        bounded = {run.seed: run for run in source_runs if run.mode == "bounded_gain"}
        iid_deltas = [
            bounded[seed].iid_test_loss_ratio - full[seed].iid_test_loss_ratio
            for seed in sorted(full)
        ]
        ood_deltas = [
            bounded[seed].ood_test_loss_ratio - full[seed].ood_test_loss_ratio
            for seed in sorted(full)
        ]
        paired[source] = {
            "iid_bounded_minus_full_mean_delta": statistics.fmean(iid_deltas),
            "iid_bounded_beats_full_seeds": sum(delta < 0.0 for delta in iid_deltas),
            "ood_bounded_minus_full_mean_delta": statistics.fmean(ood_deltas),
            "ood_bounded_beats_full_seeds": sum(delta < 0.0 for delta in ood_deltas),
            "total_seeds": len(iid_deltas),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "train_conditions": TRAIN_CONDITIONS,
        "ood_conditions": OOD_CONDITIONS,
        "history_size": HISTORY_SIZE,
        "gain_bound": GAIN_BOUND,
        "outer_lr_candidates": OUTER_LR_CANDIDATES,
        "tuned_lrs": lrs,
        "gain_normalizations": normalizations,
        "target_clip_fractions": clip_fractions,
        "teachers": teachers,
        "runs": [asdict(run) for run in runs],
        "summary": summary,
        "paired_deltas": paired,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
