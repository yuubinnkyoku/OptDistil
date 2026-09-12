from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import (
    DISTILL_WEIGHTS,
    STUDENT_SCALE_CANDIDATES,
    collect_records,
    evaluate_student,
    flatten,
    make_split,
    make_teacher,
    select_lr,
)

from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.rollout import select_student_output_scale
from optdistil.distill.secant_features import SecantFeatureState
from optdistil.distill.train import train_student
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer

HISTORY_SIZES = (1, 2, 4)
SOURCES = ("norm_gradient", "newton_025", "newton_050", "muon")


@dataclass(frozen=True, slots=True)
class ParetoRun:
    source: str
    basis: str
    history_size: int
    seed: int
    student_parameters: int
    additional_state_scalars_per_parameter: int
    total_state_scalars_per_parameter: int
    train_records: int
    final_distillation_loss: float
    validation_loss_ratio: float
    validation_scale: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the deployment-state/performance Pareto curve for a fixed 153-parameter "
            "student using L-BFGS secant histories of length 1, 2, and 4."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=191000)
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
    args.student_seeds = 5


@torch.no_grad()
def collect_secant_records(source, cases, *, lrs, steps, history_size):
    records: list[TrajectoryRecord] = []
    for initial, task in cases:
        parameter = initial.detach().clone()
        ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
        secant_state = SecantFeatureState(history_size=history_size)
        teacher = make_teacher(source, task, lrs)
        for step in range(1, steps + 1):
            grad = task.grad(parameter)
            momentum, second_moment = ema_state.observe(grad)
            features = secant_state.build(
                parameter,
                grad,
                momentum,
                second_moment,
                step=step,
                total_steps=steps,
            )
            teacher_update = teacher.step(parameter, grad).detach()
            records.append(
                TrajectoryRecord(
                    features.detach(),
                    teacher_update.reshape(-1),
                    {
                        "step": step,
                        "teacher": source,
                        "features": f"secant_lbfgs_{history_size}",
                    },
                )
            )
            parameter = parameter + teacher_update
    return records


@torch.no_grad()
def rollout_secant_student(student, initial, task, *, steps, history_size):
    parameter = initial.detach().clone()
    ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    secant_state = SecantFeatureState(history_size=history_size)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss
    student.eval()
    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        momentum, second_moment = ema_state.observe(grad)
        features = secant_state.build(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        update = student(features).reshape_as(parameter)
        parameter = parameter + update
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


@torch.no_grad()
def select_secant_scale(student, validation_cases, *, steps, history_size):
    scored = []
    for scale in STUDENT_SCALE_CANDIDATES:
        student.set_output_scale(scale)
        ratios = [
            rollout_secant_student(
                student,
                initial,
                task,
                steps=steps,
                history_size=history_size,
            )
            for initial, task in validation_cases
        ]
        scored.append((scale, statistics.fmean(ratios)))
    best_scale, best_score = min(scored, key=lambda item: item[1])
    student.set_output_scale(best_scale)
    return best_scale, best_score


@torch.no_grad()
def evaluate_secant_student(student, split, *, steps, history_size):
    by_condition = {}
    for condition, cases in split.items():
        ratios = [
            rollout_secant_student(
                student,
                initial,
                task,
                steps=steps,
                history_size=history_size,
            )
            for initial, task in cases
        ]
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def state_scalars(history_size: int) -> tuple[int, int]:
    if history_size == 0:
        return 0, 2
    additional = 2 + 2 * history_size
    return additional, 2 + additional


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
        raise ValueError("sizes, task counts, epochs, and seed counts must be positive")

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
    lr_cases = flatten(lr_validation_split)
    distill_cases = flatten(distill_split)
    student_validation_cases = flatten(student_validation_split)

    lrs = {
        source: select_lr(source, lr_cases, steps=args.steps)
        for source in ("muon", "norm_gradient")
    }

    records = {
        (source, 0): collect_records(
            source,
            distill_cases,
            lrs=lrs,
            steps=args.steps,
            feature_builder=build_matrix_aware_features,
        )
        for source in SOURCES
    }
    for source in SOURCES:
        for history_size in HISTORY_SIZES:
            records[(source, history_size)] = collect_secant_records(
                source,
                distill_cases,
                lrs=lrs,
                steps=args.steps,
                history_size=history_size,
            )

    runs: list[ParetoRun] = []
    for source_index, source in enumerate(SOURCES):
        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 1000 * source_index + seed_index
            for history_size in (0, *HISTORY_SIZES):
                torch.manual_seed(seed)
                student = TinyMLPOptimizer(feature_dim=8, hidden_dim=8, hidden_layers=2).to(device)
                history = train_student(
                    student,
                    records[(source, history_size)],
                    epochs=args.distill_epochs,
                    lr=3e-3,
                    weights=DISTILL_WEIGHTS,
                )
                if history_size == 0:
                    selection = select_student_output_scale(
                        student,
                        student_validation_cases,
                        candidates=STUDENT_SCALE_CANDIDATES,
                        steps=args.steps,
                        feature_builder=build_matrix_aware_features,
                    )
                    scale = selection.scale
                    validation_ratio = selection.validation_loss_ratio
                    test_ratio, by_condition = evaluate_student(
                        student,
                        test_split,
                        steps=args.steps,
                        feature_builder=build_matrix_aware_features,
                    )
                    basis = "row_col_rms"
                else:
                    scale, validation_ratio = select_secant_scale(
                        student,
                        student_validation_cases,
                        steps=args.steps,
                        history_size=history_size,
                    )
                    test_ratio, by_condition = evaluate_secant_student(
                        student,
                        test_split,
                        steps=args.steps,
                        history_size=history_size,
                    )
                    basis = f"secant_lbfgs_{history_size}"

                additional_state, total_state = state_scalars(history_size)
                runs.append(
                    ParetoRun(
                        source=source,
                        basis=basis,
                        history_size=history_size,
                        seed=seed,
                        student_parameters=student.parameter_count,
                        additional_state_scalars_per_parameter=additional_state,
                        total_state_scalars_per_parameter=total_state,
                        train_records=len(records[(source, history_size)]),
                        final_distillation_loss=history[-1],
                        validation_loss_ratio=validation_ratio,
                        validation_scale=scale,
                        test_loss_ratio=test_ratio,
                        test_loss_ratio_by_condition=by_condition,
                    )
                )

    summary = {}
    paired_deltas = {}
    for source in SOURCES:
        source_runs = [row for row in runs if row.source == source]
        summary[source] = {}
        for history_size in (0, *HISTORY_SIZES):
            group = [row for row in source_runs if row.history_size == history_size]
            values = [row.test_loss_ratio for row in group]
            key = "row_col_rms" if history_size == 0 else f"secant_lbfgs_{history_size}"
            summary[source][key] = {
                "history_size": history_size,
                "student_parameters": group[0].student_parameters,
                "additional_state_scalars_per_parameter": group[
                    0
                ].additional_state_scalars_per_parameter,
                "total_state_scalars_per_parameter": group[0].total_state_scalars_per_parameter,
                "test_loss_ratio_mean": statistics.fmean(values),
                "test_loss_ratio_seed_std": statistics.pstdev(values),
                "validation_loss_ratio_mean": statistics.fmean(
                    row.validation_loss_ratio for row in group
                ),
                "final_distillation_loss_mean": statistics.fmean(
                    row.final_distillation_loss for row in group
                ),
            }

        baseline = {
            row.seed: row.test_loss_ratio for row in source_runs if row.history_size == 0
        }
        paired_deltas[source] = {}
        for history_size in HISTORY_SIZES:
            candidate = {
                row.seed: row.test_loss_ratio
                for row in source_runs
                if row.history_size == history_size
            }
            deltas = [candidate[seed] - baseline[seed] for seed in sorted(baseline)]
            relative = [
                (candidate[seed] - baseline[seed]) / baseline[seed]
                for seed in sorted(baseline)
            ]
            paired_deltas[source][f"secant_lbfgs_{history_size}"] = {
                "mean_absolute_delta_vs_row_col": statistics.fmean(deltas),
                "mean_relative_delta_vs_row_col": statistics.fmean(relative),
                "improved_seeds": sum(delta < 0.0 for delta in deltas),
                "total_seeds": len(deltas),
            }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_sizes": HISTORY_SIZES,
        "tuned_lrs": lrs,
        "runs": [asdict(row) for row in runs],
        "summary": summary,
        "paired_deltas": paired_deltas,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
