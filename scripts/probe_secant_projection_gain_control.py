from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from probe_oracle_feature_bottleneck import (
    STUDENT_SCALE_CANDIDATES,
    flatten,
    make_split,
    select_lr,
)
from probe_secant_history_pareto import (
    collect_secant_records,
    evaluate_secant_student,
    select_secant_scale,
)

from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.secant_features import SecantFeatureState
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer

SOURCES = ("norm_gradient", "muon", "newton_050")
HISTORY_SIZE = 4
FULL_DISTILL_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)
EPS = 1e-8


@dataclass(frozen=True, slots=True)
class ProjectionGainRun:
    source: str
    seed: int
    mode: str
    student_parameters: int
    gain_normalization: float | None
    validation_scale: float
    validation_loss_ratio: float
    test_loss_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fair scalar-gain control for secant-aware distillation. Instead of applying the "
            "vector distillation loss to a colinear update, regress the optimal signed "
            "projection coefficient <u_teacher,d>/<d,d> with a robust scalar loss."
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
    parser.add_argument("--student-seed", type=int, default=221000)
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


def projection_target(record) -> torch.Tensor:
    direction = record.features[:, 5].reshape(-1).float()
    teacher = record.teacher_update.reshape(-1).float()
    denominator = torch.dot(direction, direction).clamp_min(EPS)
    return torch.dot(teacher, direction) / denominator


def gain_normalization(records) -> float:
    absolute_targets = torch.stack([projection_target(record).abs() for record in records])
    scale = float(absolute_targets.median())
    if not math.isfinite(scale) or scale <= EPS:
        scale = float(absolute_targets.mean())
    if not math.isfinite(scale) or scale <= EPS:
        return 1.0
    return scale


def projection_gain_update(
    student: TinyMLPOptimizer,
    features: torch.Tensor,
    *,
    normalization: float,
) -> torch.Tensor:
    normalized_gain = student(features).mean()
    gain = normalization * normalized_gain
    return gain * features[:, 5]


def train_projection_gain_student(
    student: TinyMLPOptimizer,
    records,
    *,
    epochs: int,
    lr: float,
    normalization: float,
) -> list[float]:
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    history: list[float] = []
    student.set_output_scale(1.0)
    student.train()

    for _ in range(epochs):
        total = 0.0
        for record in records:
            optimizer.zero_grad(set_to_none=True)
            predicted_normalized_gain = student(record.features).mean()
            target_normalized_gain = projection_target(record).to(
                device=predicted_normalized_gain.device,
                dtype=predicted_normalized_gain.dtype,
            ) / normalization
            loss = F.smooth_l1_loss(
                predicted_normalized_gain,
                target_normalized_gain,
                beta=0.5,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append(total / max(len(records), 1))
    return history


@torch.no_grad()
def rollout_projection_gain_student(
    student,
    initial,
    task,
    *,
    steps: int,
    normalization: float,
) -> float:
    parameter = initial.detach().clone()
    ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    secant_state = SecantFeatureState(history_size=HISTORY_SIZE)
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
        update = projection_gain_update(
            student,
            features,
            normalization=normalization,
        ).reshape_as(parameter)
        parameter = parameter + update
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf

    return final_loss / max(abs(initial_loss), 1e-12)


@torch.no_grad()
def select_projection_gain_scale(
    student,
    validation_cases,
    *,
    steps: int,
    normalization: float,
) -> tuple[float, float, bool]:
    scored = []
    for scale in STUDENT_SCALE_CANDIDATES:
        student.set_output_scale(scale)
        score = statistics.fmean(
            rollout_projection_gain_student(
                student,
                initial,
                task,
                steps=steps,
                normalization=normalization,
            )
            for initial, task in validation_cases
        )
        scored.append((scale, score))
    best_scale, best_score = min(scored, key=lambda item: item[1])
    student.set_output_scale(best_scale)
    at_boundary = best_scale in (STUDENT_SCALE_CANDIDATES[0], STUDENT_SCALE_CANDIDATES[-1])
    return best_scale, best_score, at_boundary


@torch.no_grad()
def evaluate_projection_gain_student(
    student,
    split,
    *,
    steps: int,
    normalization: float,
) -> float:
    ratios = []
    for cases in split.values():
        ratios.extend(
            rollout_projection_gain_student(
                student,
                initial,
                task,
                steps=steps,
                normalization=normalization,
            )
            for initial, task in cases
        )
    return statistics.fmean(ratios)


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

    runs: list[ProjectionGainRun] = []
    boundary_counts = {source: 0 for source in SOURCES}
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
            full_scale, full_validation = select_secant_scale(
                full_student,
                student_validation_cases,
                steps=args.steps,
                history_size=HISTORY_SIZE,
            )
            full_test, _ = evaluate_secant_student(
                full_student,
                test_split,
                steps=args.steps,
                history_size=HISTORY_SIZE,
            )
            runs.append(
                ProjectionGainRun(
                    source=source,
                    seed=seed,
                    mode="full_update",
                    student_parameters=full_student.parameter_count,
                    gain_normalization=None,
                    validation_scale=full_scale,
                    validation_loss_ratio=full_validation,
                    test_loss_ratio=full_test,
                )
            )

            torch.manual_seed(seed)
            gain_student = TinyMLPOptimizer().to(device)
            normalization = normalizations[source]
            train_projection_gain_student(
                gain_student,
                records[source],
                epochs=args.distill_epochs,
                lr=3e-3,
                normalization=normalization,
            )
            gain_scale, gain_validation, at_boundary = select_projection_gain_scale(
                gain_student,
                student_validation_cases,
                steps=args.steps,
                normalization=normalization,
            )
            boundary_counts[source] += int(at_boundary)
            gain_test = evaluate_projection_gain_student(
                gain_student,
                test_split,
                steps=args.steps,
                normalization=normalization,
            )
            runs.append(
                ProjectionGainRun(
                    source=source,
                    seed=seed,
                    mode="projection_gain_regression",
                    student_parameters=gain_student.parameter_count,
                    gain_normalization=normalization,
                    validation_scale=gain_scale,
                    validation_loss_ratio=gain_validation,
                    test_loss_ratio=gain_test,
                )
            )

    summary = {}
    paired = {}
    for source in SOURCES:
        source_runs = [run for run in runs if run.source == source]
        summary[source] = {}
        for mode in ("full_update", "projection_gain_regression"):
            group = [run for run in source_runs if run.mode == mode]
            tests = [run.test_loss_ratio for run in group]
            summary[source][mode] = {
                "student_parameters": group[0].student_parameters,
                "test_loss_ratio_mean": statistics.fmean(tests),
                "test_loss_ratio_seed_std": statistics.pstdev(tests),
                "validation_loss_ratio_mean": statistics.fmean(
                    run.validation_loss_ratio for run in group
                ),
                "validation_scale_mean": statistics.fmean(run.validation_scale for run in group),
            }
        summary[source]["projection_gain_regression"]["gain_normalization"] = normalizations[
            source
        ]
        summary[source]["projection_gain_regression"]["scale_boundary_fraction"] = (
            boundary_counts[source] / args.student_seeds
        )

        full = {
            run.seed: run.test_loss_ratio for run in source_runs if run.mode == "full_update"
        }
        gain = {
            run.seed: run.test_loss_ratio
            for run in source_runs
            if run.mode == "projection_gain_regression"
        }
        deltas = [gain[seed] - full[seed] for seed in sorted(full)]
        paired[source] = {
            "gain_minus_full_mean_absolute_delta": statistics.fmean(deltas),
            "gain_minus_full_mean_relative_delta": statistics.fmean(
                (gain[seed] - full[seed]) / full[seed] for seed in sorted(full)
            ),
            "gain_beats_full_seeds": sum(delta < 0.0 for delta in deltas),
            "total_seeds": len(deltas),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_size": HISTORY_SIZE,
        "tuned_lrs": lrs,
        "gain_normalizations": normalizations,
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
