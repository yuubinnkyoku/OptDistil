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
    flatten,
    make_split,
    select_lr,
)
from probe_secant_history_pareto import (
    collect_secant_records,
    evaluate_secant_student,
    rollout_secant_student,
    select_secant_scale,
)

from optdistil.distill.losses import distillation_loss
from optdistil.distill.secant_features import SecantFeatureState
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer

SOURCES = ("norm_gradient", "muon", "newton_050")
HISTORY_SIZE = 4


@dataclass(frozen=True, slots=True)
class GainControlRun:
    source: str
    seed: int
    mode: str
    student_parameters: int
    validation_scale: float
    validation_loss_ratio: float
    test_loss_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired same-weight comparison of a full 153-parameter secant-aware Student and "
            "the same network constrained to emit only one scalar gain multiplying the "
            "L-BFGS direction. This isolates the value of learned residual rotation."
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
    parser.add_argument("--student-seed", type=int, default=211000)
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


def scalar_gain_update(student: TinyMLPOptimizer, features: torch.Tensor) -> torch.Tensor:
    gain = student(features).mean()
    return gain * features[:, 5]


def train_scalar_gain_student(student, records, *, epochs: int, lr: float) -> list[float]:
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    history: list[float] = []
    student.train()
    for _ in range(epochs):
        total = 0.0
        for record in records:
            optimizer.zero_grad(set_to_none=True)
            predicted = scalar_gain_update(student, record.features)
            loss, _ = distillation_loss(
                predicted,
                record.teacher_update,
                weights=DISTILL_WEIGHTS,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append(total / max(len(records), 1))
    return history


@torch.no_grad()
def rollout_scalar_gain_student(student, initial, task, *, steps: int) -> float:
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
        update = scalar_gain_update(student, features).reshape_as(parameter)
        parameter = parameter + update
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf

    return final_loss / max(abs(initial_loss), 1e-12)


@torch.no_grad()
def select_scalar_gain_scale(student, validation_cases, *, steps: int) -> tuple[float, float]:
    scored = []
    for scale in STUDENT_SCALE_CANDIDATES:
        student.set_output_scale(scale)
        score = statistics.fmean(
            rollout_scalar_gain_student(student, initial, task, steps=steps)
            for initial, task in validation_cases
        )
        scored.append((scale, score))
    best_scale, best_score = min(scored, key=lambda item: item[1])
    student.set_output_scale(best_scale)
    return best_scale, best_score


@torch.no_grad()
def evaluate_scalar_gain_student(student, split, *, steps: int) -> float:
    ratios = []
    for cases in split.values():
        ratios.extend(
            rollout_scalar_gain_student(student, initial, task, steps=steps)
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

    runs: list[GainControlRun] = []
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
                weights=DISTILL_WEIGHTS,
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
                GainControlRun(
                    source=source,
                    seed=seed,
                    mode="full_update",
                    student_parameters=full_student.parameter_count,
                    validation_scale=full_scale,
                    validation_loss_ratio=full_validation,
                    test_loss_ratio=full_test,
                )
            )

            torch.manual_seed(seed)
            gain_student = TinyMLPOptimizer().to(device)
            train_scalar_gain_student(
                gain_student,
                records[source],
                epochs=args.distill_epochs,
                lr=3e-3,
            )
            gain_scale, gain_validation = select_scalar_gain_scale(
                gain_student,
                student_validation_cases,
                steps=args.steps,
            )
            gain_test = evaluate_scalar_gain_student(gain_student, test_split, steps=args.steps)
            runs.append(
                GainControlRun(
                    source=source,
                    seed=seed,
                    mode="scalar_gain_only",
                    student_parameters=gain_student.parameter_count,
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
        for mode in ("full_update", "scalar_gain_only"):
            group = [run for run in source_runs if run.mode == mode]
            tests = [run.test_loss_ratio for run in group]
            summary[source][mode] = {
                "student_parameters": group[0].student_parameters,
                "test_loss_ratio_mean": statistics.fmean(tests),
                "test_loss_ratio_seed_std": statistics.pstdev(tests),
                "validation_loss_ratio_mean": statistics.fmean(
                    run.validation_loss_ratio for run in group
                ),
            }
        full = {
            run.seed: run.test_loss_ratio for run in source_runs if run.mode == "full_update"
        }
        gain = {
            run.seed: run.test_loss_ratio
            for run in source_runs
            if run.mode == "scalar_gain_only"
        }
        deltas = [gain[seed] - full[seed] for seed in sorted(full)]
        paired[source] = {
            "gain_minus_full_mean_absolute_delta": statistics.fmean(deltas),
            "gain_beats_full_seeds": sum(delta < 0.0 for delta in deltas),
            "total_seeds": len(deltas),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_size": HISTORY_SIZE,
        "tuned_lrs": lrs,
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
