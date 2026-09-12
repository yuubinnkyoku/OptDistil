from __future__ import annotations

import argparse
import json
import math
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_stochastic_nonlinear_controls import (
    BOOTSTRAP_SCALES,
    SECANT_SCALES,
    StochasticCase,
    batch_sequence,
    make_split,
    rollout_secant,
    rollout_teacher,
    tune_secant,
    tune_teacher,
)

from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.train import train_student
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.gradient_direction import GradientDirectionTeacher

TRAIN_CONDITIONS = (30.0, 300.0)
OOD_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
REGIMES = (
    ("adamw_b32", "adamw", 32),
    ("norm_gradient_b8", "norm_gradient", 8),
)
STUDENT_SCALE_CANDIDATES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
OUTER_LR_CANDIDATES = (3e-4, 1e-3, 3e-3)
DISTILL_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)


@dataclass(frozen=True, slots=True)
class StudentRun:
    regime: str
    teacher: str
    batch_size: int
    seed: int
    mode: str
    student_parameters: int
    validation_scale: float
    selected_outer_lr: float | None
    meta_validation_loss_ratio: float | None
    test_loss_ratio: float
    test_by_condition: dict[str, float]
    ood_loss_ratio: float
    ood_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill stochastic AdamW and normalized-gradient teachers into the fixed 153p "
            "matrix-aware Student, then compare supervised distillation and closed-loop meta refinement."
        )
    )
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr-validation-tasks", type=int, default=6)
    parser.add_argument("--distill-train-tasks", type=int, default=8)
    parser.add_argument("--scale-validation-tasks", type=int, default=6)
    parser.add_argument("--meta-train-tasks", type=int, default=6)
    parser.add_argument("--meta-validation-tasks", type=int, default=6)
    parser.add_argument("--test-tasks", type=int, default=10)
    parser.add_argument("--ood-test-tasks", type=int, default=6)
    parser.add_argument("--distill-epochs", type=int, default=25)
    parser.add_argument("--meta-iterations", type=int, default=15)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=401000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 16
    args.lr_validation_tasks = 3
    args.distill_train_tasks = 3
    args.scale_validation_tasks = 3
    args.meta_train_tasks = 3
    args.meta_validation_tasks = 3
    args.test_tasks = 4
    args.ood_test_tasks = 3
    args.distill_epochs = 10
    args.meta_iterations = 6
    args.student_seeds = 2


def flatten(split) -> list[StochasticCase]:
    return [case for cases in split.values() for case in cases]


def make_teacher(method: str, lr: float):
    if method == "adamw":
        return AdamWTeacher(lr=lr)
    if method == "norm_gradient":
        return GradientDirectionTeacher(lr=lr)
    raise ValueError(f"unsupported distillation teacher: {method}")


@torch.no_grad()
def collect_records(
    method: str,
    lr: float,
    cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for task_index, case in enumerate(cases):
        parameter = case.initial.detach().clone()
        teacher = make_teacher(method, lr)
        student_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
        batches = batch_sequence(case, batch_size=batch_size, steps=steps)
        for step, indices in enumerate(batches, start=1):
            grad = case.task.grad_on_samples(parameter, indices)
            momentum, second_moment = student_state.observe(grad)
            features = build_matrix_aware_features(
                parameter,
                grad,
                momentum,
                second_moment,
                step=step,
                total_steps=steps,
            )
            update = teacher.step(parameter, grad).detach()
            records.append(
                TrajectoryRecord(
                    features=features.detach(),
                    teacher_update=update.reshape(-1),
                    metadata={
                        "teacher": method,
                        "task_index": task_index,
                        "step": step,
                        "batch_size": batch_size,
                        "full_loss_before": float(case.task.loss(parameter)),
                    },
                )
            )
            parameter = parameter + update
    return records


@torch.no_grad()
def rollout_student(
    student: TinyMLPOptimizer,
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
) -> float:
    parameter = case.initial.detach().clone()
    state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(parameter))
    student.eval()
    for step, indices in enumerate(batches, start=1):
        grad = case.task.grad_on_samples(parameter, indices)
        momentum, second_moment = state.observe(grad)
        features = build_matrix_aware_features(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        parameter = parameter + student(features).reshape_as(parameter)
        if not torch.isfinite(parameter).all():
            return math.inf
    final_loss = float(case.task.loss(parameter))
    if not math.isfinite(final_loss):
        return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def evaluate_cases(
    student: TinyMLPOptimizer,
    cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
) -> float:
    return statistics.fmean(
        rollout_student(student, case, batch_size=batch_size, steps=steps) for case in cases
    )


def evaluate_split(student, split, *, batch_size: int, steps: int):
    by_condition = {}
    ratios = []
    for condition, cases in split.items():
        values = [rollout_student(student, case, batch_size=batch_size, steps=steps) for case in cases]
        by_condition[str(condition)] = statistics.fmean(values)
        ratios.extend(values)
    return statistics.fmean(ratios), by_condition


def select_student_scale(
    student: TinyMLPOptimizer,
    validation_cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
) -> tuple[float, float]:
    scored = []
    for scale in STUDENT_SCALE_CANDIDATES:
        student.set_output_scale(scale)
        score = evaluate_cases(
            student,
            validation_cases,
            batch_size=batch_size,
            steps=steps,
        )
        scored.append((score, scale))
    score, scale = min(scored, key=lambda item: item[0])
    student.set_output_scale(scale)
    return scale, score


def differentiable_rollout(
    student: TinyMLPOptimizer,
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = case.initial.detach().clone()
    momentum = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = case.task.loss(parameter).detach().clamp_min(1e-12)
    ratios = []

    for step, indices in enumerate(batches, start=1):
        grad = case.task.grad_on_samples(parameter, indices).detach()
        momentum = 0.9 * momentum + 0.1 * grad
        second_moment = 0.999 * second_moment + 0.001 * grad.square()
        features = build_matrix_aware_features(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        parameter = parameter + student(features).reshape_as(parameter)
        ratios.append(case.task.loss(parameter) / initial_loss)
    return ratios[-1], torch.stack(ratios).mean()


def meta_objective(
    student: TinyMLPOptimizer,
    cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
    final_weight: float = 0.7,
) -> torch.Tensor:
    values = []
    for case in cases:
        final_ratio, mean_ratio = differentiable_rollout(
            student,
            case,
            batch_size=batch_size,
            steps=steps,
        )
        values.append(final_weight * final_ratio + (1.0 - final_weight) * mean_ratio)
    return torch.stack(values).mean()


def clone_state(student: TinyMLPOptimizer):
    return {name: value.detach().clone() for name, value in student.state_dict().items()}


def train_meta(
    student: TinyMLPOptimizer,
    train_cases: list[StochasticCase],
    validation_cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
    iterations: int,
    outer_lr: float,
) -> float:
    outer = torch.optim.Adam(student.parameters(), lr=outer_lr)
    best_validation = evaluate_cases(
        student,
        validation_cases,
        batch_size=batch_size,
        steps=steps,
    )
    best_state = clone_state(student)

    for _ in range(iterations):
        student.train()
        outer.zero_grad(set_to_none=True)
        objective = meta_objective(
            student,
            train_cases,
            batch_size=batch_size,
            steps=steps,
        )
        if not torch.isfinite(objective):
            break
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            break
        outer.step()
        validation = evaluate_cases(
            student,
            validation_cases,
            batch_size=batch_size,
            steps=steps,
        )
        if validation < best_validation:
            best_validation = validation
            best_state = clone_state(student)

    student.load_state_dict(best_state)
    student.eval()
    return best_validation


def select_meta_lr(
    initial_state,
    *,
    device: torch.device,
    train_cases: list[StochasticCase],
    validation_cases: list[StochasticCase],
    batch_size: int,
    steps: int,
    iterations: int,
) -> tuple[TinyMLPOptimizer, float, float]:
    candidates = []
    for outer_lr in OUTER_LR_CANDIDATES:
        student = TinyMLPOptimizer().to(device)
        student.load_state_dict(deepcopy(initial_state))
        validation = train_meta(
            student,
            train_cases,
            validation_cases,
            batch_size=batch_size,
            steps=steps,
            iterations=iterations,
            outer_lr=outer_lr,
        )
        candidates.append((validation, outer_lr, student))
    validation, outer_lr, student = min(candidates, key=lambda item: item[0])
    return student, outer_lr, validation


def analytic_results(
    split,
    *,
    methods: dict[str, float],
    batch_size: int,
    steps: int,
):
    result = {}
    for method, lr in methods.items():
        by_condition = {}
        values = []
        for condition, cases in split.items():
            ratios = [
                rollout_teacher(method, lr, case, batch_size=batch_size, steps=steps)
                for case in cases
            ]
            by_condition[str(condition)] = statistics.fmean(ratios)
            values.extend(ratios)
        result[method] = {
            "loss_ratio": statistics.fmean(values),
            "by_condition": by_condition,
        }
    return result


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.samples,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.scale_validation_tasks,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.test_tasks,
        args.ood_test_tasks,
        args.distill_epochs,
        args.meta_iterations,
        args.student_seeds,
    ) <= 0:
        raise ValueError("all dimensions, task counts, epochs, iterations, and seed counts must be positive")

    device = torch.device(args.device)
    split_specs = {
        "lr_validation": (411000, args.lr_validation_tasks),
        "distill": (421000, args.distill_train_tasks),
        "scale_validation": (431000, args.scale_validation_tasks),
        "meta_train": (441000, args.meta_train_tasks),
        "meta_validation": (451000, args.meta_validation_tasks),
        "test": (461000, args.test_tasks),
    }
    splits = {
        name: make_split(
            TRAIN_CONDITIONS,
            seed_base=seed_base,
            count=count,
            size=args.size,
            samples=args.samples,
            device=device,
        )
        for name, (seed_base, count) in split_specs.items()
    }
    ood_split = make_split(
        OOD_CONDITIONS,
        seed_base=471000,
        count=args.ood_test_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )

    lr_cases = flatten(splits["lr_validation"])
    distill_cases = flatten(splits["distill"])
    scale_validation_cases = flatten(splits["scale_validation"])
    meta_train_cases = flatten(splits["meta_train"])
    meta_validation_cases = flatten(splits["meta_validation"])

    payload_regimes = {}
    all_runs: list[StudentRun] = []
    for regime_index, (regime, teacher_method, batch_size) in enumerate(REGIMES):
        tuned_lrs = {
            method: tune_teacher(
                method,
                lr_cases,
                batch_size=batch_size,
                steps=args.steps,
            )[0]
            for method in ("adamw", "norm_gradient")
        }
        records = collect_records(
            teacher_method,
            tuned_lrs[teacher_method],
            distill_cases,
            batch_size=batch_size,
            steps=args.steps,
        )
        secant_tuning, secant_validation = tune_secant(
            scale_validation_cases,
            batch_size=batch_size,
            steps=args.steps,
        )

        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 1000 * regime_index + seed_index
            torch.manual_seed(seed)
            student = TinyMLPOptimizer().to(device)
            train_student(
                student,
                records,
                epochs=args.distill_epochs,
                lr=3e-3,
                weights=DISTILL_WEIGHTS,
            )
            scale, _ = select_student_scale(
                student,
                scale_validation_cases,
                batch_size=batch_size,
                steps=args.steps,
            )
            test_ratio, test_by_condition = evaluate_split(
                student,
                splits["test"],
                batch_size=batch_size,
                steps=args.steps,
            )
            ood_ratio, ood_by_condition = evaluate_split(
                student,
                ood_split,
                batch_size=batch_size,
                steps=args.steps,
            )
            all_runs.append(
                StudentRun(
                    regime=regime,
                    teacher=teacher_method,
                    batch_size=batch_size,
                    seed=seed,
                    mode="distill_only",
                    student_parameters=student.parameter_count,
                    validation_scale=scale,
                    selected_outer_lr=None,
                    meta_validation_loss_ratio=None,
                    test_loss_ratio=test_ratio,
                    test_by_condition=test_by_condition,
                    ood_loss_ratio=ood_ratio,
                    ood_by_condition=ood_by_condition,
                )
            )

            meta_student, outer_lr, meta_validation = select_meta_lr(
                clone_state(student),
                device=device,
                train_cases=meta_train_cases,
                validation_cases=meta_validation_cases,
                batch_size=batch_size,
                steps=args.steps,
                iterations=args.meta_iterations,
            )
            meta_test, meta_test_by_condition = evaluate_split(
                meta_student,
                splits["test"],
                batch_size=batch_size,
                steps=args.steps,
            )
            meta_ood, meta_ood_by_condition = evaluate_split(
                meta_student,
                ood_split,
                batch_size=batch_size,
                steps=args.steps,
            )
            all_runs.append(
                StudentRun(
                    regime=regime,
                    teacher=teacher_method,
                    batch_size=batch_size,
                    seed=seed,
                    mode="distill_meta",
                    student_parameters=meta_student.parameter_count,
                    validation_scale=scale,
                    selected_outer_lr=outer_lr,
                    meta_validation_loss_ratio=meta_validation,
                    test_loss_ratio=meta_test,
                    test_by_condition=meta_test_by_condition,
                    ood_loss_ratio=meta_ood,
                    ood_by_condition=meta_ood_by_condition,
                )
            )

        test_analytic = analytic_results(
            splits["test"],
            methods=tuned_lrs,
            batch_size=batch_size,
            steps=args.steps,
        )
        ood_analytic = analytic_results(
            ood_split,
            methods=tuned_lrs,
            batch_size=batch_size,
            steps=args.steps,
        )
        secant_test_by_condition = {}
        secant_test_values = []
        for condition, cases in splits["test"].items():
            values = [
                rollout_secant(
                    case,
                    batch_size=batch_size,
                    steps=args.steps,
                    secant_scale=secant_tuning["secant_scale"],
                    bootstrap_scale=secant_tuning["bootstrap_scale"],
                )
                for case in cases
            ]
            secant_test_by_condition[str(condition)] = statistics.fmean(values)
            secant_test_values.extend(values)
        secant_ood_by_condition = {}
        secant_ood_values = []
        for condition, cases in ood_split.items():
            values = [
                rollout_secant(
                    case,
                    batch_size=batch_size,
                    steps=args.steps,
                    secant_scale=secant_tuning["secant_scale"],
                    bootstrap_scale=secant_tuning["bootstrap_scale"],
                )
                for case in cases
            ]
            secant_ood_by_condition[str(condition)] = statistics.fmean(values)
            secant_ood_values.extend(values)

        payload_regimes[regime] = {
            "teacher": teacher_method,
            "batch_size": batch_size,
            "tuned_lrs": tuned_lrs,
            "train_records": len(records),
            "analytic_test": test_analytic,
            "analytic_ood": ood_analytic,
            "raw_lbfgs_two_scale": {
                "validation_loss_ratio": secant_validation,
                "tuning": secant_tuning,
                "test_loss_ratio": statistics.fmean(secant_test_values),
                "test_by_condition": secant_test_by_condition,
                "ood_loss_ratio": statistics.fmean(secant_ood_values),
                "ood_by_condition": secant_ood_by_condition,
            },
        }

    summary = {}
    paired = {}
    for regime, _, _ in REGIMES:
        regime_runs = [run for run in all_runs if run.regime == regime]
        summary[regime] = {}
        for mode in ("distill_only", "distill_meta"):
            group = [run for run in regime_runs if run.mode == mode]
            tests = [run.test_loss_ratio for run in group]
            oods = [run.ood_loss_ratio for run in group]
            summary[regime][mode] = {
                "student_parameters": group[0].student_parameters,
                "test_mean": statistics.fmean(tests),
                "test_median": statistics.median(tests),
                "test_seed_std": statistics.pstdev(tests),
                "ood_mean": statistics.fmean(oods),
                "ood_median": statistics.median(oods),
                "ood_seed_std": statistics.pstdev(oods),
                "validation_scale_mean": statistics.fmean(run.validation_scale for run in group),
            }
        distill = {run.seed: run for run in regime_runs if run.mode == "distill_only"}
        meta = {run.seed: run for run in regime_runs if run.mode == "distill_meta"}
        test_deltas = [meta[seed].test_loss_ratio - distill[seed].test_loss_ratio for seed in distill]
        ood_deltas = [meta[seed].ood_loss_ratio - distill[seed].ood_loss_ratio for seed in distill]
        paired[regime] = {
            "meta_beats_distill_test_seeds": sum(delta < 0.0 for delta in test_deltas),
            "meta_minus_distill_test_mean_delta": statistics.fmean(test_deltas),
            "meta_beats_distill_ood_seeds": sum(delta < 0.0 for delta in ood_deltas),
            "meta_minus_distill_ood_mean_delta": statistics.fmean(ood_deltas),
            "total_seeds": len(test_deltas),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "train_conditions": TRAIN_CONDITIONS,
        "ood_conditions": OOD_CONDITIONS,
        "distillation_weights": asdict(DISTILL_WEIGHTS),
        "regimes": payload_regimes,
        "runs": [asdict(run) for run in all_runs],
        "summary": summary,
        "paired": paired,
        "student_parameters": 153,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
