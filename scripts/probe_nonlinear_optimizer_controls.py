from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from optdistil.distill.secant_features import SecantFeatureState
from optdistil.students.tiny_mlp import StudentState
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.gradient_direction import GradientDirectionTeacher
from optdistil.teachers.muon import MuonTeacher

TRAIN_CONDITIONS = (30.0, 300.0)
OOD_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
HISTORY_SIZE = 4
TEACHER_LRS = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)
SECANT_SCALES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
BOOTSTRAP_SCALES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75)


@dataclass(frozen=True, slots=True)
class MethodResult:
    method: str
    validation_loss_ratio: float | None
    tuning: dict[str, float]
    iid_loss_ratio: float
    ood_loss_ratio: float
    iid_by_condition: dict[str, float]
    ood_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tuned optimizers and raw L-BFGS controls on a nonlinear matrix task."
    )
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--validation-tasks", type=int, default=6)
    parser.add_argument("--iid-test-tasks", type=int, default=10)
    parser.add_argument("--ood-test-tasks", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 16
    args.validation_tasks = 3
    args.iid_test_tasks = 4
    args.ood_test_tasks = 3


def make_split(
    conditions: tuple[float, ...],
    *,
    seed_base: int,
    count: int,
    size: int,
    samples: int,
    device: torch.device,
):
    return {
        condition: [
            make_frozen_readout_mlp(
                seed_base + 10000 * condition_index + index,
                hidden_dim=size,
                input_dim=size,
                output_dim=max(2, size // 2),
                samples=samples,
                input_condition=condition,
                device=device,
            )
            for index in range(count)
        ]
        for condition_index, condition in enumerate(conditions)
    }


def flatten(split):
    return [case for cases in split.values() for case in cases]


def make_teacher(method: str, lr: float):
    if method == "adamw":
        return AdamWTeacher(lr=lr)
    if method == "muon":
        return MuonTeacher(lr=lr)
    if method == "norm_gradient":
        return GradientDirectionTeacher(lr=lr)
    raise ValueError(f"unknown teacher method: {method}")


@torch.no_grad()
def rollout_teacher(method: str, lr: float, initial, task, *, steps: int) -> float:
    parameter = initial.detach().clone()
    teacher = make_teacher(method, lr)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss
    for _ in range(steps):
        grad = task.grad(parameter)
        parameter = parameter + teacher.step(parameter, grad)
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def tune_teacher(method: str, validation_cases, *, steps: int) -> tuple[float, float]:
    scored = []
    for lr in TEACHER_LRS:
        score = statistics.fmean(
            rollout_teacher(method, lr, initial, task, steps=steps)
            for initial, task in validation_cases
        )
        scored.append((score, lr))
    score, lr = min(scored, key=lambda item: item[0])
    return lr, score


@torch.no_grad()
def rollout_secant(
    initial,
    task,
    *,
    steps: int,
    secant_scale: float,
    bootstrap_scale: float | None,
) -> float:
    parameter = initial.detach().clone()
    ema = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    state = SecantFeatureState(history_size=HISTORY_SIZE, normalize_direction=False)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        momentum, second_moment = ema.observe(grad)
        features = state.build(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        if step == 1 and bootstrap_scale is not None:
            grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
            update = -bootstrap_scale * grad / grad_rms
        else:
            update = secant_scale * features[:, 5].reshape_as(parameter)
        parameter = parameter + update
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def tune_secant(validation_cases, *, steps: int, two_scale: bool) -> tuple[dict[str, float], float]:
    scored = []
    bootstraps = BOOTSTRAP_SCALES if two_scale else (None,)
    for bootstrap in bootstraps:
        for secant_scale in SECANT_SCALES:
            score = statistics.fmean(
                rollout_secant(
                    initial,
                    task,
                    steps=steps,
                    secant_scale=secant_scale,
                    bootstrap_scale=bootstrap,
                )
                for initial, task in validation_cases
            )
            scored.append((score, bootstrap, secant_scale))
    score, bootstrap, secant_scale = min(scored, key=lambda item: item[0])
    tuning = {"secant_scale": secant_scale}
    if bootstrap is not None:
        tuning["bootstrap_scale"] = bootstrap
    return tuning, score


@torch.no_grad()
def rollout_armijo(initial, task, *, steps: int, max_backtracks: int = 12) -> float:
    parameter = initial.detach().clone()
    ema = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    state = SecantFeatureState(history_size=HISTORY_SIZE, normalize_direction=False)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        momentum, second_moment = ema.observe(grad)
        features = state.build(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        direction = features[:, 5].reshape_as(parameter)
        directional_derivative = float(torch.sum(grad * direction))
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            direction = -grad
            directional_derivative = -float(torch.sum(grad.square()))

        before = float(task.loss(parameter))
        alpha = 1.0
        accepted = False
        for _ in range(max_backtracks + 1):
            candidate = parameter + alpha * direction
            candidate_loss = float(task.loss(candidate))
            if math.isfinite(candidate_loss) and candidate_loss <= before + 1e-4 * alpha * directional_derivative:
                parameter = candidate
                final_loss = candidate_loss
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return before / max(abs(initial_loss), 1e-12)
    return final_loss / max(abs(initial_loss), 1e-12)


def evaluate_split(split, rollout) -> tuple[float, dict[str, float]]:
    by_condition = {}
    all_ratios = []
    for condition, cases in split.items():
        ratios = [rollout(initial, task) for initial, task in cases]
        by_condition[str(condition)] = statistics.fmean(ratios)
        all_ratios.extend(ratios)
    return statistics.fmean(all_ratios), by_condition


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.samples,
        args.steps,
        args.validation_tasks,
        args.iid_test_tasks,
        args.ood_test_tasks,
    ) <= 0:
        raise ValueError("dimensions, steps, and task counts must be positive")

    device = torch.device(args.device)
    validation = make_split(
        TRAIN_CONDITIONS,
        seed_base=301000,
        count=args.validation_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )
    iid_test = make_split(
        TRAIN_CONDITIONS,
        seed_base=311000,
        count=args.iid_test_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )
    ood_test = make_split(
        OOD_CONDITIONS,
        seed_base=321000,
        count=args.ood_test_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )
    validation_cases = flatten(validation)

    results: list[MethodResult] = []
    for method in ("adamw", "muon", "norm_gradient"):
        lr, validation_score = tune_teacher(method, validation_cases, steps=args.steps)
        iid, iid_by_condition = evaluate_split(
            iid_test,
            lambda initial, task, method=method, lr=lr: rollout_teacher(
                method, lr, initial, task, steps=args.steps
            ),
        )
        ood, ood_by_condition = evaluate_split(
            ood_test,
            lambda initial, task, method=method, lr=lr: rollout_teacher(
                method, lr, initial, task, steps=args.steps
            ),
        )
        results.append(
            MethodResult(
                method=method,
                validation_loss_ratio=validation_score,
                tuning={"lr": lr},
                iid_loss_ratio=iid,
                ood_loss_ratio=ood,
                iid_by_condition=iid_by_condition,
                ood_by_condition=ood_by_condition,
            )
        )

    for name, two_scale in (("raw_lbfgs_global", False), ("raw_lbfgs_two_scale", True)):
        tuning, validation_score = tune_secant(
            validation_cases,
            steps=args.steps,
            two_scale=two_scale,
        )
        iid, iid_by_condition = evaluate_split(
            iid_test,
            lambda initial, task, tuning=tuning: rollout_secant(
                initial,
                task,
                steps=args.steps,
                secant_scale=tuning["secant_scale"],
                bootstrap_scale=tuning.get("bootstrap_scale"),
            ),
        )
        ood, ood_by_condition = evaluate_split(
            ood_test,
            lambda initial, task, tuning=tuning: rollout_secant(
                initial,
                task,
                steps=args.steps,
                secant_scale=tuning["secant_scale"],
                bootstrap_scale=tuning.get("bootstrap_scale"),
            ),
        )
        results.append(
            MethodResult(
                method=name,
                validation_loss_ratio=validation_score,
                tuning=tuning,
                iid_loss_ratio=iid,
                ood_loss_ratio=ood,
                iid_by_condition=iid_by_condition,
                ood_by_condition=ood_by_condition,
            )
        )

    armijo_iid, armijo_iid_by_condition = evaluate_split(
        iid_test,
        lambda initial, task: rollout_armijo(initial, task, steps=args.steps),
    )
    armijo_ood, armijo_ood_by_condition = evaluate_split(
        ood_test,
        lambda initial, task: rollout_armijo(initial, task, steps=args.steps),
    )
    results.append(
        MethodResult(
            method="raw_lbfgs_armijo",
            validation_loss_ratio=None,
            tuning={"max_backtracks": 12.0},
            iid_loss_ratio=armijo_iid,
            ood_loss_ratio=armijo_ood,
            iid_by_condition=armijo_iid_by_condition,
            ood_by_condition=armijo_ood_by_condition,
        )
    )

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "train_conditions": TRAIN_CONDITIONS,
        "ood_conditions": OOD_CONDITIONS,
        "history_size": HISTORY_SIZE,
        "results": [asdict(result) for result in results],
        "iid_ranking": [
            result.method for result in sorted(results, key=lambda item: item.iid_loss_ratio)
        ],
        "ood_ranking": [
            result.method for result in sorted(results, key=lambda item: item.ood_loss_ratio)
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
