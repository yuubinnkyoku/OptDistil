from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
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
DEFAULT_BATCH_SIZES = (4, 8, 16, 32, 64)
TEACHER_LRS = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)
SECANT_SCALES = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
BOOTSTRAP_SCALES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True, slots=True)
class StochasticCase:
    initial: torch.Tensor
    task: object
    batch_seed: int


@dataclass(frozen=True, slots=True)
class MethodResult:
    method: str
    validation_loss_ratio: float
    tuning: dict[str, float]
    iid_loss_ratio: float
    ood_loss_ratio: float
    iid_by_condition: dict[str, float]
    ood_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep minibatch noise on the nonlinear matrix task using identical batch "
            "sequences for AdamW, Muon, normalized gradient, and raw L-BFGS controls."
        )
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
) -> dict[float, list[StochasticCase]]:
    split = {}
    for condition_index, condition in enumerate(conditions):
        cases = []
        for index in range(count):
            seed = seed_base + 10000 * condition_index + index
            initial, task = make_frozen_readout_mlp(
                seed,
                hidden_dim=size,
                input_dim=size,
                output_dim=max(2, size // 2),
                samples=samples,
                input_condition=condition,
                device=device,
            )
            cases.append(StochasticCase(initial, task, 1_000_000 + seed))
        split[condition] = cases
    return split


def flatten(split) -> list[StochasticCase]:
    return [case for cases in split.values() for case in cases]


def batch_sequence(case: StochasticCase, *, batch_size: int, steps: int) -> list[torch.Tensor]:
    sample_count = case.task.sample_count
    if batch_size <= 0 or batch_size > sample_count:
        raise ValueError("batch_size must lie in [1, sample_count]")
    if batch_size == sample_count:
        full = torch.arange(sample_count, dtype=torch.long)
        return [full for _ in range(steps)]

    generator = torch.Generator(device="cpu").manual_seed(case.batch_seed)
    return [
        torch.randperm(sample_count, generator=generator)[:batch_size]
        for _ in range(steps)
    ]


def make_teacher(method: str, lr: float):
    if method == "adamw":
        return AdamWTeacher(lr=lr)
    if method == "muon":
        return MuonTeacher(lr=lr)
    if method == "norm_gradient":
        return GradientDirectionTeacher(lr=lr)
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def rollout_teacher(
    method: str,
    lr: float,
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
) -> float:
    parameter = case.initial.detach().clone()
    teacher = make_teacher(method, lr)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(parameter))
    for indices in batches:
        grad = case.task.grad_on_samples(parameter, indices)
        parameter = parameter + teacher.step(parameter, grad)
        if not torch.isfinite(parameter).all():
            return math.inf
    final_loss = float(case.task.loss(parameter))
    if not math.isfinite(final_loss):
        return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def tune_teacher(
    method: str,
    validation_cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
) -> tuple[float, float]:
    scored = []
    for lr in TEACHER_LRS:
        score = statistics.fmean(
            rollout_teacher(method, lr, case, batch_size=batch_size, steps=steps)
            for case in validation_cases
        )
        scored.append((score, lr))
    score, lr = min(scored, key=lambda item: item[0])
    return lr, score


@torch.no_grad()
def rollout_secant(
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
    secant_scale: float,
    bootstrap_scale: float,
) -> float:
    parameter = case.initial.detach().clone()
    ema = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    state = SecantFeatureState(history_size=HISTORY_SIZE, normalize_direction=False)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(parameter))

    for step, indices in enumerate(batches, start=1):
        grad = case.task.grad_on_samples(parameter, indices)
        momentum, second_moment = ema.observe(grad)
        features = state.build(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        if step == 1:
            grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
            update = -bootstrap_scale * grad / grad_rms
        else:
            update = secant_scale * features[:, 5].reshape_as(parameter)
        parameter = parameter + update
        if not torch.isfinite(parameter).all():
            return math.inf

    final_loss = float(case.task.loss(parameter))
    if not math.isfinite(final_loss):
        return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def tune_secant(
    validation_cases: list[StochasticCase],
    *,
    batch_size: int,
    steps: int,
) -> tuple[dict[str, float], float]:
    scored = []
    for bootstrap_scale in BOOTSTRAP_SCALES:
        for secant_scale in SECANT_SCALES:
            score = statistics.fmean(
                rollout_secant(
                    case,
                    batch_size=batch_size,
                    steps=steps,
                    secant_scale=secant_scale,
                    bootstrap_scale=bootstrap_scale,
                )
                for case in validation_cases
            )
            scored.append((score, bootstrap_scale, secant_scale))
    score, bootstrap_scale, secant_scale = min(scored, key=lambda item: item[0])
    return {
        "bootstrap_scale": bootstrap_scale,
        "secant_scale": secant_scale,
    }, score


def evaluate_split(split, rollout) -> tuple[float, dict[str, float]]:
    by_condition = {}
    ratios = []
    for condition, cases in split.items():
        condition_ratios = [rollout(case) for case in cases]
        by_condition[str(condition)] = statistics.fmean(condition_ratios)
        ratios.extend(condition_ratios)
    return statistics.fmean(ratios), by_condition


@torch.no_grad()
def gradient_noise_ratio(cases: list[StochasticCase], *, batch_size: int) -> float:
    values = []
    for case in cases:
        full_grad = case.task.grad(case.initial)
        indices = batch_sequence(case, batch_size=batch_size, steps=1)[0]
        batch_grad = case.task.grad_on_samples(case.initial, indices)
        denominator = float(full_grad.norm().clamp_min(1e-12))
        values.append(float((batch_grad - full_grad).norm()) / denominator)
    return statistics.fmean(values)


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
        seed_base=341000,
        count=args.validation_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )
    iid_test = make_split(
        TRAIN_CONDITIONS,
        seed_base=351000,
        count=args.iid_test_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )
    ood_test = make_split(
        OOD_CONDITIONS,
        seed_base=361000,
        count=args.ood_test_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )
    validation_cases = flatten(validation)

    batch_sizes = sorted({size for size in DEFAULT_BATCH_SIZES if size <= args.samples} | {args.samples})
    sweeps = {}
    for batch_size in batch_sizes:
        results: list[MethodResult] = []
        for method in ("adamw", "muon", "norm_gradient"):
            lr, validation_score = tune_teacher(
                method,
                validation_cases,
                batch_size=batch_size,
                steps=args.steps,
            )
            iid, iid_by_condition = evaluate_split(
                iid_test,
                lambda case, method=method, lr=lr, batch_size=batch_size: rollout_teacher(
                    method,
                    lr,
                    case,
                    batch_size=batch_size,
                    steps=args.steps,
                ),
            )
            ood, ood_by_condition = evaluate_split(
                ood_test,
                lambda case, method=method, lr=lr, batch_size=batch_size: rollout_teacher(
                    method,
                    lr,
                    case,
                    batch_size=batch_size,
                    steps=args.steps,
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

        tuning, validation_score = tune_secant(
            validation_cases,
            batch_size=batch_size,
            steps=args.steps,
        )
        iid, iid_by_condition = evaluate_split(
            iid_test,
            lambda case, tuning=tuning, batch_size=batch_size: rollout_secant(
                case,
                batch_size=batch_size,
                steps=args.steps,
                secant_scale=tuning["secant_scale"],
                bootstrap_scale=tuning["bootstrap_scale"],
            ),
        )
        ood, ood_by_condition = evaluate_split(
            ood_test,
            lambda case, tuning=tuning, batch_size=batch_size: rollout_secant(
                case,
                batch_size=batch_size,
                steps=args.steps,
                secant_scale=tuning["secant_scale"],
                bootstrap_scale=tuning["bootstrap_scale"],
            ),
        )
        results.append(
            MethodResult(
                method="raw_lbfgs_two_scale",
                validation_loss_ratio=validation_score,
                tuning=tuning,
                iid_loss_ratio=iid,
                ood_loss_ratio=ood,
                iid_by_condition=iid_by_condition,
                ood_by_condition=ood_by_condition,
            )
        )

        sweeps[str(batch_size)] = {
            "gradient_noise_ratio": gradient_noise_ratio(validation_cases, batch_size=batch_size),
            "results": [asdict(result) for result in results],
            "iid_ranking": [
                result.method for result in sorted(results, key=lambda item: item.iid_loss_ratio)
            ],
            "ood_ranking": [
                result.method for result in sorted(results, key=lambda item: item.ood_loss_ratio)
            ],
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "train_conditions": TRAIN_CONDITIONS,
        "ood_conditions": OOD_CONDITIONS,
        "history_size": HISTORY_SIZE,
        "batch_sizes": batch_sizes,
        "sweeps": sweeps,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
