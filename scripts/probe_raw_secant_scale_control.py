from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import make_case

from optdistil.distill.secant_features import SecantFeatureState
from optdistil.students.tiny_mlp import StudentState

TRAIN_CONDITIONS = (30.0, 300.0)
OOD_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
SCALE_CANDIDATES = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BOOTSTRAP_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
HISTORY_SIZE = 4


@dataclass(frozen=True, slots=True)
class PolicyResult:
    policy: str
    validation_loss_ratio: float
    bootstrap_scale: float | None
    secant_scale: float | None
    iid_loss_ratio: float
    ood_loss_ratio: float
    iid_by_condition: dict[str, float]
    ood_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--validation-tasks", type=int, default=8)
    parser.add_argument("--iid-test-tasks", type=int, default=12)
    parser.add_argument("--ood-test-tasks", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.validation_tasks = 3
    args.iid_test_tasks = 4
    args.ood_test_tasks = 3


def make_split(conditions, *, seed_base: int, count: int, size: int, device: torch.device):
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


def flatten(split):
    return [case for cases in split.values() for case in cases]


@torch.no_grad()
def rollout_fixed_policy(
    initial,
    task,
    *,
    steps: int,
    normalize_direction: bool,
    secant_scale: float,
    bootstrap_scale: float | None = None,
) -> float:
    parameter = initial.detach().clone()
    ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    secant_state = SecantFeatureState(
        history_size=HISTORY_SIZE,
        normalize_direction=normalize_direction,
    )
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss

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
        if step == 1 and bootstrap_scale is not None:
            grad_rms = grad.square().mean().add(1e-8).sqrt()
            update = -bootstrap_scale * grad / grad_rms
        else:
            update = secant_scale * features[:, 5].reshape_as(parameter)
        parameter = parameter + update
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


@torch.no_grad()
def rollout_exact_line_search(initial, task, *, steps: int) -> float:
    parameter = initial.detach().clone()
    ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    secant_state = SecantFeatureState(
        history_size=HISTORY_SIZE,
        normalize_direction=False,
    )
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss

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
        direction = features[:, 5].reshape_as(parameter)
        h_direction = task.hessian_vector(direction)
        denominator = torch.sum(direction * h_direction)
        numerator = -torch.sum(grad * direction)
        if float(denominator) <= 1e-12 or not torch.isfinite(denominator):
            return math.inf
        alpha = numerator / denominator
        parameter = parameter + alpha * direction
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf
    return final_loss / max(abs(initial_loss), 1e-12)


def mean_policy(cases, **kwargs) -> float:
    return statistics.fmean(rollout_fixed_policy(initial, task, **kwargs) for initial, task in cases)


def select_one_scale(cases, *, steps: int, normalize_direction: bool) -> tuple[float, float]:
    scored = [
        (
            scale,
            mean_policy(
                cases,
                steps=steps,
                normalize_direction=normalize_direction,
                secant_scale=scale,
            ),
        )
        for scale in SCALE_CANDIDATES
    ]
    return min(scored, key=lambda item: item[1])


def select_two_scales(cases, *, steps: int) -> tuple[float, float, float]:
    scored = []
    for bootstrap_scale in BOOTSTRAP_CANDIDATES:
        for secant_scale in SCALE_CANDIDATES:
            score = mean_policy(
                cases,
                steps=steps,
                normalize_direction=False,
                secant_scale=secant_scale,
                bootstrap_scale=bootstrap_scale,
            )
            scored.append((score, bootstrap_scale, secant_scale))
    return min(scored, key=lambda item: item[0])


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
        args.steps,
        args.validation_tasks,
        args.iid_test_tasks,
        args.ood_test_tasks,
    ) <= 0:
        raise ValueError("sizes, steps, and task counts must be positive")

    device = torch.device(args.device)
    validation = make_split(
        TRAIN_CONDITIONS,
        seed_base=261000,
        count=args.validation_tasks,
        size=args.size,
        device=device,
    )
    iid_test = make_split(
        TRAIN_CONDITIONS,
        seed_base=271000,
        count=args.iid_test_tasks,
        size=args.size,
        device=device,
    )
    ood_test = make_split(
        OOD_CONDITIONS,
        seed_base=281000,
        count=args.ood_test_tasks,
        size=args.size,
        device=device,
    )
    validation_cases = flatten(validation)

    normalized_scale, normalized_validation = select_one_scale(
        validation_cases,
        steps=args.steps,
        normalize_direction=True,
    )
    raw_scale, raw_validation = select_one_scale(
        validation_cases,
        steps=args.steps,
        normalize_direction=False,
    )
    two_validation, bootstrap_scale, two_secant_scale = select_two_scales(
        validation_cases,
        steps=args.steps,
    )

    policies = []
    for name, validation_score, bootstrap, secant, normalize in (
        ("normalized_global", normalized_validation, None, normalized_scale, True),
        ("raw_global", raw_validation, None, raw_scale, False),
        ("raw_two_scale", two_validation, bootstrap_scale, two_secant_scale, False),
    ):
        def rollout(initial, task, *, normalize=normalize, bootstrap=bootstrap, secant=secant):
            return rollout_fixed_policy(
                initial,
                task,
                steps=args.steps,
                normalize_direction=normalize,
                secant_scale=secant,
                bootstrap_scale=bootstrap,
            )

        iid_ratio, iid_by_condition = evaluate_split(iid_test, rollout)
        ood_ratio, ood_by_condition = evaluate_split(ood_test, rollout)
        policies.append(
            PolicyResult(
                policy=name,
                validation_loss_ratio=validation_score,
                bootstrap_scale=bootstrap,
                secant_scale=secant,
                iid_loss_ratio=iid_ratio,
                ood_loss_ratio=ood_ratio,
                iid_by_condition=iid_by_condition,
                ood_by_condition=ood_by_condition,
            )
        )

    oracle_iid, oracle_iid_by_condition = evaluate_split(
        iid_test,
        lambda initial, task: rollout_exact_line_search(initial, task, steps=args.steps),
    )
    oracle_ood, oracle_ood_by_condition = evaluate_split(
        ood_test,
        lambda initial, task: rollout_exact_line_search(initial, task, steps=args.steps),
    )

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "train_conditions": TRAIN_CONDITIONS,
        "ood_conditions": OOD_CONDITIONS,
        "history_size": HISTORY_SIZE,
        "policies": [asdict(result) for result in policies],
        "exact_line_search_oracle": {
            "iid_loss_ratio": oracle_iid,
            "ood_loss_ratio": oracle_ood,
            "iid_by_condition": oracle_iid_by_condition,
            "ood_by_condition": oracle_ood_by_condition,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
