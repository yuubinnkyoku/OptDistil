from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import make_split

from optdistil.distill.secant_features import SecantFeatureState
from optdistil.students.tiny_mlp import StudentState

HISTORY_SIZES = (1, 2, 4)
ANALYTIC_SCALE_CANDIDATES = (
    0.0001,
    0.0003,
    0.0006,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.004,
    0.006,
    0.01,
    0.02,
    0.03,
    0.06,
    0.1,
)


@dataclass(frozen=True, slots=True)
class AnalyticResult:
    mode: str
    history_size: int
    additional_state_scalars_per_parameter: int
    total_state_scalars_per_parameter: int
    validation_scale: float | None
    validation_scale_at_boundary: bool
    validation_loss_ratio: float | None
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-free analytic L-BFGS controls using exactly the normalized secant "
            "direction and history budgets exposed to the 153-parameter Student. Compare "
            "an independently tuned global scale with a quadratic-only exact line-search oracle."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.validation_tasks = 2
    args.test_tasks = 4


def _secant_direction(secant_state, ema_state, parameter, grad, *, step, steps):
    momentum, second_moment = ema_state.observe(grad)
    features = secant_state.build(
        parameter,
        grad,
        momentum,
        second_moment,
        step=step,
        total_steps=steps,
    )
    return features[:, 5].reshape_as(parameter)


@torch.no_grad()
def rollout_analytic_secant(
    initial,
    task,
    *,
    steps: int,
    history_size: int,
    scale: float,
) -> float:
    parameter = initial.detach().clone()
    ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    secant_state = SecantFeatureState(history_size=history_size)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        direction = _secant_direction(
            secant_state,
            ema_state,
            parameter,
            grad,
            step=step,
            steps=steps,
        )
        parameter = parameter + scale * direction
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf

    return final_loss / max(abs(initial_loss), 1e-12)


@torch.no_grad()
def rollout_exact_line_search_secant(
    initial,
    task,
    *,
    steps: int,
    history_size: int,
) -> float:
    if not hasattr(task, "hessian_vector"):
        raise TypeError("exact line-search control requires a task with hessian_vector")

    parameter = initial.detach().clone()
    ema_state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    secant_state = SecantFeatureState(history_size=history_size)
    initial_loss = float(task.loss(parameter))
    final_loss = initial_loss

    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        direction = _secant_direction(
            secant_state,
            ema_state,
            parameter,
            grad,
            step=step,
            steps=steps,
        )
        h_direction = task.hessian_vector(direction)
        numerator = -torch.sum(grad * direction)
        denominator = torch.sum(direction * h_direction)
        if (
            not torch.isfinite(numerator)
            or not torch.isfinite(denominator)
            or float(numerator) <= 0.0
            or float(denominator) <= 0.0
        ):
            return math.inf
        alpha = numerator / denominator
        parameter = parameter + alpha * direction
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf

    return final_loss / max(abs(initial_loss), 1e-12)


def mean_global_ratio(cases, *, steps: int, history_size: int, scale: float) -> float:
    return statistics.fmean(
        rollout_analytic_secant(
            initial,
            task,
            steps=steps,
            history_size=history_size,
            scale=scale,
        )
        for initial, task in cases
    )


def mean_line_search_ratio(cases, *, steps: int, history_size: int) -> float:
    return statistics.fmean(
        rollout_exact_line_search_secant(
            initial,
            task,
            steps=steps,
            history_size=history_size,
        )
        for initial, task in cases
    )


def select_scale(cases, *, steps: int, history_size: int) -> tuple[float, float, bool]:
    scores = [
        (
            scale,
            mean_global_ratio(cases, steps=steps, history_size=history_size, scale=scale),
        )
        for scale in ANALYTIC_SCALE_CANDIDATES
    ]
    best_scale, best_score = min(scores, key=lambda item: item[1])
    at_boundary = best_scale in (ANALYTIC_SCALE_CANDIDATES[0], ANALYTIC_SCALE_CANDIDATES[-1])
    return best_scale, best_score, at_boundary


def evaluate_global_split(split, *, steps: int, history_size: int, scale: float):
    by_condition = {
        f"{condition:g}": mean_global_ratio(
            cases,
            steps=steps,
            history_size=history_size,
            scale=scale,
        )
        for condition, cases in split.items()
    }
    return statistics.fmean(by_condition.values()), by_condition


def evaluate_line_search_split(split, *, steps: int, history_size: int):
    by_condition = {
        f"{condition:g}": mean_line_search_ratio(
            cases,
            steps=steps,
            history_size=history_size,
        )
        for condition, cases in split.items()
    }
    return statistics.fmean(by_condition.values()), by_condition


def state_scalars(history_size: int) -> tuple[int, int]:
    additional = 2 + 2 * history_size
    return additional, 2 + additional


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(args.size, args.steps, args.validation_tasks, args.test_tasks) <= 0:
        raise ValueError("sizes, steps, and task counts must be positive")

    device = torch.device(args.device)
    validation_split = make_split(
        seed_base=151000,
        count=args.validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(
        seed_base=156000,
        count=args.test_tasks,
        size=args.size,
        device=device,
    )
    validation_cases = [case for cases in validation_split.values() for case in cases]

    results: list[AnalyticResult] = []
    for history_size in HISTORY_SIZES:
        additional, total = state_scalars(history_size)

        scale, validation_ratio, at_boundary = select_scale(
            validation_cases,
            steps=args.steps,
            history_size=history_size,
        )
        test_ratio, by_condition = evaluate_global_split(
            test_split,
            steps=args.steps,
            history_size=history_size,
            scale=scale,
        )
        results.append(
            AnalyticResult(
                mode="global_scale",
                history_size=history_size,
                additional_state_scalars_per_parameter=additional,
                total_state_scalars_per_parameter=total,
                validation_scale=scale,
                validation_scale_at_boundary=at_boundary,
                validation_loss_ratio=validation_ratio,
                test_loss_ratio=test_ratio,
                test_loss_ratio_by_condition=by_condition,
            )
        )

        oracle_test_ratio, oracle_by_condition = evaluate_line_search_split(
            test_split,
            steps=args.steps,
            history_size=history_size,
        )
        results.append(
            AnalyticResult(
                mode="exact_line_search_oracle",
                history_size=history_size,
                additional_state_scalars_per_parameter=additional,
                total_state_scalars_per_parameter=total,
                validation_scale=None,
                validation_scale_at_boundary=False,
                validation_loss_ratio=None,
                test_loss_ratio=oracle_test_ratio,
                test_loss_ratio_by_condition=oracle_by_condition,
            )
        )

    global_results = [result for result in results if result.mode == "global_scale"]
    oracle_results = [result for result in results if result.mode == "exact_line_search_oracle"]
    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_sizes": HISTORY_SIZES,
        "scale_candidates": ANALYTIC_SCALE_CANDIDATES,
        "results": [asdict(result) for result in results],
        "best_global_history_by_test": min(
            global_results, key=lambda result: result.test_loss_ratio
        ).history_size,
        "best_global_test_loss_ratio": min(result.test_loss_ratio for result in global_results),
        "best_oracle_history_by_test": min(
            oracle_results, key=lambda result: result.test_loss_ratio
        ).history_size,
        "best_oracle_test_loss_ratio": min(result.test_loss_ratio for result in oracle_results),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
