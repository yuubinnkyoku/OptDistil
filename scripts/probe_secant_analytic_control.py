from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import STUDENT_SCALE_CANDIDATES, make_split

from optdistil.distill.secant_features import SecantFeatureState
from optdistil.students.tiny_mlp import StudentState

HISTORY_SIZES = (1, 2, 4)


@dataclass(frozen=True, slots=True)
class AnalyticResult:
    history_size: int
    additional_state_scalars_per_parameter: int
    total_state_scalars_per_parameter: int
    validation_scale: float
    validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-free analytic L-BFGS control using exactly the same normalized secant "
            "direction, history budgets, scale grid, and validation/test task splits as the "
            "153-parameter distilled-student secant probes."
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
        momentum, second_moment = ema_state.observe(grad)
        features = secant_state.build(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        # Feature column 5 is exactly the normalized L-BFGS direction used by the Student.
        update = scale * features[:, 5].reshape_as(parameter)
        parameter = parameter + update
        final_loss = float(task.loss(parameter))
        if not torch.isfinite(parameter).all() or not math.isfinite(final_loss):
            return math.inf

    return final_loss / max(abs(initial_loss), 1e-12)


def mean_ratio(cases, *, steps: int, history_size: int, scale: float) -> float:
    ratios = [
        rollout_analytic_secant(
            initial,
            task,
            steps=steps,
            history_size=history_size,
            scale=scale,
        )
        for initial, task in cases
    ]
    return statistics.fmean(ratios)


def select_scale(cases, *, steps: int, history_size: int) -> tuple[float, float]:
    scores = [
        (
            scale,
            mean_ratio(cases, steps=steps, history_size=history_size, scale=scale),
        )
        for scale in STUDENT_SCALE_CANDIDATES
    ]
    best_scale, best_score = min(scores, key=lambda item: item[1])
    return best_scale, best_score


def evaluate_split(split, *, steps: int, history_size: int, scale: float):
    by_condition = {
        f"{condition:g}": mean_ratio(
            cases,
            steps=steps,
            history_size=history_size,
            scale=scale,
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
        scale, validation_ratio = select_scale(
            validation_cases,
            steps=args.steps,
            history_size=history_size,
        )
        test_ratio, by_condition = evaluate_split(
            test_split,
            steps=args.steps,
            history_size=history_size,
            scale=scale,
        )
        additional, total = state_scalars(history_size)
        results.append(
            AnalyticResult(
                history_size=history_size,
                additional_state_scalars_per_parameter=additional,
                total_state_scalars_per_parameter=total,
                validation_scale=scale,
                validation_loss_ratio=validation_ratio,
                test_loss_ratio=test_ratio,
                test_loss_ratio_by_condition=by_condition,
            )
        )

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_sizes": HISTORY_SIZES,
        "results": [asdict(result) for result in results],
        "best_history_by_test": min(results, key=lambda result: result.test_loss_ratio).history_size,
        "best_test_loss_ratio": min(result.test_loss_ratio for result in results),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
