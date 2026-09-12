from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.stochastic import (
    DEFAULT_BATCH_SIZES,
    OOD_CONDITIONS,
    TRAIN_CONDITIONS,
    StochasticCase,
    batch_sequence,
    evaluate_secant_split,
    evaluate_teacher_split,
    flatten_split,
    make_split,
    tune_secant,
    tune_teacher_lr,
)


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
    validation_cases = flatten_split(validation)

    batch_sizes = sorted(
        {size for size in DEFAULT_BATCH_SIZES if size <= args.samples} | {args.samples}
    )
    sweeps = {}
    for batch_size in batch_sizes:
        results: list[MethodResult] = []
        for method in ("adamw", "muon", "norm_gradient"):
            lr, validation_score = tune_teacher_lr(
                method,  # type: ignore[arg-type]
                validation_cases,
                batch_size=batch_size,
                steps=args.steps,
            )
            iid = evaluate_teacher_split(
                method,  # type: ignore[arg-type]
                lr,
                iid_test,
                batch_size=batch_size,
                steps=args.steps,
            )
            ood = evaluate_teacher_split(
                method,  # type: ignore[arg-type]
                lr,
                ood_test,
                batch_size=batch_size,
                steps=args.steps,
            )
            results.append(
                MethodResult(
                    method=method,
                    validation_loss_ratio=validation_score,
                    tuning={"lr": lr},
                    iid_loss_ratio=iid["loss_ratio"]["mean"],
                    ood_loss_ratio=ood["loss_ratio"]["mean"],
                    iid_by_condition=iid["by_condition"],
                    ood_by_condition=ood["by_condition"],
                )
            )

        tuning, validation_score = tune_secant(
            validation_cases,
            batch_size=batch_size,
            steps=args.steps,
        )
        iid = evaluate_secant_split(
            iid_test,
            batch_size=batch_size,
            steps=args.steps,
            secant_scale=tuning["secant_scale"],
            bootstrap_scale=tuning["bootstrap_scale"],
        )
        ood = evaluate_secant_split(
            ood_test,
            batch_size=batch_size,
            steps=args.steps,
            secant_scale=tuning["secant_scale"],
            bootstrap_scale=tuning["bootstrap_scale"],
        )
        results.append(
            MethodResult(
                method="raw_lbfgs_two_scale",
                validation_loss_ratio=validation_score,
                tuning=tuning,
                iid_loss_ratio=iid["loss_ratio"]["mean"],
                ood_loss_ratio=ood["loss_ratio"]["mean"],
                iid_by_condition=iid["by_condition"],
                ood_by_condition=ood["by_condition"],
            )
        )

        sweeps[str(batch_size)] = {
            "gradient_noise_ratio": gradient_noise_ratio(
                validation_cases, batch_size=batch_size
            ),
            "results": [asdict(result) for result in results],
            "iid_ranking": [
                result.method
                for result in sorted(results, key=lambda item: item.iid_loss_ratio)
            ],
            "ood_ranking": [
                result.method
                for result in sorted(results, key=lambda item: item.ood_loss_ratio)
            ],
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "train_conditions": list(TRAIN_CONDITIONS),
        "ood_conditions": list(OOD_CONDITIONS),
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
