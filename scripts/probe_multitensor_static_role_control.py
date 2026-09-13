from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from optdistil.multitensor.mechanisms import (
    apply_direction_scales,
    coordinate_descent_role_scales,
    scales_for_names,
)
from optdistil.multitensor.stochastic import (
    OOD_CONDITIONS,
    TRAIN_CONDITIONS,
    MultiTensorCase,
    artifact_metadata,
    batch_sequence,
    flatten_split,
    git_commit_sha,
    make_split,
    summarize_ratios,
    tune_teacher_lr,
)

ARCHITECTURES = ("two_layer", "residual")
TEACHER_NAME = "norm_grad_local"
MULTIPLIERS = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


@dataclass(frozen=True, slots=True)
class RolloutResult:
    loss_ratio: float
    aulc: float
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune static per-role NormGrad scales on validation data as a strong analytic "
            "control for the multi-tensor Student allocation mechanism."
        )
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--role-validation-tasks", type=int, default=8)
    parser.add_argument("--test-tasks", type=int, default=6)
    parser.add_argument("--ood-test-tasks", type=int, default=4)
    parser.add_argument("--coordinate-passes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 12
    args.lr_validation_tasks = 2
    args.role_validation_tasks = 2
    args.test_tasks = 3
    args.ood_test_tasks = 2
    args.coordinate_passes = 2


def make_cases_by_architecture(
    *,
    seed_base: int,
    count: int,
    width: int,
    samples: int,
    conditions: tuple[float, ...],
    device: torch.device,
) -> dict[str, list[MultiTensorCase]]:
    result: dict[str, list[MultiTensorCase]] = {}
    for architecture in ARCHITECTURES:
        offset = 0 if architecture == "two_layer" else 500_000
        result[architecture] = flatten_split(
            make_split(
                architecture,
                conditions,
                seed_base=seed_base + offset,
                count=count,
                width=width,
                samples=samples,
                device=device,
            )
        )
    return result


def flatten_architectures(cases: dict[str, list[MultiTensorCase]]) -> list[MultiTensorCase]:
    return [case for architecture in ARCHITECTURES for case in cases[architecture]]


def _aulc(losses: list[float]) -> float:
    if len(losses) <= 1:
        return 1.0
    initial = max(abs(losses[0]), 1e-12)
    area = 0.0
    for left, right in pairwise(losses):
        if not (math.isfinite(left) and math.isfinite(right)):
            return math.inf
        area += 0.5 * (left + right)
    return area / ((len(losses) - 1) * initial)


@torch.no_grad()
def rollout_static_normgrad(
    case: MultiTensorCase,
    role_scales: dict[str, float],
    *,
    batch_size: int,
    steps: int,
) -> RolloutResult:
    params = case.initial.clone()
    losses = [float(case.task.loss(params))]
    scales = scales_for_names(case.task.parameter_names, role_scales)
    for indices in batch_sequence(case, batch_size=batch_size, steps=steps):
        grads = case.task.grad_on_samples(params, indices)
        params = params.add(apply_direction_scales(grads, scales))
        if not params.is_finite():
            return RolloutResult(math.inf, math.inf, False)
        loss = float(case.task.loss(params))
        losses.append(loss)
        if not math.isfinite(loss):
            return RolloutResult(math.inf, math.inf, False)
    return RolloutResult(
        loss_ratio=losses[-1] / max(abs(losses[0]), 1e-12),
        aulc=_aulc(losses),
        finite=True,
    )


def validation_score(
    cases: list[MultiTensorCase],
    role_scales: dict[str, float],
    *,
    batch_size: int,
    steps: int,
) -> float:
    return statistics.fmean(
        rollout_static_normgrad(
            case,
            role_scales,
            batch_size=batch_size,
            steps=steps,
        ).loss_ratio
        for case in cases
    )


def evaluate_cases(
    cases_by_architecture: dict[str, list[MultiTensorCase]],
    scales_for_architecture: dict[str, dict[str, float]],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    finite: list[float] = []
    by_architecture: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        architecture_ratios: list[float] = []
        architecture_aulcs: list[float] = []
        for case in cases_by_architecture[architecture]:
            rollout = rollout_static_normgrad(
                case,
                scales_for_architecture[architecture],
                batch_size=batch_size,
                steps=steps,
            )
            architecture_ratios.append(rollout.loss_ratio)
            architecture_aulcs.append(rollout.aulc)
            ratios.append(rollout.loss_ratio)
            aulcs.append(rollout.aulc)
            finite.append(float(rollout.finite))
        by_architecture[architecture] = {
            "loss_ratio": summarize_ratios(architecture_ratios).to_dict(),
            "aulc": summarize_ratios(architecture_aulcs).to_dict(),
        }
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "finite_fraction": statistics.fmean(finite),
        "by_architecture": by_architecture,
    }


def tune_shared_scales(
    validation_cases: list[MultiTensorCase],
    *,
    teacher_lr: float,
    candidates: tuple[float, ...],
    batch_size: int,
    steps: int,
    passes: int,
) -> tuple[dict[str, float], float, list[dict[str, object]]]:
    roles = tuple(
        sorted(
            {
                str(name)
                for case in validation_cases
                for name in case.task.parameter_names
            }
        )
    )

    def evaluate(scales) -> float:
        return validation_score(
            validation_cases,
            dict(scales),
            batch_size=batch_size,
            steps=steps,
        )

    return coordinate_descent_role_scales(
        roles,
        candidates,
        evaluate,
        initial_scale=teacher_lr,
        passes=passes,
    )


def tune_architecture_scales(
    validation_by_architecture: dict[str, list[MultiTensorCase]],
    *,
    teacher_lr: float,
    candidates: tuple[float, ...],
    batch_size: int,
    steps: int,
    passes: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, list[dict[str, object]]]]:
    scales: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    histories: dict[str, list[dict[str, object]]] = {}
    for architecture in ARCHITECTURES:
        cases = validation_by_architecture[architecture]
        role_names = tuple(str(name) for name in cases[0].task.parameter_names)
        evaluate = partial(
            validation_score,
            cases,
            batch_size=batch_size,
            steps=steps,
        )

        selected, score, history = coordinate_descent_role_scales(
            role_names,
            candidates,
            evaluate,
            initial_scale=teacher_lr,
            passes=passes,
        )
        scales[architecture] = selected
        scores[architecture] = score
        histories[architecture] = history
    return scales, scores, histories


def main() -> None:
    args = parse_args()
    apply_quick(args)
    device = torch.device(args.device)

    lr_by_arch = make_cases_by_architecture(
        seed_base=411000,
        count=args.lr_validation_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    role_validation = make_cases_by_architecture(
        seed_base=491000,
        count=args.role_validation_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    test = make_cases_by_architecture(
        seed_base=461000,
        count=args.test_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    ood = make_cases_by_architecture(
        seed_base=471000,
        count=args.ood_test_tasks,
        width=args.width,
        samples=args.samples,
        conditions=OOD_CONDITIONS,
        device=device,
    )

    teacher_lr, teacher_validation = tune_teacher_lr(
        TEACHER_NAME,
        flatten_architectures(lr_by_arch),
        batch_size=args.batch_size,
        steps=args.steps,
    )
    candidates = tuple(teacher_lr * multiplier for multiplier in MULTIPLIERS)

    shared_scales, shared_validation, shared_history = tune_shared_scales(
        flatten_architectures(role_validation),
        teacher_lr=teacher_lr,
        candidates=candidates,
        batch_size=args.batch_size,
        steps=args.steps,
        passes=args.coordinate_passes,
    )
    architecture_scales, architecture_validation, architecture_histories = (
        tune_architecture_scales(
            role_validation,
            teacher_lr=teacher_lr,
            candidates=candidates,
            batch_size=args.batch_size,
            steps=args.steps,
            passes=args.coordinate_passes,
        )
    )

    teacher_scales = {
        architecture: {
            str(name): teacher_lr
            for name in role_validation[architecture][0].task.parameter_names
        }
        for architecture in ARCHITECTURES
    }
    shared_by_architecture = {
        architecture: dict(shared_scales) for architecture in ARCHITECTURES
    }

    config = vars(args) | {"output": str(args.output)}
    commit_sha = git_commit_sha()
    split_specs = {
        "architectures": list(ARCHITECTURES),
        "train_conditions": list(TRAIN_CONDITIONS),
        "ood_conditions": list(OOD_CONDITIONS),
        "seed_bases": {
            "teacher_lr_validation": 411000,
            "role_validation": 491000,
            "test": 461000,
            "ood": 471000,
            "residual_offset": 500000,
        },
    }
    payload = {
        "experiment": "multitensor_static_role_normgrad_control",
        "git_sha": commit_sha,
        "metadata": artifact_metadata(
            run_id=os.environ.get("GITHUB_RUN_ID", "local"),
            commit_sha=commit_sha,
            config=config,
            split_specs=split_specs,
        ),
        "config": config,
        "teacher": {
            "lr": teacher_lr,
            "lr_validation_loss_ratio": teacher_validation,
        },
        "candidate_multipliers": list(MULTIPLIERS),
        "candidate_scales": list(candidates),
        "selected": {
            "shared_role_scales": shared_scales,
            "shared_validation_loss_ratio": shared_validation,
            "shared_history": shared_history,
            "architecture_role_scales": architecture_scales,
            "architecture_validation_loss_ratio": architecture_validation,
            "architecture_history": architecture_histories,
        },
        "test": {
            "teacher_norm": evaluate_cases(
                test,
                teacher_scales,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "tuned_static_shared": evaluate_cases(
                test,
                shared_by_architecture,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "tuned_static_architecture": evaluate_cases(
                test,
                architecture_scales,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
        },
        "ood": {
            "teacher_norm": evaluate_cases(
                ood,
                teacher_scales,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "tuned_static_shared": evaluate_cases(
                ood,
                shared_by_architecture,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "tuned_static_architecture": evaluate_cases(
                ood,
                architecture_scales,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
        },
        "mechanism_reference": {
            "run_id": 34750297519,
            "artifact_commit": "8a07650bd95254f12cf541a966d15fe5f2f39054",
            "full_student_iid": 0.06456449826272108,
            "tensor_projection_iid": 0.06388667714911869,
            "full_student_ood": 0.06362718100092321,
            "tensor_projection_ood": 0.06377441499127312,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
