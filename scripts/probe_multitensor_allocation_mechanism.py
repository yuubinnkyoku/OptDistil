from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from optdistil.distill.losses import DistillationLossWeights
from optdistil.multitensor.features import (
    build_multitensor_features,
    concatenate_features,
    split_update,
)
from optdistil.multitensor.mechanisms import (
    aggregate_role_scales,
    apply_direction_scales,
    equalized_role_scales,
    global_projection,
    scales_for_names,
    swapped_role_scales,
    tensor_projection,
)
from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.stochastic import (
    OOD_CONDITIONS,
    TRAIN_CONDITIONS,
    MultiTensorCase,
    artifact_metadata,
    batch_sequence,
    collect_records,
    flatten_split,
    git_commit_sha,
    make_split,
    select_student_scale,
    summarize_ratios,
    train_supervised_student,
    tune_teacher_lr,
)
from optdistil.multitensor.teachers import make_teacher
from optdistil.students.tiny_mlp import TinyMLPOptimizer

ARCHITECTURES = ("two_layer", "residual")
TEACHER_NAME = "norm_grad_local"
DIRECTION_ONLY = DistillationLossWeights(direction=1.0, magnitude=0.0)
VARIANTS = (
    "full_student",
    "tensor_projection",
    "global_projection",
    "frozen_role",
    "equalized_frozen",
    "swapped_frozen",
    "teacher_norm",
)


@dataclass(frozen=True, slots=True)
class RolloutResult:
    loss_ratio: float
    aulc: float
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose why a direction-only 153-param multi-tensor Student can beat "
            "its tensor-wise NormGrad teacher."
        )
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=6)
    parser.add_argument("--scale-validation-tasks", type=int, default=4)
    parser.add_argument("--allocation-calibration-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=6)
    parser.add_argument("--ood-test-tasks", type=int, default=4)
    parser.add_argument("--distill-epochs", type=int, default=15)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=401000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 12
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.scale_validation_tasks = 2
    args.allocation_calibration_tasks = 2
    args.test_tasks = 3
    args.ood_test_tasks = 2
    args.distill_epochs = 4
    args.student_seeds = 2


def _mixed_cases(
    *,
    seed_base: int,
    count: int,
    width: int,
    samples: int,
    conditions: tuple[float, ...],
    device: torch.device,
) -> list[MultiTensorCase]:
    cases: list[MultiTensorCase] = []
    for architecture in ARCHITECTURES:
        offset = 0 if architecture == "two_layer" else 500_000
        cases.extend(
            flatten_split(
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
        )
    return cases


def _observe_ema(
    momentums: list[torch.Tensor],
    second_moments: list[torch.Tensor],
    grads: list[torch.Tensor],
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> None:
    for momentum, second_moment, grad in zip(
        momentums, second_moments, grads, strict=True
    ):
        momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)


def _student_update(
    student: TinyMLPOptimizer,
    params: ParamCollection,
    grads: list[torch.Tensor],
    momentums: list[torch.Tensor],
    second_moments: list[torch.Tensor],
    *,
    step: int,
    total_steps: int,
) -> list[torch.Tensor]:
    features = concatenate_features(
        build_multitensor_features(
            params.tensors,
            grads,
            momentums,
            second_moments,
            step=step,
            total_steps=total_steps,
            include_global=True,
        )
    )
    flat_update = student(features)
    return split_update(flat_update, params.shapes())


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
def rollout_variant(
    student: TinyMLPOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
    variant: str,
    teacher_lr: float,
    frozen_scales: dict[str, float] | None = None,
) -> RolloutResult:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    params = case.initial.clone()
    momentums = [torch.zeros_like(tensor) for tensor in params]
    second_moments = [torch.zeros_like(tensor) for tensor in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    losses = [float(case.task.loss(params))]
    names = tuple(case.task.parameter_names)
    student.eval()

    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)

        if variant in {"full_student", "tensor_projection", "global_projection"}:
            raw_updates = _student_update(
                student,
                params,
                grads,
                momentums,
                second_moments,
                step=step,
                total_steps=steps,
            )
            if variant == "full_student":
                updates = raw_updates
            elif variant == "tensor_projection":
                updates, _ = tensor_projection(raw_updates, grads)
            else:
                updates, _ = global_projection(raw_updates, grads)
        elif variant == "teacher_norm":
            updates = apply_direction_scales(grads, [teacher_lr] * len(grads))
        else:
            if frozen_scales is None:
                raise ValueError(f"{variant} requires frozen_scales")
            if variant == "equalized_frozen":
                role_scales = equalized_role_scales(frozen_scales)
            elif variant == "swapped_frozen":
                role_scales = swapped_role_scales(frozen_scales)
            else:
                role_scales = frozen_scales
            updates = apply_direction_scales(grads, scales_for_names(names, role_scales))

        params = params.add(updates)
        if not params.is_finite():
            return RolloutResult(loss_ratio=math.inf, aulc=math.inf, finite=False)
        loss = float(case.task.loss(params))
        losses.append(loss)
        if not math.isfinite(loss):
            return RolloutResult(loss_ratio=math.inf, aulc=math.inf, finite=False)

    return RolloutResult(
        loss_ratio=losses[-1] / max(abs(losses[0]), 1e-12),
        aulc=_aulc(losses),
        finite=True,
    )


@torch.no_grad()
def estimate_role_scales(
    student: TinyMLPOptimizer,
    cases: list[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    named_coefficients: list[tuple[tuple[str, ...], list[float]]] = []
    per_role_values: dict[str, list[float]] = {}
    positive = 0
    total = 0

    for case in cases:
        params = case.initial.clone()
        momentums = [torch.zeros_like(tensor) for tensor in params]
        second_moments = [torch.zeros_like(tensor) for tensor in params]
        names = tuple(case.task.parameter_names)
        for step, indices in enumerate(
            batch_sequence(case, batch_size=batch_size, steps=steps), start=1
        ):
            grads = case.task.grad_on_samples(params, indices)
            _observe_ema(momentums, second_moments, grads)
            raw_updates = _student_update(
                student,
                params,
                grads,
                momentums,
                second_moments,
                step=step,
                total_steps=steps,
            )
            _, coefficients = tensor_projection(raw_updates, grads)
            named_coefficients.append((names, coefficients))
            for name, coefficient in zip(names, coefficients, strict=True):
                per_role_values.setdefault(name, []).append(coefficient)
                positive += int(coefficient > 0.0)
                total += 1
            params = params.add(raw_updates)
            if not params.is_finite():
                break

    role_scales = aggregate_role_scales(named_coefficients)
    diagnostics = {
        "role_scales": role_scales,
        "positive_fraction": positive / max(total, 1),
        "per_role": {
            name: {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }
            for name, values in per_role_values.items()
        },
    }
    return role_scales, diagnostics


def evaluate_variants(
    student: TinyMLPOptimizer,
    cases: list[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    teacher_lr: float,
    frozen_scales: dict[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        ratios: list[float] = []
        aulcs: list[float] = []
        by_arch: dict[str, list[float]] = {}
        for case in cases:
            rollout = rollout_variant(
                student,
                case,
                batch_size=batch_size,
                steps=steps,
                variant=variant,
                teacher_lr=teacher_lr,
                frozen_scales=frozen_scales,
            )
            ratios.append(rollout.loss_ratio)
            aulcs.append(rollout.aulc)
            by_arch.setdefault(case.architecture, []).append(rollout.loss_ratio)
        result[variant] = {
            "loss_ratio": summarize_ratios(ratios).to_dict(),
            "aulc": summarize_ratios(aulcs).to_dict(),
            "by_architecture": {
                name: summarize_ratios(values).to_dict()
                for name, values in by_arch.items()
            },
        }
    return result


def summarize_across_seeds(seed_results: list[dict[str, Any]], split: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        seed_means = [
            result[split][variant]["loss_ratio"]["mean"] for result in seed_results
        ]
        summary[variant] = summarize_ratios(seed_means).to_dict()
    full = [result[split]["full_student"]["loss_ratio"]["mean"] for result in seed_results]
    summary["paired_delta_vs_full"] = {
        variant: statistics.fmean(
            result[split][variant]["loss_ratio"]["mean"]
            - result[split]["full_student"]["loss_ratio"]["mean"]
            for result in seed_results
        )
        for variant in VARIANTS
        if variant != "full_student"
    }
    summary["full_student_seed_means"] = full
    return summary


def main() -> None:
    args = parse_args()
    apply_quick(args)
    device = torch.device(args.device)

    lr_cases = _mixed_cases(
        seed_base=411000,
        count=args.lr_validation_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    distill_cases = _mixed_cases(
        seed_base=421000,
        count=args.distill_train_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    scale_cases = _mixed_cases(
        seed_base=431000,
        count=args.scale_validation_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    allocation_cases = _mixed_cases(
        seed_base=481000,
        count=args.allocation_calibration_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    test_cases = _mixed_cases(
        seed_base=461000,
        count=args.test_tasks,
        width=args.width,
        samples=args.samples,
        conditions=TRAIN_CONDITIONS,
        device=device,
    )
    ood_cases = _mixed_cases(
        seed_base=471000,
        count=args.ood_test_tasks,
        width=args.width,
        samples=args.samples,
        conditions=OOD_CONDITIONS,
        device=device,
    )

    teacher_lr, teacher_validation = tune_teacher_lr(
        TEACHER_NAME,
        lr_cases,
        batch_size=args.batch_size,
        steps=args.steps,
    )
    records = collect_records(
        make_teacher(TEACHER_NAME, teacher_lr),
        distill_cases,
        batch_size=args.batch_size,
        steps=args.steps,
        teacher_name=TEACHER_NAME,
    )

    seed_results: list[dict[str, Any]] = []
    for seed_index in range(args.student_seeds):
        seed = args.student_seed + seed_index
        student, train_loss = train_supervised_student(
            records,
            device=device,
            seed=seed,
            epochs=args.distill_epochs,
            weights=DIRECTION_ONLY,
        )
        scale, scale_validation = select_student_scale(
            student,
            scale_cases,
            batch_size=args.batch_size,
            steps=args.steps,
        )
        role_scales, scale_diagnostics = estimate_role_scales(
            student,
            allocation_cases,
            batch_size=args.batch_size,
            steps=args.steps,
        )
        seed_results.append(
            {
                "seed": seed,
                "train_loss": train_loss,
                "selected_scale": scale,
                "scale_validation_loss_ratio": scale_validation,
                "allocation_calibration": scale_diagnostics,
                "relative_role_scales_vs_teacher": {
                    name: value / teacher_lr for name, value in role_scales.items()
                },
                "test": evaluate_variants(
                    student,
                    test_cases,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    teacher_lr=teacher_lr,
                    frozen_scales=role_scales,
                ),
                "ood": evaluate_variants(
                    student,
                    ood_cases,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    teacher_lr=teacher_lr,
                    frozen_scales=role_scales,
                ),
            }
        )

    config = vars(args) | {"output": str(args.output)}
    split_specs = {
        "architectures": list(ARCHITECTURES),
        "train_conditions": list(TRAIN_CONDITIONS),
        "ood_conditions": list(OOD_CONDITIONS),
        "seed_bases": {
            "lr_validation": 411000,
            "distill": 421000,
            "scale_validation": 431000,
            "allocation_calibration": 481000,
            "test": 461000,
            "ood": 471000,
            "residual_offset": 500000,
        },
    }
    commit_sha = git_commit_sha()
    payload = {
        "experiment": "multitensor_normgrad_allocation_mechanism",
        "git_sha": commit_sha,
        "metadata": artifact_metadata(
            run_id=os.environ.get("GITHUB_RUN_ID", "local"),
            commit_sha=commit_sha,
            config=config,
            split_specs=split_specs,
        ),
        "config": config,
        "teacher": {
            "name": TEACHER_NAME,
            "lr": teacher_lr,
            "lr_validation_loss_ratio": teacher_validation,
        },
        "interpretation": {
            "tensor_projection": (
                "Keeps dynamic per-tensor Student scale but removes within-tensor direction residual."
            ),
            "global_projection": (
                "Keeps one dynamic Student scale but removes tensor-wise allocation and direction residual."
            ),
            "frozen_role": (
                "Uses validation-only median Student projection scale per tensor name on NormGrad directions."
            ),
            "equalized_frozen": "Removes tensor-name allocation from the frozen policy.",
            "swapped_frozen": "Swaps W1/W2 and b1/b2 frozen scales as a causal role-identity control.",
            "teacher_norm": "Restores the original tensor-wise NormGrad teacher update norm.",
        },
        "seed_results": seed_results,
        "summary": {
            "test": summarize_across_seeds(seed_results, "test"),
            "ood": summarize_across_seeds(seed_results, "ood"),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
