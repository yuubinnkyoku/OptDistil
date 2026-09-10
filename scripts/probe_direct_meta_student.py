from __future__ import annotations

import argparse
import copy
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from optdistil.distill.direct_meta import (
    evaluate_direct_student,
    train_direct_student,
    zero_initialize_student_output,
)
from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.rollout import OptimizationTask, rollout_student, select_student_output_scale
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp

Case = tuple[torch.Tensor, OptimizationTask]
CONDITIONS = (10.0, 100.0)
OUTER_LR_CANDIDATES = (1e-3, 3e-3, 1e-2)
SCALE_CANDIDATES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True, slots=True)
class DirectRun:
    seed: int
    parameters: int
    selected_outer_lr: float
    meta_validation_loss_ratio: float
    output_scale: float
    scale_validation_loss_ratio: float
    test_loss_ratio: float
    test_loss_ratio_by_condition: dict[str, float]
    final_meta_train_objective: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly meta-train the same 153-parameter deployment student."
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--meta-train-tasks", type=int, default=4)
    parser.add_argument("--meta-validation-tasks", type=int, default=4)
    parser.add_argument("--scale-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--student-seed", type=int, default=51000)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 8
    args.iterations = 12
    args.meta_train_tasks = 2
    args.meta_validation_tasks = 2
    args.scale_validation_tasks = 2
    args.test_tasks = 4
    args.student_seeds = 3


def make_split(
    seed_base: int,
    count: int,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[float, list[Case]]:
    result: dict[float, list[Case]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        result[condition] = [
            make_frozen_readout_mlp(
                seed_base + 1000 * condition_index + index,
                hidden_dim=args.size,
                input_dim=args.size,
                output_dim=args.output_dim,
                samples=args.samples,
                input_condition=condition,
                device=device,
            )
            for index in range(count)
        ]
    return result


def flatten(split: dict[float, list[Case]]) -> list[Case]:
    return [case for condition in CONDITIONS for case in split[condition]]


def evaluate_by_condition(
    student: TinyMLPOptimizer,
    split: dict[float, list[Case]],
    *,
    steps: int,
) -> tuple[float, dict[str, float]]:
    by_condition: dict[str, float] = {}
    for condition in CONDITIONS:
        ratios: list[float] = []
        for initial, task in split[condition]:
            result = rollout_student(
                student,
                initial,
                task,
                steps=steps,
                feature_builder=build_matrix_aware_features,
            )
            ratios.append(result.loss_ratio)
        by_condition[f"{condition:g}"] = statistics.fmean(ratios)
    return statistics.fmean(by_condition.values()), by_condition


def train_candidate(
    *,
    seed: int,
    outer_lr: float,
    train_cases: list[Case],
    validation_cases: list[Case],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TinyMLPOptimizer, list]:
    torch.manual_seed(seed)
    student = TinyMLPOptimizer().to(device)
    zero_initialize_student_output(student)
    history = train_direct_student(
        student,
        train_cases,
        validation_cases,
        steps=args.steps,
        iterations=args.iterations,
        outer_lr=outer_lr,
        grad_clip=1.0,
        validation_interval=2 if args.quick else 5,
        feature_builder=build_matrix_aware_features,
    )
    return student, history


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.output_dim,
        args.samples,
        args.steps,
        args.iterations,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.scale_validation_tasks,
        args.test_tasks,
        args.student_seeds,
    ) <= 0:
        raise ValueError("dimensions, counts, steps, iterations, and seeds must be positive")

    device = torch.device(args.device)
    train_split = make_split(11000, args.meta_train_tasks, args=args, device=device)
    meta_validation_split = make_split(
        16000,
        args.meta_validation_tasks,
        args=args,
        device=device,
    )
    scale_validation_split = make_split(
        26000,
        args.scale_validation_tasks,
        args=args,
        device=device,
    )
    test_split = make_split(31000, args.test_tasks, args=args, device=device)
    train_cases = flatten(train_split)
    meta_validation_cases = flatten(meta_validation_split)
    scale_validation_cases = flatten(scale_validation_split)

    runs: list[DirectRun] = []
    histories: dict[str, dict[str, list[dict]]] = {}
    for seed_offset in range(args.student_seeds):
        seed = args.student_seed + seed_offset
        candidates = []
        histories[str(seed)] = {}
        for outer_lr in OUTER_LR_CANDIDATES:
            student, history = train_candidate(
                seed=seed,
                outer_lr=outer_lr,
                train_cases=train_cases,
                validation_cases=meta_validation_cases,
                args=args,
                device=device,
            )
            validation_ratio = evaluate_direct_student(
                student,
                meta_validation_cases,
                steps=args.steps,
                feature_builder=build_matrix_aware_features,
            )
            histories[str(seed)][f"{outer_lr:g}"] = [asdict(row) for row in history]
            candidates.append((validation_ratio, outer_lr, copy.deepcopy(student), history[-1]))

        validation_ratio, selected_lr, student, final_history = min(
            candidates,
            key=lambda item: item[0],
        )
        student.set_output_scale(1.0)
        scale_result = select_student_output_scale(
            student,
            scale_validation_cases,
            candidates=SCALE_CANDIDATES,
            steps=args.steps,
            feature_builder=build_matrix_aware_features,
        )
        test_ratio, test_by_condition = evaluate_by_condition(
            student,
            test_split,
            steps=args.steps,
        )
        runs.append(
            DirectRun(
                seed=seed,
                parameters=student.parameter_count,
                selected_outer_lr=selected_lr,
                meta_validation_loss_ratio=validation_ratio,
                output_scale=scale_result.scale,
                scale_validation_loss_ratio=scale_result.validation_loss_ratio,
                test_loss_ratio=test_ratio,
                test_loss_ratio_by_condition=test_by_condition,
                final_meta_train_objective=final_history.train_objective,
            )
        )

    ratios = [row.test_loss_ratio for row in runs]
    payload = {
        "experiment": "direct_meta_tiny_student",
        "quick": args.quick,
        "student_parameters": 153,
        "conditions": CONDITIONS,
        "steps": args.steps,
        "iterations": args.iterations,
        "outer_lr_candidates": OUTER_LR_CANDIDATES,
        "student_seeds": args.student_seeds,
        "runs": [asdict(row) for row in runs],
        "summary": {
            "test_loss_ratio_mean": statistics.fmean(ratios),
            "test_loss_ratio_seed_std": statistics.pstdev(ratios) if len(ratios) > 1 else 0.0,
            "selected_outer_lrs": [row.selected_outer_lr for row in runs],
            "output_scales": [row.output_scale for row in runs],
        },
        "histories": histories,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    print("DIRECT_META_SUMMARY_BEGIN")
    for row in runs:
        print(
            f"seed={row.seed} lr={row.selected_outer_lr:g} scale={row.output_scale:g} "
            f"test={row.test_loss_ratio:.6f} by_condition={row.test_loss_ratio_by_condition}"
        )
    print(
        f"mean={payload['summary']['test_loss_ratio_mean']:.6f} "
        f"std={payload['summary']['test_loss_ratio_seed_std']:.6f}"
    )
    print("DIRECT_META_SUMMARY_END")


if __name__ == "__main__":
    main()
