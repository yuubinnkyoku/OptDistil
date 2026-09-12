from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import flatten, make_split, select_lr
from probe_secant_history_pareto import (
    collect_secant_records,
    evaluate_secant_student,
    select_secant_scale,
)
from probe_secant_projection_gain_control import (
    evaluate_projection_gain_student,
    gain_normalization,
    select_projection_gain_scale,
    train_projection_gain_student,
)

from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.secant_meta import (
    evaluate_secant_meta_student,
    train_secant_meta_student,
)
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer

SOURCES = ("norm_gradient", "muon", "newton_050")
HISTORY_SIZE = 4
OUTER_LR_CANDIDATES = (3e-4, 1e-3, 3e-3)
FULL_DISTILL_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)


@dataclass(frozen=True, slots=True)
class MetaControlRun:
    source: str
    seed: int
    mode: str
    student_parameters: int
    gain_normalization: float | None
    pre_meta_validation_scale: float
    pre_meta_test_loss_ratio: float
    selected_outer_lr: float
    meta_validation_loss_ratio: float
    post_meta_test_loss_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Give full-update and projection-gain-only 153-parameter Students the same "
            "closed-loop meta-finetuning budget after their respective supervised "
            "distillation initializations. Outer LR is selected on an independent meta-val split."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--scale-validation-tasks", type=int, default=4)
    parser.add_argument("--meta-train-tasks", type=int, default=4)
    parser.add_argument("--meta-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--meta-iterations", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=231000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.scale_validation_tasks = 2
    args.meta_train_tasks = 2
    args.meta_validation_tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 10
    args.meta_iterations = 10
    args.student_seeds = 3


def clone_state(student: TinyMLPOptimizer) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in student.state_dict().items()}


def make_student_from_state(state, *, device: torch.device) -> TinyMLPOptimizer:
    student = TinyMLPOptimizer().to(device)
    student.load_state_dict(state)
    return student


def select_meta_lr(
    initial_state,
    *,
    device: torch.device,
    train_cases,
    validation_cases,
    steps: int,
    iterations: int,
    mode: str,
    normalization: float,
) -> tuple[TinyMLPOptimizer, float, float]:
    candidates = []
    for outer_lr in OUTER_LR_CANDIDATES:
        student = make_student_from_state(initial_state, device=device)
        train_secant_meta_student(
            student,
            train_cases,
            validation_cases,
            steps=steps,
            history_size=HISTORY_SIZE,
            mode=mode,
            gain_normalization=normalization,
            iterations=iterations,
            outer_lr=outer_lr,
            grad_clip=1.0,
            final_weight=0.7,
        )
        validation_ratio = evaluate_secant_meta_student(
            student,
            validation_cases,
            steps=steps,
            history_size=HISTORY_SIZE,
            mode=mode,
            gain_normalization=normalization,
        )
        candidates.append((validation_ratio, outer_lr, student))
    best_validation, best_lr, best_student = min(candidates, key=lambda item: item[0])
    return best_student, best_lr, best_validation


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.scale_validation_tasks,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.meta_iterations,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, task counts, iterations, epochs, and seeds must be positive")

    device = torch.device(args.device)
    split_specs = {
        "lr_validation": (141000, args.lr_validation_tasks),
        "distill": (146000, args.distill_train_tasks),
        "scale_validation": (151000, args.scale_validation_tasks),
        "meta_train": (161000, args.meta_train_tasks),
        "meta_validation": (166000, args.meta_validation_tasks),
        "test": (171000, args.test_tasks),
    }
    splits = {
        name: make_split(seed_base=seed, count=count, size=args.size, device=device)
        for name, (seed, count) in split_specs.items()
    }
    lr_cases = flatten(splits["lr_validation"])
    distill_cases = flatten(splits["distill"])
    scale_validation_cases = flatten(splits["scale_validation"])
    meta_train_cases = flatten(splits["meta_train"])
    meta_validation_cases = flatten(splits["meta_validation"])
    test_cases = flatten(splits["test"])

    lrs = {
        source: select_lr(source, lr_cases, steps=args.steps)
        for source in ("muon", "norm_gradient")
    }
    records = {
        source: collect_secant_records(
            source,
            distill_cases,
            lrs=lrs,
            steps=args.steps,
            history_size=HISTORY_SIZE,
        )
        for source in SOURCES
    }
    normalizations = {source: gain_normalization(records[source]) for source in SOURCES}

    runs: list[MetaControlRun] = []
    for source_index, source in enumerate(SOURCES):
        for seed_index in range(args.student_seeds):
            seed = args.student_seed + 1000 * source_index + seed_index

            torch.manual_seed(seed)
            full_student = TinyMLPOptimizer().to(device)
            train_student(
                full_student,
                records[source],
                epochs=args.distill_epochs,
                lr=3e-3,
                weights=FULL_DISTILL_WEIGHTS,
            )
            full_scale, _ = select_secant_scale(
                full_student,
                scale_validation_cases,
                steps=args.steps,
                history_size=HISTORY_SIZE,
            )
            pre_full_test, _ = evaluate_secant_student(
                full_student,
                splits["test"],
                steps=args.steps,
                history_size=HISTORY_SIZE,
            )
            full_best, full_outer_lr, full_meta_validation = select_meta_lr(
                clone_state(full_student),
                device=device,
                train_cases=meta_train_cases,
                validation_cases=meta_validation_cases,
                steps=args.steps,
                iterations=args.meta_iterations,
                mode="full_update",
                normalization=1.0,
            )
            post_full_test = evaluate_secant_meta_student(
                full_best,
                test_cases,
                steps=args.steps,
                history_size=HISTORY_SIZE,
                mode="full_update",
                gain_normalization=1.0,
            )
            runs.append(
                MetaControlRun(
                    source=source,
                    seed=seed,
                    mode="full_update",
                    student_parameters=full_student.parameter_count,
                    gain_normalization=None,
                    pre_meta_validation_scale=full_scale,
                    pre_meta_test_loss_ratio=pre_full_test,
                    selected_outer_lr=full_outer_lr,
                    meta_validation_loss_ratio=full_meta_validation,
                    post_meta_test_loss_ratio=post_full_test,
                )
            )

            torch.manual_seed(seed)
            gain_student = TinyMLPOptimizer().to(device)
            normalization = normalizations[source]
            train_projection_gain_student(
                gain_student,
                records[source],
                epochs=args.distill_epochs,
                lr=3e-3,
                normalization=normalization,
            )
            gain_scale, _, _ = select_projection_gain_scale(
                gain_student,
                scale_validation_cases,
                steps=args.steps,
                normalization=normalization,
            )
            pre_gain_test = evaluate_projection_gain_student(
                gain_student,
                splits["test"],
                steps=args.steps,
                normalization=normalization,
            )
            gain_best, gain_outer_lr, gain_meta_validation = select_meta_lr(
                clone_state(gain_student),
                device=device,
                train_cases=meta_train_cases,
                validation_cases=meta_validation_cases,
                steps=args.steps,
                iterations=args.meta_iterations,
                mode="scalar_gain",
                normalization=normalization,
            )
            post_gain_test = evaluate_secant_meta_student(
                gain_best,
                test_cases,
                steps=args.steps,
                history_size=HISTORY_SIZE,
                mode="scalar_gain",
                gain_normalization=normalization,
            )
            runs.append(
                MetaControlRun(
                    source=source,
                    seed=seed,
                    mode="projection_gain_meta",
                    student_parameters=gain_student.parameter_count,
                    gain_normalization=normalization,
                    pre_meta_validation_scale=gain_scale,
                    pre_meta_test_loss_ratio=pre_gain_test,
                    selected_outer_lr=gain_outer_lr,
                    meta_validation_loss_ratio=gain_meta_validation,
                    post_meta_test_loss_ratio=post_gain_test,
                )
            )

    summary = {}
    paired = {}
    for source in SOURCES:
        source_runs = [run for run in runs if run.source == source]
        summary[source] = {}
        for mode in ("full_update", "projection_gain_meta"):
            group = [run for run in source_runs if run.mode == mode]
            pre = [run.pre_meta_test_loss_ratio for run in group]
            post = [run.post_meta_test_loss_ratio for run in group]
            summary[source][mode] = {
                "student_parameters": group[0].student_parameters,
                "pre_meta_test_mean": statistics.fmean(pre),
                "post_meta_test_mean": statistics.fmean(post),
                "post_meta_test_seed_std": statistics.pstdev(post),
                "meta_validation_mean": statistics.fmean(
                    run.meta_validation_loss_ratio for run in group
                ),
                "selected_outer_lr_mean": statistics.fmean(
                    run.selected_outer_lr for run in group
                ),
            }
        full = {
            run.seed: run.post_meta_test_loss_ratio
            for run in source_runs
            if run.mode == "full_update"
        }
        gain = {
            run.seed: run.post_meta_test_loss_ratio
            for run in source_runs
            if run.mode == "projection_gain_meta"
        }
        deltas = [gain[seed] - full[seed] for seed in sorted(full)]
        paired[source] = {
            "gain_minus_full_mean_absolute_delta": statistics.fmean(deltas),
            "gain_minus_full_mean_relative_delta": statistics.fmean(
                (gain[seed] - full[seed]) / full[seed] for seed in sorted(full)
            ),
            "gain_beats_full_seeds": sum(delta < 0.0 for delta in deltas),
            "total_seeds": len(deltas),
        }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "history_size": HISTORY_SIZE,
        "outer_lr_candidates": OUTER_LR_CANDIDATES,
        "tuned_lrs": lrs,
        "gain_normalizations": normalizations,
        "runs": [asdict(run) for run in runs],
        "summary": summary,
        "paired_deltas": paired,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
