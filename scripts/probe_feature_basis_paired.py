from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from probe_oracle_feature_bottleneck import (
    DISTILL_WEIGHTS,
    STUDENT_SCALE_CANDIDATES,
    Run,
    collect_records,
    evaluate_student,
    flatten,
    make_split,
    select_lr,
)

from optdistil.distill.experimental_features import build_hybrid_gram_features
from optdistil.distill.features import build_gram_matrix_features, build_matrix_aware_features
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.distill.rollout import select_student_output_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired comparison of scalar summaries, Gram directions, and a hybrid feature "
            "basis. Every feature set starts from the identical student initialization."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--student-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=8)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=171000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.student_validation_tasks = 2
    args.test_tasks = 4
    args.distill_epochs = 10
    args.student_seeds = 5


def main() -> None:
    args = parse_args()
    apply_quick(args)
    if min(
        args.size,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.student_validation_tasks,
        args.test_tasks,
        args.distill_epochs,
        args.student_seeds,
    ) <= 0:
        raise ValueError("sizes, task counts, epochs, and seed counts must be positive")

    device = torch.device(args.device)
    lr_validation_split = make_split(
        seed_base=141000,
        count=args.lr_validation_tasks,
        size=args.size,
        device=device,
    )
    distill_split = make_split(
        seed_base=146000,
        count=args.distill_train_tasks,
        size=args.size,
        device=device,
    )
    student_validation_split = make_split(
        seed_base=151000,
        count=args.student_validation_tasks,
        size=args.size,
        device=device,
    )
    test_split = make_split(
        seed_base=156000,
        count=args.test_tasks,
        size=args.size,
        device=device,
    )
    lr_cases = flatten(lr_validation_split)
    distill_cases = flatten(distill_split)
    student_validation_cases = flatten(student_validation_split)

    lrs = {
        source: select_lr(source, lr_cases, steps=args.steps)
        for source in ("muon", "norm_gradient")
    }
    sources = ("muon", "norm_gradient", "newton_025", "newton_050")
    feature_specs = (
        ("row_col_rms", build_matrix_aware_features),
        ("gram_cubic", build_gram_matrix_features),
        ("hybrid_grad_gram", build_hybrid_gram_features),
    )

    records_by_pair = {
        (source, feature_name): collect_records(
            source,
            distill_cases,
            lrs=lrs,
            steps=args.steps,
            feature_builder=feature_builder,
        )
        for source in sources
        for feature_name, feature_builder in feature_specs
    }

    runs: list[Run] = []
    for source_index, source in enumerate(sources):
        for seed_index in range(args.student_seeds):
            # Deliberately reuse this exact seed for every feature set in the pair.
            seed = args.student_seed + 1000 * source_index + seed_index
            for feature_name, feature_builder in feature_specs:
                records = records_by_pair[(source, feature_name)]
                torch.manual_seed(seed)
                student = TinyMLPOptimizer(feature_dim=8, hidden_dim=8, hidden_layers=2).to(device)
                history = train_student(
                    student,
                    records,
                    epochs=args.distill_epochs,
                    lr=3e-3,
                    weights=DISTILL_WEIGHTS,
                )
                scale = select_student_output_scale(
                    student,
                    student_validation_cases,
                    candidates=STUDENT_SCALE_CANDIDATES,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                test_ratio, by_condition = evaluate_student(
                    student,
                    test_split,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                runs.append(
                    Run(
                        source=source,
                        features=feature_name,
                        seed=seed,
                        student_parameters=student.parameter_count,
                        train_records=len(records),
                        final_distillation_loss=history[-1],
                        validation_loss_ratio=scale.validation_loss_ratio,
                        validation_scale=scale.scale,
                        test_loss_ratio=test_ratio,
                        test_loss_ratio_by_condition=by_condition,
                    )
                )

    summary = {}
    paired_deltas = {}
    for source in sources:
        summary[source] = {}
        source_rows = [row for row in runs if row.source == source]
        by_feature = {
            feature_name: [row for row in source_rows if row.features == feature_name]
            for feature_name, _ in feature_specs
        }
        for feature_name, group in by_feature.items():
            tests = [row.test_loss_ratio for row in group]
            summary[source][feature_name] = {
                "student_parameters": group[0].student_parameters,
                "test_loss_ratio_mean": statistics.fmean(tests),
                "test_loss_ratio_seed_std": statistics.pstdev(tests),
                "validation_loss_ratio_mean": statistics.fmean(
                    row.validation_loss_ratio for row in group
                ),
                "final_distillation_loss_mean": statistics.fmean(
                    row.final_distillation_loss for row in group
                ),
            }

        baseline = {row.seed: row.test_loss_ratio for row in by_feature["row_col_rms"]}
        paired_deltas[source] = {}
        for feature_name in ("gram_cubic", "hybrid_grad_gram"):
            candidate = {row.seed: row.test_loss_ratio for row in by_feature[feature_name]}
            deltas = [candidate[seed] - baseline[seed] for seed in sorted(baseline)]
            relative = [
                (candidate[seed] - baseline[seed]) / baseline[seed]
                for seed in sorted(baseline)
            ]
            paired_deltas[source][feature_name] = {
                "mean_absolute_delta_vs_row_col": statistics.fmean(deltas),
                "mean_relative_delta_vs_row_col": statistics.fmean(relative),
                "improved_seeds": sum(delta < 0.0 for delta in deltas),
                "total_seeds": len(deltas),
            }

    payload = {
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "tuned_lrs": lrs,
        "runs": [asdict(row) for row in runs],
        "summary": summary,
        "paired_deltas": paired_deltas,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
