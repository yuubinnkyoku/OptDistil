from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import uuid
from pathlib import Path
from typing import Any

import torch

from optdistil.distill.experimental_features import (
    build_hybrid_gram_features,
    build_matrix_aware_no_ema,
    build_matrix_aware_no_progress,
)
from optdistil.distill.features import (
    build_elementwise_features,
    build_gram_matrix_features,
    build_matrix_aware_features,
)
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.stochastic import (
    DEFAULT_BATCH_SIZES,
    OOD_CONDITIONS,
    TRAIN_CONDITIONS,
    StochasticCase,
    batch_transfer_matrix,
    clone_state_dict,
    collect_stochastic_records,
    evaluate_secant_split,
    evaluate_student_split,
    evaluate_teacher_split,
    flatten_split,
    make_split,
    paired_differences,
    select_outer_lr_and_train,
    select_student_scale,
    summarize_ratios,
    train_direct_meta_student,
    train_supervised_student,
    tune_secant,
    tune_teacher_lr,
)

REGIMES = (
    ("adamw_b32", "adamw", 32),
    ("norm_gradient_b8", "norm_gradient", 8),
)
JOINT_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)
DIRECTION_ONLY_WEIGHTS = DistillationLossWeights(direction=1.0, magnitude=0.0)
FEATURE_SETS = {
    "matrix_aware": build_matrix_aware_features,
    "elementwise": build_elementwise_features,
    "gram_cubic": build_gram_matrix_features,
    "hybrid_gram": build_hybrid_gram_features,
    "no_ema": build_matrix_aware_no_ema,
    "no_progress": build_matrix_aware_no_progress,
}
LABEL_CONTROLS = ("shuffle_tasks", "permute_coords", "norm_only", "random_unit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stochastic optimizer-distillation benchmark on FrozenReadoutMLP with "
            "AdamW@batch32 and NormGrad@batch8 teacher regimes."
        )
    )
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr-validation-tasks", type=int, default=6)
    parser.add_argument("--distill-train-tasks", type=int, default=8)
    parser.add_argument("--scale-validation-tasks", type=int, default=6)
    parser.add_argument("--meta-train-tasks", type=int, default=6)
    parser.add_argument("--meta-validation-tasks", type=int, default=6)
    parser.add_argument("--test-tasks", type=int, default=10)
    parser.add_argument("--ood-test-tasks", type=int, default=6)
    parser.add_argument("--distill-epochs", type=int, default=25)
    parser.add_argument("--meta-iterations", type=int, default=15)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=401000)
    parser.add_argument("--control-seeds", type=int, default=3)
    parser.add_argument("--ablation-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--regime",
        choices=[name for name, _, _ in REGIMES],
        action="append",
        help="Run only the named regime(s). Default: all regimes.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip negative-control label corruptions.",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip feature-family ablations.",
    )
    parser.add_argument(
        "--skip-batch-transfer",
        action="store_true",
        help="Skip evaluation at batch sizes other than the training batch.",
    )
    parser.add_argument(
        "--skip-main-students",
        action="store_true",
        help="Skip distilled/meta/direct-meta student modes; keep analytic, controls, ablations.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 12
    args.lr_validation_tasks = 2
    args.distill_train_tasks = 2
    args.scale_validation_tasks = 2
    args.meta_train_tasks = 2
    args.meta_validation_tasks = 2
    args.test_tasks = 3
    args.ood_test_tasks = 2
    args.distill_epochs = 4
    args.meta_iterations = 3
    args.student_seeds = 2
    args.control_seeds = 1
    args.ablation_seeds = 1


def git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate_student_bundle(
    student: torch.nn.Module,
    *,
    test_split: dict[float, list[StochasticCase]],
    ood_split: dict[float, list[StochasticCase]],
    batch_size: int,
    steps: int,
    feature_builder,
    reference: str,
    reference_lr: float,
) -> dict[str, Any]:
    test_metrics = evaluate_student_split(
        student,
        test_split,
        batch_size=batch_size,
        steps=steps,
        feature_builder=feature_builder,
        reference=reference,  # type: ignore[arg-type]
        reference_lr=reference_lr,
    )
    ood_metrics = evaluate_student_split(
        student,
        ood_split,
        batch_size=batch_size,
        steps=steps,
        feature_builder=feature_builder,
        reference=reference,  # type: ignore[arg-type]
        reference_lr=reference_lr,
    )
    return {"test": test_metrics, "ood": ood_metrics}


def run_regime(
    *,
    regime: str,
    teacher_method: str,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
    splits: dict[str, dict[float, list[StochasticCase]]],
    ood_split: dict[float, list[StochasticCase]],
) -> dict[str, Any]:
    lr_cases = flatten_split(splits["lr_validation"])
    distill_cases = flatten_split(splits["distill"])
    scale_validation_cases = flatten_split(splits["scale_validation"])
    meta_train_cases = flatten_split(splits["meta_train"])
    meta_validation_cases = flatten_split(splits["meta_validation"])
    test_split = splits["test"]

    tuned_lrs = {
        method: tune_teacher_lr(
            method,  # type: ignore[arg-type]
            lr_cases,
            batch_size=batch_size,
            steps=args.steps,
        )[0]
        for method in ("adamw", "norm_gradient", "muon")
    }
    teacher_lr = tuned_lrs[teacher_method]

    analytic_test = {
        method: evaluate_teacher_split(
            method,  # type: ignore[arg-type]
            lr,
            test_split,
            batch_size=batch_size,
            steps=args.steps,
        )
        for method, lr in tuned_lrs.items()
    }
    analytic_ood = {
        method: evaluate_teacher_split(
            method,  # type: ignore[arg-type]
            lr,
            ood_split,
            batch_size=batch_size,
            steps=args.steps,
        )
        for method, lr in tuned_lrs.items()
    }

    secant_tuning, secant_validation = tune_secant(
        scale_validation_cases,
        batch_size=batch_size,
        steps=args.steps,
    )
    raw_lbfgs = {
        "validation_loss_ratio": secant_validation,
        "tuning": secant_tuning,
        "test": evaluate_secant_split(
            test_split,
            batch_size=batch_size,
            steps=args.steps,
            secant_scale=secant_tuning["secant_scale"],
            bootstrap_scale=secant_tuning["bootstrap_scale"],
        ),
        "ood": evaluate_secant_split(
            ood_split,
            batch_size=batch_size,
            steps=args.steps,
            secant_scale=secant_tuning["secant_scale"],
            bootstrap_scale=secant_tuning["bootstrap_scale"],
        ),
    }

    strongest_analytic_name = min(
        analytic_test,
        key=lambda name: analytic_test[name]["loss_ratio"]["mean"],
    )
    strongest_analytic_ratio = analytic_test[strongest_analytic_name]["loss_ratio"]["mean"]
    teacher_test_ratio = analytic_test[teacher_method]["loss_ratio"]["mean"]

    joint_records = collect_stochastic_records(
        teacher_method,  # type: ignore[arg-type]
        teacher_lr,
        distill_cases,
        batch_size=batch_size,
        steps=args.steps,
        feature_builder=build_matrix_aware_features,
    )
    direction_records = collect_stochastic_records(
        teacher_method,  # type: ignore[arg-type]
        teacher_lr,
        distill_cases,
        batch_size=batch_size,
        steps=args.steps,
        feature_builder=build_matrix_aware_features,
    )

    feature_builder = FEATURE_SETS["matrix_aware"]
    runs: list[dict[str, Any]] = []
    batch_transfer: dict[str, Any] | None = None
    per_seed_test_ratios: dict[str, list[float]] = {
        "distill_joint": [],
        "distill_direction": [],
        "distill_joint_meta": [],
        "direct_meta": [],
    }
    per_seed_ood_ratios: dict[str, list[float]] = {key: [] for key in per_seed_test_ratios}

    for seed_index in range(0 if args.skip_main_students else args.student_seeds):
        seed = args.student_seed + seed_index

        # 1) Supervised distillation, direction+magnitude.
        joint_student, joint_train_loss = train_supervised_student(
            joint_records,
            device=device,
            seed=seed,
            epochs=args.distill_epochs,
            weights=JOINT_WEIGHTS,
        )
        joint_scale, joint_scale_score = select_student_scale(
            joint_student,
            scale_validation_cases,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
        )
        joint_eval = evaluate_student_bundle(
            joint_student,
            test_split=test_split,
            ood_split=ood_split,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
            reference=teacher_method,
            reference_lr=teacher_lr,
        )
        runs.append(
            {
                "mode": "distill_joint",
                "seed": seed,
                "student_parameters": joint_student.parameter_count,
                "validation_scale": joint_scale,
                "validation_scale_score": joint_scale_score,
                "train_distillation_loss": joint_train_loss,
                "test": joint_eval["test"],
                "ood": joint_eval["ood"],
            }
        )
        per_seed_test_ratios["distill_joint"].append(joint_eval["test"]["loss_ratio"]["mean"])
        per_seed_ood_ratios["distill_joint"].append(joint_eval["ood"]["loss_ratio"]["mean"])

        # 2) Supervised distillation, direction-only.
        direction_student, direction_train_loss = train_supervised_student(
            direction_records,
            device=device,
            seed=seed + 50_000,
            epochs=args.distill_epochs,
            weights=DIRECTION_ONLY_WEIGHTS,
        )
        direction_scale, direction_scale_score = select_student_scale(
            direction_student,
            scale_validation_cases,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
        )
        direction_eval = evaluate_student_bundle(
            direction_student,
            test_split=test_split,
            ood_split=ood_split,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
            reference=teacher_method,
            reference_lr=teacher_lr,
        )
        runs.append(
            {
                "mode": "distill_direction",
                "seed": seed,
                "student_parameters": direction_student.parameter_count,
                "validation_scale": direction_scale,
                "validation_scale_score": direction_scale_score,
                "train_distillation_loss": direction_train_loss,
                "test": direction_eval["test"],
                "ood": direction_eval["ood"],
            }
        )
        per_seed_test_ratios["distill_direction"].append(
            direction_eval["test"]["loss_ratio"]["mean"]
        )
        per_seed_ood_ratios["distill_direction"].append(
            direction_eval["ood"]["loss_ratio"]["mean"]
        )

        # 3) Closed-loop meta-finetuning from the joint distilled student.
        meta_state = clone_state_dict(joint_student)
        meta_student, outer_lr, meta_validation = select_outer_lr_and_train(
            meta_state,
            device=device,
            train_cases=meta_train_cases,
            validation_cases=meta_validation_cases,
            batch_size=batch_size,
            steps=args.steps,
            iterations=args.meta_iterations,
            feature_builder=feature_builder,
        )
        meta_eval = evaluate_student_bundle(
            meta_student,
            test_split=test_split,
            ood_split=ood_split,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
            reference=teacher_method,
            reference_lr=teacher_lr,
        )
        runs.append(
            {
                "mode": "distill_joint_meta",
                "seed": seed,
                "student_parameters": meta_student.parameter_count,
                "validation_scale": meta_student.output_scale.item(),
                "selected_outer_lr": outer_lr,
                "meta_validation_loss_ratio": meta_validation,
                "test": meta_eval["test"],
                "ood": meta_eval["ood"],
            }
        )
        per_seed_test_ratios["distill_joint_meta"].append(meta_eval["test"]["loss_ratio"]["mean"])
        per_seed_ood_ratios["distill_joint_meta"].append(meta_eval["ood"]["loss_ratio"]["mean"])

        # 4) Direct meta-trained student without distillation.
        direct_student, direct_outer_lr, direct_validation = train_direct_meta_student(
            meta_train_cases,
            meta_validation_cases,
            device=device,
            batch_size=batch_size,
            steps=args.steps,
            iterations=args.meta_iterations,
            feature_builder=feature_builder,
            seed=seed + 70_000,
        )
        direct_scale, direct_scale_score = select_student_scale(
            direct_student,
            scale_validation_cases,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
        )
        direct_eval = evaluate_student_bundle(
            direct_student,
            test_split=test_split,
            ood_split=ood_split,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
            reference=teacher_method,
            reference_lr=teacher_lr,
        )
        runs.append(
            {
                "mode": "direct_meta",
                "seed": seed,
                "student_parameters": direct_student.parameter_count,
                "validation_scale": direct_scale,
                "validation_scale_score": direct_scale_score,
                "selected_outer_lr": direct_outer_lr,
                "meta_validation_loss_ratio": direct_validation,
                "test": direct_eval["test"],
                "ood": direct_eval["ood"],
            }
        )
        per_seed_test_ratios["direct_meta"].append(direct_eval["test"]["loss_ratio"]["mean"])
        per_seed_ood_ratios["direct_meta"].append(direct_eval["ood"]["loss_ratio"]["mean"])

        if seed_index == 0 and not args.skip_batch_transfer:
            batch_transfer = {
                "seed": seed,
                "matrix": batch_transfer_matrix(
                    joint_student,
                    test_split,
                    train_batch_size=batch_size,
                    eval_batch_sizes=DEFAULT_BATCH_SIZES,
                    steps=args.steps,
                    feature_builder=feature_builder,
                ),
            }

    if args.skip_batch_transfer:
        batch_transfer = None

    controls: dict[str, Any] = {}
    if not args.skip_controls:
        print(f"[stoch-distill] regime={regime} negative controls", flush=True)
        for control_index, label_control in enumerate(LABEL_CONTROLS):
            control_ratios_test = []
            control_ratios_ood = []
            for seed_index in range(args.control_seeds):
                seed = args.student_seed + 200_000 + seed_index
                controlled_records = collect_stochastic_records(
                    teacher_method,  # type: ignore[arg-type]
                    teacher_lr,
                    distill_cases,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=feature_builder,
                    label_control=label_control,  # type: ignore[arg-type]
                    control_seed=seed + control_index,
                )
                student, _ = train_supervised_student(
                    controlled_records,
                    device=device,
                    seed=seed,
                    epochs=args.distill_epochs,
                    weights=JOINT_WEIGHTS,
                )
                select_student_scale(
                    student,
                    scale_validation_cases,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                test_eval = evaluate_student_split(
                    student,
                    test_split,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                ood_eval = evaluate_student_split(
                    student,
                    ood_split,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )
                control_ratios_test.append(test_eval["loss_ratio"]["mean"])
                control_ratios_ood.append(ood_eval["loss_ratio"]["mean"])
            controls[label_control] = {
                "test": summarize_ratios(control_ratios_test).to_dict(),
                "ood": summarize_ratios(control_ratios_ood).to_dict(),
                "seeds": args.control_seeds,
            }

        # Analytic-baseline trajectory student: learn from the strongest non-teacher
        # analytic optimizer instead of the regime teacher.
        baseline_method = (
            "norm_gradient" if teacher_method == "adamw" else "adamw"
        )
        baseline_lr = tuned_lrs[baseline_method]
        baseline_records = collect_stochastic_records(
            baseline_method,  # type: ignore[arg-type]
            baseline_lr,
            distill_cases,
            batch_size=batch_size,
            steps=args.steps,
            feature_builder=feature_builder,
        )
        baseline_test = []
        baseline_ood = []
        for seed_index in range(args.control_seeds):
            seed = args.student_seed + 250_000 + seed_index
            student, _ = train_supervised_student(
                baseline_records,
                device=device,
                seed=seed,
                epochs=args.distill_epochs,
                weights=JOINT_WEIGHTS,
            )
            select_student_scale(
                student,
                scale_validation_cases,
                batch_size=batch_size,
                steps=args.steps,
                feature_builder=feature_builder,
            )
            baseline_test.append(
                evaluate_student_split(
                    student,
                    test_split,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )["loss_ratio"]["mean"]
            )
            baseline_ood.append(
                evaluate_student_split(
                    student,
                    ood_split,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=feature_builder,
                )["loss_ratio"]["mean"]
            )
        controls["analytic_baseline_trajectory"] = {
            "source_teacher": baseline_method,
            "source_lr": baseline_lr,
            "test": summarize_ratios(baseline_test).to_dict(),
            "ood": summarize_ratios(baseline_ood).to_dict(),
            "seeds": args.control_seeds,
        }

    ablations: dict[str, Any] = {}
    if not args.skip_ablations:
        print(f"[stoch-distill] regime={regime} feature ablations", flush=True)
        for feature_name, builder in FEATURE_SETS.items():
            print(f"[stoch-distill] regime={regime} ablation={feature_name}", flush=True)
            ablation_test = []
            ablation_ood = []
            for seed_index in range(args.ablation_seeds):
                seed = args.student_seed + 300_000 + seed_index
                records = collect_stochastic_records(
                    teacher_method,  # type: ignore[arg-type]
                    teacher_lr,
                    distill_cases,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=builder,
                )
                student, _ = train_supervised_student(
                    records,
                    device=device,
                    seed=seed,
                    epochs=args.distill_epochs,
                    weights=JOINT_WEIGHTS,
                )
                select_student_scale(
                    student,
                    scale_validation_cases,
                    batch_size=batch_size,
                    steps=args.steps,
                    feature_builder=builder,
                )
                ablation_test.append(
                    evaluate_student_split(
                        student,
                        test_split,
                        batch_size=batch_size,
                        steps=args.steps,
                        feature_builder=builder,
                    )["loss_ratio"]["mean"]
                )
                ablation_ood.append(
                    evaluate_student_split(
                        student,
                        ood_split,
                        batch_size=batch_size,
                        steps=args.steps,
                        feature_builder=builder,
                    )["loss_ratio"]["mean"]
                )
            ablations[feature_name] = {
                "test": summarize_ratios(ablation_test).to_dict(),
                "ood": summarize_ratios(ablation_ood).to_dict(),
                "student_parameters": 153,
                "seeds": args.ablation_seeds,
            }

    mode_summary = {}
    for mode, test_values in per_seed_test_ratios.items():
        ood_values = per_seed_ood_ratios[mode]
        if not test_values:
            continue
        mode_summary[mode] = {
            "test": summarize_ratios(test_values).to_dict(),
            "ood": summarize_ratios(ood_values).to_dict(),
            "fraction_beats_teacher_test": statistics.fmean(
                float(value < teacher_test_ratio) for value in test_values
            ),
            "fraction_beats_strongest_analytic_test": statistics.fmean(
                float(value < strongest_analytic_ratio) for value in test_values
            ),
        }

    if all(not values for values in per_seed_test_ratios.values()):
        paired: dict[str, Any] = {}
    else:
        paired = {
            "meta_minus_distill_test": paired_differences(
                per_seed_test_ratios["distill_joint_meta"],
                per_seed_test_ratios["distill_joint"],
            ),
            "meta_minus_distill_ood": paired_differences(
                per_seed_ood_ratios["distill_joint_meta"],
                per_seed_ood_ratios["distill_joint"],
            ),
            "distill_minus_direct_meta_test": paired_differences(
                per_seed_test_ratios["distill_joint"],
                per_seed_test_ratios["direct_meta"],
            ),
            "distill_minus_direct_meta_ood": paired_differences(
                per_seed_ood_ratios["distill_joint"],
                per_seed_ood_ratios["direct_meta"],
            ),
            "direction_minus_joint_test": paired_differences(
                per_seed_test_ratios["distill_direction"],
                per_seed_test_ratios["distill_joint"],
            ),
        }

    return {
        "teacher_method": teacher_method,
        "batch_size": batch_size,
        "tuned_lrs": tuned_lrs,
        "teacher_lr": teacher_lr,
        "train_records": len(joint_records),
        "analytic_test": analytic_test,
        "analytic_ood": analytic_ood,
        "raw_lbfgs_two_scale": raw_lbfgs,
        "strongest_analytic_test": {
            "name": strongest_analytic_name,
            "loss_ratio_mean": strongest_analytic_ratio,
        },
        "teacher_test_loss_ratio_mean": teacher_test_ratio,
        "runs": runs,
        "mode_summary": mode_summary,
        "paired": paired,
        "batch_transfer": batch_transfer,
        "negative_controls": controls,
        "feature_ablations": ablations,
    }


def main() -> None:
    args = parse_args()
    apply_quick(args)
    positive_fields = (
        args.size,
        args.samples,
        args.steps,
        args.lr_validation_tasks,
        args.distill_train_tasks,
        args.scale_validation_tasks,
        args.meta_train_tasks,
        args.meta_validation_tasks,
        args.test_tasks,
        args.ood_test_tasks,
        args.distill_epochs,
        args.meta_iterations,
        args.student_seeds,
        args.control_seeds,
        args.ablation_seeds,
    )
    if min(positive_fields) <= 0:
        raise ValueError("all dimensions, task counts, epochs, iterations, and seed counts must be positive")

    device = torch.device(args.device)
    run_id = f"stoch-distill-{uuid.uuid4().hex[:12]}"
    commit_sha = git_commit_sha()

    split_specs = {
        "lr_validation": (411000, args.lr_validation_tasks),
        "distill": (421000, args.distill_train_tasks),
        "scale_validation": (431000, args.scale_validation_tasks),
        "meta_train": (441000, args.meta_train_tasks),
        "meta_validation": (451000, args.meta_validation_tasks),
        "test": (461000, args.test_tasks),
    }
    splits = {
        name: make_split(
            TRAIN_CONDITIONS,
            seed_base=seed_base,
            count=count,
            size=args.size,
            samples=args.samples,
            device=device,
        )
        for name, (seed_base, count) in split_specs.items()
    }
    ood_split = make_split(
        OOD_CONDITIONS,
        seed_base=471000,
        count=args.ood_test_tasks,
        size=args.size,
        samples=args.samples,
        device=device,
    )

    regimes = {}
    selected = [item for item in REGIMES if not args.regime or item[0] in set(args.regime)]
    for regime, teacher_method, batch_size in selected:
        print(f"[stoch-distill] start regime={regime}", flush=True)
        regimes[regime] = run_regime(
            regime=regime,
            teacher_method=teacher_method,
            batch_size=batch_size,
            args=args,
            device=device,
            splits=splits,
            ood_split=ood_split,
        )
        print(f"[stoch-distill] done regime={regime}", flush=True)

    payload = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "experiment": "stochastic_optimizer_distillation",
        "config": {
            **{key: value for key, value in vars(args).items() if key != "output"},
            "output": str(args.output) if args.output else None,
            "device": str(device),
        },
        "train_conditions": list(TRAIN_CONDITIONS),
        "ood_conditions": list(OOD_CONDITIONS),
        "split_specs": split_specs,
        "ood_split_seed_base": 471000,
        "student_parameter_count": 153,
        "regimes": regimes,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
