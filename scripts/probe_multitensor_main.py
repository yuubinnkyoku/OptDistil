from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import torch

from optdistil.distill.losses import DistillationLossWeights
from optdistil.multitensor.stochastic import (
    OOD_CONDITIONS,
    TRAIN_CONDITIONS,
    MultiTensorCase,
    artifact_metadata,
    batch_sequence,
    batch_transfer_matrix,
    collect_records,
    evaluate_student_split,
    evaluate_teacher_split,
    flatten_split,
    git_commit_sha,
    make_split,
    paired_differences,
    select_outer_lr_and_train,
    select_student_scale,
    summarize_ratios,
    train_direct_meta_student,
    train_supervised_student,
    tune_teacher_lr,
    width_transfer_matrix,
)
from optdistil.multitensor.teachers import TEACHER_METHODS, make_teacher
from optdistil.students.tiny_mlp import TinyMLPOptimizer

ARCHITECTURES = ("two_layer", "residual")
JOINT_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)
DIRECTION_ONLY_WEIGHTS = DistillationLossWeights(direction=1.0, magnitude=0.0)
LABEL_CONTROLS = (
    "shuffle_tasks",
    "permute_coords",
    "norm_only",
    "random_unit",
)
FEATURE_MODES = ("local_global", "local_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-tensor stochastic optimizer-distillation benchmark."
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--lr-validation-tasks", type=int, default=4)
    parser.add_argument("--distill-train-tasks", type=int, default=6)
    parser.add_argument("--scale-validation-tasks", type=int, default=4)
    parser.add_argument("--meta-train-tasks", type=int, default=4)
    parser.add_argument("--meta-validation-tasks", type=int, default=4)
    parser.add_argument("--test-tasks", type=int, default=6)
    parser.add_argument("--ood-test-tasks", type=int, default=4)
    parser.add_argument("--distill-epochs", type=int, default=15)
    parser.add_argument("--meta-iterations", type=int, default=8)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=401000)
    parser.add_argument("--control-seeds", type=int, default=2)
    parser.add_argument("--ablation-seeds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        action="append",
        help="Run only the named architecture(s). Default: both.",
    )
    parser.add_argument(
        "--teacher",
        default="norm_grad_local",
        choices=list(TEACHER_METHODS),
        help="Primary distillation teacher method.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--skip-width-ood", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
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


def build_splits(
    *,
    architectures: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[float, list[MultiTensorCase]]]:
    def mixed_split(kind: str, seed_base: int, count: int) -> dict[float, list[MultiTensorCase]]:
        cases: list[MultiTensorCase] = []
        for architecture in architectures:
            split = make_split(
                architecture,
                TRAIN_CONDITIONS if kind != "ood" else OOD_CONDITIONS,
                seed_base=seed_base + (0 if architecture == "two_layer" else 500_000),
                count=count,
                width=args.width,
                samples=args.samples,
                device=device,
            )
            cases.extend(flatten_split(split))
        return {"all": cases}

    splits = {
        "lr_validation": mixed_split("train", 411000, args.lr_validation_tasks),
        "distill": mixed_split("train", 421000, args.distill_train_tasks),
        "scale_validation": mixed_split("train", 431000, args.scale_validation_tasks),
        "meta_train": mixed_split("train", 441000, args.meta_train_tasks),
        "meta_validation": mixed_split("train", 451000, args.meta_validation_tasks),
        "test": mixed_split("train", 461000, args.test_tasks),
        "ood": mixed_split("ood", 471000, args.ood_test_tasks),
    }
    return splits


def evaluate_bundle(
    student: TinyMLPOptimizer,
    *,
    test_split: dict[float, list[MultiTensorCase]],
    ood_split: dict[float, list[MultiTensorCase]] | None,
    batch_size: int,
    steps: int,
    include_global: bool,
    reference_teacher=None,
) -> dict[str, Any]:
    test_metrics = evaluate_student_split(
        student,
        test_split,
        batch_size=batch_size,
        steps=steps,
        include_global=include_global,
        reference_teacher=reference_teacher,
    )
    payload: dict[str, Any] = {"test": test_metrics}
    if ood_split is not None:
        payload["ood"] = evaluate_student_split(
            student,
            ood_split,
            batch_size=batch_size,
            steps=steps,
            include_global=include_global,
            reference_teacher=reference_teacher,
        )
    return payload


def run_normgrad_local_vs_global(
    *,
    args: argparse.Namespace,
    device: torch.device,
    distill_cases: list[MultiTensorCase],
    scale_cases: list[MultiTensorCase],
    test_split: dict[float, list[MultiTensorCase]],
    ood_split: dict[float, list[MultiTensorCase]],
) -> dict[str, Any]:
    """Paired local vs global NormGrad teacher + student comparison."""
    result: dict[str, Any] = {}
    lr_val = flatten_split(
        make_split(
            "two_layer",
            TRAIN_CONDITIONS,
            seed_base=411000,
            count=2,
            width=args.width,
            samples=args.samples,
        )
    ) + flatten_split(
        make_split(
            "residual",
            TRAIN_CONDITIONS,
            seed_base=611000,
            count=2,
            width=args.width,
            samples=args.samples,
        )
    )
    for method in ("norm_grad_local", "norm_grad_global"):
        lr, score = tune_teacher_lr(
            method,
            lr_val,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        teacher = evaluate_teacher_split(
            method,
            lr,
            test_split,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        records = collect_records(
            make_teacher(method, lr),
            distill_cases,
            batch_size=args.train_batch_size,
            steps=args.steps,
            teacher_name=method,
        )
        students = []
        n_seeds = min(2, args.student_seeds)
        for seed_index in range(n_seeds):
            seed = args.student_seed + 10_000 + seed_index
            student, train_loss = train_supervised_student(
                records,
                device=device,
                seed=seed,
                epochs=args.distill_epochs,
                weights=JOINT_WEIGHTS,
            )
            scale, _ = select_student_scale(
                student,
                scale_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
            )
            bundle = evaluate_bundle(
                student,
                test_split=test_split,
                ood_split=None if args.skip_ood else ood_split,
                batch_size=args.train_batch_size,
                steps=args.steps,
                include_global=True,
                reference_teacher=make_teacher(method, lr),
            )
            students.append(
                {
                    "seed": seed,
                    "scale": scale,
                    "train_loss": train_loss,
                    "test_mean": bundle["test"]["loss_ratio"]["mean"],
                    "bundle": bundle,
                }
            )
        result[method] = {
            "tuned_lr": lr,
            "validation_score": score,
            "teacher_test": teacher,
            "student_seeds": students,
            "student_test_summary": summarize_ratios([s["test_mean"] for s in students]).to_dict(),
        }
    return result


def run_negative_controls(
    *,
    args: argparse.Namespace,
    device: torch.device,
    distill_cases: list[MultiTensorCase],
    scale_cases: list[MultiTensorCase],
    test_split: dict[float, list[MultiTensorCase]],
    teacher_name: str,
    teacher_lr: float,
    analytic_control_name: str,
    analytic_control_lr: float,
) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    for label_control in LABEL_CONTROLS:
        ratios = []
        for seed_index in range(args.control_seeds):
            seed = args.student_seed + 200_000 + seed_index
            records = collect_records(
                make_teacher(teacher_name, teacher_lr),
                distill_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
                teacher_name=teacher_name,
                label_control=label_control,
                control_seed=seed + seed_index,
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
                scale_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
            )
            eval_payload = evaluate_student_split(
                student,
                test_split,
                batch_size=args.train_batch_size,
                steps=args.steps,
            )
            ratios.append(eval_payload["loss_ratio"]["mean"])
        controls[label_control] = {
            "test": summarize_ratios(ratios).to_dict(),
            "seeds": args.control_seeds,
        }

    # Analytic imitation: train student directly on the analytic rule generator trajectory
    # (not the primary teacher trajectory).
    ratios = []
    for seed_index in range(args.control_seeds):
        seed = args.student_seed + 250_000 + seed_index
        records = collect_records(
            make_teacher(analytic_control_name, analytic_control_lr),
            distill_cases,
            batch_size=args.train_batch_size,
            steps=args.steps,
            teacher_name=analytic_control_name,
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
            scale_cases,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        eval_payload = evaluate_student_split(
            student,
            test_split,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        ratios.append(eval_payload["loss_ratio"]["mean"])
    controls["analytic_imitation"] = {
        "source_teacher": analytic_control_name,
        "source_lr": analytic_control_lr,
        "test": summarize_ratios(ratios).to_dict(),
        "seeds": args.control_seeds,
        "interpretation": (
            "If this matches distill_joint, the result is NormGrad rule compression "
            "rather than teacher-trajectory-specific knowledge."
        ),
    }
    return controls


def run_feature_ablation(
    *,
    args: argparse.Namespace,
    device: torch.device,
    distill_cases: list[MultiTensorCase],
    scale_cases: list[MultiTensorCase],
    test_split: dict[float, list[MultiTensorCase]],
    teacher_name: str,
    teacher_lr: float,
) -> dict[str, Any]:
    ablations: dict[str, Any] = {}
    for mode in FEATURE_MODES:
        include_global = mode == "local_global"
        ratios = []
        for seed_index in range(args.ablation_seeds):
            seed = args.student_seed + 300_000 + seed_index
            records = collect_records(
                make_teacher(teacher_name, teacher_lr),
                distill_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
                include_global=include_global,
                teacher_name=teacher_name,
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
                scale_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
                include_global=include_global,
            )
            eval_payload = evaluate_student_split(
                student,
                test_split,
                batch_size=args.train_batch_size,
                steps=args.steps,
                include_global=include_global,
            )
            ratios.append(eval_payload["loss_ratio"]["mean"])
        ablations[mode] = {
            "include_global": include_global,
            "test": summarize_ratios(ratios).to_dict(),
            "seeds": args.ablation_seeds,
        }
    return ablations


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    architectures = args.architecture or list(ARCHITECTURES)
    splits = build_splits(architectures=architectures, args=args, device=device)
    lr_cases = flatten_split(splits["lr_validation"])
    distill_cases = flatten_split(splits["distill"])
    scale_cases = flatten_split(splits["scale_validation"])
    meta_train_cases = flatten_split(splits["meta_train"])
    meta_validation_cases = flatten_split(splits["meta_validation"])
    test_split = splits["test"]
    ood_split = splits["ood"]

    print("[mt] tuning analytic teacher LRs", flush=True)
    tuned_lrs: dict[str, float] = {}
    for method in TEACHER_METHODS:
        lr, _ = tune_teacher_lr(
            method,
            lr_cases,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        tuned_lrs[method] = lr

    analytic_test = {
        method: evaluate_teacher_split(
            method,
            lr,
            test_split,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        for method, lr in tuned_lrs.items()
    }
    analytic_ood = None
    if not args.skip_ood:
        analytic_ood = {
            method: evaluate_teacher_split(
                method,
                lr,
                ood_split,
                batch_size=args.train_batch_size,
                steps=args.steps,
            )
            for method, lr in tuned_lrs.items()
        }

    teacher_name = args.teacher
    teacher_lr = tuned_lrs[teacher_name]
    strongest_analytic = min(
        analytic_test,
        key=lambda name: analytic_test[name]["loss_ratio"]["mean"],
    )

    print(f"[mt] collecting teacher trajectories for {teacher_name}@lr={teacher_lr}", flush=True)
    joint_records = collect_records(
        make_teacher(teacher_name, teacher_lr),
        distill_cases,
        batch_size=args.train_batch_size,
        steps=args.steps,
        teacher_name=teacher_name,
    )
    direction_records = list(joint_records)

    runs: list[dict[str, Any]] = []
    per_seed_test: dict[str, list[float]] = {
        "distill_joint": [],
        "distill_direction": [],
        "distill_joint_meta": [],
        "direct_meta": [],
    }
    per_seed_ood: dict[str, list[float]] = {key: [] for key in per_seed_test}

    batch_transfer = None
    width_ood = None

    for seed_index in range(args.student_seeds):
        seed = args.student_seed + seed_index
        print(f"[mt] student seed {seed} ({seed_index + 1}/{args.student_seeds})", flush=True)

        joint_student, joint_loss = train_supervised_student(
            joint_records,
            device=device,
            seed=seed,
            epochs=args.distill_epochs,
            weights=JOINT_WEIGHTS,
        )
        joint_scale, _ = select_student_scale(
            joint_student,
            scale_cases,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        joint_eval = evaluate_bundle(
            joint_student,
            test_split=test_split,
            ood_split=None if args.skip_ood else ood_split,
            batch_size=args.train_batch_size,
            steps=args.steps,
            include_global=True,
            reference_teacher=make_teacher(teacher_name, teacher_lr),
        )
        runs.append(
            {
                "mode": "distill_joint",
                "seed": seed,
                "student_parameters": joint_student.parameter_count,
                "validation_scale": joint_scale,
                "train_distillation_loss": joint_loss,
                "test": joint_eval["test"],
                "ood": joint_eval.get("ood"),
            }
        )
        per_seed_test["distill_joint"].append(joint_eval["test"]["loss_ratio"]["mean"])
        if "ood" in joint_eval:
            per_seed_ood["distill_joint"].append(joint_eval["ood"]["loss_ratio"]["mean"])

        direction_student, direction_loss = train_supervised_student(
            direction_records,
            device=device,
            seed=seed + 50_000,
            epochs=args.distill_epochs,
            weights=DIRECTION_ONLY_WEIGHTS,
        )
        direction_scale, _ = select_student_scale(
            direction_student,
            scale_cases,
            batch_size=args.train_batch_size,
            steps=args.steps,
        )
        direction_eval = evaluate_bundle(
            direction_student,
            test_split=test_split,
            ood_split=None if args.skip_ood else ood_split,
            batch_size=args.train_batch_size,
            steps=args.steps,
            include_global=True,
            reference_teacher=make_teacher(teacher_name, teacher_lr),
        )
        runs.append(
            {
                "mode": "distill_direction",
                "seed": seed,
                "student_parameters": direction_student.parameter_count,
                "validation_scale": direction_scale,
                "train_distillation_loss": direction_loss,
                "test": direction_eval["test"],
                "ood": direction_eval.get("ood"),
            }
        )
        per_seed_test["distill_direction"].append(direction_eval["test"]["loss_ratio"]["mean"])
        if "ood" in direction_eval:
            per_seed_ood["distill_direction"].append(direction_eval["ood"]["loss_ratio"]["mean"])

        if not args.skip_meta:
            meta_state = {
                name: value.detach().clone()
                for name, value in joint_student.state_dict().items()
            }
            meta_student, outer_lr, meta_val = select_outer_lr_and_train(
                meta_state,
                device=device,
                train_cases=meta_train_cases,
                validation_cases=meta_validation_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
                iterations=args.meta_iterations,
            )
            meta_eval = evaluate_bundle(
                meta_student,
                test_split=test_split,
                ood_split=None if args.skip_ood else ood_split,
                batch_size=args.train_batch_size,
                steps=args.steps,
                include_global=True,
                reference_teacher=make_teacher(teacher_name, teacher_lr),
            )
            runs.append(
                {
                    "mode": "distill_joint_meta",
                    "seed": seed,
                    "student_parameters": meta_student.parameter_count,
                    "selected_outer_lr": outer_lr,
                    "meta_validation_loss_ratio": meta_val,
                    "test": meta_eval["test"],
                    "ood": meta_eval.get("ood"),
                }
            )
            per_seed_test["distill_joint_meta"].append(meta_eval["test"]["loss_ratio"]["mean"])
            if "ood" in meta_eval:
                per_seed_ood["distill_joint_meta"].append(meta_eval["ood"]["loss_ratio"]["mean"])

            direct_student, direct_lr, direct_val = train_direct_meta_student(
                meta_train_cases,
                meta_validation_cases,
                device=device,
                batch_size=args.train_batch_size,
                steps=args.steps,
                iterations=args.meta_iterations,
                seed=seed + 70_000,
            )
            select_student_scale(
                direct_student,
                scale_cases,
                batch_size=args.train_batch_size,
                steps=args.steps,
            )
            direct_eval = evaluate_bundle(
                direct_student,
                test_split=test_split,
                ood_split=None if args.skip_ood else ood_split,
                batch_size=args.train_batch_size,
                steps=args.steps,
                include_global=True,
                reference_teacher=make_teacher(teacher_name, teacher_lr),
            )
            runs.append(
                {
                    "mode": "direct_meta",
                    "seed": seed,
                    "student_parameters": direct_student.parameter_count,
                    "selected_outer_lr": direct_lr,
                    "meta_validation_loss_ratio": direct_val,
                    "test": direct_eval["test"],
                    "ood": direct_eval.get("ood"),
                }
            )
            per_seed_test["direct_meta"].append(direct_eval["test"]["loss_ratio"]["mean"])
            if "ood" in direct_eval:
                per_seed_ood["direct_meta"].append(direct_eval["ood"]["loss_ratio"]["mean"])

        if seed_index == 0:
            batch_transfer = batch_transfer_matrix(
                joint_student,
                test_split,
                train_batch_size=args.train_batch_size,
                eval_batch_sizes=(4, 8, 16, 32),
                steps=args.steps,
            )
            if not args.skip_width_ood:
                width_ood = width_transfer_matrix(
                    joint_student,
                    architecture=architectures[0],
                    widths=(args.width, args.width + 4, args.width * 2),
                    seed_base=990000,
                    count=max(2, args.test_tasks // 2),
                    samples=args.samples,
                    batch_size=args.train_batch_size,
                    steps=args.steps,
                    device=device,
                )

    print("[mt] local vs global NormGrad", flush=True)
    normgrad_compare = run_normgrad_local_vs_global(
        args=args,
        device=device,
        distill_cases=distill_cases,
        scale_cases=scale_cases,
        test_split=test_split,
        ood_split=ood_split,
    )

    controls: dict[str, Any] = {}
    if not args.skip_controls:
        print("[mt] negative controls", flush=True)
        analytic_control_name = "norm_grad_global" if teacher_name == "norm_grad_local" else "norm_grad_local"
        controls = run_negative_controls(
            args=args,
            device=device,
            distill_cases=distill_cases,
            scale_cases=scale_cases,
            test_split=test_split,
            teacher_name=teacher_name,
            teacher_lr=teacher_lr,
            analytic_control_name=analytic_control_name,
            analytic_control_lr=tuned_lrs[analytic_control_name],
        )

    ablations: dict[str, Any] = {}
    if not args.skip_ablations:
        print("[mt] feature ablations", flush=True)
        ablations = run_feature_ablation(
            args=args,
            device=device,
            distill_cases=distill_cases,
            scale_cases=scale_cases,
            test_split=test_split,
            teacher_name=teacher_name,
            teacher_lr=teacher_lr,
        )

    # Shared batch-index check across methods on one case.
    sample_case = distill_cases[0]
    batch_a = batch_sequence(sample_case, batch_size=args.train_batch_size, steps=5)
    batch_b = batch_sequence(sample_case, batch_size=args.train_batch_size, steps=5)
    shared_batch_ok = all(torch.equal(a, b) for a, b in zip(batch_a, batch_b, strict=True))

    summary = {
        "teacher_method": teacher_name,
        "teacher_lr": teacher_lr,
        "architectures": architectures,
        "student_parameters": TinyMLPOptimizer().parameter_count,
        "strongest_analytic": strongest_analytic,
        "strongest_analytic_test_mean": analytic_test[strongest_analytic]["loss_ratio"]["mean"],
        "teacher_test_mean": analytic_test[teacher_name]["loss_ratio"]["mean"],
        "per_seed_test": {
            mode: summarize_ratios(values).to_dict() if values else None
            for mode, values in per_seed_test.items()
        },
        "per_seed_ood": {
            mode: summarize_ratios(values).to_dict() if values else None
            for mode, values in per_seed_ood.items()
        },
        "paired_joint_vs_direct_meta": (
            paired_differences(per_seed_test["distill_joint"], per_seed_test["direct_meta"])
            if per_seed_test["direct_meta"]
            else None
        ),
        "shared_batch_index_sequence": shared_batch_ok,
    }

    payload = {
        "run_id": f"multitensor-{uuid.uuid4().hex[:8]}",
        "config": {
            "width": args.width,
            "samples": args.samples,
            "steps": args.steps,
            "train_batch_size": args.train_batch_size,
            "student_seeds": args.student_seeds,
            "student_seed_base": args.student_seed,
            "distill_epochs": args.distill_epochs,
            "meta_iterations": args.meta_iterations,
            "teacher": teacher_name,
            "architectures": architectures,
            "quick": args.quick,
        },
        "split_specs": {
            "teacher_lr_validation_seed_base": 411000,
            "distill_train_seed_base": 421000,
            "scale_validation_seed_base": 431000,
            "meta_train_seed_base": 441000,
            "meta_validation_seed_base": 451000,
            "iid_test_seed_base": 461000,
            "ood_test_seed_base": 471000,
            "train_conditions": list(TRAIN_CONDITIONS),
            "ood_conditions": list(OOD_CONDITIONS),
        },
        "tuned_lrs": tuned_lrs,
        "analytic_test": analytic_test,
        "analytic_ood": analytic_ood,
        "runs": runs,
        "batch_transfer": batch_transfer,
        "width_ood": width_ood,
        "normgrad_local_vs_global": normgrad_compare,
        "negative_controls": controls,
        "feature_ablations": ablations,
        "summary": summary,
    }
    meta = artifact_metadata(
        run_id=payload["run_id"],
        commit_sha=git_commit_sha(),
        config=payload["config"],
        split_specs=payload["split_specs"],
    )
    payload.update(meta)
    return payload


def main() -> None:
    args = parse_args()
    apply_quick(args)
    payload = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    summary = payload["summary"]
    print("[mt] done", flush=True)
    print(
        f"  teacher={summary['teacher_method']} test_mean={summary['teacher_test_mean']:.4f} "
        f"strongest={summary['strongest_analytic']} "
        f"({summary['strongest_analytic_test_mean']:.4f})",
        flush=True,
    )
    for mode, stats in summary["per_seed_test"].items():
        if stats:
            print(
                f"  {mode}: mean={stats['mean']:.4f} median={stats['median']:.4f} "
                f"finite={stats['finite_fraction']:.2f}",
                flush=True,
            )
    print(f"  wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
