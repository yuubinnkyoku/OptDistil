"""Exact multi-tensor reparameterization stress benchmark.

Answers whether the 153-param Student is explained by fixed tensor-role LRs
or truly infers state-dependent adaptive scales under p_i = s_i * theta_i.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import uuid
from pathlib import Path
from typing import Any

import torch

from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.multitensor.effective_scale import (
    analyze_effective_scales,
    collect_effective_scale_trace,
)
from optdistil.multitensor.features import build_multitensor_features, concatenate_features
from optdistil.multitensor.methods import (
    MethodSpec,
    estimate_role_scales_from_student,
    make_ordinary_method,
    make_static_arch_method,
    make_static_shared_method,
)
from optdistil.multitensor.reparam import (
    OOD_SCALE_RANGE,
    STRONG_OOD_SCALE_RANGE,
    TRAIN_SCALE_RANGE,
    PrivilegedFunctionSpaceNormGrad,
    make_reparameterized_case,
    sample_log_uniform_scales,
)
from optdistil.multitensor.static_role import (
    expand_role_lrs,
    roles_for_case,
    tune_architecture_tensor_lrs,
    tune_shared_role_lrs,
)
from optdistil.multitensor.stats_utils import paired_comparison
from optdistil.multitensor.stochastic import (
    MultiTensorCase,
    _observe_ema,
    artifact_metadata,
    batch_sequence,
    flatten_split,
    git_commit_sha,
    make_split,
    select_student_scale,
    train_supervised_student,
    tune_teacher_lr,
)
from optdistil.multitensor.structured import (
    evaluate_structured_split,
    fit_structured_from_privileged,
    rollout_structured,
)
from optdistil.students.tiny_mlp import TinyMLPOptimizer

ARCHITECTURES = ("two_layer", "residual")
JOINT_WEIGHTS = DistillationLossWeights(direction=0.7, magnitude=0.3)
DIRECTION_ONLY_WEIGHTS = DistillationLossWeights(direction=1.0, magnitude=0.0)

# Seed bases for reparameterization stress protocol.
SCALE_VALIDATION_SEED = 511000
DISTILL_SEED = 521000
ROLE_VALIDATION_SEED = 531000
IID_TEST_SEED = 561000
OOD_TEST_SEED = 571000
STRONG_OOD_SEED = 581000
REPARAM_SCALE_SEED_BASE = 601000
STRUCTURED_SEED = 611000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reparameterization stress benchmark.")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scale-validation-tasks", type=int, default=3)
    parser.add_argument("--distill-train-tasks", type=int, default=4)
    parser.add_argument("--role-validation-tasks", type=int, default=3)
    parser.add_argument("--iid-test-tasks", type=int, default=4)
    parser.add_argument("--ood-test-tasks", type=int, default=3)
    parser.add_argument("--distill-epochs", type=int, default=12)
    parser.add_argument("--student-seeds", type=int, default=5)
    parser.add_argument("--student-seed", type=int, default=401000)
    parser.add_argument("--structured-epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-structured", action="store_true")
    parser.add_argument("--skip-experiment-b", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 12
    args.scale_validation_tasks = 2
    args.distill_train_tasks = 2
    args.role_validation_tasks = 2
    args.iid_test_tasks = 2
    args.ood_test_tasks = 2
    args.distill_epochs = 5
    args.student_seeds = 2
    args.structured_epochs = 8


def mixed_base_split(
    *,
    architectures: tuple[str, ...],
    conditions: tuple[float, ...],
    seed_base: int,
    count: int,
    width: int,
    samples: int,
    device: torch.device,
) -> list[MultiTensorCase]:
    cases: list[MultiTensorCase] = []
    for architecture in architectures:
        split = make_split(
            architecture,
            conditions,
            seed_base=seed_base + (0 if architecture == "two_layer" else 500_000),
            count=count,
            width=width,
            samples=samples,
            device=device,
        )
        cases.extend(flatten_split(split))
    return cases


def reparameterize_cases(
    cases: list[MultiTensorCase],
    *,
    scale_range: tuple[float, float],
    seed_base: int,
) -> tuple[list[MultiTensorCase], list[list[float]]]:
    reparam_cases: list[MultiTensorCase] = []
    scales_list: list[list[float]] = []
    for index, case in enumerate(cases):
        scales = sample_log_uniform_scales(
            len(case.initial),
            low=scale_range[0],
            high=scale_range[1],
            seed=seed_base + index,
        )
        reparam_case, values = make_reparameterized_case(case, scales=scales)
        reparam_cases.append(reparam_case)
        scales_list.append(values)
    return reparam_cases, scales_list


@torch.no_grad()
def collect_privileged_records(
    case: MultiTensorCase,
    scales: list[float],
    *,
    lr: float,
    batch_size: int,
    steps: int,
) -> list[TrajectoryRecord]:
    teacher = PrivilegedFunctionSpaceNormGrad(lr=lr, scales=scales)
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    records: list[TrajectoryRecord] = []
    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)
        features = concatenate_features(
            build_multitensor_features(
                params.tensors,
                grads,
                momentums,
                second_moments,
                step=step,
                total_steps=steps,
                include_global=True,
            )
        )
        updates = teacher.step(params, grads)
        update_flat = torch.cat([u.reshape(-1) for u in updates])
        records.append(
            TrajectoryRecord(
                features=features.detach(),
                teacher_update=update_flat.detach(),
                metadata={
                    "step": step,
                    "architecture": case.architecture,
                    "scales": list(scales),
                },
            )
        )
        params = params.add(updates)
    return records


def make_frozen_method(role_lrs: dict[str, float], case: MultiTensorCase) -> MethodSpec:
    from optdistil.multitensor.methods import make_frozen_role_method

    return make_frozen_role_method(role_lrs, case, source="student")


def _rollout(spec: MethodSpec, case: MultiTensorCase, *, batch_size: int, steps: int):
    from optdistil.multitensor.methods import rollout_method

    return rollout_method(spec, case, batch_size=batch_size, steps=steps)


def tune_baseline_lrs(
    validation_cases: list[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ordinary_lr, ordinary_score = tune_teacher_lr(
        "norm_grad_local", validation_cases, batch_size=batch_size, steps=steps
    )
    privileged_lr, privileged_score = tune_teacher_lr(
        "norm_grad_local", validation_cases, batch_size=batch_size, steps=steps
    )
    role_lrs, role_score = tune_shared_role_lrs(
        validation_cases, batch_size=batch_size, steps=steps
    )
    arch_lrs: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        subset = [c for c in validation_cases if c.architecture == architecture]
        if not subset:
            continue
        init = expand_role_lrs(role_lrs, roles_for_case(subset[0]))
        tensor_lrs, score = tune_architecture_tensor_lrs(
            subset,
            batch_size=batch_size,
            steps=steps,
            init_scales=init,
            rounds=1 if len(validation_cases) <= 4 else 2,
        )
        arch_lrs[architecture] = {"tensor_lrs": tensor_lrs, "score": score}
    return {
        "ordinary_lr": ordinary_lr,
        "ordinary_score": ordinary_score,
        "privileged_lr": privileged_lr,
        "privileged_score": privileged_score,
        "shared_role_lrs": role_lrs,
        "shared_role_score": role_score,
        "arch_tensor_lrs": arch_lrs,
    }


def arch_lrs_for_case(
    arch_tensor_lrs: dict[str, dict[str, Any]],
    case: MultiTensorCase,
    shared_role_lrs: dict[str, float],
) -> list[float]:
    if case.architecture in arch_tensor_lrs:
        return list(arch_tensor_lrs[case.architecture]["tensor_lrs"])
    return expand_role_lrs(shared_role_lrs, roles_for_case(case))


def train_students_on_privileged(
    distill_cases: list[MultiTensorCase],
    distill_scales: list[list[float]],
    *,
    privileged_lr: float,
    batch_size: int,
    steps: int,
    epochs: int,
    device: torch.device,
    seed_base: int,
    n_seeds: int,
    weights: DistillationLossWeights,
) -> list[TinyMLPOptimizer]:
    records: list[TrajectoryRecord] = []
    for case, scales in zip(distill_cases, distill_scales, strict=True):
        records.extend(
            collect_privileged_records(
                case,
                scales,
                lr=privileged_lr,
                batch_size=batch_size,
                steps=steps,
            )
        )
    students: list[TinyMLPOptimizer] = []
    for seed_offset in range(n_seeds):
        student, _loss = train_supervised_student(
            records,
            device=device,
            seed=seed_base + seed_offset,
            epochs=epochs,
            weights=weights,
        )
        students.append(student)
    return students


def experiment_b_static_role_transfer(
    *,
    device: torch.device,
    args: argparse.Namespace,
    base_role_lrs: dict[str, float],
) -> dict[str, Any]:
    """Transfer frozen static role LRs across batch/width/arch/condition shifts."""
    transfers: dict[str, Any] = {}

    def eval_transfer(
        name: str,
        cases: list[MultiTensorCase],
        scales_list: list[list[float]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for case, scales in zip(cases, scales_list, strict=True):
            spec = make_static_shared_method(base_role_lrs, case)
            ratio, aulc, finite = _rollout(spec, case, batch_size=args.batch_size, steps=args.steps)
            results.setdefault("ratios", []).append(ratio)
            results.setdefault("aulcs", []).append(aulc)
            results.setdefault("finite", []).append(finite)
            results.setdefault("scales", []).append(scales)
        from optdistil.multitensor.stats_utils import summarize_method_values

        return {
            "loss_ratio": summarize_method_values(results["ratios"]),
            "aulc": summarize_method_values(results["aulcs"]),
            "n": len(results["ratios"]),
        }

    # Batch transfer
    for batch in (4, 16, 32):
        cases = mixed_base_split(
            architectures=("two_layer", "residual"),
            conditions=(30.0, 300.0),
            seed_base=ROLE_VALIDATION_SEED + 90_000,
            count=args.role_validation_tasks,
            width=args.width,
            samples=args.samples,
            device=device,
        )
        reparam, scales_list = reparameterize_cases(
            cases, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + batch
        )
        # Re-evaluate at this batch size.
        ratios: list[float] = []
        for case in reparam:
            spec = make_static_shared_method(base_role_lrs, case)
            ratio, _aulc, _ = _rollout(spec, case, batch_size=batch, steps=args.steps)
            ratios.append(ratio)
        from optdistil.multitensor.stats_utils import summarize_method_values

        transfers[f"batch_{batch}"] = {
            "loss_ratio": summarize_method_values(ratios),
            "batch_size": batch,
        }
        _ = scales_list

    # Width transfer
    for width in (12, 16, 24):
        cases = mixed_base_split(
            architectures=("two_layer", "residual"),
            conditions=(30.0, 300.0),
            seed_base=ROLE_VALIDATION_SEED + 120_000 + width,
            count=max(1, args.ood_test_tasks - 1),
            width=width,
            samples=args.samples,
            device=device,
        )
        reparam, scales_list = reparameterize_cases(
            cases, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 1000 + width
        )
        transfers[f"width_{width}"] = eval_transfer(f"width_{width}", reparam, scales_list)

    # Architecture transfer: tune already done on mixed; evaluate pure arch subsets.
    for architecture in ARCHITECTURES:
        cases = mixed_base_split(
            architectures=(architecture,),
            conditions=(30.0, 300.0),
            seed_base=ROLE_VALIDATION_SEED + 200_000,
            count=args.ood_test_tasks,
            width=args.width,
            samples=args.samples,
            device=device,
        )
        reparam, scales_list = reparameterize_cases(
            cases, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 2000
        )
        transfers[f"arch_{architecture}"] = eval_transfer(
            f"arch_{architecture}", reparam, scales_list
        )

    # Also tune residual-only and transfer to two_layer (and reverse).
    residual_cases = mixed_base_split(
        architectures=("residual",),
        conditions=(30.0, 300.0),
        seed_base=ROLE_VALIDATION_SEED + 300_000,
        count=args.role_validation_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    residual_reparam, _ = reparameterize_cases(
        residual_cases, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 3000
    )
    residual_role_lrs, residual_score = tune_shared_role_lrs(
        residual_reparam, batch_size=args.batch_size, steps=args.steps
    )
    two_layer_cases = mixed_base_split(
        architectures=("two_layer",),
        conditions=(30.0, 300.0),
        seed_base=ROLE_VALIDATION_SEED + 310_000,
        count=args.ood_test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    two_reparam, two_scales = reparameterize_cases(
        two_layer_cases, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 3100
    )
    transfers["residual_tuned_to_two_layer"] = eval_transfer(
        "residual_tuned_to_two_layer", two_reparam, two_scales
    )
    transfers["residual_tuned_lrs"] = {
        "role_lrs": residual_role_lrs,
        "validation_score": residual_score,
    }

    # Condition OOD
    condition_cases = mixed_base_split(
        architectures=("two_layer", "residual"),
        conditions=(10.0, 100.0, 1000.0, 3000.0),
        seed_base=ROLE_VALIDATION_SEED + 400_000,
        count=max(1, args.ood_test_tasks - 1),
        width=args.width,
        samples=args.samples,
        device=device,
    )
    condition_reparam, condition_scales = reparameterize_cases(
        condition_cases, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 4000
    )
    transfers["conditions_ood"] = eval_transfer(
        "conditions_ood", condition_reparam, condition_scales
    )

    transfers["base_role_lrs"] = base_role_lrs
    return transfers


def interpret_results(
    experiment_a: dict[str, Any],
    experiment_c: dict[str, Any] | None,
    experiment_d: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map results onto claims A/B/C/D using pre-registered decision rules."""
    methods = experiment_a.get("methods", {})

    def mean_of(name: str, split_key: str = "iid") -> float:
        # method names may be seeded; find prefix matches
        values = []
        for key, payload in methods.items():
            if key == name or key.startswith(name):
                values.append(payload["loss_ratio"]["mean"])
        if not values:
            return math.inf
        return statistics.fmean(values)

    student_key_prefix = "student_153p"
    student_means = [
        payload["loss_ratio"]["mean"]
        for key, payload in methods.items()
        if key.startswith(student_key_prefix)
    ]
    static_mean = mean_of("static_shared_role_normgrad")
    ordinary_mean = mean_of("ordinary_local_normgrad")
    privileged_mean = mean_of("privileged_function_space_normgrad")
    student_mean = statistics.fmean(student_means) if student_means else math.inf

    # OOD section stored separately under experiment_a["ood"]
    ood = experiment_a.get("ood", {})
    ood_methods = ood.get("methods", {})
    ood_student_means = [
        payload["loss_ratio"]["mean"]
        for key, payload in ood_methods.items()
        if key.startswith(student_key_prefix)
    ]
    ood_static = ood_methods.get("static_shared_role_normgrad", {}).get("loss_ratio", {}).get(
        "mean", math.inf
    )
    ood_student = statistics.fmean(ood_student_means) if ood_student_means else math.inf
    strong = experiment_a.get("strong_ood", {})
    strong_methods = strong.get("methods", {})
    strong_student_means = [
        payload["loss_ratio"]["mean"]
        for key, payload in strong_methods.items()
        if key.startswith(student_key_prefix)
    ]
    strong_static = strong_methods.get("static_shared_role_normgrad", {}).get("loss_ratio", {}).get(
        "mean", math.inf
    )
    strong_student = (
        statistics.fmean(strong_student_means) if strong_student_means else math.inf
    )

    structured_global = None
    structured_per = None
    if experiment_d:
        structured_global = experiment_d.get("global", {}).get("loss_ratio", {}).get("mean")
        structured_per = experiment_d.get("per_tensor", {}).get("loss_ratio", {}).get("mean")

    # Decision rules
    # A: fixed role LR is sufficient if Student does NOT clearly beat static on IID or OOD.
    student_beats_static_iid = math.isfinite(student_mean) and math.isfinite(static_mean) and (
        student_mean < 0.95 * static_mean
    )
    student_beats_static_ood = math.isfinite(ood_student) and math.isfinite(ood_static) and (
        ood_student < 0.95 * ood_static
    )
    student_beats_static_strong = math.isfinite(strong_student) and math.isfinite(strong_static) and (
        strong_student < 0.95 * strong_static
    )

    structured_matches_student = False
    if (
        experiment_d
        and structured_per is not None
        and math.isfinite(structured_per)
        and math.isfinite(student_mean)
    ):
        structured_matches_student = structured_per <= 1.10 * student_mean
    structured_matches_student_global = False
    if (
        experiment_d
        and structured_global is not None
        and math.isfinite(structured_global)
        and math.isfinite(student_mean)
    ):
        structured_matches_student_global = structured_global <= 1.10 * student_mean
    structured_beats_static = False
    if (
        experiment_d
        and structured_per is not None
        and math.isfinite(structured_per)
        and math.isfinite(static_mean)
    ):
        structured_beats_static = structured_per < 0.95 * static_mean

    # Hidden-scale signal from Experiment C
    hidden_signal = False
    hidden_delta = None
    if experiment_c:
        hidden_delta = experiment_c.get("hidden_delta_r2")
        hidden_corr = (
            experiment_c.get("correlations", {}).get("hidden_scale", {}).get("spearman")
        )
        if hidden_delta is not None and hidden_delta > 0.05:
            hidden_signal = True
        if hidden_corr is not None and math.isfinite(hidden_corr) and abs(hidden_corr) > 0.5:
            hidden_signal = True

    claims: dict[str, bool | str] = {}
    # A: fixed tensor-role LR is sufficient when Student/structured do not beat it.
    claims["A_fixed_tensor_role_lr_sufficient"] = bool(
        (not student_beats_static_iid)
        and (not student_beats_static_ood)
        and (not student_beats_static_strong)
        and (not structured_beats_static)
    )
    # B: dynamic tensor-wise scale adaptation is necessary
    claims["B_dynamic_tensorwise_scale_necessary"] = bool(
        student_beats_static_ood or student_beats_static_strong or structured_beats_static
    )
    # C: within-tensor direction adaptation is necessary
    projected_means = [
        payload["loss_ratio"]["mean"]
        for key, payload in methods.items()
        if key.startswith("student_projected")
    ]
    projected_mean = statistics.fmean(projected_means) if projected_means else math.inf
    claims["C_within_tensor_direction_necessary"] = bool(
        math.isfinite(student_mean)
        and math.isfinite(projected_mean)
        and student_mean < 0.95 * projected_mean
        and student_beats_static_iid
    )
    # D: <50-param structured optimizer captures useful behavior
    claims["D_structured_under_50p_captures"] = bool(
        structured_matches_student or structured_matches_student_global
    )

    recommended = None
    # Prefer the minimal sufficient explanation.
    # A wins when static-role already covers the behavior (student/structured do not beat it).
    # D is deployment-relevant only when structured matches a competitive student.
    structured_competitive = False
    if (
        experiment_d
        and structured_per is not None
        and math.isfinite(structured_per)
        and math.isfinite(static_mean)
    ):
        structured_competitive = structured_per <= 1.10 * static_mean
    student_competitive = math.isfinite(student_mean) and math.isfinite(static_mean) and (
        student_mean <= 1.10 * static_mean
    )
    if claims["A_fixed_tensor_role_lr_sufficient"] and structured_competitive and student_competitive:
        recommended = "A" if not claims["D_structured_under_50p_captures"] else "D"
    elif claims["D_structured_under_50p_captures"] and student_competitive:
        recommended = "D"
    elif claims["B_dynamic_tensorwise_scale_necessary"]:
        recommended = "B"
    elif claims["C_within_tensor_direction_necessary"]:
        recommended = "C"
    elif claims["A_fixed_tensor_role_lr_sufficient"]:
        recommended = "A"
    else:
        recommended = "inconclusive"

    return {
        "claims": claims,
        "recommended_primary": recommended,
        "key_numbers": {
            "iid_static_shared_mean": static_mean,
            "iid_student_mean": student_mean,
            "iid_ordinary_mean": ordinary_mean,
            "iid_privileged_mean": privileged_mean,
            "iid_student_projected_mean": projected_mean,
            "iid_student_beats_static": student_beats_static_iid,
            "ood_static_shared_mean": ood_static,
            "ood_student_mean": ood_student,
            "strong_ood_static_shared_mean": strong_static,
            "strong_ood_student_mean": strong_student,
            "structured_global_mean": structured_global,
            "structured_per_tensor_mean": structured_per,
            "hidden_delta_r2": hidden_delta,
            "hidden_signal": hidden_signal,
        },
        "notes": [
            "A holds if Student matches static shared-role on IID and does not clearly beat it on OOD.",
            "B is supported if Student clearly beats static-role on unseen reparameterization OOD.",
            "C is supported if full Student beats its NormGrad-direction projection.",
            "D is supported if a <50p structured controller reaches within 10% of the 153p Student.",
            "No test/OOD information was used for hyperparameter tuning.",
        ],
    }


def main() -> None:
    args = parse_args()
    apply_quick(args)
    device = torch.device(args.device)
    run_id = str(uuid.uuid4())
    commit_sha = git_commit_sha()

    # Base function-space splits
    scale_val_base = mixed_base_split(
        architectures=ARCHITECTURES,
        conditions=(30.0, 300.0),
        seed_base=SCALE_VALIDATION_SEED,
        count=args.scale_validation_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    distill_base = mixed_base_split(
        architectures=ARCHITECTURES,
        conditions=(30.0, 300.0),
        seed_base=DISTILL_SEED,
        count=args.distill_train_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    role_val_base = mixed_base_split(
        architectures=ARCHITECTURES,
        conditions=(30.0, 300.0),
        seed_base=ROLE_VALIDATION_SEED,
        count=args.role_validation_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    iid_base = mixed_base_split(
        architectures=ARCHITECTURES,
        conditions=(30.0, 300.0),
        seed_base=IID_TEST_SEED,
        count=args.iid_test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    ood_base = mixed_base_split(
        architectures=ARCHITECTURES,
        conditions=(30.0, 300.0),
        seed_base=OOD_TEST_SEED,
        count=args.ood_test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    strong_base = mixed_base_split(
        architectures=ARCHITECTURES,
        conditions=(30.0, 300.0),
        seed_base=STRONG_OOD_SEED,
        count=args.ood_test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )

    scale_val, _scale_val_scales = reparameterize_cases(
        scale_val_base, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE
    )
    distill, distill_scales = reparameterize_cases(
        distill_base, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 10_000
    )
    role_val, _role_val_scales = reparameterize_cases(
        role_val_base, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 20_000
    )
    iid, iid_scales = reparameterize_cases(
        iid_base, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 30_000
    )
    ood, ood_scales = reparameterize_cases(
        ood_base, scale_range=OOD_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED_BASE + 40_000
    )
    strong, strong_scales = reparameterize_cases(
        strong_base,
        scale_range=STRONG_OOD_SCALE_RANGE,
        seed_base=REPARAM_SCALE_SEED_BASE + 50_000,
    )

    print("Tuning baseline LRs on validation (train-range scales only)...")
    tuned = tune_baseline_lrs(
        role_val, batch_size=args.batch_size, steps=args.steps
    )
    ordinary_lr = tuned["ordinary_lr"]
    privileged_lr = tuned["privileged_lr"]
    shared_role_lrs = tuned["shared_role_lrs"]
    arch_tensor_lrs = tuned["arch_tensor_lrs"]

    print(f"Training {args.student_seeds} students on privileged trajectories...")
    students = train_students_on_privileged(
        distill,
        distill_scales,
        privileged_lr=privileged_lr,
        batch_size=args.batch_size,
        steps=args.steps,
        epochs=args.distill_epochs,
        device=device,
        seed_base=args.student_seed,
        n_seeds=args.student_seeds,
        weights=JOINT_WEIGHTS,
    )
    for student in students:
        select_student_scale(
            student,
            scale_val,
            batch_size=args.batch_size,
            steps=args.steps,
            include_global=True,
        )

    frozen_role_lrs = estimate_role_scales_from_student(
        students[0], role_val, batch_size=args.batch_size, steps=args.steps
    )

    structured_global = None
    structured_per = None
    experiment_d: dict[str, Any] | None = None
    if not args.skip_structured:
        print("Fitting structured tiny controllers...")
        structured_global = fit_structured_from_privileged(
            distill,
            distill_scales,
            batch_size=args.batch_size,
            steps=args.steps,
            mode="global",
            role_lrs=shared_role_lrs,
            seed=STRUCTURED_SEED,
            epochs=args.structured_epochs,
        )
        structured_per = fit_structured_from_privileged(
            distill,
            distill_scales,
            batch_size=args.batch_size,
            steps=args.steps,
            mode="per_tensor",
            role_lrs=shared_role_lrs,
            seed=STRUCTURED_SEED + 1,
            epochs=args.structured_epochs,
        )

    def run_experiment_a(
        cases: list[MultiTensorCase], scales_list: list[list[float]], tag: str
    ) -> dict[str, Any]:
        # Methods that do not depend on student seed are evaluated once;
        # student methods share the same case list.
        static_payloads: list[dict[str, Any]] = []
        student_payloads: list[dict[str, Any]] = []
        projected_payloads: list[dict[str, Any]] = []
        structured_payloads: list[dict[str, Any]] = []
        frozen_payloads: list[dict[str, Any]] = []
        ordinary_payloads: list[dict[str, Any]] = []
        privileged_payloads: list[dict[str, Any]] = []

        for case, scales in zip(cases, scales_list, strict=True):
            arch_lrs = arch_lrs_for_case(arch_tensor_lrs, case, shared_role_lrs)
            static_specs = [
                make_ordinary_method(ordinary_lr),
                MethodSpec(
                    name="privileged_function_space_normgrad",
                    kind="teacher",
                    teacher=PrivilegedFunctionSpaceNormGrad(
                        lr=privileged_lr, scales=scales
                    ),
                    uses_hidden_scales=True,
                    metadata={"lr": privileged_lr},
                ),
                make_static_shared_method(shared_role_lrs, case),
                make_static_arch_method(arch_lrs, case),
                make_frozen_method(frozen_role_lrs, case),
            ]
            for spec in static_specs:
                ratio, aulc, finite = _rollout(
                    spec, case, batch_size=args.batch_size, steps=args.steps
                )
                record = {
                    "name": spec.name,
                    "loss_ratio": ratio,
                    "aulc": aulc,
                    "finite": finite,
                    "metadata": spec.metadata,
                    "uses_hidden_scales": spec.uses_hidden_scales,
                }
                if spec.name == "ordinary_local_normgrad":
                    ordinary_payloads.append(record)
                elif spec.name == "privileged_function_space_normgrad":
                    privileged_payloads.append(record)
                elif spec.name == "static_shared_role_normgrad" or spec.name == "static_architecture_role_normgrad":
                    static_payloads.append(record)
                elif spec.name == "frozen_student_role_scales":
                    frozen_payloads.append(record)

            for student_index, student in enumerate(students):
                for name, kind in (
                    (f"student_153p_seed{student_index}", "student"),
                    (f"student_projected_seed{student_index}", "student_projected"),
                ):
                    spec = MethodSpec(
                        name=name,
                        kind=kind,
                        student=student,
                        metadata={"student_index": student_index},
                    )
                    ratio, aulc, finite = _rollout(
                        spec, case, batch_size=args.batch_size, steps=args.steps
                    )
                    record = {
                        "name": name,
                        "loss_ratio": ratio,
                        "aulc": aulc,
                        "finite": finite,
                        "student_index": student_index,
                    }
                    if kind == "student":
                        student_payloads.append(record)
                    else:
                        projected_payloads.append(record)

            if structured_global is not None and structured_per is not None:
                for label, opt in (
                    ("structured_global", structured_global),
                    ("structured_per_tensor", structured_per),
                ):
                    ratio, aulc, finite = rollout_structured(
                        opt, case, batch_size=args.batch_size, steps=args.steps
                    )
                    structured_payloads.append(
                        {
                            "name": label,
                            "loss_ratio": ratio,
                            "aulc": aulc,
                            "finite": finite,
                            "parameter_count": opt.parameter_count(),
                        }
                    )

        from optdistil.multitensor.stats_utils import summarize_method_values

        def merge_by_name(records: list[dict[str, Any]]) -> dict[str, Any]:
            by_name: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                by_name.setdefault(record["name"], []).append(record)
            out: dict[str, Any] = {}
            for name, items in by_name.items():
                ratios = [item["loss_ratio"] for item in items]
                aulcs = [item["aulc"] for item in items]
                out[name] = {
                    "name": name,
                    "loss_ratio": summarize_method_values(ratios, seed=0),
                    "aulc": summarize_method_values(aulcs, seed=1),
                    "n": len(items),
                    "uses_hidden_scales": items[0].get("uses_hidden_scales", False),
                    "metadata": items[0].get("metadata", {}),
                }
            return out

        methods_payload: dict[str, Any] = {}
        for group in (
            ordinary_payloads,
            privileged_payloads,
            static_payloads,
            frozen_payloads,
            student_payloads,
            projected_payloads,
            structured_payloads,
        ):
            methods_payload.update(merge_by_name(group))

        # Pair student seeds vs static shared
        static_vals = [
            r["loss_ratio"] for r in static_payloads if r["name"] == "static_shared_role_normgrad"
        ]
        paired: dict[str, Any] = {}
        for student_index in range(len(students)):
            student_vals = [
                r["loss_ratio"]
                for r in student_payloads
                if r["student_index"] == student_index
            ]
            paired[f"student_seed{student_index}"] = paired_comparison(
                student_vals, static_vals, seed=student_index
            )
            projected_vals = [
                r["loss_ratio"]
                for r in projected_payloads
                if r["student_index"] == student_index
            ]
            paired[f"projected_seed{student_index}"] = paired_comparison(
                projected_vals, static_vals, seed=100 + student_index
            )
            paired[f"student_vs_projected_seed{student_index}"] = paired_comparison(
                student_vals, projected_vals, seed=200 + student_index
            )

        return {
            "tag": tag,
            "n_cases": len(cases),
            "methods": methods_payload,
            "paired_vs_static_shared": paired,
        }

    print("Running Experiment A (IID)...")
    experiment_a_iid = run_experiment_a(iid, iid_scales, "iid")
    print("Running Experiment A (OOD mild)...")
    experiment_a_ood = run_experiment_a(ood, ood_scales, "ood_mild")
    print("Running Experiment A (OOD strong)...")
    experiment_a_strong = run_experiment_a(strong, strong_scales, "ood_strong")

    experiment_b = None
    if not args.skip_experiment_b:
        print("Running Experiment B (static-role transfer)...")
        experiment_b = experiment_b_static_role_transfer(
            device=device, args=args, base_role_lrs=shared_role_lrs
        )

    print("Running Experiment C (effective scale analysis)...")
    c_traces = []
    for case, scales in zip(iid, iid_scales, strict=True):
        c_traces.append(
            collect_effective_scale_trace(
                students[0],
                case,
                batch_size=args.batch_size,
                steps=args.steps,
                hidden_scales=scales,
            )
        )
    ood_traces = []
    for case, scales in zip(ood, ood_scales, strict=True):
        ood_traces.append(
            collect_effective_scale_trace(
                students[0],
                case,
                batch_size=args.batch_size,
                steps=args.steps,
                hidden_scales=scales,
            )
        )
    experiment_c_iid = analyze_effective_scales(c_traces)
    experiment_c_ood = analyze_effective_scales(ood_traces)

    if not args.skip_structured and structured_global is not None and structured_per is not None:
        print("Running Experiment D evaluation...")
        experiment_d = {
            "global": evaluate_structured_split(
                structured_global,
                {"all": iid},
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "per_tensor": evaluate_structured_split(
                structured_per,
                {"all": iid},
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "global_ood": evaluate_structured_split(
                structured_global,
                {"all": ood},
                batch_size=args.batch_size,
                steps=args.steps,
            ),
            "per_tensor_ood": evaluate_structured_split(
                structured_per,
                {"all": ood},
                batch_size=args.batch_size,
                steps=args.steps,
            ),
        }
        # Static shared-role across heterogeneous architectures (per-case scale vectors).
        static_ratios = []
        for case in iid:
            spec = make_static_shared_method(shared_role_lrs, case)
            ratio, _aulc, _ = _rollout(spec, case, batch_size=args.batch_size, steps=args.steps)
            static_ratios.append(ratio)
        from optdistil.multitensor.stats_utils import summarize_method_values

        experiment_d["static_shared"] = {
            "loss_ratio": summarize_method_values(static_ratios),
            "parameter_count": len(shared_role_lrs),
        }

    # Merge experiment A for interpretation
    experiment_a = {
        "iid": experiment_a_iid,
        "ood": experiment_a_ood,
        "strong_ood": experiment_a_strong,
        "methods": experiment_a_iid["methods"],
        "paired_vs_static_shared": experiment_a_iid["paired_vs_static_shared"],
    }
    interpretation = interpret_results(experiment_a, experiment_c_iid, experiment_d)

    config = {
        "width": args.width,
        "samples": args.samples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "scale_validation_tasks": args.scale_validation_tasks,
        "distill_train_tasks": args.distill_train_tasks,
        "role_validation_tasks": args.role_validation_tasks,
        "iid_test_tasks": args.iid_test_tasks,
        "ood_test_tasks": args.ood_test_tasks,
        "distill_epochs": args.distill_epochs,
        "student_seeds": args.student_seeds,
        "student_seed": args.student_seed,
        "structured_epochs": args.structured_epochs,
        "quick": args.quick,
        "train_scale_range": list(TRAIN_SCALE_RANGE),
        "ood_scale_range": list(OOD_SCALE_RANGE),
        "strong_ood_scale_range": list(STRONG_OOD_SCALE_RANGE),
        "loss_weights": {"direction": JOINT_WEIGHTS.direction, "magnitude": JOINT_WEIGHTS.magnitude},
        "architectures": list(ARCHITECTURES),
    }
    split_specs = {
        "scale_validation_seed": SCALE_VALIDATION_SEED,
        "distill_seed": DISTILL_SEED,
        "role_validation_seed": ROLE_VALIDATION_SEED,
        "iid_test_seed": IID_TEST_SEED,
        "ood_test_seed": OOD_TEST_SEED,
        "strong_ood_seed": STRONG_OOD_SEED,
        "reparam_scale_seed_base": REPARAM_SCALE_SEED_BASE,
        "structured_seed": STRUCTURED_SEED,
        "conditions_train": [30.0, 300.0],
    }

    payload = {
        **artifact_metadata(
            run_id=run_id,
            commit_sha=commit_sha,
            config=config,
            split_specs=split_specs,
        ),
        "tuned": tuned,
        "frozen_role_lrs": frozen_role_lrs,
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "experiment_c": {
            "iid": experiment_c_iid,
            "ood": experiment_c_ood,
        },
        "experiment_d": experiment_d,
        "interpretation": interpretation,
        "student_parameter_count": students[0].parameter_count,
        "structured_parameter_counts": {
            "global": structured_global.parameter_count() if structured_global else None,
            "per_tensor": structured_per.parameter_count() if structured_per else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    print("Interpretation:", json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()
