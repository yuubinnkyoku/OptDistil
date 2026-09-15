"""Minimum optimizer computation + role-rule transfer/falsification probe.

Experiments
-----------
A. Computation ladder on the base multitensor family (SGD / NormGrad / AdamW x uniform/role)
B. Computation ladder on IID reparameterization stress cases
C. Role-label permutation (true / swap / random partitions) on base family
D. Frozen-ratio transfer to three_layer and width OOD
E. Inverted role-aligned reparameterization family (s_matrix ≫ s_vector)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from optdistil.multitensor.ladder import (
    build_adamw,
    build_normgrad,
    build_sgd,
    collect_ratios_for_method,
    fixed_role_teacher,
    frozen_ratio_teacher,
    paired_mean_difference,
    tune_global_scale,
    tune_role_grid,
)
from optdistil.multitensor.reparam import (
    make_reparameterized_case,
    role_aligned_scales,
    sample_log_uniform_scales,
)
from optdistil.multitensor.roles import random_balanced_binary_partition
from optdistil.multitensor.static_role import roles_for_case
from optdistil.multitensor.stochastic import (
    flatten_split,
    git_commit_sha,
    make_split,
    summarize_ratios,
    tune_teacher_lr,
)
from optdistil.multitensor.teachers import NormGradTensorWise

TRAIN_CONDITIONS = (30.0, 300.0)
IID_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
TRAIN_SCALE_RANGE = (0.5, 2.0)
# Privileged match: c_i = lr / s_i. Large s_matrix ⇒ small matrix θ-LR.
INVERT_SCALE_MATRIX = 10.0
INVERT_SCALE_VECTOR = 0.1

# Distinct seed bases; never used for both tuning and test.
SOURCE_VAL_SEED = 711000
SOURCE_TEST_SEED = 721000
THREE_LAYER_VAL_SEED = 751000
THREE_LAYER_TEST_SEED = 761000
WIDTH_OOD_SEED = 771000
INVERT_VAL_SEED = 781000
INVERT_TEST_SEED = 791000
REPARAM_SCALE_SEED = 801000
PERM_SEED = 910000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimum-computation / role-transfer probe.")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-tasks", type=int, default=3)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--random-permutations", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.steps = 10
    args.val_tasks = 2
    args.test_tasks = 2
    args.random_permutations = 2


def mixed_cases(
    architectures: tuple[str, ...],
    conditions: tuple[float, ...],
    *,
    seed_base: int,
    count: int,
    width: int,
    samples: int,
    device: str,
) -> list:
    cases = []
    for architecture in architectures:
        split = make_split(
            architecture,
            conditions,
            seed_base=seed_base + (0 if architecture == "two_layer" else 500_000)
            + (100_000 if architecture == "three_layer" else 0),
            count=count,
            width=width,
            samples=samples,
            device=device,
        )
        cases.extend(flatten_split(split))
    return cases


def reparameterize_list(
    cases: list,
    *,
    scale_range: tuple[float, float],
    seed_base: int,
) -> list:
    out = []
    for index, case in enumerate(cases):
        scales = sample_log_uniform_scales(
            len(case.initial),
            low=scale_range[0],
            high=scale_range[1],
            seed=seed_base + index,
        )
        reparam_case, _values = make_reparameterized_case(case, scales=scales)
        out.append(reparam_case)
    return out


def invert_reparameterize_list(
    cases: list,
    *,
    seed_base: int,
    matrix_scale: float = INVERT_SCALE_MATRIX,
    vector_scale: float = INVERT_SCALE_VECTOR,
) -> list:
    """Role-aligned constant scales that invert the usual matrix>vector θ-step need.

    With ``s_matrix ≫ s_vector``, privileged NormGrad needs
    ``c_matrix = lr/s_matrix ≪ c_vector = lr/s_vector``.
    """
    out = []
    for index, case in enumerate(cases):
        roles = roles_for_case(case)
        scales = role_aligned_scales(
            roles, matrix_scale=matrix_scale, vector_scale=vector_scale
        )
        # Add mild multiplicative jitter so cases are not identical.
        jitter = sample_log_uniform_scales(
            len(scales), low=0.9, high=1.1, seed=seed_base + index
        )
        scales = [s * j for s, j in zip(scales, jitter, strict=True)]
        reparam_case, _values = make_reparameterized_case(case, scales=scales)
        out.append(reparam_case)
    return out


def summarize_method_list(ratios: list[float]) -> dict[str, Any]:
    summary = summarize_ratios(ratios).to_dict()
    return {
        "mean": summary["mean"],
        "std": summary["std"],
        "median": summary["median"],
        "finite_fraction": summary["finite_fraction"],
        "n": len(ratios),
        "ratios": ratios,
    }


def experiment_ladder(
    val_cases: list,
    test_cases: list,
    *,
    batch_size: int,
    steps: int,
    label: str,
) -> dict[str, Any]:
    """Experiment A/B: fair LR budgets across geometry x role split."""
    result: dict[str, Any] = {"label": label}

    uniform_candidates = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)

    def tune_uniform(builder_name: str) -> tuple[float, float]:
        scored = []
        for lr in uniform_candidates:
            if builder_name == "sgd":
                teacher = build_sgd([lr] * len(val_cases[0].initial))
            elif builder_name == "normgrad":
                teacher = build_normgrad([lr] * len(val_cases[0].initial))
            elif builder_name == "adamw":
                teacher = build_adamw([lr] * len(val_cases[0].initial))
            else:
                raise ValueError(builder_name)
            scores = [
                collect_ratios_for_method(
                    lambda _c, t=teacher: t,
                    [case],
                    batch_size=batch_size,
                    steps=steps,
                )[0]
                for case in val_cases
            ]
            scored.append((statistics.fmean(scores), float(lr)))
        score, lr = min(scored, key=lambda item: item[0])
        return lr, score

    methods: dict[str, Any] = {}

    # Also compare raw analytic teachers via existing path for NormGrad/SGD/AdamW.
    for method_name in ("sgd", "norm_grad_local", "adamw"):
        lr, score = tune_teacher_lr(
            method_name, val_cases, batch_size=batch_size, steps=steps
        )
        if method_name == "norm_grad_local":
            teacher_factory = lambda lr=lr: NormGradTensorWise(lr=lr)
        elif method_name == "sgd":
            from optdistil.multitensor.teachers import SGDMultiTensor

            teacher_factory = lambda lr=lr: SGDMultiTensor(lr=lr)
        else:
            from optdistil.multitensor.teachers import AdamWMultiTensor

            teacher_factory = lambda lr=lr: AdamWMultiTensor(lr=lr)
        ratios = collect_ratios_for_method(
            lambda _c, f=teacher_factory: f(),
            test_cases,
            batch_size=batch_size,
            steps=steps,
        )
        methods[f"{method_name}_uniform"] = {
            "kind": "uniform",
            "free_scalars": 1,
            "tuned_lr": lr,
            "val_score": score,
            "test": summarize_method_list(ratios),
        }

    builders = {
        "sgd_role": build_sgd,
        "normgrad_role": build_normgrad,
        "adamw_role": build_adamw,
    }
    for name, builder in builders.items():
        role_lrs, score = tune_role_grid(
            builder, val_cases, batch_size=batch_size, steps=steps
        )
        ratios = collect_ratios_for_method(
            lambda _c, b=builder, rl=role_lrs: fixed_role_teacher(b, rl, _c),
            test_cases,
            batch_size=batch_size,
            steps=steps,
        )
        methods[name] = {
            "kind": "role",
            "free_scalars": len(role_lrs),
            "role_lrs": role_lrs,
            "val_score": score,
            "test": summarize_method_list(ratios),
        }

    # Identity / no-op control.
    identity_ratios = [1.0 for _ in test_cases]
    methods["identity_noop"] = {
        "kind": "noop",
        "free_scalars": 0,
        "test": summarize_method_list(identity_ratios),
    }

    result["methods"] = methods
    # Ranking by test mean.
    ranked = sorted(
        (
            (name, payload["test"]["mean"], payload["free_scalars"])
            for name, payload in methods.items()
            if payload["test"]["finite_fraction"] > 0
        ),
        key=lambda item: (item[1], item[2]),
    )
    result["ranking"] = [
        {"name": n, "test_mean": m, "free_scalars": f} for n, m, f in ranked
    ]
    result["minimum_sufficient"] = ranked[0] if ranked else None
    return result


def experiment_permutation(
    val_cases: list,
    test_cases: list,
    *,
    batch_size: int,
    steps: int,
    n_random: int,
) -> dict[str, Any]:
    """Experiment C: is the true matrix/vector partition special?

    Partitions are generated per architecture because two_layer has 4 tensors
    and residual has 5; a single assignment cannot span both.
    """
    architectures = sorted({c.architecture for c in val_cases})
    out: dict[str, Any] = {"architectures": architectures, "partitions": []}

    def roles_for_spec(kind: str, n_tensors: int, seed: int | None) -> list[str]:
        true_roles = ["matrix" if i % 2 == 0 else "vector" for i in range(n_tensors)]
        if kind == "true":
            return true_roles
        if kind == "swap":
            from optdistil.multitensor.roles import swap_binary_role_labels

            return swap_binary_role_labels(true_roles)
        if kind == "random":
            return random_balanced_binary_partition(n_tensors, seed=int(seed))
        raise ValueError(kind)

    kinds: list[tuple[str, str, int | None]] = [
        ("true_matrix_vector", "true", None),
        ("swapped_labels", "swap", None),
    ]
    for index in range(n_random):
        kinds.append((f"random_partition_{index}", "random", PERM_SEED + index))

    payload_by_kind: dict[str, dict[str, Any]] = {}
    for name, kind, seed in kinds:
        ratios_all: list[float] = []
        role_lrs_report: dict[str, Any] = {}
        val_scores: list[float] = []
        for architecture in architectures:
            val_sub = [c for c in val_cases if c.architecture == architecture]
            test_sub = [c for c in test_cases if c.architecture == architecture]
            if not val_sub:
                continue
            n_tensors = len(val_sub[0].initial)
            assigned = roles_for_spec(kind, n_tensors, seed)
            role_lrs, score = tune_role_grid(
                build_normgrad,
                val_sub,
                batch_size=batch_size,
                steps=steps,
                assigned_roles=assigned,
            )
            role_lrs_report[architecture] = {"roles": assigned, "role_lrs": role_lrs}
            val_scores.append(score)
            ratios = collect_ratios_for_method(
                lambda _c, rl=role_lrs, ar=assigned: fixed_role_teacher(
                    build_normgrad, rl, _c, assigned_roles=ar
                ),
                test_sub,
                batch_size=batch_size,
                steps=steps,
            )
            ratios_all.extend(ratios)
        payload_by_kind[name] = {
            "name": name,
            "kind": kind,
            "seed": seed,
            "role_lrs_by_arch": role_lrs_report,
            "val_score_mean": statistics.fmean(val_scores) if val_scores else math.nan,
            "test": summarize_method_list(ratios_all),
            "ratios": ratios_all,
        }

    true_payload = payload_by_kind["true_matrix_vector"]
    for payload in payload_by_kind.values():
        payload["paired_vs_true"] = paired_mean_difference(
            payload["test"]["ratios"], true_payload["test"]["ratios"]
        )
    randoms = [p for p in payload_by_kind.values() if p["kind"] == "random"]
    if randoms:
        best_random = min(randoms, key=lambda p: p["test"]["mean"])
        out["best_random_vs_true"] = best_random["paired_vs_true"]
        out["best_random_name"] = best_random["name"]
    out["partitions"] = list(payload_by_kind.values())
    return out


def experiment_transfer(
    source_role_lrs: dict[str, float],
    source_val: list,
    targets: dict[str, list],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    """Experiment D: freeze ratio, retune only a global scalar per target."""
    out: dict[str, Any] = {
        "source_role_lrs": source_role_lrs,
        "targets": {},
    }
    for name, (val_cases, test_cases) in targets.items():
        global_scale, scale_score = tune_global_scale(
            build_normgrad,
            source_role_lrs,
            val_cases,
            batch_size=batch_size,
            steps=steps,
        )
        frozen_ratios = collect_ratios_for_method(
            lambda _c, gs=global_scale: frozen_ratio_teacher(
                build_normgrad, source_role_lrs, _c, global_scale=gs
            ),
            test_cases,
            batch_size=batch_size,
            steps=steps,
        )
        # Fully retuned 2-role control on the target validation split.
        retuned_lrs, retuned_score = tune_role_grid(
            build_normgrad, val_cases, batch_size=batch_size, steps=steps
        )
        retuned_ratios = collect_ratios_for_method(
            lambda _c, rl=retuned_lrs: fixed_role_teacher(build_normgrad, rl, _c),
            test_cases,
            batch_size=batch_size,
            steps=steps,
        )
        uniform_lr, uniform_score = tune_teacher_lr(
            "norm_grad_local", val_cases, batch_size=batch_size, steps=steps
        )
        uniform_ratios = collect_ratios_for_method(
            lambda _c, lr=uniform_lr: NormGradTensorWise(lr=lr),
            test_cases,
            batch_size=batch_size,
            steps=steps,
        )
        out["targets"][name] = {
            "global_scale": global_scale,
            "global_scale_val_score": scale_score,
            "frozen_ratio": summarize_method_list(frozen_ratios),
            "retuned_role_lrs": retuned_lrs,
            "retuned_val_score": retuned_score,
            "retuned_role": summarize_method_list(retuned_ratios),
            "uniform_lr": uniform_lr,
            "uniform_val_score": uniform_score,
            "uniform": summarize_method_list(uniform_ratios),
            "frozen_vs_retuned": paired_mean_difference(frozen_ratios, retuned_ratios),
            "frozen_vs_uniform": paired_mean_difference(frozen_ratios, uniform_ratios),
        }
    return out


def experiment_inversion(
    source_role_lrs: dict[str, float],
    invert_val: list,
    invert_test: list,
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    """Experiment E: does the source-family matrix>vector ratio transfer when s flips?"""
    global_scale, scale_score = tune_global_scale(
        build_normgrad,
        source_role_lrs,
        invert_val,
        batch_size=batch_size,
        steps=steps,
    )
    frozen_ratios = collect_ratios_for_method(
        lambda _c: frozen_ratio_teacher(
            build_normgrad, source_role_lrs, _c, global_scale=global_scale
        ),
        invert_test,
        batch_size=batch_size,
        steps=steps,
    )
    retuned_lrs, retuned_score = tune_role_grid(
        build_normgrad, invert_val, batch_size=batch_size, steps=steps
    )
    retuned_ratios = collect_ratios_for_method(
        lambda _c: fixed_role_teacher(build_normgrad, retuned_lrs, _c),
        invert_test,
        batch_size=batch_size,
        steps=steps,
    )
    uniform_lr, uniform_score = tune_teacher_lr(
        "norm_grad_local", invert_val, batch_size=batch_size, steps=steps
    )
    uniform_ratios = collect_ratios_for_method(
        lambda _c: NormGradTensorWise(lr=uniform_lr),
        invert_test,
        batch_size=batch_size,
        steps=steps,
    )
    # Ratio diagnostic.
    def _ratio(lrs: dict[str, float]) -> float | None:
        if "matrix" in lrs and "vector" in lrs and lrs["vector"] > 0:
            return float(lrs["matrix"]) / float(lrs["vector"])
        return None

    return {
        "source_role_lrs": source_role_lrs,
        "source_matrix_over_vector": _ratio(source_role_lrs),
        "inverted_scales": {
            "matrix": INVERT_SCALE_MATRIX,
            "vector": INVERT_SCALE_VECTOR,
        },
        "global_scale": global_scale,
        "global_scale_val_score": scale_score,
        "frozen_ratio": summarize_method_list(frozen_ratios),
        "retuned_role_lrs": retuned_lrs,
        "retuned_matrix_over_vector": _ratio(retuned_lrs),
        "retuned_val_score": retuned_score,
        "retuned_role": summarize_method_list(retuned_ratios),
        "uniform_lr": uniform_lr,
        "uniform_val_score": uniform_score,
        "uniform": summarize_method_list(uniform_ratios),
        "frozen_vs_retuned": paired_mean_difference(frozen_ratios, retuned_ratios),
        "frozen_vs_uniform": paired_mean_difference(frozen_ratios, uniform_ratios),
    }


def main() -> None:
    args = parse_args()
    apply_quick(args)
    started = time.time()
    device = args.device
    arch_source = ("two_layer", "residual")

    source_val = mixed_cases(
        arch_source,
        TRAIN_CONDITIONS,
        seed_base=SOURCE_VAL_SEED,
        count=args.val_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    source_test = mixed_cases(
        arch_source,
        IID_CONDITIONS,
        seed_base=SOURCE_TEST_SEED,
        count=args.test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    reparam_val = reparameterize_list(
        source_val, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED
    )
    reparam_test = reparameterize_list(
        source_test, scale_range=TRAIN_SCALE_RANGE, seed_base=REPARAM_SCALE_SEED + 10_000
    )

    exp_a = experiment_ladder(
        source_val,
        source_test,
        batch_size=args.batch_size,
        steps=args.steps,
        label="base_multitensor",
    )
    exp_b = experiment_ladder(
        reparam_val,
        reparam_test,
        batch_size=args.batch_size,
        steps=args.steps,
        label="reparam_iid",
    )
    exp_c = experiment_permutation(
        source_val,
        source_test,
        batch_size=args.batch_size,
        steps=args.steps,
        n_random=args.random_permutations,
    )

    # Source shared-role LRs for transfer (tune on source validation).
    source_role_lrs, source_role_score = tune_role_grid(
        build_normgrad, source_val, batch_size=args.batch_size, steps=args.steps
    )

    three_val = mixed_cases(
        ("three_layer",),
        TRAIN_CONDITIONS,
        seed_base=THREE_LAYER_VAL_SEED,
        count=args.val_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    three_test = mixed_cases(
        ("three_layer",),
        IID_CONDITIONS,
        seed_base=THREE_LAYER_TEST_SEED,
        count=args.test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    wide_val = mixed_cases(
        arch_source,
        TRAIN_CONDITIONS,
        seed_base=WIDTH_OOD_SEED,
        count=args.val_tasks,
        width=args.width + 8,
        samples=args.samples,
        device=device,
    )
    wide_test = mixed_cases(
        arch_source,
        IID_CONDITIONS,
        seed_base=WIDTH_OOD_SEED + 50_000,
        count=args.test_tasks,
        width=args.width + 8,
        samples=args.samples,
        device=device,
    )
    exp_d = experiment_transfer(
        source_role_lrs,
        source_val,
        {
            "three_layer": (three_val, three_test),
            "width_ood": (wide_val, wide_test),
        },
        batch_size=args.batch_size,
        steps=args.steps,
    )

    invert_base_val = mixed_cases(
        arch_source,
        TRAIN_CONDITIONS,
        seed_base=INVERT_VAL_SEED,
        count=args.val_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    invert_base_test = mixed_cases(
        arch_source,
        IID_CONDITIONS,
        seed_base=INVERT_TEST_SEED,
        count=args.test_tasks,
        width=args.width,
        samples=args.samples,
        device=device,
    )
    invert_val = invert_reparameterize_list(
        invert_base_val, seed_base=INVERT_VAL_SEED + 7_000
    )
    invert_test = invert_reparameterize_list(
        invert_base_test, seed_base=INVERT_TEST_SEED + 7_000
    )
    exp_e = experiment_inversion(
        source_role_lrs,
        invert_val,
        invert_test,
        batch_size=args.batch_size,
        steps=args.steps,
    )

    payload = {
        "config": {
            "width": args.width,
            "samples": args.samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "val_tasks": args.val_tasks,
            "test_tasks": args.test_tasks,
            "architectures_source": list(arch_source),
            "random_permutations": args.random_permutations,
            "quick": args.quick,
            "device": device,
        },
        "seed_bases": {
            "source_val": SOURCE_VAL_SEED,
            "source_test": SOURCE_TEST_SEED,
            "three_layer_val": THREE_LAYER_VAL_SEED,
            "three_layer_test": THREE_LAYER_TEST_SEED,
            "width_ood": WIDTH_OOD_SEED,
            "invert_val": INVERT_VAL_SEED,
            "invert_test": INVERT_TEST_SEED,
            "reparam_scale": REPARAM_SCALE_SEED,
            "permutation": PERM_SEED,
        },
        "source_role_lrs": source_role_lrs,
        "source_role_val_score": source_role_score,
        "experiment_a_base_ladder": exp_a,
        "experiment_b_reparam_ladder": exp_b,
        "experiment_c_role_permutation": exp_c,
        "experiment_d_frozen_transfer": exp_d,
        "experiment_e_inversion": exp_e,
        "runtime_seconds": time.time() - started,
    }
    payload["artifact"] = {
        "commit_sha": git_commit_sha(),
        "script": "scripts/probe_minimum_computation.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps({"ranking_a": exp_a["ranking"], "ranking_b": exp_b["ranking"]}, indent=2))


if __name__ == "__main__":
    main()
