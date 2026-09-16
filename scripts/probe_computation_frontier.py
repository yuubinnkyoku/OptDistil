"""Minimum-optimizer-computation frontier probe.

Answers whether the 2-parameter anisotropic NormGrad explanation can be reduced
further, what the gain is made of, and whether property-based partitions work
without validation search.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from optdistil.multitensor.frontier import (
    evaluate_assigned_lrs,
    formula_lrs_for_case,
    per_tensor_oracle,
    property_partition_from_case,
    tune_groups_cd,
)
from optdistil.multitensor.ladder import build_normgrad, collect_ratios_for_method
from optdistil.multitensor.roles import random_balanced_binary_partition
from optdistil.multitensor.static_role import roles_for_case, unique_roles
from optdistil.multitensor.stochastic import (
    flatten_split,
    git_commit_sha,
    make_split,
    summarize_ratios,
)

SOURCE_VAL_SEED = 711000
SOURCE_TEST_SEED = 721000
ALL_MATRIX_VAL_SEED = 831000
ALL_MATRIX_TEST_SEED = 841000
ALL_VECTOR_VAL_SEED = 851000
ALL_VECTOR_TEST_SEED = 861000
ISO_VAL_SEED = 871000
ISO_TEST_SEED = 881000
RANDOM_PART_SEED = 910000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimum-computation frontier probe.")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-tasks", type=int, default=3)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--cd-rounds", type=int, default=3)
    parser.add_argument("--random-partitions", type=int, default=4)
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
    args.cd_rounds = 2
    args.random_partitions = 2


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
        offset = 0
        if architecture == "residual":
            offset = 500_000
        elif architecture == "three_layer":
            offset = 100_000
        elif architecture == "all_matrix":
            offset = 200_000
        elif architecture == "all_vector":
            offset = 300_000
        elif architecture == "iso_shape":
            offset = 400_000
        split = make_split(
            architecture,
            conditions,
            seed_base=seed_base + offset,
            count=count,
            width=width,
            samples=samples,
            device=device,
        )
        cases.extend(flatten_split(split))
    return cases


def summarize(ratios: list[float]) -> dict[str, Any]:
    return summarize_ratios(ratios).to_dict() | {"n": len(ratios), "ratios": ratios}


def bootstrap_ci(values: list[float], *, n_boot: int = 2000, seed: int = 0) -> list[float]:
    import random

    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return [means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]]


def paired_stats(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired lengths differ")
    diffs = [a - b for a, b in zip(left, right, strict=True)]
    wins = sum(1 for d in diffs if d < 0)
    mean = statistics.fmean(diffs)
    # crude sign-test two-sided via normal approx
    n = len(diffs)
    if n == 0:
        p = 1.0
    else:
        z = (wins - n / 2) / math.sqrt(n / 4 + 1e-12)
        p = math.erfc(abs(z) / math.sqrt(2))
    return {
        "mean_diff": mean,
        "median_diff": statistics.median(diffs),
        "wins_left_better": wins,
        "n": n,
        "bootstrap_95": bootstrap_ci(diffs, seed=7),
        "sign_test_p_approx": p,
        "diffs": diffs,
    }


def assignment_map(cases: list, mode: str, seed: int | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for case in cases:
        if case.architecture in out:
            continue
        if mode == "random":
            n = len(case.initial)
            out[case.architecture] = random_balanced_binary_partition(
                n, seed=int(seed) if seed is not None else RANDOM_PART_SEED
            )
        else:
            out[case.architecture] = property_partition_from_case(case, mode=mode)
    return out


def run_group(
    val_cases: list,
    test_cases: list,
    *,
    batch_size: int,
    steps: int,
    rounds: int,
    name: str,
    assigned_by_arch: dict[str, list[str]] | None = None,
    assigned_roles: list[str] | None = None,
) -> dict[str, Any]:
    _roles, role_lrs, val_score = tune_groups_cd(
        build_normgrad,
        val_cases,
        batch_size=batch_size,
        steps=steps,
        assigned_by_arch=assigned_by_arch,
        assigned_roles=assigned_roles,
        rounds=rounds,
    )
    ratios = evaluate_assigned_lrs(
        build_normgrad,
        role_lrs,
        test_cases,
        batch_size=batch_size,
        steps=steps,
        assigned_by_arch=assigned_by_arch,
        assigned_roles=assigned_roles,
    )
    groups = unique_roles(
        assigned_roles
        if assigned_roles is not None
        else (next(iter(assigned_by_arch.values())) if assigned_by_arch else roles_for_case(val_cases[0]))
    )
    return {
        "name": name,
        "n_groups": len(groups),
        "free_scalars": len(role_lrs),
        "role_lrs": role_lrs,
        "val_score": val_score,
        "test": summarize(ratios),
    }


def experiment_frontier(val_cases: list, test_cases: list, args: argparse.Namespace) -> dict[str, Any]:
    """k = 1 .. n groups plus property partitions and random controls."""
    modes = [
        ("k1_uniform", "single"),
        ("k2_true_roles", "true_roles"),
        ("k2_ndim", "ndim"),
        ("k2_numel", "numel2"),
        ("k3_numel", "numel3"),
        ("k2_index_mod", "index_mod2"),
        ("k2_first_half", "first_half"),
        ("per_tensor_oracle", "per_tensor"),
    ]
    results = []
    for name, mode in modes:
        assigned = assignment_map(val_cases, mode)
        results.append(
            run_group(
                val_cases,
                test_cases,
                batch_size=args.batch_size,
                steps=args.steps,
                rounds=args.cd_rounds,
                name=name,
                assigned_by_arch=assigned,
            )
        )
    for index in range(args.random_partitions):
        assigned = assignment_map(val_cases, "random", seed=RANDOM_PART_SEED + index)
        results.append(
            run_group(
                val_cases,
                test_cases,
                batch_size=args.batch_size,
                steps=args.steps,
                rounds=args.cd_rounds,
                name=f"k2_random_{index}",
                assigned_by_arch=assigned,
            )
        )
    by_name = {r["name"]: r for r in results}
    k1 = by_name["k1_uniform"]["test"]["ratios"]
    for r in results:
        r["paired_vs_k1"] = paired_stats(r["test"]["ratios"], k1)
    ranked = sorted(results, key=lambda r: r["test"]["mean"])
    return {
        "methods": results,
        "ranking": [
            {
                "name": r["name"],
                "free_scalars": r["free_scalars"],
                "test_mean": r["test"]["mean"],
                "mean_diff_vs_k1": r["paired_vs_k1"]["mean_diff"],
                "ci_vs_k1": r["paired_vs_k1"]["bootstrap_95"],
                "sign_p_vs_k1": r["paired_vs_k1"]["sign_test_p_approx"],
            }
            for r in ranked
        ],
    }


def experiment_formulas(val_cases: list, test_cases: list, args: argparse.Namespace) -> dict[str, Any]:
    """Property formulas with no validation search (absolute LRs)."""
    modes = (
        "uniform",
        "inv_sqrt_numel",
        "inv_numel",
        "inv_sqrt_fan_in",
        "ndim_scaled",
        "numel_rank",
        "lars_init_norm",
        "inv_sqrt_numel_normalized",
    )
    out = []
    for mode in modes:
        ratios = []
        for case in test_cases:
            lrs = formula_lrs_for_case(case, mode=mode, base=0.1)
            teacher = build_normgrad(lrs)
            ratios.append(
                collect_ratios_for_method(
                    lambda _c, t=teacher: t, [case], batch_size=args.batch_size, steps=args.steps
                )[0]
            )
        out.append({"mode": mode, "base": 0.1, "test": summarize(ratios)})
    # Also: property partition + single global scale tuned on val (1 free param).
    for mode in ("ndim", "numel2", "true_roles"):
        assigned = assignment_map(val_cases, mode)
        _roles, role_lrs, _score = tune_groups_cd(
            build_normgrad,
            val_cases,
            batch_size=args.batch_size,
            steps=args.steps,
            assigned_by_arch=assigned,
            rounds=1,
            candidates=(0.01, 0.03, 0.1, 0.3),
        )
        # freeze ratio, retune only a global scalar on a second pass is already
        # the role_lrs; report as "partition+2val" not formula.
        ratios = evaluate_assigned_lrs(
            build_normgrad,
            role_lrs,
            test_cases,
            batch_size=args.batch_size,
            steps=args.steps,
            assigned_by_arch=assigned,
        )
        out.append(
            {
                "mode": f"partition_{mode}_val_tuned",
                "free_scalars": len(role_lrs),
                "role_lrs": role_lrs,
                "test": summarize(ratios),
            }
        )
    return {"methods": out}


def experiment_oracle(val_cases: list, test_cases: list, args: argparse.Namespace) -> dict[str, Any]:
    assigned_by_arch, role_lrs, val_score = per_tensor_oracle(
        val_cases,
        batch_size=args.batch_size,
        steps=args.steps,
        rounds=args.cd_rounds,
    )
    ratios = evaluate_assigned_lrs(
        build_normgrad,
        role_lrs,
        test_cases,
        batch_size=args.batch_size,
        steps=args.steps,
        assigned_by_arch=assigned_by_arch,
    )
    uniform_lr = 0.1
    k1_ratios = []
    for case in test_cases:
        teacher = build_normgrad([uniform_lr] * len(case.initial))
        k1_ratios.append(
            collect_ratios_for_method(
                lambda _c, t=teacher: t, [case], batch_size=args.batch_size, steps=args.steps
            )[0]
        )
    return {
        "assigned_by_arch": assigned_by_arch,
        "per_tensor_role_lrs": role_lrs,
        "val_score": val_score,
        "test": summarize(ratios),
        "uniform_0.1": summarize(k1_ratios),
        "paired_vs_uniform": paired_stats(ratios, k1_ratios),
    }


def experiment_adversarial(args: argparse.Namespace) -> dict[str, Any]:
    """Families without matrix/vector structure."""
    families = {
        "all_matrix": (ALL_MATRIX_VAL_SEED, ALL_MATRIX_TEST_SEED),
        "all_vector": (ALL_VECTOR_VAL_SEED, ALL_VECTOR_TEST_SEED),
        "iso_shape": (ISO_VAL_SEED, ISO_TEST_SEED),
    }
    out = {}
    for name, (val_seed, test_seed) in families.items():
        val_cases = mixed_cases(
            (name,),
            (30.0, 300.0),
            seed_base=val_seed,
            count=args.val_tasks,
            width=args.width,
            samples=args.samples,
            device=args.device,
        )
        test_cases = mixed_cases(
            (name,),
            (10.0, 100.0, 1000.0, 3000.0),
            seed_base=test_seed,
            count=args.test_tasks,
            width=args.width,
            samples=args.samples,
            device=args.device,
        )
        frontier = experiment_frontier(val_cases, test_cases, args)
        out[name] = {
            "n_tensors": len(val_cases[0].initial),
            "roles": case_roles_safe(val_cases[0]),
            "ranking": frontier["ranking"],
        }
    return out


def case_roles_safe(case) -> list[str]:
    return list(roles_for_case(case))


def main() -> None:
    args = parse_args()
    apply_quick(args)
    started = time.time()
    arch_source = ("two_layer", "residual")

    val_cases = mixed_cases(
        arch_source,
        (30.0, 300.0),
        seed_base=SOURCE_VAL_SEED,
        count=args.val_tasks,
        width=args.width,
        samples=args.samples,
        device=args.device,
    )
    test_cases = mixed_cases(
        arch_source,
        (10.0, 100.0, 1000.0, 3000.0),
        seed_base=SOURCE_TEST_SEED,
        count=args.test_tasks,
        width=args.width,
        samples=args.samples,
        device=args.device,
    )

    frontier = experiment_frontier(val_cases, test_cases, args)
    formulas = experiment_formulas(val_cases, test_cases, args)
    oracle = experiment_oracle(val_cases, test_cases, args)
    adversarial = experiment_adversarial(args)

    # Focus: is k=1 distinguishable from best k=2?
    k1 = next(m for m in frontier["methods"] if m["name"] == "k1_uniform")
    k2_candidates = [
        m
        for m in frontier["methods"]
        if m["free_scalars"] == 2 and m["name"].startswith("k2_")
    ]
    best_k2 = min(k2_candidates, key=lambda m: m["test"]["mean"])
    one_vs_two = paired_stats(best_k2["test"]["ratios"], k1["test"]["ratios"])

    payload = {
        "config": {
            "width": args.width,
            "samples": args.samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "val_tasks": args.val_tasks,
            "test_tasks": args.test_tasks,
            "cd_rounds": args.cd_rounds,
            "random_partitions": args.random_partitions,
            "quick": args.quick,
            "device": args.device,
        },
        "seed_bases": {
            "source_val": SOURCE_VAL_SEED,
            "source_test": SOURCE_TEST_SEED,
            "all_matrix_val": ALL_MATRIX_VAL_SEED,
            "all_matrix_test": ALL_MATRIX_TEST_SEED,
            "all_vector_val": ALL_VECTOR_VAL_SEED,
            "all_vector_test": ALL_VECTOR_TEST_SEED,
            "random_part": RANDOM_PART_SEED,
        },
        "frontier": frontier,
        "formulas": formulas,
        "oracle": oracle,
        "adversarial": adversarial,
        "one_vs_two": {
            "k1_mean": k1["test"]["mean"],
            "best_k2_name": best_k2["name"],
            "best_k2_mean": best_k2["test"]["mean"],
            "paired_k2_minus_k1": one_vs_two,
        },
        "runtime_seconds": time.time() - started,
        "artifact": {"commit_sha": git_commit_sha(), "script": "scripts/probe_computation_frontier.py"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps(payload["frontier"]["ranking"], indent=2))
    print(json.dumps(payload["one_vs_two"], indent=2)[:800])


if __name__ == "__main__":
    main()
