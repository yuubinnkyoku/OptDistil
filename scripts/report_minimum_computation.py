"""Human-readable report for the minimum-computation probe artifact."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_index = int((alpha / 2) * n_boot)
    hi_index = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return means[lo_index], means[hi_index]


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _ranking_row(item: dict[str, Any]) -> str:
    return (
        f"| {item['name']} | {item['free_scalars']} | {_fmt(item['test_mean'])} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report min-compute probe results.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cfg = payload["config"]

    print("# Minimum-computation probe report")
    print()
    print(f"- steps={cfg['steps']} batch={cfg['batch_size']} width={cfg['width']}")
    print(
        f"- val_tasks={cfg['val_tasks']} test_tasks={cfg['test_tasks']} "
        f"quick={cfg['quick']}"
    )
    print(f"- source_role_lrs={payload['source_role_lrs']}")
    print(f"- commit={payload['artifact']['commit_sha']}")
    print()

    for key, title in (
        ("experiment_a_base_ladder", "A. Base multitensor ladder"),
        ("experiment_b_reparam_ladder", "B. Reparam ladder"),
    ):
        exp = payload[key]
        print(f"## {title}")
        print()
        print("| method | free scalars | test mean |")
        print("|---|---:|---:|")
        for item in exp["ranking"]:
            print(_ranking_row(item))
        print()

    exp_c = payload["experiment_c_role_permutation"]
    print("## C. Role-label permutation")
    print()
    print("| partition | kind | test mean | vs true mean diff | wins better than true |")
    print("|---|---|---:|---:|---:|")
    for part in exp_c["partitions"]:
        paired = part["paired_vs_true"]
        print(
            f"| {part['name']} | {part['kind']} | {_fmt(part['test']['mean'])} | "
            f"{_fmt(paired['mean_diff'])} | {paired['wins_left_better']}/{paired['n']} |"
        )
    if "best_random_vs_true" in exp_c:
        print()
        print(
            f"Best random `{exp_c['best_random_name']}` vs true: "
            f"mean_diff={_fmt(exp_c['best_random_vs_true']['mean_diff'])}"
        )
    print()

    exp_d = payload["experiment_d_frozen_transfer"]
    print("## D. Frozen-ratio transfer")
    print()
    print("| target | global scale | frozen | retuned | uniform | frozen-retuned |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, target in exp_d["targets"].items():
        print(
            f"| {name} | {_fmt(target['global_scale'])} | "
            f"{_fmt(target['frozen_ratio']['mean'])} | "
            f"{_fmt(target['retuned_role']['mean'])} | "
            f"{_fmt(target['uniform']['mean'])} | "
            f"{_fmt(target['frozen_vs_retuned']['mean_diff'])} |"
        )
    print()

    exp_e = payload["experiment_e_inversion"]
    print("## E. Inversion (s_matrix ≫ s_vector)")
    print()
    print(f"- source ratio matrix/vector = {_fmt(exp_e['source_matrix_over_vector'])}")
    print(
        f"- retuned ratio matrix/vector = {_fmt(exp_e['retuned_matrix_over_vector'])}"
    )
    print(f"- retuned LRs = {exp_e['retuned_role_lrs']}")
    print(
        f"- frozen mean={_fmt(exp_e['frozen_ratio']['mean'])} "
        f"retuned mean={_fmt(exp_e['retuned_role']['mean'])} "
        f"uniform mean={_fmt(exp_e['uniform']['mean'])}"
    )
    print(
        f"- frozen vs retuned: mean_diff={_fmt(exp_e['frozen_vs_retuned']['mean_diff'])} "
        f"wins={exp_e['frozen_vs_retuned']['wins_left_better']}/"
        f"{exp_e['frozen_vs_retuned']['n']}"
    )
    print(
        f"- frozen vs uniform: mean_diff={_fmt(exp_e['frozen_vs_uniform']['mean_diff'])}"
    )
    print()

    # Bootstrap CIs for the two strongest falsification claims.
    true_ratios = next(
        p["test"]["ratios"]
        for p in exp_c["partitions"]
        if p["kind"] == "true"
    )
    best_random = min(
        (p for p in exp_c["partitions"] if p["kind"] == "random"),
        key=lambda p: p["test"]["mean"],
    )
    diffs_c = [a - b for a, b in zip(best_random["test"]["ratios"], true_ratios, strict=True)]
    lo_c, hi_c = bootstrap_mean_ci(diffs_c, seed=1)
    print("## Bootstrap 95% CI (paired mean difference)")
    print()
    print(
        f"- C best random − true: {_fmt(sum(diffs_c)/len(diffs_c))} "
        f"[{_fmt(lo_c)}, {_fmt(hi_c)}]"
    )
    diffs_e = exp_e["frozen_vs_retuned"]["diffs"]
    # frozen − retuned; positive means retuned better
    lo_e, hi_e = bootstrap_mean_ci(diffs_e, seed=2)
    print(
        f"- E frozen − retuned: {_fmt(exp_e['frozen_vs_retuned']['mean_diff'])} "
        f"[{_fmt(lo_e)}, {_fmt(hi_e)}]"
    )
    print()
    print(f"runtime_seconds={payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
