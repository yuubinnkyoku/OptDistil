"""Print a compact report from reparameterization stress JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "inf"
            return f"{value:.4f}"
        return str(value)
    return str(value)


def method_row(name: str, payload: dict[str, Any]) -> str:
    ratio = payload.get("loss_ratio", {})
    return (
        f"| {name} | {fmt(ratio.get('mean'))} | {fmt(ratio.get('median'))} | "
        f"{fmt(ratio.get('std'))} | {fmt(ratio.get('ci_low'))}–{fmt(ratio.get('ci_high'))} | "
        f"{fmt(ratio.get('finite_fraction'))} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report reparameterization stress results.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/reparam-stress-main.json"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    print("# Reparameterization stress report")
    print()
    print(f"commit: `{payload.get('commit_sha')}`")
    print(f"run_id: `{payload.get('run_id')}`")
    config = payload.get("config", {})
    print(
        "config: "
        f"steps={config.get('steps')} batch={config.get('batch_size')} "
        f"width={config.get('width')} students={config.get('student_seeds')} "
        f"quick={config.get('quick')}"
    )
    print()
    print("## Tuned hyperparameters (validation only)")
    tuned = payload.get("tuned", {})
    print(f"- ordinary lr: {fmt(tuned.get('ordinary_lr'))}")
    print(f"- privileged lr: {fmt(tuned.get('privileged_lr'))}")
    print(f"- shared role lrs: {tuned.get('shared_role_lrs')}")
    print(f"- frozen student role scales: {payload.get('frozen_role_lrs')}")
    print(f"- student params: {payload.get('student_parameter_count')}")
    print(f"- structured params: {payload.get('structured_parameter_counts')}")
    print()

    exp_a = payload.get("experiment_a", {})
    for split_key, title in (
        ("iid", "Experiment A — IID (train-range scales, unseen seeds)"),
        ("ood", "Experiment A — OOD mild s∈[0.25,4]"),
        ("strong_ood", "Experiment A — strong OOD s∈[0.1,10]"),
    ):
        section = exp_a.get(split_key, {})
        methods = section.get("methods", {})
        if not methods:
            continue
        print(f"## {title}")
        print()
        print("| method | mean | median | std | boot 95% CI | finite |")
        print("|---|---:|---:|---:|---|---:|")
        preferred_order = [
            "ordinary_local_normgrad",
            "privileged_function_space_normgrad",
            "static_shared_role_normgrad",
            "static_architecture_role_normgrad",
            "frozen_student_role_scales",
            "structured_global",
            "structured_per_tensor",
        ]
        seen = set()
        for name in preferred_order:
            if name in methods:
                print(method_row(name, methods[name]))
                seen.add(name)
        for name in sorted(methods):
            if name in seen:
                continue
            if name.startswith(("student_153p", "student_projected")):
                print(method_row(name, methods[name]))
        print()

    exp_b = payload.get("experiment_b")
    if exp_b:
        print("## Experiment B — static-role transfer (tuned once, no retune)")
        print()
        print("| transfer | mean loss ratio |")
        print("|---|---:|")
        for key in sorted(exp_b):
            if key in {"base_role_lrs", "residual_tuned_lrs"}:
                continue
            value = exp_b[key]
            if isinstance(value, dict) and "loss_ratio" in value:
                print(f"| {key} | {fmt(value['loss_ratio'].get('mean'))} |")
        print()

    exp_c = payload.get("experiment_c", {})
    if exp_c:
        print("## Experiment C — effective scale analysis (seed0 student)")
        print()
        for split in ("iid", "ood"):
            analysis = exp_c.get(split, {})
            if not analysis:
                continue
            print(f"### {split}")
            print(f"- n: {analysis.get('n')}")
            print(f"- hidden_delta_r2: {fmt(analysis.get('hidden_delta_r2'))}")
            print(f"- r2_multivariate_observables: {fmt(analysis.get('r2_multivariate_observables'))}")
            corr = analysis.get("correlations", {})
            print(f"- hidden_scale spearman: {fmt(corr.get('hidden_scale', {}).get('spearman'))}")
            print(f"- grad_rms spearman: {fmt(corr.get('grad_rms', {}).get('spearman'))}")
            print(f"- param_rms spearman: {fmt(corr.get('param_rms', {}).get('spearman'))}")
            print()

    exp_d = payload.get("experiment_d")
    if exp_d:
        print("## Experiment D — structured tiny controls")
        print()
        for key in ("static_shared", "global", "per_tensor", "global_ood", "per_tensor_ood"):
            value = exp_d.get(key)
            if not value:
                continue
            ratio = value.get("loss_ratio", {})
            print(
                f"- {key}: mean={fmt(ratio.get('mean'))} "
                f"params={value.get('parameter_count')} mode={value.get('mode')}"
            )
        print()

    interp = payload.get("interpretation", {})
    print("## Interpretation")
    print()
    claims = interp.get("claims", {})
    for key, value in claims.items():
        print(f"- {key}: {fmt(value)}")
    print(f"- recommended_primary: **{interp.get('recommended_primary')}**")
    print()
    print("### Key numbers")
    print()
    for key, value in interp.get("key_numbers", {}).items():
        print(f"- {key}: {fmt(value)}")
    print()
    print("### Notes")
    print()
    for note in interp.get("notes", []):
        print(f"- {note}")


if __name__ == "__main__":
    main()
