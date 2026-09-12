from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stochastic distillation artifacts.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stochastic-distillation-merged.json"))
    args = parser.parse_args()

    main_adamw = load(args.artifacts / "stoch-adamw-b32-main.json")
    main_norm = load(args.artifacts / "stoch-normgrad-b8-main.json")
    extra_adamw = load(args.artifacts / "stoch-adamw-b32-controls-ablations.json")
    extra_norm = load(args.artifacts / "stoch-normgrad-b8-controls-ablations.json")

    merged = {
        "experiment": "stochastic_optimizer_distillation_merged",
        "commit_sha": main_adamw["commit_sha"],
        "student_parameter_count": main_adamw["student_parameter_count"],
        "train_conditions": main_adamw["train_conditions"],
        "ood_conditions": main_adamw["ood_conditions"],
        "sources": {
            "adamw_b32_main": main_adamw["run_id"],
            "norm_gradient_b8_main": main_norm["run_id"],
            "adamw_b32_controls_ablations": extra_adamw["run_id"],
            "norm_gradient_b8_controls_ablations": extra_norm["run_id"],
        },
        "config_notes": {
            "main": "5 student seeds, full train/test splits",
            "controls_ablations": "2 seeds, reduced task counts for runtime",
        },
        "regimes": {},
    }

    for regime, main_payload, extra_payload in (
        ("adamw_b32", main_adamw, extra_adamw),
        ("norm_gradient_b8", main_norm, extra_norm),
    ):
        main_regime = main_payload["regimes"][regime]
        extra_regime = extra_payload["regimes"][regime]
        merged_regime = {
            key: value for key, value in main_regime.items()
        }
        merged_regime["negative_controls"] = extra_regime.get("negative_controls") or {}
        merged_regime["feature_ablations"] = extra_regime.get("feature_ablations") or {}
        merged_regime["controls_ablations_config"] = {
            "control_seeds": extra_payload["config"]["control_seeds"],
            "ablation_seeds": extra_payload["config"]["ablation_seeds"],
            "test_tasks": extra_payload["config"]["test_tasks"],
            "ood_test_tasks": extra_payload["config"]["ood_test_tasks"],
            "distill_train_tasks": extra_payload["config"]["distill_train_tasks"],
            "distill_epochs": extra_payload["config"]["distill_epochs"],
        }
        merged["regimes"][regime] = merged_regime

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
