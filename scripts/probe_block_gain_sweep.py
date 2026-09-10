from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep element, row, and whole-tensor BlockGain granularities while keeping "
            "the learned student architecture fixed."
        )
    )
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--train-tasks", type=int, default=2)
    parser.add_argument("--validation-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-seed", type=int, default=9100)
    parser.add_argument("--student-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _direction_only_summary(payload: dict) -> dict[tuple[float, str], dict]:
    return {
        (float(row["condition"]), row["teacher"]): row
        for row in payload["student_summaries"]
        if row["objective"] == "direction_only"
    }


def _baseline_summary(payload: dict) -> dict[float, dict]:
    return {float(row["condition"]): row for row in payload["baselines"]}


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("size must be positive")

    granularities = (
        ("element", 1),
        ("row", args.size),
        ("tensor", args.size * args.size),
    )
    if len({block_size for _, block_size in granularities}) != len(granularities):
        raise ValueError("granularity sweep requires a matrix size greater than 1")

    primitive = Path(__file__).with_name("probe_block_gain_distillation.py")
    work_dir = (args.output.parent if args.output is not None else Path("artifacts")) / "block_gain_sweep"
    work_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict] = {}
    rows: list[dict] = []

    for granularity, block_size in granularities:
        child_output = work_dir / f"{granularity}.json"
        command = [
            sys.executable,
            str(primitive),
            "--size",
            str(args.size),
            "--block-size",
            str(block_size),
            "--steps",
            str(args.steps),
            "--pool-size",
            str(args.pool_size),
            "--train-tasks",
            str(args.train_tasks),
            "--validation-tasks",
            str(args.validation_tasks),
            "--test-tasks",
            str(args.test_tasks),
            "--epochs",
            str(args.epochs),
            "--student-seed",
            str(args.student_seed),
            "--student-seeds",
            str(args.student_seeds),
            "--device",
            args.device,
            "--output",
            str(child_output),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise RuntimeError(
                f"BlockGain primitive failed for {granularity} blocks with exit code "
                f"{completed.returncode}"
            )

        payload = json.loads(child_output.read_text(encoding="utf-8"))
        runs[granularity] = payload
        baselines = _baseline_summary(payload)
        summaries = _direction_only_summary(payload)

        for condition in payload["conditions"]:
            condition = float(condition)
            baseline = baselines[condition]
            muon = summaries[(condition, "muon")]
            norm_gradient = summaries[(condition, "muon_norm_gradient")]
            rows.append(
                {
                    "granularity": granularity,
                    "block_size": block_size,
                    "condition": condition,
                    "student_parameters": muon["student_parameters"],
                    "baseline_loss_ratio": baseline["test_loss_ratio"],
                    "baseline_aulc": baseline["test_aulc"],
                    "muon_loss_ratio": muon["student_loss_ratio_mean"],
                    "muon_seed_std": muon["student_loss_ratio_seed_std"],
                    "muon_aulc": muon["student_aulc_mean"],
                    "norm_gradient_loss_ratio": norm_gradient["student_loss_ratio_mean"],
                    "norm_gradient_seed_std": norm_gradient["student_loss_ratio_seed_std"],
                    "norm_gradient_aulc": norm_gradient["student_aulc_mean"],
                    "muon_minus_norm_gradient_loss_ratio": (
                        muon["student_loss_ratio_mean"]
                        - norm_gradient["student_loss_ratio_mean"]
                    ),
                    "muon_gain_over_baseline": (
                        baseline["test_loss_ratio"] - muon["student_loss_ratio_mean"]
                    ),
                    "norm_gradient_gain_over_baseline": (
                        baseline["test_loss_ratio"]
                        - norm_gradient["student_loss_ratio_mean"]
                    ),
                }
            )

    payload = {
        "experiment": "block_gain_granularity_sweep",
        "interpretation": {
            "element": "one gain/normalization block per parameter",
            "row": "one contiguous block per matrix row",
            "tensor": "one gain/normalization block for the whole matrix",
            "muon_minus_norm_gradient_loss_ratio": (
                "negative means Muon-distilled BlockGain is better"
            ),
        },
        "size": args.size,
        "steps": args.steps,
        "pool_size": args.pool_size,
        "train_tasks": args.train_tasks,
        "validation_tasks": args.validation_tasks,
        "test_tasks": args.test_tasks,
        "epochs": args.epochs,
        "student_seeds": args.student_seeds,
        "device": args.device,
        "rows": rows,
        "runs": runs,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    print("BLOCK_GAIN_SWEEP_BEGIN")
    for row in rows:
        print(
            "SWEEP "
            f"granularity={row['granularity']} condition={row['condition']:g} "
            f"base={row['baseline_loss_ratio']:.6f} "
            f"muon={row['muon_loss_ratio']:.6f} "
            f"normgrad={row['norm_gradient_loss_ratio']:.6f} "
            f"delta={row['muon_minus_norm_gradient_loss_ratio']:+.6f} "
            f"muon_aulc={row['muon_aulc']:.6f} "
            f"normgrad_aulc={row['norm_gradient_aulc']:.6f}"
        )
    print("BLOCK_GAIN_SWEEP_END")


if __name__ == "__main__":
    main()
