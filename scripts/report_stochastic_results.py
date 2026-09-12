from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fmt(value: float) -> str:
    return f"{value:.4f}"


def summarize_regime(name: str, data: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"REGIME {name} teacher={data['teacher_method']} batch={data['batch_size']}")
    print(f"teacher_lr={data['teacher_lr']} teacher_test={fmt(data['teacher_test_loss_ratio_mean'])}")
    analytic = {
        method: fmt(result["loss_ratio"]["mean"])
        for method, result in data["analytic_test"].items()
    }
    print(f"analytic_test {analytic}")
    print(f"raw_lbfgs_test {fmt(data['raw_lbfgs_two_scale']['test']['loss_ratio']['mean'])}")
    print(f"strongest_analytic {data['strongest_analytic_test']}")

    print("-- modes --")
    for mode, summary in data.get("mode_summary", {}).items():
        print(
            f"  {mode:18s} test={fmt(summary['test']['mean'])} "
            f"geo={fmt(summary['test']['geometric_mean'])} "
            f"ood={fmt(summary['ood']['mean'])} "
            f"beatT={summary['fraction_beats_teacher_test']:.2f} "
            f"beatA={summary['fraction_beats_strongest_analytic_test']:.2f} "
            f"finite={summary['test']['finite_fraction']:.2f}"
        )

    print("-- paired --")
    for key, value in data.get("paired", {}).items():
        if value:
            print(
                f"  {key}: mean_delta={value['mean_delta']:.4f} "
                f"wins={value['left_beats_right']}/{value['total']}"
            )

    transfer = data.get("batch_transfer")
    if transfer:
        print("-- batch transfer --")
        for batch, metrics in sorted(transfer["matrix"].items(), key=lambda item: int(item[0])):
            mark = " (train)" if metrics["matches_train_batch"] else ""
            print(f"  b={batch:2s}{mark}: {fmt(metrics['loss_ratio_mean'])}")

    controls = data.get("negative_controls") or {}
    if controls:
        print("-- negative controls --")
        for key, value in controls.items():
            print(
                f"  {key:28s} test={fmt(value['test']['mean'])} "
                f"ood={fmt(value['ood']['mean'])}"
            )

    ablations = data.get("feature_ablations") or {}
    if ablations:
        print("-- feature ablations --")
        for key, value in ablations.items():
            print(f"  {key:16s} test={fmt(value['test']['mean'])} ood={fmt(value['ood']['mean'])}")

    alignments = []
    for run in data.get("runs", []):
        if run["mode"] == "distill_joint":
            alignment = run["test"].get("alignment") or {}
            if alignment:
                alignments.append(alignment)
    if alignments:
        cos_t = sum(item["cosine_teacher_mean"] for item in alignments) / len(alignments)
        cos_g = sum(item["cosine_negative_gradient_mean"] for item in alignments) / len(alignments)
        norm = sum(item["update_norm_ratio_mean"] for item in alignments) / len(alignments)
        print(f"-- distill_joint alignment (n={len(alignments)}) --")
        print(f"  cos_teacher={cos_t:.3f} cos_neg_grad={cos_g:.3f} norm_ratio={norm:.3f}")


def main() -> None:
    path = Path("artifacts/stochastic-distillation-merged.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"commit={payload['commit_sha'][:10]} params={payload['student_parameter_count']}")
    for name, data in payload["regimes"].items():
        summarize_regime(name, data)
        print()


if __name__ == "__main__":
    main()
