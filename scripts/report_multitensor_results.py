from __future__ import annotations

import json
from pathlib import Path

p = json.loads(Path("artifacts/multitensor-main.json").read_text(encoding="utf-8"))
s = p["summary"]
print("=== SUMMARY ===")
print("teacher", s["teacher_method"], "lr", s["teacher_lr"])
print("teacher_test", s["teacher_test_mean"])
print("strongest", s["strongest_analytic"], s["strongest_analytic_test_mean"])
print("shared_batch", s["shared_batch_index_sequence"])
print("arch", s["architectures"])
print("params", s["student_parameters"])
print()
print("=== ANALYTIC TEST ===")
for m, v in p["analytic_test"].items():
    print(f"{m:20s} mean={v['loss_ratio']['mean']:.4f} median={v['loss_ratio']['median']:.4f} lr={v['lr']}")
print()
print("=== PER SEED STUDENTS ===")
for mode, st in s["per_seed_test"].items():
    if st:
        print(
            f"{mode:20s} mean={st['mean']:.4f} median={st['median']:.4f} "
            f"std={st['std']:.4f} finite={st['finite_fraction']:.2f}"
        )
print()
print("=== OOD ===")
for mode, st in s["per_seed_ood"].items():
    if st:
        print(f"{mode:20s} mean={st['mean']:.4f} median={st['median']:.4f}")
print()
print("=== BY ARCH (analytic) ===")
for m, v in p["analytic_test"].items():
    for a, st in v.get("by_architecture", {}).items():
        print(f"{m:16s} {a:12s} {st['mean']:.4f}")
print()
print("=== BATCH TRANSFER ===")
bt = p.get("batch_transfer")
if bt:
    matrix = bt.get("matrix", bt)
    for b, st in matrix.items():
        if isinstance(st, dict) and "loss_ratio_mean" in st:
            print(f"batch {b}: {st['loss_ratio_mean']:.4f} train={st.get('matches_train_batch')}")
print()
print("=== WIDTH OOD ===")
wo = p.get("width_ood")
if wo:
    for w, st in wo.items():
        if isinstance(st, dict) and "loss_ratio_mean" in st:
            print(f"width {w}: {st['loss_ratio_mean']:.4f}")
print()
print("=== NORMGRAD LOCAL vs GLOBAL ===")
for method, v in p["normgrad_local_vs_global"].items():
    print(
        method,
        "lr",
        v["tuned_lr"],
        "teacher",
        v["teacher_test"]["loss_ratio"]["mean"],
        "student",
        v["student_test_summary"]["mean"],
    )
print()
print("=== CONTROLS ===")
for name, v in p.get("negative_controls", {}).items():
    print(f"{name:28s} test={v['test']['mean']:.4f} src={v.get('source_teacher', '')}")
print()
print("=== ABLATIONS ===")
for name, v in p.get("feature_ablations", {}).items():
    print(f"{name:16s} test={v['test']['mean']:.4f}")
print()
print("=== PAIRED joint vs direct_meta ===")
print(s.get("paired_joint_vs_direct_meta"))
print()
print("=== TENSOR DIAG (distill_joint seed0) ===")
for run in p["runs"]:
    if run["mode"] == "distill_joint" and run["seed"] == 401000:
        td = run["test"].get("tensor_diagnostics")
        al = run["test"].get("alignment")
        print("alignment", al)
        if td:
            for i, n in enumerate(td["names"]):
                print(
                    f"  {n:10s} energy={td['update_norm_fraction_mean'][i]:.4f} "
                    f"cos={td['cosine_teacher_per_tensor'][i]:.4f} "
                    f"rel={td['relative_update_scale'][i]:.4f}"
                )
