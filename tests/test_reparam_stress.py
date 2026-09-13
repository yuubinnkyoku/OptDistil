from __future__ import annotations

import math

import torch

from optdistil.multitensor.effective_scale import (
    analyze_effective_scales,
    collect_effective_scale_trace,
    pearson_correlation,
    project_onto_normgrad_direction,
    ridge_r2,
    spearman_correlation,
)
from optdistil.multitensor.methods import (
    estimate_role_scales_from_student,
    evaluate_method_split,
    make_privileged_method,
    make_static_shared_method,
    make_student_method,
    rollout_method,
)
from optdistil.multitensor.reparam import (
    PrivilegedFunctionSpaceNormGrad,
    make_reparameterized_case,
    sample_log_uniform_scales,
)
from optdistil.multitensor.static_role import (
    StaticRoleNormGrad,
    expand_role_lrs,
    roles_for_case,
    tune_shared_role_lrs,
)
from optdistil.multitensor.stats_utils import (
    bootstrap_mean_ci,
    paired_comparison,
    summarize_method_values,
)
from optdistil.multitensor.stochastic import (
    batch_sequence,
    flatten_split,
    make_split,
    select_student_scale,
    train_supervised_student,
    tune_teacher_lr,
)
from optdistil.multitensor.structured import (
    GlobalScalarController,
    PerTensorController,
    StructuredTinyOptimizer,
    evaluate_structured_split,
    fit_structured_from_privileged,
    global_cheap_stats,
    per_tensor_cheap_stats,
    rollout_structured,
)
from optdistil.multitensor.teachers import NormGradTensorWise
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def _reparam_case(seed: int = 7, *, architecture: str = "two_layer", width: int = 4):
    split = make_split(
        architecture,
        (30.0,),
        seed_base=seed,
        count=1,
        width=width,
        samples=16,
        device="cpu",
    )
    case = flatten_split(split)[0]
    scales = sample_log_uniform_scales(len(case.initial), low=0.5, high=2.0, seed=seed + 99)
    reparam, values = make_reparameterized_case(case, scales=scales)
    return case, reparam, values


def test_sample_log_uniform_scales_in_range() -> None:
    scales = sample_log_uniform_scales(5, low=0.25, high=4.0, seed=3)
    assert len(scales) == 5
    assert all(0.25 <= s <= 4.0 for s in scales)
    again = sample_log_uniform_scales(5, low=0.25, high=4.0, seed=3)
    assert scales == again


def test_reparameterization_preserves_function_objective() -> None:
    base, reparam, scales = _reparam_case()
    # Same function: loss(base) == loss(theta0) with p = s * theta0
    loss_base = float(base.task.loss(base.initial))
    loss_reparam = float(reparam.task.loss(reparam.initial))
    assert math.isclose(loss_base, loss_reparam, rel_tol=1e-5, abs_tol=1e-6)

    # Gradient transform: g_theta = s * g_p
    g_p = base.task.grad(base.initial)
    g_theta = reparam.task.grad(reparam.initial)
    for gp, gt, s in zip(g_p, g_theta, scales, strict=True):
        torch.testing.assert_close(gt, gp * s, rtol=1e-5, atol=1e-6)


def test_privileged_normgrad_equals_function_space_step() -> None:
    base, reparam, scales = _reparam_case()
    lr = 0.1
    ordinary = NormGradTensorWise(lr=lr)
    privileged = PrivilegedFunctionSpaceNormGrad(lr=lr, scales=scales)

    grads = reparam.task.grad(reparam.initial)
    u_ordinary = ordinary.step(reparam.initial, grads)
    u_priv = privileged.step(reparam.initial, grads)

    # Privileged theta-update equals ordinary / s
    for uo, up, s in zip(u_ordinary, u_priv, scales, strict=True):
        torch.testing.assert_close(up, uo / s, rtol=1e-5, atol=1e-6)

    # Function-space update p: s * u_priv equals ordinary function-space step
    # from the base (p-space) gradient.
    g_p = base.task.grad(base.initial)
    u_p_expected = NormGradTensorWise(lr=lr).step(base.initial, g_p)
    for up, s, expected in zip(u_priv, scales, u_p_expected, strict=True):
        torch.testing.assert_close(up * s, expected, rtol=1e-5, atol=1e-6)


def test_static_role_normgrad_uses_role_scales() -> None:
    _, reparam, _ = _reparam_case()
    roles = roles_for_case(reparam)
    role_lrs = {"matrix": 0.05, "vector": 0.2}
    teacher = StaticRoleNormGrad(scales=expand_role_lrs(role_lrs, roles))
    grads = reparam.task.grad(reparam.initial)
    updates = teacher.step(reparam.initial, grads)
    for update, grad, role in zip(updates, grads, roles, strict=True):
        g = grad.reshape(-1).float()
        expected = -role_lrs[role] * g / g.norm().clamp_min(1e-8)
        torch.testing.assert_close(
            update.reshape(-1).float(), expected.reshape(-1).float(), rtol=1e-5, atol=1e-6
        )


def test_project_onto_normgrad_direction() -> None:
    grad = torch.randn(4, 3)
    direction = -grad.reshape(-1) / grad.reshape(-1).norm()
    scale = 0.37
    update = (scale * direction).reshape(4, 3)
    assert math.isclose(project_onto_normgrad_direction(update, grad), scale, rel_tol=1e-5)


def test_cheap_stats_shapes() -> None:
    _, reparam, _ = _reparam_case()
    grads = reparam.task.grad(reparam.initial)
    moms = [torch.zeros_like(g) for g in grads]
    vs = [torch.ones_like(g) for g in grads]
    gstats = global_cheap_stats(grads, moms, vs, reparam.initial, step=1, total_steps=4)
    assert gstats.shape == (4,)
    tstats = per_tensor_cheap_stats(
        grads[0], moms[0], vs[0], reparam.initial[0], global_grad_rms=gstats[0], progress=0.25
    )
    assert tstats.shape == (6,)


def test_structured_controller_param_counts() -> None:
    global_ctrl = GlobalScalarController()
    per_ctrl = PerTensorController()
    g_params = sum(p.numel() for p in global_ctrl.parameters())
    p_params = sum(p.numel() for p in per_ctrl.parameters())
    assert 10 <= g_params <= 50
    assert 10 <= p_params <= 50
    role_lrs = {"matrix": 0.05, "vector": 0.1}
    structured = StructuredTinyOptimizer(
        role_lrs=role_lrs, controller=global_ctrl, mode="global"
    )
    total = structured.parameter_count()
    assert total < 153
    assert 20 <= total <= 50


def test_structured_rollout_runs() -> None:
    _, reparam, _ = _reparam_case()
    controller = GlobalScalarController()
    with torch.no_grad():
        for param in controller.parameters():
            param.zero_()
    # output bias path: controller starts near 0; add output_bias=1 for a usable step
    opt = StructuredTinyOptimizer(
        role_lrs={"matrix": 0.05, "vector": 0.05},
        controller=controller,
        mode="global",
        output_bias=1.0,
    )
    ratio, aulc, finite = rollout_structured(opt, reparam, batch_size=8, steps=4)
    assert finite
    assert math.isfinite(ratio)
    assert math.isfinite(aulc)


def test_collect_and_analyze_effective_scale() -> None:
    _, reparam, scales = _reparam_case()
    student = TinyMLPOptimizer()
    torch.manual_seed(0)
    for p in student.parameters():
        torch.nn.init.normal_(p, std=0.1)
    trace = collect_effective_scale_trace(
        student,
        reparam,
        batch_size=8,
        steps=6,
        hidden_scales=scales,
    )
    assert len(trace.scales) == 6
    assert len(trace.scales[0]) == len(scales)
    analysis = analyze_effective_scales([trace])
    assert analysis["n"] == 6 * len(scales)
    assert "correlations" in analysis
    assert "r2_multivariate_observables" in analysis


def test_correlation_helpers() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    assert pearson_correlation(x, y) > 0.99
    assert spearman_correlation(x, y) > 0.99
    r2 = ridge_r2([[1.0], [2.0], [3.0], [4.0]], [2.0, 4.0, 6.0, 8.0], alpha=1e-6)
    assert r2 > 0.9


def test_bootstrap_and_paired() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    summary = bootstrap_mean_ci(values, n_boot=500, seed=1)
    assert math.isfinite(summary["mean"])
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    paired = paired_comparison([0.1, 0.2, 0.3], [0.2, 0.2, 0.4], n_boot=200, seed=2)
    assert paired["left_wins"] == 2
    assert paired["n"] == 3
    method = summarize_method_values(values, n_boot=200, seed=3)
    assert method["finite_fraction"] == 1.0


def test_tune_shared_role_and_evaluate() -> None:
    split = make_split("two_layer", (30.0,), seed_base=21, count=2, width=4, samples=16)
    cases = flatten_split(split)
    role_lrs, score = tune_shared_role_lrs(
        cases, batch_size=8, steps=4, candidates=(0.03, 0.1, 0.3)
    )
    assert set(role_lrs) == {"matrix", "vector"}
    assert math.isfinite(score)
    teacher = StaticRoleNormGrad(scales=expand_role_lrs(role_lrs, roles_for_case(cases[0])))
    spec_kind = make_static_shared_method(role_lrs, cases[0])
    assert spec_kind.kind == "teacher"
    _ratio, _aulc, finite = rollout_method(spec_kind, cases[0], batch_size=8, steps=4)
    assert finite
    assert teacher.scales[0] > 0


def test_privileged_method_evaluation_without_leak() -> None:
    base, reparam, scales = _reparam_case(seed=13)
    _ = base
    lr, _ = tune_teacher_lr("norm_grad_local", [reparam], batch_size=8, steps=4)
    spec = make_privileged_method(lr=lr, case=reparam, hidden_scales=scales)
    result = evaluate_method_split(
        spec, {"30.0": [reparam]}, batch_size=8, steps=4, n_boot=100, seed=0
    )
    assert result["uses_hidden_scales"] is True
    assert math.isfinite(result["loss_ratio"]["mean"])


def test_quick_distill_on_reparam_and_role_estimate() -> None:
    _, reparam, scales = _reparam_case(seed=31)
    lr, _ = tune_teacher_lr("norm_grad_local", [reparam], batch_size=8, steps=6)
    privileged = PrivilegedFunctionSpaceNormGrad(lr=lr, scales=scales)

    records = []
    params = reparam.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(reparam, batch_size=8, steps=6)
    from optdistil.distill.trajectory import TrajectoryRecord
    from optdistil.multitensor.features import build_multitensor_features, concatenate_features

    for step, indices in enumerate(batches, start=1):
        grads = reparam.task.grad_on_samples(params, indices)
        from optdistil.multitensor.stochastic import _observe_ema

        _observe_ema(momentums, second_moments, grads)
        features = concatenate_features(
            build_multitensor_features(
                params.tensors,
                grads,
                momentums,
                second_moments,
                step=step,
                total_steps=6,
                include_global=True,
            )
        )
        updates = privileged.step(params, grads)
        update_flat = torch.cat([u.reshape(-1) for u in updates])
        records.append(
            TrajectoryRecord(features=features.detach(), teacher_update=update_flat.detach())
        )
        params = params.add(updates)

    student, _loss = train_supervised_student(
        records, device=torch.device("cpu"), seed=1, epochs=3
    )
    assert student.parameter_count == 153
    role_lrs = estimate_role_scales_from_student(
        student, [reparam], batch_size=8, steps=6
    )
    assert set(role_lrs) == {"matrix", "vector"}
    assert all(v > 0 for v in role_lrs.values())

    spec = make_student_method(student)
    result = evaluate_method_split(
        spec, {"30.0": [reparam]}, batch_size=8, steps=6, n_boot=50, seed=0
    )
    assert math.isfinite(result["loss_ratio"]["mean"])


def test_fit_structured_from_privileged_param_count() -> None:
    cases = []
    scales_list = []
    for seed in (41, 42):
        _, reparam, scales = _reparam_case(seed=seed)
        cases.append(reparam)
        scales_list.append(scales)
    role_lrs = {"matrix": 0.1, "vector": 0.1}
    structured = fit_structured_from_privileged(
        cases,
        scales_list,
        batch_size=8,
        steps=4,
        mode="per_tensor",
        role_lrs=role_lrs,
        seed=0,
        epochs=5,
    )
    assert structured.parameter_count() < 153
    payload = evaluate_structured_split(
        structured, {"30.0": cases}, batch_size=8, steps=4
    )
    assert payload["parameter_count"] == structured.parameter_count()
    assert math.isfinite(payload["loss_ratio"]["mean"])


def test_select_student_scale_still_works_on_reparam() -> None:
    _, reparam, _ = _reparam_case(seed=51)
    student = TinyMLPOptimizer()
    scale, score = select_student_scale(
        student, [reparam], batch_size=8, steps=4, candidates=(0.1, 0.3)
    )
    assert scale > 0
    assert math.isfinite(score)


def test_existing_single_tensor_path_untouched() -> None:
    # Smoke: standard multitensor teacher still runs on non-reparameterized case.
    split = make_split("two_layer", (30.0,), seed_base=99, count=1, width=4, samples=12)
    case = flatten_split(split)[0]
    teacher = NormGradTensorWise(lr=0.1)
    params = case.initial.clone()
    for indices in batch_sequence(case, batch_size=8, steps=3):
        grads = case.task.grad_on_samples(params, indices)
        params = params.add(teacher.step(params, grads))
    assert params.is_finite()
