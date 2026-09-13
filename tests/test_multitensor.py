from __future__ import annotations

import math

import torch

from optdistil.multitensor.features import build_multitensor_features, concatenate_features
from optdistil.multitensor.params import ParamCollection, flatten_updates, unflatten_updates
from optdistil.multitensor.stochastic import (
    apply_label_control,
    batch_sequence,
    collect_records,
    evaluate_student_split,
    evaluate_teacher_split,
    flatten_split,
    make_split,
    paired_differences,
    rollout_student,
    rollout_teacher,
    select_student_scale,
    summarize_ratios,
    train_supervised_student,
    tune_teacher_lr,
)
from optdistil.multitensor.tasks import make_residual_mlp, make_task, make_two_layer_mlp
from optdistil.multitensor.teachers import (
    NormGradGlobal,
    NormGradTensorWise,
    make_teacher,
)
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def _tiny_case(architecture: str = "two_layer", seed: int = 3, *, samples: int = 16) -> object:
    split = make_split(
        architecture,
        (30.0,),
        seed_base=seed,
        count=1,
        width=4,
        samples=samples,
        device="cpu",
    )
    return flatten_split(split)[0]


def test_param_collection_roundtrip() -> None:
    initial, _ = make_two_layer_mlp(1, input_dim=4, hidden_dim=4, output_dim=2, samples=8)
    flat = initial.flat_copy()
    assert flat.numel() == initial.total_numel()
    restored = initial.apply_flat(flat)
    for left, right in zip(initial, restored, strict=True):
        torch.testing.assert_close(left, right)
    updates = [0.1 * t for t in initial]
    moved = initial.add(updates)
    for left, right, u in zip(initial, moved, updates, strict=True):
        torch.testing.assert_close(left + u, right)
    assert not moved.is_finite() is False  # finite


def test_flatten_unflatten_updates() -> None:
    shapes = [torch.Size((3, 2)), torch.Size((2,))]
    tensors = [torch.arange(6.0).reshape(3, 2), torch.tensor([1.0, 2.0])]
    flat = flatten_updates(tensors)
    restored = unflatten_updates(flat, shapes)
    for left, right in zip(tensors, restored, strict=True):
        torch.testing.assert_close(left, right)


def test_tasks_are_deterministic_and_analytic_grads_match_autograd() -> None:
    for maker in (make_two_layer_mlp, make_residual_mlp):
        initial_a, task_a = maker(11, input_dim=4, hidden_dim=5, output_dim=2, samples=12)
        initial_b, task_b = maker(11, input_dim=4, hidden_dim=5, output_dim=2, samples=12)
        torch.testing.assert_close(task_a.inputs, task_b.inputs)
        for left, right in zip(initial_a, initial_b, strict=True):
            torch.testing.assert_close(left, right)

        params = ParamCollection([t.clone().requires_grad_(True) for t in initial_a])
        loss = task_a.loss(params)
        loss.backward()
        analytic = task_a.grad(initial_a)
        for a, g in zip(analytic, params, strict=True):
            torch.testing.assert_close(a, g.grad, rtol=1e-4, atol=1e-5)

        indices = torch.tensor([0, 2, 5, 7], dtype=torch.long)
        params_b = ParamCollection([t.clone().requires_grad_(True) for t in initial_a])
        loss_b = task_a.loss_on_samples(params_b, indices)
        loss_b.backward()
        analytic_b = task_a.grad_on_samples(initial_a, indices)
        for a, g in zip(analytic_b, params_b, strict=True):
            torch.testing.assert_close(a, g.grad, rtol=1e-4, atol=1e-5)


def test_task_architectures_have_different_parameter_geometry() -> None:
    _, two = make_task("two_layer", 1, width=8, samples=16)
    _, residual = make_task("residual", 1, width=8, samples=16)
    assert len(two.parameter_shapes) == 4
    assert len(residual.parameter_shapes) == 5
    assert two.parameter_shapes != residual.parameter_shapes


def test_normgrad_local_and_global_definitions_differ() -> None:
    case = _tiny_case()
    grads = case.task.grad(case.initial)
    local = NormGradTensorWise(lr=0.1).step(case.initial, grads)
    global_u = NormGradGlobal(lr=0.1).step(case.initial, grads)
    # Tensor-wise: each tensor has update norm ≈ lr.
    for update in local:
        assert abs(float(update.norm()) - 0.1) < 1e-4
    # Global: concatenated update norm ≈ lr, not each tensor.
    flat_global = torch.cat([u.reshape(-1) for u in global_u])
    assert abs(float(flat_global.norm()) - 0.1) < 1e-4
    # Local and global updates are not identical.
    flat_local = torch.cat([u.reshape(-1) for u in local])
    assert not torch.allclose(flat_local, flat_global)


def test_features_shape_and_global_channel() -> None:
    case = _tiny_case()
    grads = case.task.grad(case.initial)
    momentums = [torch.zeros_like(t) for t in case.initial]
    seconds = [torch.zeros_like(t) for t in case.initial]
    with_global = build_multitensor_features(
        case.initial.tensors, grads, momentums, seconds, step=1, total_steps=4, include_global=True
    )
    local_only = build_multitensor_features(
        case.initial.tensors,
        grads,
        momentums,
        seconds,
        step=1,
        total_steps=4,
        include_global=False,
    )
    assert len(with_global) == len(case.initial)
    for g, l in zip(with_global, local_only, strict=True):
        assert g.shape == l.shape
        assert g.shape[1] == 8
    flat = concatenate_features(with_global)
    assert flat.shape[0] == case.initial.total_numel()
    # Global channel is constant across elements of the same tensor and matches global RMS.
    g0 = with_global[0]
    assert torch.allclose(g0[:, 6], g0[0, 6])


def test_teacher_rollouts_are_finite_and_student_path_works() -> None:
    case = _tiny_case(samples=24)
    for method in ("sgd", "momentum", "norm_grad_local", "norm_grad_global", "adamw", "muon", "lbfgs"):
        metrics = rollout_teacher(method, 0.05, case, batch_size=4, steps=8)
        assert math.isfinite(metrics.loss_ratio)
        assert metrics.finite

    torch.manual_seed(0)
    student = TinyMLPOptimizer()
    assert student.parameter_count == 153
    student.set_output_scale(0.1)
    metrics, alignment = rollout_student(
        student,
        case,
        batch_size=4,
        steps=8,
        reference=NormGradTensorWise(lr=0.1),
    )
    assert metrics.finite
    assert alignment is not None
    assert alignment.tensor_diagnostics is not None
    assert len(alignment.tensor_diagnostics.names) == len(case.initial)
    fractions = alignment.tensor_diagnostics.update_norm_fraction_mean
    # Energy fractions ||u_l||^2 / ||u||^2 average to ~1 per step.
    assert abs(sum(fractions) - 1.0) < 1e-3


def test_collect_and_distill_short_path() -> None:
    case = _tiny_case(seed=21, samples=16)
    records = collect_records(
        NormGradTensorWise(lr=0.1),
        [case],
        batch_size=4,
        steps=6,
        teacher_name="norm_grad_local",
    )
    assert len(records) == 6
    student, loss = train_supervised_student(
        records,
        device=torch.device("cpu"),
        seed=0,
        epochs=3,
    )
    assert math.isfinite(loss)
    scale, score = select_student_scale(
        student,
        [case],
        batch_size=4,
        steps=6,
        candidates=(0.1, 0.3, 1.0),
    )
    assert scale > 0
    assert math.isfinite(score)


def test_label_controls_preserve_features() -> None:
    case = _tiny_case(seed=31)
    records = collect_records(
        NormGradTensorWise(lr=0.1),
        [case],
        batch_size=4,
        steps=4,
        teacher_name="norm_grad_local",
    )
    for control in ("shuffle_tasks", "permute_coords", "norm_only", "random_unit"):
        controlled = apply_label_control(records, label_control=control, seed=7)
        assert len(controlled) == len(records)
        torch.testing.assert_close(controlled[0].features, records[0].features)


def test_shuffle_control_handles_mixed_architectures() -> None:
    two = _tiny_case("two_layer", seed=51)
    residual = _tiny_case("residual", seed=52)
    records = collect_records(
        NormGradTensorWise(lr=0.1),
        [two, residual],
        batch_size=4,
        steps=3,
        teacher_name="norm_grad_local",
    )
    assert records[0].teacher_update.numel() != records[3].teacher_update.numel()
    controlled = apply_label_control(records, label_control="shuffle_tasks", seed=9)
    for original, corrupted in zip(records, controlled, strict=True):
        assert original.features.shape == corrupted.features.shape
        assert original.teacher_update.numel() == corrupted.teacher_update.numel()


def test_batch_sequence_identical_across_methods() -> None:
    case = _tiny_case(seed=41)
    first = batch_sequence(case, batch_size=4, steps=5)
    second = batch_sequence(case, batch_size=4, steps=5)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b)
    full = batch_sequence(case, batch_size=case.task.sample_count, steps=3)
    assert all(int(i.numel()) == case.task.sample_count for i in full)


def test_summarize_and_paired_differences() -> None:
    summary = summarize_ratios([1.0, 2.0, math.inf])
    assert summary.finite_fraction == 2 / 3
    paired = paired_differences([0.2, 0.1], [0.3, 0.05])
    assert paired["left_beats_right"] == 1


def test_mixed_architecture_split_evaluation() -> None:
    two_split = make_split("two_layer", (30.0,), seed_base=100, count=2, width=4, samples=16)
    residual_split = make_split("residual", (30.0,), seed_base=200, count=2, width=4, samples=16)
    mixed = {
        30.0: two_split[30.0] + residual_split[30.0],
    }
    lr, _ = tune_teacher_lr(
        "norm_grad_local",
        mixed[30.0][:2],
        batch_size=4,
        steps=6,
        candidates=(0.05, 0.1),
    )
    teacher_eval = evaluate_teacher_split(
        "norm_grad_local",
        lr,
        mixed,
        batch_size=4,
        steps=6,
    )
    assert "by_architecture" in teacher_eval
    assert set(teacher_eval["by_architecture"]) == {"two_layer", "residual"}

    records = collect_records(
        make_teacher("norm_grad_local", lr),
        mixed[30.0],
        batch_size=4,
        steps=6,
        teacher_name="norm_grad_local",
    )
    student, _ = train_supervised_student(
        records, device=torch.device("cpu"), seed=0, epochs=4
    )
    select_student_scale(student, mixed[30.0], batch_size=4, steps=6)
    student_eval = evaluate_student_split(student, mixed, batch_size=4, steps=6)
    assert student_eval["loss_ratio"]["mean"] < 10.0
    assert "by_architecture" in student_eval


def test_residual_vs_two_layer_grad_structure_differs() -> None:
    initial_t, task_t = make_task("two_layer", 5, width=6, samples=12)
    initial_r, task_r = make_task("residual", 5, width=6, samples=12)
    grads_t = task_t.grad(initial_t)
    grads_r = task_r.grad(initial_r)
    assert len(grads_t) != len(grads_r)
    # W1 gradients should not be identical even under same seed/data construction.
    assert not torch.allclose(grads_t[0], grads_r[0])
