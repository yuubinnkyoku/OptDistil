from __future__ import annotations

import math

import torch

from optdistil.distill.experimental_features import (
    build_matrix_aware_no_ema,
    build_matrix_aware_no_progress,
)
from optdistil.distill.features import build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights
from optdistil.distill.stochastic import (
    StochasticCase,
    apply_label_control,
    batch_sequence,
    batch_transfer_matrix,
    collect_stochastic_records,
    evaluate_student_split,
    flatten_split,
    geometric_mean,
    make_split,
    paired_differences,
    rollout_student,
    rollout_teacher,
    select_student_scale,
    summarize_ratios,
    train_supervised_student,
    tune_teacher_lr,
)
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def _tiny_case(seed: int = 3, *, samples: int = 12) -> StochasticCase:
    split = make_split(
        (30.0,),
        seed_base=seed,
        count=1,
        size=4,
        samples=samples,
        device="cpu",
    )
    return flatten_split(split)[0]


def test_make_split_is_deterministic_and_disjoint() -> None:
    split_a = make_split((30.0, 300.0), seed_base=1000, count=2, size=4, samples=16)
    split_b = make_split((30.0, 300.0), seed_base=1000, count=2, size=4, samples=16)
    split_c = make_split((30.0, 300.0), seed_base=2000, count=2, size=4, samples=16)

    torch.testing.assert_close(split_a[30.0][0].initial, split_b[30.0][0].initial)
    assert not torch.allclose(split_a[30.0][0].initial, split_c[30.0][0].initial)
    assert split_a[30.0][0].batch_seed != split_a[300.0][0].batch_seed


def test_batch_sequence_is_reproducible_and_valid() -> None:
    case = _tiny_case()
    first = batch_sequence(case, batch_size=4, steps=5)
    second = batch_sequence(case, batch_size=4, steps=5)
    assert len(first) == 5
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right)
        assert left.numel() == 4
        assert int(left.max()) < case.task.sample_count


def test_teacher_and_student_rollouts_are_finite() -> None:
    case = _tiny_case()
    teacher_metrics = rollout_teacher("adamw", 0.05, case, batch_size=4, steps=6)
    assert teacher_metrics.finite
    assert math.isfinite(teacher_metrics.loss_ratio)
    assert 0.0 <= teacher_metrics.aulc <= 1.5

    torch.manual_seed(0)
    student = TinyMLPOptimizer()
    student.set_output_scale(0.1)
    student_metrics, alignment = rollout_student(
        student,
        case,
        batch_size=4,
        steps=6,
        reference="adamw",
        reference_lr=0.05,
    )
    assert student_metrics.finite
    assert alignment is not None
    assert -1.0 <= alignment.cosine_teacher_mean <= 1.0
    assert alignment.steps == 6


def test_label_controls_preserve_features_and_change_updates() -> None:
    case = _tiny_case()
    records = collect_stochastic_records(
        "adamw",
        0.05,
        [case],
        batch_size=4,
        steps=4,
        feature_builder=build_matrix_aware_features,
    )
    shuffled = apply_label_control(records, label_control="shuffle_tasks", seed=1)
    permuted = apply_label_control(records, label_control="permute_coords", seed=2)
    norm_only = apply_label_control(records, label_control="norm_only", seed=3)
    random_unit = apply_label_control(records, label_control="random_unit", seed=4)

    assert len(shuffled) == len(records)
    torch.testing.assert_close(shuffled[0].features, records[0].features)
    assert not torch.allclose(shuffled[0].teacher_update, records[0].teacher_update)
    torch.testing.assert_close(
        permuted[0].teacher_update.norm(), records[0].teacher_update.norm()
    )
    torch.testing.assert_close(
        norm_only[0].teacher_update.norm(), records[0].teacher_update.norm()
    )
    torch.testing.assert_close(
        random_unit[0].teacher_update.norm(), records[0].teacher_update.norm()
    )


def test_feature_ablations_keep_width_and_zero_channels() -> None:
    case = _tiny_case()
    parameter = case.initial
    grad = case.task.grad(parameter)
    zeros = torch.zeros_like(parameter)
    base = build_matrix_aware_features(parameter, grad, zeros, zeros, step=1, total_steps=4)
    no_ema = build_matrix_aware_no_ema(parameter, grad, zeros, zeros, step=1, total_steps=4)
    no_progress = build_matrix_aware_no_progress(
        parameter, grad, zeros, zeros, step=1, total_steps=4
    )

    assert base.shape == no_ema.shape == no_progress.shape
    assert base.shape[1] == 8
    assert torch.all(no_ema[:, 1] == 0)
    assert torch.all(no_ema[:, 2] == 0)
    assert torch.all(no_progress[:, 7] == 0)


def test_select_scale_and_short_distillation_path() -> None:
    case = _tiny_case(seed=11)
    validation = [case]
    records = collect_stochastic_records(
        "norm_gradient",
        0.1,
        validation,
        batch_size=4,
        steps=4,
    )
    student, loss = train_supervised_student(
        records,
        device=torch.device("cpu"),
        seed=5,
        epochs=3,
        weights=DistillationLossWeights(direction=1.0, magnitude=0.0),
    )
    assert math.isfinite(loss)
    assert student.parameter_count == 153
    scale, score = select_student_scale(
        student,
        validation,
        batch_size=4,
        steps=4,
    )
    assert scale > 0.0
    assert math.isfinite(score)


def test_summaries_and_paired_differences() -> None:
    summary = summarize_ratios([0.2, 0.4, math.inf])
    assert summary.finite_fraction < 1.0
    assert math.isfinite(summary.mean)
    assert geometric_mean([1.0, 4.0]) == 2.0

    paired = paired_differences([0.1, 0.2], [0.2, 0.15])
    assert paired["left_beats_right"] == 1
    assert paired["total"] == 2


def test_batch_transfer_matrix_uses_requested_sizes() -> None:
    split = make_split((30.0,), seed_base=77, count=1, size=4, samples=16)
    torch.manual_seed(0)
    student = TinyMLPOptimizer()
    student.set_output_scale(0.2)
    matrix = batch_transfer_matrix(
        student,
        split,
        train_batch_size=8,
        eval_batch_sizes=(4, 8, 16),
        steps=4,
    )
    assert set(matrix) == {"4", "8", "16"}
    assert matrix["8"]["matches_train_batch"] is True


def test_evaluate_student_split_reports_alignment() -> None:
    split = make_split((30.0,), seed_base=91, count=1, size=4, samples=12)
    torch.manual_seed(1)
    student = TinyMLPOptimizer()
    student.set_output_scale(0.1)
    payload = evaluate_student_split(
        student,
        split,
        batch_size=4,
        steps=4,
        reference="adamw",
        reference_lr=0.05,
    )
    assert "alignment" in payload
    assert "loss_ratio" in payload
    assert "aulc" in payload
    assert "by_condition" in payload


def test_tune_teacher_lr_returns_candidate() -> None:
    case = _tiny_case(seed=13)
    lr, score = tune_teacher_lr("adamw", [case], batch_size=4, steps=4)
    assert lr in (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)
    assert math.isfinite(score) or math.isinf(score)


def test_trajectory_record_metadata_from_collection() -> None:
    case = _tiny_case(seed=17)
    records = collect_stochastic_records(
        "adamw",
        0.05,
        [case],
        batch_size=4,
        steps=3,
    )
    assert all(isinstance(record, TrajectoryRecord) for record in records)
    assert records[0].metadata["batch_size"] == 4
    assert records[0].metadata["label_control"] == "none"
