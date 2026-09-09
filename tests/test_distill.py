from pathlib import Path

import torch

from optdistil.distill.collect import collect_teacher_step
from optdistil.distill.losses import direction_loss, magnitude_loss
from optdistil.distill.trajectory import TrajectoryDataset, TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer
from optdistil.teachers.adamw import AdamWTeacher


def test_identical_updates_have_zero_distillation_losses() -> None:
    update = torch.tensor([1.0, -2.0, 3.0])
    assert torch.allclose(direction_loss(update, update), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(magnitude_loss(update, update), torch.tensor(0.0), atol=1e-6)


def test_tiny_student_preserves_element_count() -> None:
    student = TinyMLPOptimizer(feature_dim=8, hidden_dim=8, hidden_layers=2)
    features = torch.randn(32, 8)
    update = student(features)
    assert update.shape == (32,)
    assert student.parameter_count == 153


def test_collect_teacher_step_uses_teacher_independent_features() -> None:
    parameter = torch.randn(4, 4)
    grad = torch.randn_like(parameter)
    teacher = AdamWTeacher(lr=0.01)
    state = StudentState(parameter.shape, dtype=parameter.dtype)

    record = collect_teacher_step(
        parameter,
        grad,
        teacher=teacher,
        student_state=state,
        step=1,
        total_steps=10,
    )

    assert record.features.shape == (16, 8)
    assert record.teacher_update.shape == (16,)
    assert record.metadata["teacher"] == "AdamWTeacher"


def test_trajectory_dataset_roundtrip(tmp_path: Path) -> None:
    record = TrajectoryRecord(torch.randn(5, 8), torch.randn(5), {"step": 1})
    path = tmp_path / "trajectory.pt"
    TrajectoryDataset([record]).save(path)
    loaded = TrajectoryDataset.load(path)

    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0].features, record.features)
    torch.testing.assert_close(loaded[0].teacher_update, record.teacher_update)
    assert loaded[0].metadata == record.metadata
