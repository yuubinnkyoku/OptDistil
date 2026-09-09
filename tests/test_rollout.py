import torch

from optdistil.distill.rollout import collect_teacher_trajectory, evaluate_imitation, rollout_student
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import TinyMLPOptimizer
from optdistil.tasks.quadratic import QuadraticTask
from optdistil.teachers.adamw import AdamWTeacher


def test_teacher_trajectory_and_student_rollout_are_finite() -> None:
    torch.manual_seed(7)
    initial = torch.randn(4, 4) * 0.2
    target = torch.zeros_like(initial)
    task = QuadraticTask(target, curvature=1.0)

    records, teacher_rollout = collect_teacher_trajectory(
        initial,
        task,
        teacher=AdamWTeacher(lr=0.03),
        steps=6,
    )

    assert len(records) == 6
    assert len(teacher_rollout.losses) == 7
    assert teacher_rollout.finite
    assert records[0].metadata["loss_before"] >= records[0].metadata["loss_after"]

    torch.manual_seed(11)
    student = TinyMLPOptimizer()
    train_student(student, records, epochs=4, lr=3e-3)
    imitation = evaluate_imitation(student, records)
    student_rollout = rollout_student(student, initial, task, steps=4)

    assert set(imitation) == {"total", "direction", "magnitude"}
    assert all(torch.isfinite(torch.tensor(value)) for value in imitation.values())
    assert student_rollout.finite
