from __future__ import annotations

import torch

from optdistil.distill.collect import collect_teacher_step
from optdistil.distill.train import train_student
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer
from optdistil.tasks.quadratic import QuadraticTask
from optdistil.teachers.adamw import AdamWTeacher


def main() -> None:
    torch.manual_seed(0)
    steps = 48
    parameter = torch.randn(8, 8)
    task = QuadraticTask(torch.zeros_like(parameter))
    teacher = AdamWTeacher(lr=0.03)
    state = StudentState(parameter.shape, dtype=parameter.dtype)

    records = []
    for step in range(1, steps + 1):
        grad = task.grad(parameter)
        record = collect_teacher_step(
            parameter,
            grad,
            teacher=teacher,
            student_state=state,
            step=step,
            total_steps=steps,
        )
        records.append(record.cpu())
        parameter.add_(record.teacher_update.reshape_as(parameter))

    student = TinyMLPOptimizer(hidden_dim=8, hidden_layers=2)
    history = train_student(student, records, epochs=150, lr=3e-3)

    print(f"student parameters: {student.parameter_count}")
    print(f"initial distillation loss: {history[0]:.6f}")
    print(f"final distillation loss:   {history[-1]:.6f}")


if __name__ == "__main__":
    main()
