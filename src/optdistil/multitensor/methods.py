"""Unified rollout evaluation for reparameterization stress methods."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from optdistil.multitensor.effective_scale import (
    collect_effective_scale_trace,
)
from optdistil.multitensor.features import build_multitensor_features, concatenate_features
from optdistil.multitensor.reparam import (
    PrivilegedFunctionSpaceNormGrad,
    ProjectedNormGradDirection,
)
from optdistil.multitensor.static_role import (
    StaticRoleNormGrad,
    expand_role_lrs,
    roles_for_case,
)
from optdistil.multitensor.stats_utils import summarize_method_values
from optdistil.multitensor.stochastic import (
    MultiTensorCase,
    _observe_ema,
    batch_sequence,
    split_update,
)
from optdistil.multitensor.structured import StructuredTinyOptimizer
from optdistil.multitensor.teachers import MultiTensorTeacher
from optdistil.students.tiny_mlp import TinyMLPOptimizer


@dataclass
class MethodSpec:
    name: str
    kind: str  # teacher | student | structured | student_projected
    teacher: MultiTensorTeacher | None = None
    student: TinyMLPOptimizer | None = None
    structured: StructuredTinyOptimizer | None = None
    role_lrs: dict[str, float] | None = None
    uses_hidden_scales: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def _rollout_teacher_method(
    teacher: MultiTensorTeacher,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> tuple[float, float, bool]:
    params = case.initial.clone()
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    for indices in batches:
        grads = case.task.grad_on_samples(params, indices)
        updates = teacher.step(params, grads)
        params = params.add(updates)
        if not params.is_finite():
            losses.append(math.inf)
            return math.inf, math.inf, False
        losses.append(float(case.task.loss(params)))
    final = losses[-1]
    ratio = final / max(abs(initial_loss), 1e-12)
    area = 0.0
    for left, right in itertools.pairwise(losses):
        area += 0.5 * (left + right)
    aulc = area / ((len(losses) - 1) * max(abs(initial_loss), 1e-12))
    return ratio, aulc, True


@torch.no_grad()
def _rollout_student_method(
    student: TinyMLPOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    project_onto_normgrad: bool = False,
) -> tuple[float, float, bool]:
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    shapes = params.shapes()
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    student.eval()
    projector = ProjectedNormGradDirection()
    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)
        features = build_multitensor_features(
            params.tensors,
            grads,
            momentums,
            second_moments,
            step=step,
            total_steps=steps,
            include_global=include_global,
        )
        flat = student(concatenate_features(features))
        update_list = split_update(flat, shapes)
        if project_onto_normgrad:
            update_list = projector.project(update_list, grads)
        params = params.add(update_list)
        if not params.is_finite():
            losses.append(math.inf)
            return math.inf, math.inf, False
        losses.append(float(case.task.loss(params)))
    final = losses[-1]
    ratio = final / max(abs(initial_loss), 1e-12)
    area = 0.0
    for left, right in itertools.pairwise(losses):
        area += 0.5 * (left + right)
    aulc = area / ((len(losses) - 1) * max(abs(initial_loss), 1e-12))
    return ratio, aulc, True


@torch.no_grad()
def _rollout_structured_method(
    optimizer: StructuredTinyOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> tuple[float, float, bool]:
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    roles = roles_for_case(case)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)
        updates = optimizer.step_with_stats(
            params,
            grads,
            momentums,
            second_moments,
            step=step,
            total_steps=steps,
            roles=roles,
        )
        params = params.add(updates)
        if not params.is_finite():
            losses.append(math.inf)
            return math.inf, math.inf, False
        losses.append(float(case.task.loss(params)))
    final = losses[-1]
    ratio = final / max(abs(initial_loss), 1e-12)
    area = 0.0
    for left, right in itertools.pairwise(losses):
        area += 0.5 * (left + right)
    aulc = area / ((len(losses) - 1) * max(abs(initial_loss), 1e-12))
    return ratio, aulc, True


def rollout_method(
    spec: MethodSpec,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> tuple[float, float, bool]:
    if spec.kind == "teacher":
        if spec.teacher is None:
            raise ValueError("teacher method requires teacher")
        return _rollout_teacher_method(spec.teacher, case, batch_size=batch_size, steps=steps)
    if spec.kind == "student":
        if spec.student is None:
            raise ValueError("student method requires student")
        return _rollout_student_method(spec.student, case, batch_size=batch_size, steps=steps)
    if spec.kind == "student_projected":
        if spec.student is None:
            raise ValueError("student_projected method requires student")
        return _rollout_student_method(
            spec.student,
            case,
            batch_size=batch_size,
            steps=steps,
            project_onto_normgrad=True,
        )
    if spec.kind == "structured":
        if spec.structured is None:
            raise ValueError("structured method requires structured optimizer")
        return _rollout_structured_method(
            spec.structured, case, batch_size=batch_size, steps=steps
        )
    raise ValueError(f"unknown method kind: {spec.kind}")


def evaluate_method_split(
    spec: MethodSpec,
    split: dict[float, list[MultiTensorCase]],
    *,
    batch_size: int,
    steps: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    for cases in split.values():
        for case in cases:
            ratio, aulc, _finite = rollout_method(spec, case, batch_size=batch_size, steps=steps)
            ratios.append(ratio)
            aulcs.append(aulc)
    return {
        "name": spec.name,
        "kind": spec.kind,
        "uses_hidden_scales": spec.uses_hidden_scales,
        "loss_ratio": summarize_method_values(ratios, n_boot=n_boot, seed=seed),
        "aulc": summarize_method_values(aulcs, n_boot=n_boot, seed=seed + 1),
        "parameter_count": spec.metadata.get("parameter_count"),
        "metadata": spec.metadata,
    }


def make_privileged_method(
    *,
    lr: float,
    case: MultiTensorCase,
    hidden_scales: Sequence[float],
) -> MethodSpec:
    teacher = PrivilegedFunctionSpaceNormGrad(lr=lr, scales=hidden_scales)
    return MethodSpec(
        name="privileged_function_space_normgrad",
        kind="teacher",
        teacher=teacher,
        uses_hidden_scales=True,
        metadata={"lr": lr},
    )


def make_static_shared_method(
    role_lrs: dict[str, float],
    case: MultiTensorCase,
) -> MethodSpec:
    teacher = StaticRoleNormGrad(scales=expand_role_lrs(role_lrs, roles_for_case(case)))
    return MethodSpec(
        name="static_shared_role_normgrad",
        kind="teacher",
        teacher=teacher,
        uses_hidden_scales=False,
        metadata={"role_lrs": dict(role_lrs)},
    )


def make_static_arch_method(
    tensor_lrs: Sequence[float],
    case: MultiTensorCase,
) -> MethodSpec:
    teacher = StaticRoleNormGrad(scales=list(tensor_lrs))
    return MethodSpec(
        name="static_architecture_role_normgrad",
        kind="teacher",
        teacher=teacher,
        uses_hidden_scales=False,
        metadata={"tensor_lrs": list(tensor_lrs), "architecture": case.architecture},
    )


def make_ordinary_method(lr: float) -> MethodSpec:
    from optdistil.multitensor.teachers import NormGradTensorWise

    return MethodSpec(
        name="ordinary_local_normgrad",
        kind="teacher",
        teacher=NormGradTensorWise(lr=lr),
        uses_hidden_scales=False,
        metadata={"lr": lr},
    )


def make_student_method(student: TinyMLPOptimizer, name: str = "student_153p") -> MethodSpec:
    return MethodSpec(
        name=name,
        kind="student",
        student=student,
        uses_hidden_scales=False,
        metadata={"parameter_count": student.parameter_count},
    )


def make_student_projected_method(
    student: TinyMLPOptimizer, name: str = "student_projected_normgrad"
) -> MethodSpec:
    return MethodSpec(
        name=name,
        kind="student_projected",
        student=student,
        uses_hidden_scales=False,
        metadata={"parameter_count": student.parameter_count},
    )


def make_frozen_role_method(
    role_lrs: dict[str, float],
    case: MultiTensorCase,
    *,
    source: str = "student",
) -> MethodSpec:
    teacher = StaticRoleNormGrad(scales=expand_role_lrs(role_lrs, roles_for_case(case)))
    return MethodSpec(
        name="frozen_student_role_scales",
        kind="teacher",
        teacher=teacher,
        uses_hidden_scales=False,
        metadata={"role_lrs": dict(role_lrs), "source": source},
    )


def make_structured_method(
    optimizer: StructuredTinyOptimizer,
    name: str,
) -> MethodSpec:
    return MethodSpec(
        name=name,
        kind="structured",
        structured=optimizer,
        uses_hidden_scales=False,
        metadata={
            "parameter_count": optimizer.parameter_count(),
            "mode": optimizer.mode,
            "role_lrs": dict(optimizer.role_lrs),
        },
    )


def estimate_role_scales_from_student(
    student: TinyMLPOptimizer,
    cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, float]:
    """Estimate frozen role LRs as mean student effective scale per role on validation."""
    by_role: dict[str, list[float]] = {}
    for case in cases:
        trace = collect_effective_scale_trace(
            student, case, batch_size=batch_size, steps=steps
        )
        for row in trace.records:
            by_role.setdefault(str(row["role"]), []).append(float(row["effective_scale"]))
    role_lrs: dict[str, float] = {}
    for role, values in by_role.items():
        finite = [v for v in values if math.isfinite(v)]
        role_lrs[role] = statistics_fmean_positive(finite)
    return role_lrs


def statistics_fmean_positive(values: Sequence[float]) -> float:
    if not values:
        return 0.05
    mean = sum(values) / len(values)
    return mean if mean > 0 else 0.05
