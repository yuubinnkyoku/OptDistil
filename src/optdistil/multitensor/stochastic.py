from __future__ import annotations

import math
import statistics
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor

from optdistil.distill.alignment import flattened_cosine
from optdistil.distill.losses import DistillationLossWeights, distillation_loss
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.multitensor.features import (
    build_multitensor_features,
    concatenate_features,
    split_update,
)
from optdistil.multitensor.params import ParamCollection
from optdistil.multitensor.tasks import MultiTensorTask, make_task
from optdistil.multitensor.teachers import (
    TEACHER_METHODS,
    MultiTensorTeacher,
    make_teacher,
)
from optdistil.students.tiny_mlp import TinyMLPOptimizer

TRAIN_CONDITIONS = (30.0, 300.0)
OOD_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
DEFAULT_BATCH_SIZES = (4, 8, 16, 32)
TEACHER_LRS = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)
STUDENT_SCALE_CANDIDATES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
OUTER_LR_CANDIDATES = (1e-3, 3e-3)

FeatureMode = str  # "local_global" | "local_only"


@dataclass(frozen=True, slots=True)
class MultiTensorCase:
    initial: ParamCollection
    task: MultiTensorTask
    batch_seed: int
    architecture: str
    width: int
    condition: float


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    loss_ratio: float
    aulc: float
    finite: bool
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TensorDiagnostics:
    """Per-tensor update geometry averaged over a rollout."""

    names: tuple[str, ...]
    update_norm_fraction_mean: list[float]
    cosine_teacher_per_tensor: list[float]
    relative_update_scale: list[float]
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "update_norm_fraction_mean": self.update_norm_fraction_mean,
            "cosine_teacher_per_tensor": self.cosine_teacher_per_tensor,
            "relative_update_scale": self.relative_update_scale,
            "steps": self.steps,
        }


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    cosine_teacher_mean: float
    cosine_negative_gradient_mean: float
    update_norm_ratio_mean: float
    steps: int
    tensor_diagnostics: TensorDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.tensor_diagnostics is not None:
            payload["tensor_diagnostics"] = self.tensor_diagnostics.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class RatioSummary:
    mean: float
    median: float
    geometric_mean: float
    std: float
    finite_fraction: float
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "geometric_mean": self.geometric_mean,
            "std": self.std,
            "finite_fraction": self.finite_fraction,
            "values": list(self.values),
        }


def geometric_mean(values: Sequence[float], *, eps: float = 1e-12) -> float:
    finite = [max(float(value), eps) for value in values if math.isfinite(value)]
    if not finite:
        return math.inf
    return math.exp(statistics.fmean(math.log(value) for value in finite))


def summarize_ratios(values: Sequence[float]) -> RatioSummary:
    if not values:
        raise ValueError("at least one ratio is required")
    floats = [float(value) for value in values]
    finite_flags = [math.isfinite(value) for value in floats]
    finite_values = [value for value, flag in zip(floats, finite_flags, strict=True) if flag]
    if not finite_values:
        return RatioSummary(
            mean=math.inf,
            median=math.inf,
            geometric_mean=math.inf,
            std=0.0,
            finite_fraction=0.0,
            values=tuple(floats),
        )
    penalized = [value if math.isfinite(value) else 1e6 for value in floats]
    return RatioSummary(
        mean=statistics.fmean(penalized),
        median=statistics.median(penalized),
        geometric_mean=geometric_mean(penalized),
        std=statistics.pstdev(penalized) if len(penalized) > 1 else 0.0,
        finite_fraction=statistics.fmean(finite_flags),
        values=tuple(floats),
    )


def git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_split(
    architecture: str,
    conditions: Sequence[float],
    *,
    seed_base: int,
    count: int,
    width: int,
    samples: int,
    device: torch.device | str = "cpu",
) -> dict[float, list[MultiTensorCase]]:
    device = torch.device(device)
    split: dict[float, list[MultiTensorCase]] = {}
    for condition_index, condition in enumerate(conditions):
        cases: list[MultiTensorCase] = []
        for index in range(count):
            seed = seed_base + 10000 * condition_index + index
            initial, task = make_task(
                architecture,
                seed,
                width=width,
                samples=samples,
                input_condition=condition,
                device=device,
            )
            cases.append(
                MultiTensorCase(
                    initial=initial,
                    task=task,
                    batch_seed=1_000_000 + seed,
                    architecture=architecture,
                    width=width,
                    condition=float(condition),
                )
            )
        split[float(condition)] = cases
    return split


def flatten_split(split: dict[float, list[MultiTensorCase]]) -> list[MultiTensorCase]:
    return [case for cases in split.values() for case in cases]


def batch_sequence(case: MultiTensorCase, *, batch_size: int, steps: int) -> list[Tensor]:
    sample_count = case.task.sample_count
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or batch_size > sample_count:
        raise ValueError("batch_size must lie in [1, sample_count]")
    if batch_size >= sample_count:
        full = torch.arange(sample_count, dtype=torch.long)
        return [full for _ in range(steps)]
    generator = torch.Generator(device="cpu").manual_seed(case.batch_seed)
    return [
        torch.randperm(sample_count, generator=generator)[:batch_size] for _ in range(steps)
    ]


def _normalized_aulc(losses: Sequence[float], initial_loss: float) -> float:
    if len(losses) <= 1:
        return 1.0
    denominator = max(abs(initial_loss), 1e-12)
    area = 0.0
    for left, right in pairwise(losses):
        if not (math.isfinite(left) and math.isfinite(right)):
            return math.inf
        area += 0.5 * (left + right)
    return area / ((len(losses) - 1) * denominator)


def _finalize(
    losses: Sequence[float],
    initial_loss: float,
) -> RolloutMetrics:
    final_loss = losses[-1]
    finite = all(math.isfinite(value) for value in losses)
    return RolloutMetrics(
        loss_ratio=final_loss / max(abs(initial_loss), 1e-12),
        aulc=_normalized_aulc(losses, initial_loss),
        finite=finite,
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=tuple(losses),
    )


def _feature_builder(
    parameters: Sequence[Tensor],
    grads: Sequence[Tensor],
    momentums: Sequence[Tensor],
    second_moments: Sequence[Tensor],
    *,
    step: int,
    total_steps: int,
    include_global: bool,
) -> Tensor:
    per_tensor = build_multitensor_features(
        parameters,
        grads,
        momentums,
        second_moments,
        step=step,
        total_steps=total_steps,
        include_global=include_global,
    )
    return concatenate_features(per_tensor)


def _observe_ema(
    momentums: list[Tensor],
    second_moments: list[Tensor],
    grads: Sequence[Tensor],
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> tuple[list[Tensor], list[Tensor]]:
    for index, grad in enumerate(grads):
        momentums[index].mul_(beta1).add_(grad, alpha=1.0 - beta1)
        second_moments[index].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    return momentums, second_moments


@torch.no_grad()
def rollout_teacher(
    method: str,
    lr: float,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
) -> RolloutMetrics:
    params = case.initial.clone()
    teacher = make_teacher(method, lr)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    for indices in batches:
        grads = case.task.grad_on_samples(params, indices)
        updates = teacher.step(params, grads)
        params = params.add(updates)
        if not params.is_finite():
            losses.append(math.inf)
            return _finalize(losses, initial_loss)
        losses.append(float(case.task.loss(params)))
    return _finalize(losses, initial_loss)


def tune_teacher_lr(
    method: str,
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    candidates: Sequence[float] = TEACHER_LRS,
) -> tuple[float, float]:
    if not candidates:
        raise ValueError("at least one learning-rate candidate is required")
    scored = []
    for lr in candidates:
        score = statistics.fmean(
            rollout_teacher(method, lr, case, batch_size=batch_size, steps=steps).loss_ratio
            for case in validation_cases
        )
        scored.append((score, lr))
    score, lr = min(scored, key=lambda item: item[0])
    return lr, score


def evaluate_teacher_split(
    method: str,
    lr: float,
    split: dict[float, list[MultiTensorCase]],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    by_arch: dict[str, list[float]] = {}
    by_condition: dict[str, float] = {}
    for condition, cases in split.items():
        condition_ratios = []
        for case in cases:
            metrics = rollout_teacher(method, lr, case, batch_size=batch_size, steps=steps)
            ratios.append(metrics.loss_ratio)
            aulcs.append(metrics.aulc)
            condition_ratios.append(metrics.loss_ratio)
            by_arch.setdefault(case.architecture, []).append(metrics.loss_ratio)
        by_condition[str(condition)] = statistics.fmean(condition_ratios)
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "by_condition": by_condition,
        "by_architecture": {
            name: summarize_ratios(values).to_dict() for name, values in by_arch.items()
        },
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
        "lr": lr,
    }


@torch.no_grad()
def rollout_student(
    student: TinyMLPOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    reference: MultiTensorTeacher | None = None,
    reference_grads_fn=None,
) -> tuple[RolloutMetrics, AlignmentMetrics | None]:
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(params))
    losses = [initial_loss]
    student.eval()

    shapes = params.shapes()
    names = tuple(
        getattr(case.task, "parameter_names", tuple(f"t{i}" for i in range(len(params))))
    )

    cosine_teacher: list[float] = []
    cosine_neg_grad: list[float] = []
    norm_ratios: list[float] = []
    tensor_cosines: list[list[float]] = [[] for _ in range(len(params))]
    tensor_norm_fractions: list[list[float]] = [[] for _ in range(len(params))]
    tensor_update_norms: list[list[float]] = [[] for _ in range(len(params))]

    for step, indices in enumerate(batches, start=1):
        grads = case.task.grad_on_samples(params, indices)
        _observe_ema(momentums, second_moments, grads)
        features = _feature_builder(
            params.tensors,
            grads,
            momentums,
            second_moments,
            step=step,
            total_steps=steps,
            include_global=include_global,
        )
        flat_update = student(features)
        update_list = split_update(flat_update, shapes)

        if reference is not None:
            reference_update_list = reference.step(params, grads)
            student_flat = flat_update
            teacher_flat = torch.cat([u.reshape(-1) for u in reference_update_list])
            cosine_teacher.append(flattened_cosine(student_flat, teacher_flat))
            neg_grad_flat = torch.cat([-g.reshape(-1) for g in grads])
            cosine_neg_grad.append(flattened_cosine(student_flat, neg_grad_flat))
            update_norm = float(student_flat.float().norm())
            reference_norm = float(teacher_flat.float().norm())
            norm_ratios.append(update_norm / max(reference_norm, 1e-12))

            total_energy = max(update_norm * update_norm, 1e-24)
            for t_index, (u, ru) in enumerate(zip(update_list, reference_update_list, strict=True)):
                u_flat = u.reshape(-1)
                ru_flat = ru.reshape(-1)
                tensor_cosines[t_index].append(flattened_cosine(u_flat, ru_flat))
                u_sq = float(u_flat.float().square().sum())
                tensor_norm_fractions[t_index].append(u_sq / total_energy)
                tensor_update_norms[t_index].append(
                    float(u_flat.float().norm()) / max(float(ru_flat.float().norm()), 1e-12)
                )

        params = params.add(update_list)
        if not params.is_finite():
            losses.append(math.inf)
            metrics = _finalize(losses, initial_loss)
            alignment = _build_alignment(
                cosine_teacher,
                cosine_neg_grad,
                norm_ratios,
                names,
                tensor_cosines,
                tensor_norm_fractions,
                tensor_update_norms,
            )
            return metrics, alignment
        losses.append(float(case.task.loss(params)))

    metrics = _finalize(losses, initial_loss)
    alignment = _build_alignment(
        cosine_teacher,
        cosine_neg_grad,
        norm_ratios,
        names,
        tensor_cosines,
        tensor_norm_fractions,
        tensor_update_norms,
    )
    return metrics, alignment


def _safe_mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _build_alignment(
    cosine_teacher: Sequence[float],
    cosine_neg_grad: Sequence[float],
    norm_ratios: Sequence[float],
    names: Sequence[str],
    tensor_cosines: Sequence[Sequence[float]],
    tensor_norm_fractions: Sequence[Sequence[float]],
    tensor_update_norms: Sequence[Sequence[float]],
) -> AlignmentMetrics | None:
    if not cosine_teacher:
        return None
    diagnostics = TensorDiagnostics(
        names=tuple(names),
        update_norm_fraction_mean=[_safe_mean(values) for values in tensor_norm_fractions],
        cosine_teacher_per_tensor=[_safe_mean(values) for values in tensor_cosines],
        relative_update_scale=[_safe_mean(values) for values in tensor_update_norms],
        steps=len(cosine_teacher),
    )
    return AlignmentMetrics(
        cosine_teacher_mean=_safe_mean(cosine_teacher),
        cosine_negative_gradient_mean=_safe_mean(cosine_neg_grad),
        update_norm_ratio_mean=_safe_mean(norm_ratios),
        steps=len(cosine_teacher),
        tensor_diagnostics=diagnostics,
    )


def evaluate_student_split(
    student: TinyMLPOptimizer,
    split: dict[float, list[MultiTensorCase]],
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    reference_teacher: MultiTensorTeacher | None = None,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    cosine_teacher: list[float] = []
    cosine_neg_grad: list[float] = []
    norm_ratios: list[float] = []
    by_condition: dict[str, float] = {}
    by_arch: dict[str, list[float]] = {}
    tensor_diag_accum: list[dict[str, Any]] = []

    for condition, cases in split.items():
        condition_ratios = []
        for case in cases:
            metrics, alignment = rollout_student(
                student,
                case,
                batch_size=batch_size,
                steps=steps,
                include_global=include_global,
                reference=reference_teacher,
            )
            ratios.append(metrics.loss_ratio)
            aulcs.append(metrics.aulc)
            condition_ratios.append(metrics.loss_ratio)
            by_arch.setdefault(case.architecture, []).append(metrics.loss_ratio)
            if alignment is not None:
                cosine_teacher.append(alignment.cosine_teacher_mean)
                cosine_neg_grad.append(alignment.cosine_negative_gradient_mean)
                norm_ratios.append(alignment.update_norm_ratio_mean)
                if alignment.tensor_diagnostics is not None:
                    tensor_diag_accum.append(alignment.tensor_diagnostics.to_dict())
        by_condition[str(condition)] = statistics.fmean(condition_ratios)

    payload: dict[str, Any] = {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "by_condition": by_condition,
        "by_architecture": {
            name: summarize_ratios(values).to_dict() for name, values in by_arch.items()
        },
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
    }
    if cosine_teacher:
        payload["alignment"] = {
            "cosine_teacher_mean": statistics.fmean(cosine_teacher),
            "cosine_negative_gradient_mean": statistics.fmean(cosine_neg_grad),
            "update_norm_ratio_mean": statistics.fmean(norm_ratios),
        }
        if tensor_diag_accum:
            payload["tensor_diagnostics"] = _average_tensor_diagnostics(tensor_diag_accum)
    return payload


def _average_tensor_diagnostics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = items[0]["names"]
    fractions = [
        statistics.fmean(item["update_norm_fraction_mean"][i] for item in items)
        for i in range(len(names))
    ]
    cosines = [
        statistics.fmean(item["cosine_teacher_per_tensor"][i] for item in items)
        for i in range(len(names))
    ]
    scales = [
        statistics.fmean(item["relative_update_scale"][i] for item in items)
        for i in range(len(names))
    ]
    return {
        "names": names,
        "update_norm_fraction_mean": fractions,
        "cosine_teacher_per_tensor": cosines,
        "relative_update_scale": scales,
    }


@torch.no_grad()
def collect_records(
    teacher: MultiTensorTeacher,
    cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    teacher_name: str = "unknown",
    label_control: str = "none",
    control_seed: int = 12345,
    control_source_teacher: MultiTensorTeacher | None = None,
) -> list[TrajectoryRecord]:
    """Collect (features, teacher_update) pairs across cases.

    ``control_source_teacher`` optionally provides analytic updates used by the
    ``analytic_imitation`` control without rolling out that teacher's trajectory.
    """
    records: list[TrajectoryRecord] = []
    for task_index, case in enumerate(cases):
        params = case.initial.clone()
        # Each case needs a fresh optimizer-state teacher (AdamW/Muon/L-BFGS).
        local_teacher = _fresh_teacher_like(teacher, teacher_name)
        momentums = [torch.zeros_like(t) for t in params]
        second_moments = [torch.zeros_like(t) for t in params]
        batches = batch_sequence(case, batch_size=batch_size, steps=steps)
        for step, indices in enumerate(batches, start=1):
            grads = case.task.grad_on_samples(params, indices)
            _observe_ema(momentums, second_moments, grads)
            features = _feature_builder(
                params.tensors,
                grads,
                momentums,
                second_moments,
                step=step,
                total_steps=steps,
                include_global=include_global,
            )
            update_list = local_teacher.step(params, grads)
            update_flat = torch.cat([u.reshape(-1) for u in update_list])
            records.append(
                TrajectoryRecord(
                    features=features.detach(),
                    teacher_update=update_flat.detach(),
                    metadata={
                        "teacher": teacher_name,
                        "task_index": task_index,
                        "step": step,
                        "batch_size": batch_size,
                        "architecture": case.architecture,
                        "width": case.width,
                        "full_loss_before": float(case.task.loss(params)),
                        "label_control": label_control,
                        "include_global": include_global,
                    },
                )
            )
            params = params.add(update_list)
    if label_control == "none":
        return records
    return apply_label_control(
        records,
        label_control=label_control,
        seed=control_seed,
        case_shapes=[case.initial.shapes() for case in cases],
        features_grad_slices=None,
    )


def _fresh_teacher_like(teacher: MultiTensorTeacher, teacher_name: str) -> MultiTensorTeacher:
    if teacher_name in TEACHER_METHODS:
        lr = float(getattr(teacher, "lr", 0.1))
        return make_teacher(teacher_name, lr)
    return teacher


def apply_label_control(
    records: Sequence[TrajectoryRecord],
    *,
    label_control: str,
    seed: int,
    case_shapes: Sequence[Sequence[torch.Size]] | None = None,
    features_grad_slices: Any = None,
) -> list[TrajectoryRecord]:
    if label_control == "none":
        return list(records)
    if not records:
        raise ValueError("records must be non-empty")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if label_control == "shuffle_tasks":
        # Mixed architectures can have different total numel; shuffle only within
        # groups that share the same update length.
        by_numel: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            by_numel.setdefault(record.teacher_update.numel(), []).append(index)
        shuffled_updates: list[Tensor | None] = [None] * len(records)
        for indices in by_numel.values():
            order = torch.randperm(len(indices), generator=generator).tolist()
            for slot, source in enumerate(order):
                shuffled_updates[indices[slot]] = (
                    records[indices[source]].teacher_update.detach().clone()
                )
        return [
            TrajectoryRecord(
                features=record.features.detach().clone(),
                teacher_update=shuffled_updates[index],
                metadata={**record.metadata, "label_control": label_control},
            )
            for index, record in enumerate(records)
        ]

    if label_control == "permute_coords":
        return [
            TrajectoryRecord(
                features=record.features.detach().clone(),
                teacher_update=record.teacher_update.detach()[
                    torch.randperm(record.teacher_update.numel(), generator=generator)
                ].clone(),
                metadata={**record.metadata, "label_control": label_control},
            )
            for record in records
        ]

    if label_control == "norm_only":
        controlled: list[TrajectoryRecord] = []
        for record in records:
            update = record.teacher_update.detach().clone()
            norm = update.norm().clamp_min(1e-12)
            direction = -record.features[:, 0]
            direction_norm = direction.norm().clamp_min(1e-12)
            controlled.append(
                TrajectoryRecord(
                    features=record.features.detach().clone(),
                    teacher_update=direction * (norm / direction_norm),
                    metadata={**record.metadata, "label_control": label_control},
                )
            )
        return controlled

    if label_control == "random_unit":
        controlled = []
        for record in records:
            update = record.teacher_update.detach()
            norm = update.norm().clamp_min(1e-12)
            noise = torch.randn(update.shape, generator=generator)
            noise = noise / noise.norm().clamp_min(1e-12)
            controlled.append(
                TrajectoryRecord(
                    features=record.features.detach().clone(),
                    teacher_update=noise * norm,
                    metadata={**record.metadata, "label_control": label_control},
                )
            )
        return controlled

    if label_control == "analytic_imitation":
        # Labels replaced by an analytic rule generator; caller should pass
        # pre-built analytic records instead of using this branch.
        raise ValueError(
            "analytic_imitation must be generated by collecting from the analytic teacher directly"
        )

    raise ValueError(f"unknown label control: {label_control}")


def select_student_scale(
    student: TinyMLPOptimizer,
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    candidates: Sequence[float] = STUDENT_SCALE_CANDIDATES,
) -> tuple[float, float]:
    if not candidates:
        raise ValueError("at least one scale candidate is required")
    scored = []
    for scale in candidates:
        student.set_output_scale(scale)
        score = statistics.fmean(
            rollout_student(
                student,
                case,
                batch_size=batch_size,
                steps=steps,
                include_global=include_global,
            )[0].loss_ratio
            for case in validation_cases
        )
        scored.append((score, scale))
    score, scale = min(scored, key=lambda item: item[0])
    student.set_output_scale(scale)
    return scale, score


def train_supervised_student(
    records: Sequence[TrajectoryRecord],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    lr: float = 3e-3,
    weights: DistillationLossWeights | None = None,
) -> tuple[TinyMLPOptimizer, float]:
    torch.manual_seed(seed)
    student = TinyMLPOptimizer().to(device)
    outer = torch.optim.AdamW(student.parameters(), lr=lr)
    records = list(records)
    history: list[float] = []
    student.train()
    for _ in range(epochs):
        total = 0.0
        for record in records:
            outer.zero_grad(set_to_none=True)
            predicted = student(record.features.to(device))
            loss, _ = distillation_loss(
                predicted,
                record.teacher_update.to(device),
                weights=weights,
            )
            loss.backward()
            outer.step()
            total += float(loss.detach())
        history.append(total / len(records))
    student.eval()
    return student, history[-1]


def differentiable_stochastic_rollout(
    student: TinyMLPOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
) -> tuple[Tensor, Tensor]:
    """Closed-loop rollout with first-order meta-gradients."""
    params = [t.detach().clone().requires_grad_(False) for t in case.initial.tensors]
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = case.task.loss(ParamCollection(params)).detach().clamp_min(1e-12)
    ratios: list[Tensor] = []
    shapes = case.initial.shapes()

    for step, indices in enumerate(batches, start=1):
        with torch.no_grad():
            grads = case.task.grad_on_samples(ParamCollection(params), indices)
            _observe_ema(momentums, second_moments, grads)
            features = _feature_builder(
                params,
                grads,
                momentums,
                second_moments,
                step=step,
                total_steps=steps,
                include_global=include_global,
            )
        flat_update = student(features)
        update_list = split_update(flat_update, shapes)
        params = [p + u for p, u in zip(params, update_list, strict=True)]
        ratios.append(case.task.loss(ParamCollection(params)) / initial_loss)
    return ratios[-1], torch.stack(ratios).mean()


def meta_objective(
    student: TinyMLPOptimizer,
    cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    final_weight: float = 0.7,
) -> Tensor:
    values = []
    for case in cases:
        final_ratio, mean_ratio = differentiable_stochastic_rollout(
            student,
            case,
            batch_size=batch_size,
            steps=steps,
            include_global=include_global,
        )
        values.append(final_weight * final_ratio + (1.0 - final_weight) * mean_ratio)
    return torch.stack(values).mean()


def clone_state_dict(module: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def train_closed_loop_meta(
    student: TinyMLPOptimizer,
    train_cases: Sequence[MultiTensorCase],
    validation_cases: Sequence[MultiTensorCase],
    *,
    batch_size: int,
    steps: int,
    iterations: int,
    outer_lr: float,
    include_global: bool = True,
    grad_clip: float = 1.0,
    validation_interval: int = 3,
) -> float:
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive")
    outer = torch.optim.Adam(student.parameters(), lr=outer_lr)
    best_validation = statistics.fmean(
        rollout_student(
            student,
            case,
            batch_size=batch_size,
            steps=steps,
            include_global=include_global,
        )[0].loss_ratio
        for case in validation_cases
    )
    best_state = clone_state_dict(student)

    for iteration in range(iterations):
        student.train()
        outer.zero_grad(set_to_none=True)
        objective = meta_objective(
            student,
            train_cases,
            batch_size=batch_size,
            steps=steps,
            include_global=include_global,
        )
        if not torch.isfinite(objective):
            break
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            break
        outer.step()
        if (iteration + 1) % validation_interval != 0 and iteration + 1 != iterations:
            continue
        validation = statistics.fmean(
            rollout_student(
                student,
                case,
                batch_size=batch_size,
                steps=steps,
                include_global=include_global,
            )[0].loss_ratio
            for case in validation_cases
        )
        if validation < best_validation:
            best_validation = validation
            best_state = clone_state_dict(student)

    student.load_state_dict(best_state)
    student.eval()
    return best_validation


def select_outer_lr_and_train(
    initial_state: dict[str, Tensor],
    *,
    device: torch.device,
    train_cases: Sequence[MultiTensorCase],
    validation_cases: Sequence[MultiTensorCase],
    batch_size: int,
    steps: int,
    iterations: int,
    include_global: bool = True,
    candidates: Sequence[float] = OUTER_LR_CANDIDATES,
) -> tuple[TinyMLPOptimizer, float, float]:
    scored = []
    for outer_lr in candidates:
        student = TinyMLPOptimizer().to(device)
        student.load_state_dict(
            {name: value.detach().clone() for name, value in initial_state.items()}
        )
        validation = train_closed_loop_meta(
            student,
            train_cases,
            validation_cases,
            batch_size=batch_size,
            steps=steps,
            iterations=iterations,
            outer_lr=outer_lr,
            include_global=include_global,
        )
        scored.append((validation, outer_lr, student))
    validation, outer_lr, student = min(scored, key=lambda item: item[0])
    return student, outer_lr, validation


def train_direct_meta_student(
    train_cases: Sequence[MultiTensorCase],
    validation_cases: Sequence[MultiTensorCase],
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    iterations: int,
    include_global: bool = True,
    seed: int = 0,
    candidates: Sequence[float] = OUTER_LR_CANDIDATES,
) -> tuple[TinyMLPOptimizer, float, float]:
    scored = []
    for outer_lr in candidates:
        torch.manual_seed(seed)
        student = TinyMLPOptimizer().to(device)
        validation = train_closed_loop_meta(
            student,
            train_cases,
            validation_cases,
            batch_size=batch_size,
            steps=steps,
            iterations=iterations,
            outer_lr=outer_lr,
            include_global=include_global,
        )
        scored.append((validation, outer_lr, student))
    validation, outer_lr, student = min(scored, key=lambda item: item[0])
    return student, outer_lr, validation


def paired_differences(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("paired sequences must have equal length")
    deltas = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    return {
        "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
        "median_delta": statistics.median(deltas) if deltas else 0.0,
        "left_beats_right": sum(delta < 0.0 for delta in deltas),
        "total": len(deltas),
        "deltas": deltas,
    }


def batch_transfer_matrix(
    student: TinyMLPOptimizer,
    split: dict[float, list[MultiTensorCase]],
    *,
    train_batch_size: int,
    eval_batch_sizes: Sequence[int],
    steps: int,
    include_global: bool = True,
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    first_case = flatten_split(split)[0]
    max_samples = first_case.task.sample_count
    for batch_size in eval_batch_sizes:
        if batch_size > max_samples:
            continue
        metrics = evaluate_student_split(
            student,
            split,
            batch_size=batch_size,
            steps=steps,
            include_global=include_global,
        )
        matrix[str(batch_size)] = {
            "loss_ratio_mean": metrics["loss_ratio"]["mean"],
            "loss_ratio_median": metrics["loss_ratio"]["median"],
            "aulc_mean": metrics["aulc"]["mean"],
            "matches_train_batch": batch_size == train_batch_size,
        }
    return matrix


def width_transfer_matrix(
    student: TinyMLPOptimizer,
    *,
    architecture: str,
    widths: Sequence[int],
    seed_base: int,
    count: int,
    samples: int,
    batch_size: int,
    steps: int,
    include_global: bool = True,
    condition: float = 30.0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for width in widths:
        split = make_split(
            architecture,
            (condition,),
            seed_base=seed_base + 50_000 * width,
            count=count,
            width=width,
            samples=samples,
            device=device,
        )
        metrics = evaluate_student_split(
            student,
            split,
            batch_size=batch_size,
            steps=steps,
            include_global=include_global,
        )
        matrix[str(width)] = {
            "loss_ratio_mean": metrics["loss_ratio"]["mean"],
            "loss_ratio_median": metrics["loss_ratio"]["median"],
            "finite_fraction": metrics["finite_fraction"],
        }
    return matrix


def artifact_metadata(
    *,
    run_id: str,
    commit_sha: str,
    config: dict[str, Any],
    split_specs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "config": config,
        "split_specs": split_specs,
        "student_parameters": TinyMLPOptimizer().parameter_count,
        "teacher_methods": list(TEACHER_METHODS),
    }
