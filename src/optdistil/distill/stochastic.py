from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Tensor

from optdistil.distill.alignment import flattened_cosine
from optdistil.distill.features import FeatureBuilder, build_matrix_aware_features
from optdistil.distill.losses import DistillationLossWeights, distillation_loss
from optdistil.distill.secant_features import SecantFeatureState
from optdistil.distill.trajectory import TrajectoryRecord
from optdistil.students.tiny_mlp import StudentState, TinyMLPOptimizer
from optdistil.tasks.frozen_readout_mlp import make_frozen_readout_mlp
from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.gradient_direction import GradientDirectionTeacher
from optdistil.teachers.muon import MuonTeacher

TeacherMethod = Literal["adamw", "norm_gradient", "muon"]
LabelControl = Literal["none", "shuffle_tasks", "permute_coords", "norm_only", "random_unit"]

TRAIN_CONDITIONS = (30.0, 300.0)
OOD_CONDITIONS = (10.0, 100.0, 1000.0, 3000.0)
DEFAULT_BATCH_SIZES = (4, 8, 16, 32, 64)
TEACHER_LRS = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)
STUDENT_SCALE_CANDIDATES = (0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
OUTER_LR_CANDIDATES = (1e-3, 3e-3)
HISTORY_SIZE = 4
SECANT_SCALES = (0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
BOOTSTRAP_SCALES = (0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True, slots=True)
class StochasticCase:
    """One FrozenReadoutMLP task with a deterministic minibatch RNG stream."""

    initial: Tensor
    task: Any
    batch_seed: int


@dataclass(frozen=True, slots=True)
class StochasticRolloutMetrics:
    """Downstream full-data metrics for one stochastic rollout."""

    loss_ratio: float
    aulc: float
    finite: bool
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    """Student-vs-teacher / student-vs--grad geometry averaged over a rollout."""

    cosine_teacher_mean: float
    cosine_negative_gradient_mean: float
    update_norm_ratio_mean: float
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    # Non-finite rollouts count as failures, not as silently dropped outliers.
    penalized = [value if math.isfinite(value) else 1e6 for value in floats]
    return RatioSummary(
        mean=statistics.fmean(penalized),
        median=statistics.median(penalized),
        geometric_mean=geometric_mean(penalized),
        std=statistics.pstdev(penalized) if len(penalized) > 1 else 0.0,
        finite_fraction=statistics.fmean(finite_flags),
        values=tuple(floats),
    )


def make_teacher(method: TeacherMethod, lr: float):
    if method == "adamw":
        return AdamWTeacher(lr=lr)
    if method == "norm_gradient":
        return GradientDirectionTeacher(lr=lr)
    if method == "muon":
        return MuonTeacher(lr=lr)
    raise ValueError(f"unknown teacher method: {method}")


def make_split(
    conditions: Sequence[float],
    *,
    seed_base: int,
    count: int,
    size: int,
    samples: int,
    device: torch.device | str = "cpu",
) -> dict[float, list[StochasticCase]]:
    device = torch.device(device)
    split: dict[float, list[StochasticCase]] = {}
    for condition_index, condition in enumerate(conditions):
        cases: list[StochasticCase] = []
        for index in range(count):
            seed = seed_base + 10000 * condition_index + index
            initial, task = make_frozen_readout_mlp(
                seed,
                hidden_dim=size,
                input_dim=size,
                output_dim=max(2, size // 2),
                samples=samples,
                input_condition=condition,
                device=device,
            )
            cases.append(StochasticCase(initial, task, 1_000_000 + seed))
        split[float(condition)] = cases
    return split


def flatten_split(split: dict[float, list[StochasticCase]]) -> list[StochasticCase]:
    return [case for cases in split.values() for case in cases]


def batch_sequence(case: StochasticCase, *, batch_size: int, steps: int) -> list[Tensor]:
    sample_count = case.task.sample_count
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or batch_size > sample_count:
        raise ValueError("batch_size must lie in [1, sample_count]")
    if batch_size == sample_count:
        full = torch.arange(sample_count, dtype=torch.long)
        return [full for _ in range(steps)]

    generator = torch.Generator(device="cpu").manual_seed(case.batch_seed)
    return [
        torch.randperm(sample_count, generator=generator)[:batch_size] for _ in range(steps)
    ]


@torch.no_grad()
def rollout_teacher(
    method: TeacherMethod,
    lr: float,
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
) -> StochasticRolloutMetrics:
    parameter = case.initial.detach().clone()
    teacher = make_teacher(method, lr)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(parameter))
    losses = [initial_loss]

    for indices in batches:
        grad = case.task.grad_on_samples(parameter, indices)
        parameter = parameter + teacher.step(parameter, grad)
        if not torch.isfinite(parameter).all():
            losses.append(math.inf)
            return StochasticRolloutMetrics(
                loss_ratio=math.inf,
                aulc=math.inf,
                finite=False,
                initial_loss=initial_loss,
                final_loss=math.inf,
                losses=tuple(losses),
            )
        losses.append(float(case.task.loss(parameter)))

    final_loss = losses[-1]
    finite = all(math.isfinite(value) for value in losses)
    loss_ratio = final_loss / max(abs(initial_loss), 1e-12)
    aulc = _normalized_aulc(losses, initial_loss)
    return StochasticRolloutMetrics(
        loss_ratio=loss_ratio,
        aulc=aulc,
        finite=finite,
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=tuple(losses),
    )


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


def tune_teacher_lr(
    method: TeacherMethod,
    validation_cases: Sequence[StochasticCase],
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


@torch.no_grad()
def rollout_secant(
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
    secant_scale: float,
    bootstrap_scale: float,
    history_size: int = HISTORY_SIZE,
) -> StochasticRolloutMetrics:
    parameter = case.initial.detach().clone()
    ema = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    state = SecantFeatureState(history_size=history_size, normalize_direction=False)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(parameter))
    losses = [initial_loss]

    for step, indices in enumerate(batches, start=1):
        grad = case.task.grad_on_samples(parameter, indices)
        momentum, second_moment = ema.observe(grad)
        features = state.build(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        if step == 1:
            grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
            update = -bootstrap_scale * grad / grad_rms
        else:
            update = secant_scale * features[:, 5].reshape_as(parameter)
        parameter = parameter + update
        if not torch.isfinite(parameter).all():
            losses.append(math.inf)
            return StochasticRolloutMetrics(
                loss_ratio=math.inf,
                aulc=math.inf,
                finite=False,
                initial_loss=initial_loss,
                final_loss=math.inf,
                losses=tuple(losses),
            )
        losses.append(float(case.task.loss(parameter)))

    final_loss = losses[-1]
    finite = all(math.isfinite(value) for value in losses)
    return StochasticRolloutMetrics(
        loss_ratio=final_loss / max(abs(initial_loss), 1e-12),
        aulc=_normalized_aulc(losses, initial_loss),
        finite=finite,
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=tuple(losses),
    )


def tune_secant(
    validation_cases: Sequence[StochasticCase],
    *,
    batch_size: int,
    steps: int,
) -> tuple[dict[str, float], float]:
    scored = []
    for bootstrap_scale in BOOTSTRAP_SCALES:
        for secant_scale in SECANT_SCALES:
            score = statistics.fmean(
                rollout_secant(
                    case,
                    batch_size=batch_size,
                    steps=steps,
                    secant_scale=secant_scale,
                    bootstrap_scale=bootstrap_scale,
                ).loss_ratio
                for case in validation_cases
            )
            scored.append((score, bootstrap_scale, secant_scale))
    score, bootstrap_scale, secant_scale = min(scored, key=lambda item: item[0])
    return {"bootstrap_scale": bootstrap_scale, "secant_scale": secant_scale}, score


@torch.no_grad()
def rollout_student(
    student: TinyMLPOptimizer,
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    reference: TeacherMethod | None = None,
    reference_lr: float | None = None,
) -> tuple[StochasticRolloutMetrics, AlignmentMetrics | None]:
    parameter = case.initial.detach().clone()
    state = StudentState(parameter.shape, device=parameter.device, dtype=parameter.dtype)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = float(case.task.loss(parameter))
    losses = [initial_loss]
    student.eval()

    reference_teacher = None
    if reference is not None:
        if reference_lr is None:
            raise ValueError("reference_lr is required when measuring alignment")
        reference_teacher = make_teacher(reference, reference_lr)

    cosine_teacher: list[float] = []
    cosine_neg_grad: list[float] = []
    norm_ratios: list[float] = []

    for step, indices in enumerate(batches, start=1):
        grad = case.task.grad_on_samples(parameter, indices)
        momentum, second_moment = state.observe(grad)
        features = feature_builder(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        update = student(features).reshape_as(parameter)
        if reference_teacher is not None:
            reference_update = reference_teacher.step(parameter, grad)
            cosine_teacher.append(flattened_cosine(update, reference_update))
            cosine_neg_grad.append(flattened_cosine(update, -grad))
            update_norm = float(update.float().norm())
            reference_norm = float(reference_update.float().norm())
            norm_ratios.append(update_norm / max(reference_norm, 1e-12))
        parameter = parameter + update
        if not torch.isfinite(parameter).all():
            losses.append(math.inf)
            metrics = StochasticRolloutMetrics(
                loss_ratio=math.inf,
                aulc=math.inf,
                finite=False,
                initial_loss=initial_loss,
                final_loss=math.inf,
                losses=tuple(losses),
            )
            alignment = (
                AlignmentMetrics(
                    cosine_teacher_mean=_safe_mean(cosine_teacher),
                    cosine_negative_gradient_mean=_safe_mean(cosine_neg_grad),
                    update_norm_ratio_mean=_safe_mean(norm_ratios),
                    steps=len(cosine_teacher),
                )
                if reference_teacher is not None
                else None
            )
            return metrics, alignment
        losses.append(float(case.task.loss(parameter)))

    final_loss = losses[-1]
    finite = all(math.isfinite(value) for value in losses)
    metrics = StochasticRolloutMetrics(
        loss_ratio=final_loss / max(abs(initial_loss), 1e-12),
        aulc=_normalized_aulc(losses, initial_loss),
        finite=finite,
        initial_loss=initial_loss,
        final_loss=final_loss,
        losses=tuple(losses),
    )
    alignment = (
        AlignmentMetrics(
            cosine_teacher_mean=_safe_mean(cosine_teacher),
            cosine_negative_gradient_mean=_safe_mean(cosine_neg_grad),
            update_norm_ratio_mean=_safe_mean(norm_ratios),
            steps=len(cosine_teacher),
        )
        if reference_teacher is not None
        else None
    )
    return metrics, alignment


def _safe_mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_student_split(
    student: TinyMLPOptimizer,
    split: dict[float, list[StochasticCase]],
    *,
    batch_size: int,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    reference: TeacherMethod | None = None,
    reference_lr: float | None = None,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    cosine_teacher: list[float] = []
    cosine_neg_grad: list[float] = []
    norm_ratios: list[float] = []
    by_condition: dict[str, float] = {}

    for condition, cases in split.items():
        condition_ratios = []
        for case in cases:
            metrics, alignment = rollout_student(
                student,
                case,
                batch_size=batch_size,
                steps=steps,
                feature_builder=feature_builder,
                reference=reference,
                reference_lr=reference_lr,
            )
            ratios.append(metrics.loss_ratio)
            aulcs.append(metrics.aulc)
            condition_ratios.append(metrics.loss_ratio)
            if alignment is not None:
                cosine_teacher.append(alignment.cosine_teacher_mean)
                cosine_neg_grad.append(alignment.cosine_negative_gradient_mean)
                norm_ratios.append(alignment.update_norm_ratio_mean)
        by_condition[str(condition)] = statistics.fmean(condition_ratios)

    ratio_summary = summarize_ratios(ratios)
    aulc_summary = summarize_ratios(aulcs)
    payload: dict[str, Any] = {
        "loss_ratio": ratio_summary.to_dict(),
        "aulc": aulc_summary.to_dict(),
        "by_condition": by_condition,
        "finite_fraction": ratio_summary.finite_fraction,
    }
    if cosine_teacher:
        payload["alignment"] = {
            "cosine_teacher_mean": statistics.fmean(cosine_teacher),
            "cosine_negative_gradient_mean": statistics.fmean(cosine_neg_grad),
            "update_norm_ratio_mean": statistics.fmean(norm_ratios),
        }
    return payload


def evaluate_teacher_split(
    method: TeacherMethod,
    lr: float,
    split: dict[float, list[StochasticCase]],
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    by_condition: dict[str, float] = {}
    for condition, cases in split.items():
        condition_ratios = []
        for case in cases:
            metrics = rollout_teacher(method, lr, case, batch_size=batch_size, steps=steps)
            ratios.append(metrics.loss_ratio)
            aulcs.append(metrics.aulc)
            condition_ratios.append(metrics.loss_ratio)
        by_condition[str(condition)] = statistics.fmean(condition_ratios)
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "by_condition": by_condition,
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
        "lr": lr,
    }


def evaluate_secant_split(
    split: dict[float, list[StochasticCase]],
    *,
    batch_size: int,
    steps: int,
    secant_scale: float,
    bootstrap_scale: float,
) -> dict[str, Any]:
    ratios: list[float] = []
    aulcs: list[float] = []
    by_condition: dict[str, float] = {}
    for condition, cases in split.items():
        condition_ratios = []
        for case in cases:
            metrics = rollout_secant(
                case,
                batch_size=batch_size,
                steps=steps,
                secant_scale=secant_scale,
                bootstrap_scale=bootstrap_scale,
            )
            ratios.append(metrics.loss_ratio)
            aulcs.append(metrics.aulc)
            condition_ratios.append(metrics.loss_ratio)
        by_condition[str(condition)] = statistics.fmean(condition_ratios)
    return {
        "loss_ratio": summarize_ratios(ratios).to_dict(),
        "aulc": summarize_ratios(aulcs).to_dict(),
        "by_condition": by_condition,
        "finite_fraction": summarize_ratios(ratios).finite_fraction,
    }


@torch.no_grad()
def collect_stochastic_records(
    method: TeacherMethod,
    lr: float,
    cases: Sequence[StochasticCase],
    *,
    batch_size: int,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    label_control: LabelControl = "none",
    control_seed: int = 12345,
) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for task_index, case in enumerate(cases):
        parameter = case.initial.detach().clone()
        teacher = make_teacher(method, lr)
        student_state = StudentState(
            parameter.shape, device=parameter.device, dtype=parameter.dtype
        )
        batches = batch_sequence(case, batch_size=batch_size, steps=steps)
        for step, indices in enumerate(batches, start=1):
            grad = case.task.grad_on_samples(parameter, indices)
            momentum, second_moment = student_state.observe(grad)
            features = feature_builder(
                parameter,
                grad,
                momentum,
                second_moment,
                step=step,
                total_steps=steps,
            )
            update = teacher.step(parameter, grad).detach()
            records.append(
                TrajectoryRecord(
                    features=features.detach(),
                    teacher_update=update.reshape(-1),
                    metadata={
                        "teacher": method,
                        "task_index": task_index,
                        "step": step,
                        "batch_size": batch_size,
                        "full_loss_before": float(case.task.loss(parameter)),
                        "label_control": label_control,
                    },
                )
            )
            parameter = parameter + update

    if label_control == "none":
        return records
    return apply_label_control(records, label_control=label_control, seed=control_seed)


def apply_label_control(
    records: Sequence[TrajectoryRecord],
    *,
    label_control: LabelControl,
    seed: int,
) -> list[TrajectoryRecord]:
    """Corrupt only teacher labels so observation features stay realistic."""
    if label_control == "none":
        return list(records)
    if not records:
        raise ValueError("records must be non-empty")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if label_control == "shuffle_tasks":
        order = torch.randperm(len(records), generator=generator).tolist()
        return [
            TrajectoryRecord(
                features=record.features.detach().clone(),
                teacher_update=records[order[index]].teacher_update.detach().clone(),
                metadata={**record.metadata, "label_control": label_control},
            )
            for index, record in enumerate(records)
        ]

    if label_control == "permute_coords":
        numel = records[0].teacher_update.numel()
        permutation = torch.randperm(numel, generator=generator)
        return [
            TrajectoryRecord(
                features=record.features.detach().clone(),
                teacher_update=record.teacher_update.detach()[permutation].clone(),
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

    raise ValueError(f"unknown label control: {label_control}")


def select_student_scale(
    student: TinyMLPOptimizer,
    validation_cases: Sequence[StochasticCase],
    *,
    batch_size: int,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    candidates: Sequence[float] = STUDENT_SCALE_CANDIDATES,
) -> tuple[float, float]:
    """Select deployment scale on validation rollouts, never on teacher update norms."""
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
                feature_builder=feature_builder,
            )[0].loss_ratio
            for case in validation_cases
        )
        scored.append((score, scale))
    score, scale = min(scored, key=lambda item: item[0])
    student.set_output_scale(scale)
    return scale, score


def differentiable_stochastic_rollout(
    student: TinyMLPOptimizer,
    case: StochasticCase,
    *,
    batch_size: int,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Closed-loop rollout with first-order meta-gradients.

    Mini-batch gradients are stop-gradient observations. Meta-gradients flow through the
    student update into the parameter and then into the full-data task loss. The gradient
    itself is not differentiated, matching deployment observations.
    """
    parameter = case.initial.detach().clone()
    momentum = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    initial_loss = case.task.loss(parameter).detach().clamp_min(eps)
    ratios: list[Tensor] = []

    for step, indices in enumerate(batches, start=1):
        grad = case.task.grad_on_samples(parameter, indices).detach()
        momentum = beta1 * momentum + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * grad.square()
        features = feature_builder(
            parameter,
            grad,
            momentum,
            second_moment,
            step=step,
            total_steps=steps,
        )
        parameter = parameter + student(features).reshape_as(parameter)
        ratios.append(case.task.loss(parameter) / initial_loss)
    return ratios[-1], torch.stack(ratios).mean()


def meta_objective(
    student: TinyMLPOptimizer,
    cases: Sequence[StochasticCase],
    *,
    batch_size: int,
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    final_weight: float = 0.7,
) -> Tensor:
    values = []
    for case in cases:
        final_ratio, mean_ratio = differentiable_stochastic_rollout(
            student,
            case,
            batch_size=batch_size,
            steps=steps,
            feature_builder=feature_builder,
        )
        values.append(final_weight * final_ratio + (1.0 - final_weight) * mean_ratio)
    return torch.stack(values).mean()


def clone_state_dict(module: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def train_closed_loop_meta(
    student: TinyMLPOptimizer,
    train_cases: Sequence[StochasticCase],
    validation_cases: Sequence[StochasticCase],
    *,
    batch_size: int,
    steps: int,
    iterations: int,
    outer_lr: float,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
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
            feature_builder=feature_builder,
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
            feature_builder=feature_builder,
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
                feature_builder=feature_builder,
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
    train_cases: Sequence[StochasticCase],
    validation_cases: Sequence[StochasticCase],
    batch_size: int,
    steps: int,
    iterations: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    candidates: Sequence[float] = OUTER_LR_CANDIDATES,
) -> tuple[TinyMLPOptimizer, float, float]:
    scored = []
    for outer_lr in candidates:
        student = TinyMLPOptimizer().to(device)
        student.load_state_dict(clone_state_dict_dict(initial_state))
        validation = train_closed_loop_meta(
            student,
            train_cases,
            validation_cases,
            batch_size=batch_size,
            steps=steps,
            iterations=iterations,
            outer_lr=outer_lr,
            feature_builder=feature_builder,
        )
        scored.append((validation, outer_lr, student))
    validation, outer_lr, student = min(scored, key=lambda item: item[0])
    return student, outer_lr, validation


def clone_state_dict_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def train_direct_meta_student(
    train_cases: Sequence[StochasticCase],
    validation_cases: Sequence[StochasticCase],
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    iterations: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
    seed: int = 0,
    candidates: Sequence[float] = OUTER_LR_CANDIDATES,
) -> tuple[TinyMLPOptimizer, float, float]:
    """Meta-train a fresh 153-parameter student without teacher trajectories."""
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
            feature_builder=feature_builder,
        )
        scored.append((validation, outer_lr, student))
    validation, outer_lr, student = min(scored, key=lambda item: item[0])
    return student, outer_lr, validation


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


def paired_differences(
    left: Sequence[float],
    right: Sequence[float],
) -> dict[str, Any]:
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
    split: dict[float, list[StochasticCase]],
    *,
    train_batch_size: int,
    eval_batch_sizes: Sequence[int],
    steps: int,
    feature_builder: FeatureBuilder = build_matrix_aware_features,
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for batch_size in eval_batch_sizes:
        if batch_size > split[next(iter(split))][0].task.sample_count:
            continue
        metrics = evaluate_student_split(
            student,
            split,
            batch_size=batch_size,
            steps=steps,
            feature_builder=feature_builder,
        )
        matrix[str(batch_size)] = {
            "loss_ratio_mean": metrics["loss_ratio"]["mean"],
            "loss_ratio_median": metrics["loss_ratio"]["median"],
            "aulc_mean": metrics["aulc"]["mean"],
            "matches_train_batch": batch_size == train_batch_size,
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
    }
