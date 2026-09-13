"""Effective Student scale analysis via NormGrad-direction projection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from optdistil.multitensor.features import build_multitensor_features, concatenate_features
from optdistil.multitensor.stochastic import (
    MultiTensorCase,
    _observe_ema,
    batch_sequence,
    split_update,
)
from optdistil.students.tiny_mlp import TinyMLPOptimizer


def project_onto_normgrad_direction(
    update: Tensor,
    grad: Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Return scale ``c`` such that ``c * (-g/||g||)`` is the projection of ``u``."""
    g = grad.reshape(-1).float()
    u = update.reshape(-1).float()
    norm = g.norm().clamp_min(eps)
    direction = -g / norm
    return float(torch.dot(u, direction))


@dataclass(slots=True)
class EffectiveScaleTrace:
    """Per-step, per-tensor effective scales and observable statistics."""

    scales: list[list[float]]  # [step][tensor]
    tensor_names: tuple[str, ...]
    hidden_scales: tuple[float, ...]
    records: list[dict[str, Any]]

    def flat_scales(self) -> list[float]:
        return [value for row in self.scales for value in row]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_names": list(self.tensor_names),
            "hidden_scales": list(self.hidden_scales),
            "scales": self.scales,
            "records": self.records,
            "n_steps": len(self.scales),
            "n_tensors": len(self.tensor_names),
        }


@torch.no_grad()
def collect_effective_scale_trace(
    student: TinyMLPOptimizer,
    case: MultiTensorCase,
    *,
    batch_size: int,
    steps: int,
    hidden_scales: Sequence[float] | None = None,
    include_global: bool = True,
) -> EffectiveScaleTrace:
    """Roll out the student and record effective_scale(t,l) plus observables."""
    params = case.initial.clone()
    momentums = [torch.zeros_like(t) for t in params]
    second_moments = [torch.zeros_like(t) for t in params]
    batches = batch_sequence(case, batch_size=batch_size, steps=steps)
    shapes = params.shapes()
    names = tuple(
        getattr(case.task, "parameter_names", tuple(f"t{i}" for i in range(len(params))))
    )
    roles = tuple(getattr(case.task, "parameter_roles", tuple("t" for _ in names)))
    student.eval()

    scale_rows: list[list[float]] = []
    records: list[dict[str, Any]] = []

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
        updates = split_update(flat, shapes)
        row: list[float] = []
        global_grad_sq = torch.zeros((), dtype=torch.float32)
        total_numel = 0
        for grad in grads:
            global_grad_sq = global_grad_sq + grad.reshape(-1).float().square().sum()
            total_numel += grad.numel()
        global_grad_rms = float((global_grad_sq / max(total_numel, 1)).sqrt())
        progress = min(max(step / max(steps, 1), 0.0), 1.0)

        for tensor_index, (update, grad, mom, v, param) in enumerate(
            zip(updates, grads, momentums, second_moments, params, strict=True)
        ):
            scale = project_onto_normgrad_direction(update, grad)
            row.append(scale)
            grad_rms = float(grad.square().mean().sqrt())
            param_rms = float(param.square().mean().sqrt())
            mom_rms = float(mom.square().mean().sqrt())
            v_rms = float(v.clamp_min(0).mean().sqrt())
            hidden = None
            if hidden_scales is not None and tensor_index < len(hidden_scales):
                hidden = float(hidden_scales[tensor_index])
            records.append(
                {
                    "step": step,
                    "tensor_index": tensor_index,
                    "tensor_name": names[tensor_index],
                    "role": roles[tensor_index],
                    "effective_scale": scale,
                    "grad_rms": grad_rms,
                    "param_rms": param_rms,
                    "momentum_rms": mom_rms,
                    "second_moment_rms": v_rms,
                    "global_grad_rms": global_grad_rms,
                    "progress": progress,
                    "hidden_scale": hidden,
                }
            )
        scale_rows.append(row)
        params = params.add(updates)
        if not params.is_finite():
            break

    return EffectiveScaleTrace(
        scales=scale_rows,
        tensor_names=names,
        hidden_scales=tuple(float(s) for s in (hidden_scales or ())),
        records=records,
    )


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    tx = torch.tensor(list(x), dtype=torch.float64)
    ty = torch.tensor(list(y), dtype=torch.float64)
    tx = tx - tx.mean()
    ty = ty - ty.mean()
    denom = tx.norm() * ty.norm()
    if float(denom) < 1e-12:
        return float("nan")
    return float(torch.dot(tx, ty) / denom)


def rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_correlation(rankdata(x), rankdata(y))


def ridge_r2(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    alpha: float = 1.0,
) -> float:
    """Return out-of-sample-style R^2 via leave-one-out ridge when n is small."""
    n = len(targets)
    if n < 3 or not features or len(features[0]) == 0:
        return float("nan")
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    # Center for numerical stability.
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean
    d = xc.shape[1]
    # Leave-one-out ridge with hat-matrix shortcut.
    xtx = xc.T @ xc + alpha * torch.eye(d, dtype=torch.float64)
    # Use LOO residual formula: r_i = e_i / (1 - h_ii)
    try:
        xtx_inv = torch.linalg.inv(xtx)
    except RuntimeError:
        return float("nan")
    beta = xtx_inv @ (xc.T @ yc)
    fitted = xc @ beta
    residual = yc - fitted
    hat = torch.sum((xc @ xtx_inv) * xc, dim=1)
    loo = residual / (1.0 - hat).clamp_min(1e-8)
    ss_res = float((loo * loo).sum())
    ss_tot = float((yc * yc).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


OBSERVABLE_KEYS = (
    "grad_rms",
    "param_rms",
    "momentum_rms",
    "second_moment_rms",
    "global_grad_rms",
    "progress",
    "hidden_scale",
)


def analyze_effective_scales(traces: Sequence[EffectiveScaleTrace]) -> dict[str, Any]:
    """Correlate effective_scale(t,l) with observables and report R^2."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.extend(trace.records)
    if not rows:
        return {"n": 0}

    targets = [float(row["effective_scale"]) for row in rows]
    correlations: dict[str, Any] = {}
    for key in OBSERVABLE_KEYS:
        xs = [row[key] for row in rows if row.get(key) is not None]
        ys = [row["effective_scale"] for row in rows if row.get(key) is not None]
        correlations[key] = {
            "pearson": pearson_correlation(xs, ys),
            "spearman": spearman_correlation(xs, ys),
            "n": len(xs),
        }

    # Role-wise mean effective scales.
    by_role: dict[str, list[float]] = {}
    for row in rows:
        by_role.setdefault(str(row["role"]), []).append(float(row["effective_scale"]))
    role_means = {
        role: {
            "mean": sum(values) / len(values),
            "std": (
                math.sqrt(
                    sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
                )
                if len(values) > 1
                else 0.0
            ),
            "n": len(values),
        }
        for role, values in by_role.items()
    }

    # Univariate R^2 for each observable; multivariate without hidden_scale.
    r2_univariate: dict[str, float] = {}
    for key in OBSERVABLE_KEYS:
        pairs = [( [float(row[key])], float(row["effective_scale"]) ) for row in rows if row.get(key) is not None]
        if len(pairs) >= 3:
            feats = [p[0] for p in pairs]
            targs = [p[1] for p in pairs]
            r2_univariate[key] = ridge_r2(feats, targs, alpha=1.0)
        else:
            r2_univariate[key] = float("nan")

    observable_only = [k for k in OBSERVABLE_KEYS if k != "hidden_scale"]
    multi_rows = [row for row in rows if all(row.get(k) is not None for k in observable_only)]
    multi_features = [[float(row[k]) for k in observable_only] for row in multi_rows]
    multi_targets = [float(row["effective_scale"]) for row in multi_rows]
    r2_multivariate = ridge_r2(multi_features, multi_targets, alpha=1.0)

    all_keys = [k for k in OBSERVABLE_KEYS]
    multi_all_rows = [row for row in rows if all(row.get(k) is not None for k in all_keys)]
    multi_all_features = [[float(row[k]) for k in all_keys] for row in multi_all_rows]
    multi_all_targets = [float(row["effective_scale"]) for row in multi_all_rows]
    r2_with_hidden = ridge_r2(multi_all_features, multi_all_targets, alpha=1.0)

    # Does hidden scale add explanatory power beyond observables?
    hidden_delta_r2 = float("nan")
    if math.isfinite(r2_with_hidden) and math.isfinite(r2_multivariate):
        hidden_delta_r2 = r2_with_hidden - r2_multivariate

    return {
        "n": len(rows),
        "correlations": correlations,
        "role_means": role_means,
        "r2_univariate": r2_univariate,
        "r2_multivariate_observables": r2_multivariate,
        "r2_with_hidden_scale": r2_with_hidden,
        "hidden_delta_r2": hidden_delta_r2,
        "targets_mean": sum(targets) / len(targets),
        "targets_std": (
            math.sqrt(
                sum((t - sum(targets) / len(targets)) ** 2 for t in targets) / len(targets)
            )
            if len(targets) > 1
            else 0.0
        ),
    }
