"""Paired comparison statistics and bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    floats = [float(v) for v in values if math.isfinite(v)]
    if not floats:
        return {"mean": math.inf, "median": math.inf, "std": math.inf, "ci_low": math.inf, "ci_high": math.inf, "n": 0}
    rng = random.Random(seed)
    n = len(floats)
    if n == 1:
        value = floats[0]
        return {
            "mean": value,
            "median": value,
            "std": 0.0,
            "ci_low": value,
            "ci_high": value,
            "n": 1,
        }
    means: list[float] = []
    for _ in range(n_boot):
        sample = [floats[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    low_index = math.floor((alpha / 2.0) * n_boot)
    high_index = math.ceil((1.0 - alpha / 2.0) * n_boot) - 1
    low_index = max(0, min(low_index, n_boot - 1))
    high_index = max(0, min(high_index, n_boot - 1))
    return {
        "mean": statistics.fmean(floats),
        "median": statistics.median(floats),
        "std": statistics.pstdev(floats),
        "ci_low": means[low_index],
        "ci_high": means[high_index],
        "n": n,
    }


def paired_comparison(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired task comparison: negative delta means left is better (lower loss)."""
    if len(left) != len(right):
        raise ValueError("paired sequences must have equal length")
    deltas = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    finite = [(a, b, d) for a, b, d in zip(left, right, deltas, strict=True) if math.isfinite(d)]
    if not finite:
        return {
            "mean_delta": math.inf,
            "median_delta": math.inf,
            "win_fraction": 0.0,
            "left_wins": 0,
            "n": 0,
            "delta_ci": bootstrap_mean_ci([], seed=seed),
        }
    only_deltas = [d for _, _, d in finite]
    left_wins = sum(1 for d in only_deltas if d < 0)
    return {
        "mean_delta": statistics.fmean(only_deltas),
        "median_delta": statistics.median(only_deltas),
        "win_fraction": left_wins / len(only_deltas),
        "left_wins": left_wins,
        "n": len(only_deltas),
        "delta_ci": bootstrap_mean_ci(only_deltas, n_boot=n_boot, seed=seed),
    }


def summarize_method_values(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    summary = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    finite = [float(v) for v in values if math.isfinite(v)]
    summary["finite_fraction"] = (len(finite) / len(values)) if values else 0.0
    summary["values"] = [float(v) for v in values]
    return summary
