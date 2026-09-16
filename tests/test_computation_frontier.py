from __future__ import annotations

import math

import torch

from optdistil.multitensor.frontier import (
    evaluate_assigned_lrs,
    formula_lrs_for_case,
    per_tensor_oracle,
    property_partition_from_case,
    tune_groups_cd,
)
from optdistil.multitensor.ladder import build_normgrad
from optdistil.multitensor.stochastic import flatten_split, make_split
from optdistil.multitensor.tasks import make_all_matrix, make_all_vector, make_task


def _cases(architecture: str, seed: int = 21, *, count: int = 2):
    split = make_split(
        architecture,
        (30.0,),
        seed_base=seed,
        count=count,
        width=6,
        samples=16,
        device="cpu",
    )
    return flatten_split(split)


def test_all_matrix_analytic_grads() -> None:
    initial, task = make_all_matrix(3, input_dim=4, samples=8)
    assert len(initial) == 3
    assert task.parameter_roles == ("matrix", "matrix", "matrix")
    params = [t.detach().clone().requires_grad_(True) for t in initial]
    loss = task.loss(params)  # type: ignore[arg-type]
    loss.backward()
    analytic = task.grad(initial)
    for auto, ana in zip([p.grad for p in params], analytic, strict=True):
        torch.testing.assert_close(auto, ana, rtol=1e-5, atol=1e-6)


def test_all_vector_analytic_grads() -> None:
    initial, task = make_all_vector(4, input_dim=4, samples=8)
    assert len(initial) == 3
    assert task.parameter_roles == ("vector", "vector", "vector")
    params = [t.detach().clone().requires_grad_(True) for t in initial]
    loss = task.loss(params)  # type: ignore[arg-type]
    loss.backward()
    analytic = task.grad(initial)
    for auto, ana in zip([p.grad for p in params], analytic, strict=True):
        torch.testing.assert_close(auto, ana, rtol=1e-5, atol=1e-6)


def test_make_task_adversarial_names() -> None:
    for name in ("all_matrix", "all_vector"):
        initial, task = make_task(name, 2, width=4, samples=8)
        assert len(initial) == len(task.parameter_roles)


def test_numel_partition_buckets() -> None:
    shapes = [torch.Size((8, 8)), torch.Size((2,)), torch.Size((4, 8)), torch.Size((4,))]
    labels = None
    from optdistil.multitensor.frontier import numel_partition

    labels = numel_partition(shapes, n_groups=2)
    assert len(labels) == 4
    assert set(labels) <= {"g0", "g1"}
    # smallest two should share a bucket or be g0
    assert labels[1] in {"g0", "g1"}


def test_formula_lrs_modes() -> None:
    case = _cases("two_layer", 5, count=1)[0]
    for mode in (
        "uniform",
        "inv_sqrt_numel",
        "inv_numel",
        "inv_sqrt_fan_in",
        "ndim_scaled",
        "numel_rank",
    ):
        lrs = formula_lrs_for_case(case, mode=mode, base=0.1)
        assert len(lrs) == len(case.initial)
        assert all(x > 0 and math.isfinite(x) for x in lrs)


def test_tune_groups_cd_and_evaluate() -> None:
    val = _cases("two_layer", 6, count=2)
    test = _cases("two_layer", 7, count=1)
    _r, lrs, score = tune_groups_cd(
        build_normgrad,
        val,
        batch_size=8,
        steps=5,
        rounds=1,
        candidates=(0.03, 0.1),
    )
    assert score >= 0
    ratios = evaluate_assigned_lrs(
        build_normgrad, lrs, test, batch_size=8, steps=5
    )
    assert len(ratios) == len(test)


def test_per_tensor_oracle_runs() -> None:
    val = _cases("two_layer", 8, count=2)
    assigned, role_lrs, score = per_tensor_oracle(val, batch_size=8, steps=4, rounds=1)
    assert "two_layer" in assigned
    assert len(assigned["two_layer"]) == 4
    assert all(k.startswith("t") for k in role_lrs)
    assert math.isfinite(score)


def test_iso_shape_task_grads() -> None:
    from optdistil.multitensor.tasks import make_iso_shape_spectrum

    initial, task = make_iso_shape_spectrum(3, width=4, samples=12)
    assert len(initial) == 4
    assert task.parameter_roles == ("matrix", "matrix", "matrix", "matrix")
    params = [t.detach().clone().requires_grad_(True) for t in initial]
    loss = task.loss(params)  # type: ignore[arg-type]
    loss.backward()
    analytic = task.grad(initial)
    for auto, ana in zip([p.grad for p in params], analytic, strict=True):
        torch.testing.assert_close(auto, ana, rtol=1e-5, atol=1e-6)


def test_make_task_iso_shape() -> None:
    initial, _task = make_task("iso_shape", 2, width=4, samples=12)
    assert len(initial) == 4


def test_property_partition_modes() -> None:
    case = _cases("residual", 9, count=1)[0]
    for mode in ("true_roles", "ndim", "numel2", "index_mod2", "single", "per_tensor"):
        labels = property_partition_from_case(case, mode=mode)
        assert len(labels) == len(case.initial)
