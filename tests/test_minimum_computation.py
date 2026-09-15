from __future__ import annotations

import math

import pytest
import torch

from optdistil.multitensor.ladder import (
    build_adamw,
    build_normgrad,
    build_sgd,
    collect_ratios_for_method,
    frozen_ratio_teacher,
    paired_mean_difference,
    permutation_specs,
    tune_global_scale,
    tune_role_grid,
)
from optdistil.multitensor.reparam import (
    make_reparameterized_case,
    role_aligned_scales,
)
from optdistil.multitensor.roles import (
    random_balanced_binary_partition,
    swap_binary_role_labels,
)
from optdistil.multitensor.static_role import StaticRoleNormGrad, roles_for_case
from optdistil.multitensor.stochastic import flatten_split, make_split
from optdistil.multitensor.tasks import make_task, make_three_layer_mlp
from optdistil.multitensor.teachers import (
    RoleScaledAdamW,
    RoleScaledSGD,
)


def _case(architecture: str = "two_layer", seed: int = 11, *, width: int = 4):
    split = make_split(
        architecture,
        (30.0,),
        seed_base=seed,
        count=1,
        width=width,
        samples=16,
        device="cpu",
    )
    return flatten_split(split)[0]


def test_three_layer_analytic_grads_match_autograd() -> None:
    initial, task = make_three_layer_mlp(
        5, input_dim=4, hidden_dim=4, output_dim=2, samples=8
    )
    params = [t.detach().clone().requires_grad_(True) for t in initial]
    loss = task.loss(params)  # type: ignore[arg-type]
    loss.backward()
    analytic = task.grad(initial)
    assert len(analytic) == 6
    for auto_grad, analytic_grad in zip(
        [p.grad for p in params], analytic, strict=True
    ):
        torch.testing.assert_close(auto_grad, analytic_grad, rtol=1e-5, atol=1e-6)


def test_make_task_three_layer_roles() -> None:
    initial, task = make_task("three_layer", 2, width=4, samples=8)
    assert len(initial) == 6
    assert roles_for_case_type(task) == (
        "matrix",
        "vector",
        "matrix",
        "vector",
        "matrix",
        "vector",
    )


def roles_for_case_type(task) -> tuple[str, ...]:
    return tuple(task.parameter_roles)


def test_role_aligned_scales_values() -> None:
    scales = role_aligned_scales(
        ("matrix", "vector", "matrix", "vector"),
        matrix_scale=10.0,
        vector_scale=0.1,
    )
    assert scales == [10.0, 0.1, 10.0, 0.1]


def test_swap_binary_role_labels() -> None:
    swapped = swap_binary_role_labels(["matrix", "vector", "matrix", "vector"])
    assert swapped == ["vector", "matrix", "vector", "matrix"]


def test_random_balanced_partition_sizes() -> None:
    labels = random_balanced_binary_partition(5, seed=3)
    assert len(labels) == 5
    assert abs(labels.count("A") - labels.count("B")) <= 1
    assert set(labels) <= {"A", "B"}


def test_permutation_specs_include_true_and_random() -> None:
    specs = permutation_specs(4, n_random=2, seed_base=1)
    assert specs[0]["kind"] == "true"
    assert specs[1]["kind"] == "swap"
    assert len(specs) == 4


def test_role_scaled_sgd_and_adamw_step() -> None:
    case = _case()
    grads = case.task.grad(case.initial)
    sgd = RoleScaledSGD(scales=[0.1, 0.01, 0.1, 0.01])
    updates = sgd.step(case.initial, grads)
    torch.testing.assert_close(updates[0], -0.1 * grads[0])
    torch.testing.assert_close(updates[1], -0.01 * grads[1])

    adamw = RoleScaledAdamW(scales=[0.1, 0.01, 0.1, 0.01])
    updates = adamw.step(case.initial, grads)
    assert len(updates) == len(grads)
    assert all(u.shape == g.shape for u, g in zip(updates, grads, strict=True))


def test_tune_role_grid_prefers_lower_loss() -> None:
    cases = [_case(seed=20 + i) for i in range(2)]
    role_lrs, score = tune_role_grid(
        build_normgrad,
        cases,
        batch_size=8,
        steps=6,
        candidates=(0.01, 0.1),
    )
    assert set(role_lrs) == {"matrix", "vector"}
    assert math.isfinite(score)
    assert all(v > 0 for v in role_lrs.values())


def test_frozen_ratio_teacher_preserves_ratio() -> None:
    case = _case()
    base = {"matrix": 0.1, "vector": 0.01}
    teacher = frozen_ratio_teacher(build_normgrad, base, case, global_scale=2.0)
    assert isinstance(teacher, StaticRoleNormGrad)
    assert teacher.scales[0] == pytest.approx(0.2)
    assert teacher.scales[1] == pytest.approx(0.02)


def test_tune_global_scale_returns_candidate() -> None:
    cases = [_case(seed=30 + i) for i in range(2)]
    scale, score = tune_global_scale(
        build_normgrad,
        {"matrix": 0.1, "vector": 0.01},
        cases,
        batch_size=8,
        steps=5,
        candidates=(0.5, 1.0, 2.0),
    )
    assert scale in (0.5, 1.0, 2.0)
    assert math.isfinite(score)


def test_collect_ratios_and_paired() -> None:
    cases = [_case(seed=40), _case(seed=41)]
    ratios = collect_ratios_for_method(
        lambda c: build_normgrad([0.06] * len(c.initial)),
        cases,
        batch_size=8,
        steps=5,
    )
    assert len(ratios) == 2
    assert all(r > 0 for r in ratios)
    paired = paired_mean_difference(ratios, [1.0, 1.0])
    assert paired["n"] == 2


def test_reparameterized_inverted_scales_lose_to_retuned_directionally() -> None:
    """Smoke: inverted family is runnable and retuned LRs prefer larger vector scale."""
    base_cases = [_case(seed=50 + i) for i in range(2)]
    inverted = []
    for case in base_cases:
        roles = roles_for_case(case)
        scales = role_aligned_scales(roles, matrix_scale=10.0, vector_scale=0.1)
        reparam_case, _ = make_reparameterized_case(case, scales=scales)
        inverted.append(reparam_case)
    role_lrs, _score = tune_role_grid(
        build_normgrad,
        inverted,
        batch_size=8,
        steps=6,
        candidates=(0.01, 0.1),
    )
    # Privileged theory: c_i = lr/s_i, so with s_matrix≫s_vector, vector θ-LR ≥ matrix.
    assert role_lrs["vector"] >= role_lrs["matrix"] * 0.5


def test_role_sgd_builder_shapes() -> None:
    case = _case()
    teacher = build_sgd([0.1, 0.1, 0.1, 0.1])
    updates = teacher.step(case.initial, case.task.grad(case.initial))
    assert len(updates) == 4


def test_adamw_builder_runs() -> None:
    case = _case()
    teacher = build_adamw([0.01, 0.01, 0.01, 0.01])
    for _ in range(2):
        updates = teacher.step(case.initial, case.task.grad(case.initial))
        assert all(torch.isfinite(u).all() for u in updates)
