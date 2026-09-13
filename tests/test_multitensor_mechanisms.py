import pytest
import torch

from optdistil.multitensor.mechanisms import (
    aggregate_role_scales,
    apply_direction_scales,
    coordinate_descent_role_scales,
    equalized_role_scales,
    global_projection,
    scales_for_names,
    swapped_role_scales,
    tensor_projection,
)


def test_tensor_projection_preserves_only_normgrad_component() -> None:
    grads = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])]
    updates = [torch.tensor([-0.6, -0.8]) + torch.tensor([0.8, -0.6]), torch.tensor([1.0, -3.0])]

    projected, coefficients = tensor_projection(updates, grads)

    torch.testing.assert_close(projected[0], torch.tensor([-0.6, -0.8]))
    torch.testing.assert_close(projected[1], torch.tensor([0.0, -3.0]))
    assert coefficients == pytest.approx([1.0, 3.0])


def test_global_projection_forces_one_shared_scale() -> None:
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])]
    updates = [torch.tensor([-2.0, 3.0]), torch.tensor([4.0, -6.0])]

    projected, coefficient = global_projection(updates, grads)

    assert coefficient == pytest.approx(4.0)
    torch.testing.assert_close(projected[0], torch.tensor([-4.0, 0.0]))
    torch.testing.assert_close(projected[1], torch.tensor([0.0, -4.0]))


def test_apply_direction_scales_sets_per_tensor_update_norms() -> None:
    grads = [torch.tensor([3.0, 4.0]), torch.tensor([2.0])]

    updates = apply_direction_scales(grads, [0.25, -0.5])

    assert float(updates[0].norm()) == pytest.approx(0.25)
    assert float(updates[1].norm()) == pytest.approx(0.5)
    assert float(updates[0] @ grads[0]) < 0.0
    assert float(updates[1] @ grads[1]) > 0.0


def test_role_scale_aggregation_equalize_swap_and_resolution() -> None:
    role_scales = aggregate_role_scales(
        [
            (("W1", "b1", "W2", "b2"), (0.4, 0.1, 0.2, 0.3)),
            (("W1", "b1", "W2", "b2", "W_skip"), (0.6, 0.3, 0.4, 0.5, 0.7)),
        ]
    )

    assert role_scales == pytest.approx(
        {"W1": 0.5, "b1": 0.2, "W2": 0.3, "b2": 0.4, "W_skip": 0.7}
    )
    assert equalized_role_scales(role_scales) == pytest.approx(
        {"W1": 0.4, "b1": 0.4, "W2": 0.4, "b2": 0.4, "W_skip": 0.4}
    )
    assert swapped_role_scales(role_scales) == pytest.approx(
        {"W1": 0.3, "b1": 0.4, "W2": 0.5, "b2": 0.2, "W_skip": 0.7}
    )
    assert scales_for_names(("W2", "W_skip", "unknown"), role_scales) == pytest.approx(
        [0.3, 0.7, 0.4]
    )


def test_coordinate_descent_role_scales_recovers_grid_optimum() -> None:
    target = {"W1": 0.5, "b1": 0.25, "W2": 1.0}

    def evaluate(scales) -> float:
        return sum((float(scales[name]) - value) ** 2 for name, value in target.items())

    scales, score, history = coordinate_descent_role_scales(
        tuple(target),
        (0.25, 0.5, 1.0, 2.0),
        evaluate,
        initial_scale=1.0,
        passes=3,
    )

    assert scales == pytest.approx(target)
    assert score == pytest.approx(0.0)
    assert history[0]["role"] == "initial"
