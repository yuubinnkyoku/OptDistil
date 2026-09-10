import torch

from optdistil.students.block_gain import BlockGainOptimizer, build_block_gain_features
from optdistil.students.row_col_gain import RowColGainOptimizer, build_row_col_gain_features
from optdistil.students.tiny_mlp import StudentState


def test_element_block_base_is_bias_corrected_on_first_step() -> None:
    grad = torch.tensor([[1.0, -2.0], [4.0, -8.0]])
    parameter = torch.zeros_like(grad)
    state = StudentState(grad.shape)
    momentum, second_moment = state.observe(grad)

    features = build_block_gain_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=8,
        block_size=1,
    )
    expected = -grad.reshape(-1) / grad.abs().reshape(-1).square().add(1e-8).sqrt()

    torch.testing.assert_close(features[:, 0], expected)


def test_row_col_global_base_is_bias_corrected_on_first_step() -> None:
    grad = torch.tensor([[1.0, -2.0], [4.0, -8.0]])
    parameter = torch.zeros_like(grad)
    state = StudentState(grad.shape)
    momentum, second_moment = state.observe(grad)

    features = build_row_col_gain_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=1,
        total_steps=8,
    )
    denominator = grad.square().mean().add(1e-8).sqrt()
    expected = -grad.reshape(-1) / denominator

    torch.testing.assert_close(features[:, 0], expected)


def test_zero_initialized_gain_students_start_as_base_optimizer() -> None:
    block = BlockGainOptimizer()
    row_col = RowColGainOptimizer()

    block_features = torch.tensor(
        [[-1.0, 0.2, -0.1, 0.3, 0.5], [2.0, -0.4, 0.7, 0.1, 0.5]]
    )
    row_col_features = torch.tensor(
        [
            [-1.0, 0.2, -0.1, 0.3, 0.4, 0.1, -0.2, 0.5],
            [2.0, -0.4, 0.7, 0.1, -0.3, 0.2, 0.6, 0.5],
        ]
    )

    torch.testing.assert_close(block(block_features), block_features[:, 0])
    torch.testing.assert_close(row_col(row_col_features), row_col_features[:, 0])
