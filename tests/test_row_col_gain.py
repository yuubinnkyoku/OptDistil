import torch

from optdistil.students.row_col_gain import RowColGainOptimizer, build_row_col_gain_features


def _features(
    student_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    parameter, grad, momentum, second_moment = student_inputs
    return build_row_col_gain_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=2,
        total_steps=8,
    )


def test_row_col_gain_features_have_expected_shape() -> None:
    inputs = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]),
        torch.tensor([[0.5, 1.0, 1.5], [2.0, 1.0, 2.5]]),
        torch.tensor([[1.0, 9.0, 4.0], [16.0, 4.0, 25.0]]),
    )
    features = _features(inputs)

    assert features.shape == (6, 8)
    # Row statistics repeat across columns.
    torch.testing.assert_close(features[0, 1:4], features[1, 1:4])
    torch.testing.assert_close(features[1, 1:4], features[2, 1:4])
    # Column statistics repeat down rows.
    torch.testing.assert_close(features[0, 4:7], features[3, 4:7])
    torch.testing.assert_close(features[1, 4:7], features[4, 4:7])


def test_row_col_gain_is_transpose_equivariant() -> None:
    torch.manual_seed(7)
    student = RowColGainOptimizer()
    inputs = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]]),
        torch.tensor([[0.5, 1.0, 1.5], [2.0, 1.0, 2.5]]),
        torch.tensor([[1.0, 9.0, 4.0], [16.0, 4.0, 25.0]]),
    )
    transposed = tuple(value.mT.contiguous() for value in inputs)

    update = student(_features(inputs)).reshape(inputs[0].shape)
    update_t = student(_features(transposed)).reshape(transposed[0].shape)

    torch.testing.assert_close(update_t, update.mT)


def test_default_row_col_gain_student_has_121_parameters() -> None:
    assert RowColGainOptimizer().parameter_count == 121


def test_row_col_gain_requires_matrix_parameter() -> None:
    values = torch.ones(4)
    try:
        build_row_col_gain_features(
            values,
            values,
            values,
            values,
            step=1,
            total_steps=2,
        )
    except ValueError as error:
        assert "2-D" in str(error)
    else:
        raise AssertionError("expected ValueError")
