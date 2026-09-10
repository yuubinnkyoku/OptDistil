import torch

from optdistil.students.block_gain import BlockGainOptimizer, build_block_gain_features


def _example_features() -> torch.Tensor:
    parameter = torch.tensor([1.0, 2.0, 3.0, 4.0])
    grad = torch.tensor([1.0, 3.0, 2.0, 4.0])
    momentum = torch.tensor([0.5, 1.0, 1.5, 2.0])
    second_moment = torch.tensor([1.0, 9.0, 4.0, 16.0])
    return build_block_gain_features(
        parameter,
        grad,
        momentum,
        second_moment,
        step=2,
        total_steps=4,
        block_size=2,
    )


def test_block_gain_features_repeat_block_statistics() -> None:
    features = _example_features()

    assert features.shape == (4, 5)
    torch.testing.assert_close(features[0, 1:], features[1, 1:])
    torch.testing.assert_close(features[2, 1:], features[3, 1:])
    assert not torch.allclose(features[0, 1:], features[2, 1:])


def test_block_gain_starts_as_identity_correction() -> None:
    features = _example_features()
    student = BlockGainOptimizer()

    torch.testing.assert_close(student(features), features[:, 0])


def test_block_gain_predicts_one_shared_gain_per_block_after_training_signal() -> None:
    features = _example_features()
    student = BlockGainOptimizer()
    output_layer = student.network[-1]
    with torch.no_grad():
        output_layer.weight.fill_(0.1)
        output_layer.bias.fill_(0.05)

    update = student(features)
    base_update = features[:, 0]
    gain = update / base_update

    torch.testing.assert_close(gain[0], gain[1])
    torch.testing.assert_close(gain[2], gain[3])


def test_default_block_gain_student_has_121_parameters() -> None:
    assert BlockGainOptimizer().parameter_count == 121


def test_block_gain_rejects_invalid_block_size() -> None:
    values = torch.ones(2)
    try:
        build_block_gain_features(
            values,
            values,
            values,
            values,
            step=0,
            total_steps=1,
            block_size=0,
        )
    except ValueError as error:
        assert "block_size" in str(error)
    else:
        raise AssertionError("expected ValueError")
