import torch

from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.muon import MuonTeacher, zeropower_via_newton_schulz5


def test_adamw_first_step_matches_sign_scaled_update() -> None:
    parameter = torch.tensor([1.0, -2.0])
    grad = torch.tensor([0.5, -0.25])
    teacher = AdamWTeacher(lr=0.1, eps=1e-12)

    update = teacher.step(parameter, grad)

    torch.testing.assert_close(update, torch.tensor([-0.1, 0.1]), rtol=1e-5, atol=1e-5)


def test_muon_update_is_finite_and_shape_preserving() -> None:
    torch.manual_seed(0)
    parameter = torch.randn(6, 4)
    grad = torch.randn_like(parameter)
    teacher = MuonTeacher(lr=0.02)

    update = teacher.step(parameter, grad)

    assert update.shape == parameter.shape
    assert torch.isfinite(update).all()


def test_newton_schulz_preserves_shape() -> None:
    matrix = torch.randn(3, 7)
    result = zeropower_via_newton_schulz5(matrix)
    assert result.shape == matrix.shape
