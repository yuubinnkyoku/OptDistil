import torch

from optdistil.teachers.muon import MuonTeacher
from optdistil.teachers.muon_controls import MuonNormGradientTeacher, PermutedMuonTeacher


def test_muon_norm_gradient_matches_muon_norm_but_uses_gradient_direction() -> None:
    parameter = torch.tensor([[0.4, -0.2], [0.1, 0.7]])
    grad = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

    reference = MuonTeacher(lr=0.2, momentum=0.95, ns_steps=5)
    control = MuonNormGradientTeacher(lr=0.2, momentum=0.95, ns_steps=5)

    muon_update = reference.step(parameter, grad)
    control_update = control.step(parameter, grad)

    assert torch.allclose(control_update.norm(), muon_update.norm(), rtol=1e-5, atol=1e-7)
    cosine = torch.nn.functional.cosine_similarity(
        control_update.reshape(1, -1),
        (-grad).reshape(1, -1),
    )
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)


def test_permuted_muon_preserves_update_values_and_norm() -> None:
    parameter = torch.tensor([[0.4, -0.2, 0.3], [0.1, 0.7, -0.5]])
    grad = torch.tensor([[1.0, -2.0, 0.2], [0.5, 3.0, -1.5]])

    reference = MuonTeacher(lr=0.2, momentum=0.95, ns_steps=5)
    control = PermutedMuonTeacher(lr=0.2, momentum=0.95, ns_steps=5, seed=17)

    muon_update = reference.step(parameter, grad)
    control_update = control.step(parameter, grad)

    assert torch.allclose(control_update.norm(), muon_update.norm(), rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        torch.sort(control_update.flatten()).values,
        torch.sort(muon_update.flatten()).values,
    )
    assert not torch.allclose(control_update, muon_update)
