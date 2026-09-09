from __future__ import annotations

import torch
from torch import Tensor

from optdistil.teachers.muon import MuonTeacher


class MuonNormGradientTeacher:
    """Use the raw gradient direction with Muon's per-step update norm.

    This control preserves the exact Frobenius norm schedule produced by a Muon teacher
    while discarding Muon's matrix-valued update direction. If Muon's advantage survives
    against this control, it cannot be explained by update magnitude alone.
    """

    def __init__(
        self,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-12,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.muon = MuonTeacher(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        muon_update = self.muon.step(parameter, grad)
        grad_norm = grad.float().norm()
        if float(grad_norm) <= self.eps:
            return torch.zeros_like(grad)
        update_norm = muon_update.float().norm()
        scale = update_norm / grad_norm
        return -grad * scale.to(device=grad.device, dtype=grad.dtype)


class PermutedMuonTeacher:
    """Apply a fixed coordinate permutation to every Muon update.

    The control preserves Muon's per-step update norm and complete multiset of update
    values, but destroys the correspondence between matrix coordinates and update values.
    A fixed permutation avoids injecting fresh stochastic noise at every optimization step.
    """

    def __init__(
        self,
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        seed: int = 2026,
    ) -> None:
        self.muon = MuonTeacher(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        self.seed = seed
        self._permutation: Tensor | None = None
        self._numel: int | None = None

    def _get_permutation(self, update: Tensor) -> Tensor:
        if self._permutation is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self._permutation = torch.randperm(update.numel(), generator=generator)
            self._numel = update.numel()
        elif self._numel != update.numel():
            raise ValueError("teacher state shape changed; create a separate teacher per tensor")
        return self._permutation.to(update.device)

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        update = self.muon.step(parameter, grad)
        permutation = self._get_permutation(update)
        return update.reshape(-1)[permutation].reshape_as(update)
