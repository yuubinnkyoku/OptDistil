from __future__ import annotations

import torch
from torch import Tensor


class CoupledNewtonOracleTeacher:
    """Task-aware exact Newton teacher for controlled coupled-quadratic experiments.

    For

        loss(W) = 0.5 * ||L @ (W - W*) @ R||_F^2,

    the Hessian action is ``A @ X @ B`` with ``A=L.T@L`` and ``B=R@R.T``.
    This teacher is deliberately privileged: it receives ``L`` and ``R`` and solves the
    Newton system exactly. It is not a deployable optimizer baseline; it is a teacher-side
    information ceiling used to test how much second-order knowledge can be compressed
    into a tiny student that never sees the factors.
    """

    def __init__(self, left: Tensor, right: Tensor, *, lr: float = 0.5) -> None:
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("left and right must be matrices")
        if left.shape[0] != left.shape[1] or right.shape[0] != right.shape[1]:
            raise ValueError("Newton oracle currently requires square factors")
        if not 0.0 < lr <= 1.0:
            raise ValueError("lr must lie in (0, 1]")
        self.left = left.detach().clone()
        self.right = right.detach().clone()
        self.lr = float(lr)
        self.left_hessian = self.left.mT @ self.left
        self.right_hessian = self.right @ self.right.mT

    @torch.no_grad()
    def step(self, parameter: Tensor, grad: Tensor) -> Tensor:
        if parameter.ndim != 2 or grad.ndim != 2:
            raise ValueError("Newton oracle requires matrix parameters and gradients")
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        if self.left_hessian.shape[0] != parameter.shape[0]:
            raise ValueError("left factor is incompatible with parameter rows")
        if self.right_hessian.shape[0] != parameter.shape[1]:
            raise ValueError("right factor is incompatible with parameter columns")

        # Solve A @ delta @ B = -grad without materializing the Kronecker Hessian.
        left_solved = torch.linalg.solve(self.left_hessian, grad)
        newton_direction = torch.linalg.solve(
            self.right_hessian.mT,
            left_solved.mT,
        ).mT
        return -self.lr * newton_direction
