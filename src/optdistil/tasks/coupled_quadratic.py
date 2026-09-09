from __future__ import annotations

from torch import Tensor


class CoupledMatrixQuadraticTask:
    """Matrix quadratic with row/column coupling through dense linear factors.

    The objective is

        0.5 * || left @ (parameter - target) @ right ||_F^2.

    Unlike the elementwise quadratic smoke task, this produces a dense Hessian over
    vec(parameter) and therefore tests whether an optimizer can exploit matrix structure.
    """

    def __init__(self, target: Tensor, left: Tensor, right: Tensor) -> None:
        if target.ndim != 2:
            raise ValueError("target must be a matrix")
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("left and right factors must be matrices")
        if left.shape[1] != target.shape[0]:
            raise ValueError("left factor is incompatible with target rows")
        if right.shape[0] != target.shape[1]:
            raise ValueError("right factor is incompatible with target columns")

        self.target = target.detach().clone()
        self.left = left.detach().clone().to(device=target.device, dtype=target.dtype)
        self.right = right.detach().clone().to(device=target.device, dtype=target.dtype)

    def loss(self, parameter: Tensor) -> Tensor:
        if parameter.shape != self.target.shape:
            raise ValueError("parameter shape must match target shape")
        residual = self.left @ (parameter - self.target) @ self.right
        return 0.5 * residual.square().sum()

    def grad(self, parameter: Tensor) -> Tensor:
        if parameter.shape != self.target.shape:
            raise ValueError("parameter shape must match target shape")
        delta = parameter - self.target
        return self.left.mT @ (self.left @ delta @ self.right) @ self.right.mT
