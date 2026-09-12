from __future__ import annotations

import torch


class SecantFeatureState:
    """One-pair L-BFGS feature state for a fixed-size tiny optimizer.

    The state stores only the previous parameter and gradient. It adds no trainable
    parameters and constructs a curvature-aware descent direction from the latest secant
    pair using reductions and elementwise vector operations.
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.previous_parameter: torch.Tensor | None = None
        self.previous_grad: torch.Tensor | None = None

    def reset(self) -> None:
        self.previous_parameter = None
        self.previous_grad = None

    def _direction(self, parameter: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self.previous_parameter is None or self.previous_grad is None:
            return -grad
        if self.previous_parameter.shape != parameter.shape or self.previous_grad.shape != grad.shape:
            raise ValueError("secant state shape changed")

        s = (parameter - self.previous_parameter).reshape(-1).float()
        y = (grad - self.previous_grad).reshape(-1).float()
        g = grad.reshape(-1).float()
        sy = torch.dot(s, y)
        yy = torch.dot(y, y)
        if not torch.isfinite(sy) or not torch.isfinite(yy):
            return -grad
        if float(sy) <= self.eps or float(yy) <= self.eps:
            return -grad

        # Standard one-pair L-BFGS two-loop recursion. For H=hI this recovers H^{-1}g
        # exactly after one valid secant pair.
        rho = sy.reciprocal()
        alpha = rho * torch.dot(s, g)
        q = g - alpha * y
        gamma = sy / yy
        r = gamma * q
        beta = rho * torch.dot(y, r)
        r = r + s * (alpha - beta)

        direction = -r.reshape_as(grad).to(dtype=grad.dtype, device=grad.device)
        if not torch.isfinite(direction).all():
            return -grad
        return direction

    def build(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        second_moment: torch.Tensor,
        *,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Return an 8-feature matrix observation and advance the secant memory."""
        if parameter.ndim != 2:
            raise ValueError("secant features require a 2-D parameter tensor")
        if parameter.shape != grad.shape:
            raise ValueError("parameter and grad must have the same shape")
        if momentum.shape != parameter.shape or second_moment.shape != parameter.shape:
            raise ValueError("student state tensors must match parameter shape")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")

        eps = self.eps
        direction = self._direction(parameter, grad)
        # Keep this feature primarily directional: match its RMS to the current gradient.
        grad_rms = grad.square().mean().add(eps).sqrt()
        direction_rms = direction.square().mean().add(eps).sqrt()
        direction = direction * (grad_rms / direction_rms)

        rms = second_moment.clamp_min(0).add(eps).sqrt()
        row_grad_rms = grad.square().mean(dim=1, keepdim=True).add(eps).sqrt().expand_as(grad)
        col_grad_rms = grad.square().mean(dim=0, keepdim=True).add(eps).sqrt().expand_as(grad)
        parameter_rms = parameter.square().mean().add(eps).sqrt()
        progress = min(max(step / total_steps, 0.0), 1.0)

        def flat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(-1)

        flat_grad = flat(grad)
        features = torch.stack(
            (
                flat_grad,
                flat(momentum),
                flat(rms),
                flat(row_grad_rms),
                flat(col_grad_rms),
                flat(direction),
                parameter_rms.expand_as(flat_grad),
                torch.full_like(flat_grad, progress),
            ),
            dim=-1,
        )
        self.previous_parameter = parameter.detach().clone()
        self.previous_grad = grad.detach().clone()
        return features
