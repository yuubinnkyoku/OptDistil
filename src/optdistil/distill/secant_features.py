from __future__ import annotations

import torch


class SecantFeatureState:
    """Limited-memory BFGS feature state for a fixed-size tiny optimizer.

    The student network stays fixed-size. Increasing ``history_size`` only increases
    optimizer state: the most recent secant pairs are summarized into one curvature-aware
    direction by the standard L-BFGS two-loop recursion.
    """

    def __init__(
        self,
        *,
        history_size: int = 1,
        eps: float = 1e-8,
        normalize_direction: bool = True,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.history_size = int(history_size)
        self.eps = float(eps)
        self.normalize_direction = bool(normalize_direction)
        self.previous_parameter: torch.Tensor | None = None
        self.previous_grad: torch.Tensor | None = None
        self._s_history: list[torch.Tensor] = []
        self._y_history: list[torch.Tensor] = []

    @property
    def stored_pairs(self) -> int:
        return len(self._s_history)

    @property
    def max_additional_state_tensors(self) -> int:
        """Full-sized tensors needed beyond the existing momentum/RMS student state."""
        return 2 + 2 * self.history_size

    def reset(self) -> None:
        self.previous_parameter = None
        self.previous_grad = None
        self._s_history.clear()
        self._y_history.clear()

    def _append_latest_pair(self, parameter: torch.Tensor, grad: torch.Tensor) -> None:
        if self.previous_parameter is None or self.previous_grad is None:
            return
        if self.previous_parameter.shape != parameter.shape or self.previous_grad.shape != grad.shape:
            raise ValueError("secant state shape changed")

        s = (parameter - self.previous_parameter).detach().reshape(-1).float()
        y = (grad - self.previous_grad).detach().reshape(-1).float()
        sy = torch.dot(s, y)
        yy = torch.dot(y, y)
        if not torch.isfinite(sy) or not torch.isfinite(yy):
            return
        if float(sy) <= self.eps or float(yy) <= self.eps:
            return

        self._s_history.append(s)
        self._y_history.append(y)
        if len(self._s_history) > self.history_size:
            del self._s_history[0]
            del self._y_history[0]

    def _direction(self, parameter: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        self._append_latest_pair(parameter, grad)
        if not self._s_history:
            return -grad

        q = grad.reshape(-1).float()
        alphas = [torch.zeros((), device=q.device, dtype=q.dtype) for _ in self._s_history]

        for index in range(len(self._s_history) - 1, -1, -1):
            s = self._s_history[index]
            y = self._y_history[index]
            rho = torch.dot(s, y).reciprocal()
            alpha = rho * torch.dot(s, q)
            alphas[index] = alpha
            q = q - alpha * y

        newest_s = self._s_history[-1]
        newest_y = self._y_history[-1]
        gamma = torch.dot(newest_s, newest_y) / torch.dot(newest_y, newest_y)
        r = gamma * q

        for index, (s, y) in enumerate(zip(self._s_history, self._y_history, strict=True)):
            rho = torch.dot(s, y).reciprocal()
            beta = rho * torch.dot(y, r)
            r = r + s * (alphas[index] - beta)

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
        if self.normalize_direction:
            # Historical OptDistil behavior: retain only L-BFGS geometry while removing its
            # inverse-curvature scale, leaving deployment step size to the downstream policy.
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
