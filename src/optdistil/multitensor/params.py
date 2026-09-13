from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class ParamCollection:
    """Lightweight ordered list of parameter tensors (no generic PyTree machinery)."""

    def __init__(self, tensors: Sequence[Tensor]) -> None:
        if not tensors:
            raise ValueError("ParamCollection requires at least one tensor")
        self.tensors: list[Tensor] = list(tensors)
        for tensor in self.tensors:
            if not isinstance(tensor, Tensor):
                raise TypeError("ParamCollection elements must be torch.Tensor")

    @classmethod
    def zeros_like(cls, other: ParamCollection, *, requires_grad: bool = False) -> ParamCollection:
        return cls(
            [
                torch.zeros_like(tensor, requires_grad=requires_grad)
                for tensor in other.tensors
            ]
        )

    def clone(self) -> ParamCollection:
        return ParamCollection([tensor.detach().clone() for tensor in self.tensors])

    def detach(self) -> ParamCollection:
        return ParamCollection([tensor.detach() for tensor in self.tensors])

    def numels(self) -> list[int]:
        return [tensor.numel() for tensor in self.tensors]

    def total_numel(self) -> int:
        return sum(self.numels())

    def shapes(self) -> list[torch.Size]:
        return [tensor.shape for tensor in self.tensors]

    def is_finite(self) -> bool:
        return all(bool(torch.isfinite(tensor).all()) for tensor in self.tensors)

    def per_tensor_l2_norms(self) -> list[Tensor]:
        return [tensor.reshape(-1).float().norm() for tensor in self.tensors]

    def global_l2_norm(self) -> Tensor:
        total = torch.zeros((), dtype=torch.float32)
        for tensor in self.tensors:
            total = total + tensor.reshape(-1).float().square().sum()
        return total.sqrt()

    def flat_copy(self) -> Tensor:
        return torch.cat([tensor.reshape(-1).detach() for tensor in self.tensors])

    def apply_flat(self, flat: Tensor) -> ParamCollection:
        if flat.ndim != 1:
            raise ValueError("flat update must be 1-D")
        if flat.numel() != self.total_numel():
            raise ValueError("flat update numel mismatch")
        pieces: list[Tensor] = []
        offset = 0
        for tensor in self.tensors:
            size = tensor.numel()
            pieces.append(flat[offset : offset + size].reshape(tensor.shape))
            offset += size
        return ParamCollection(pieces)

    def add(self, updates: Sequence[Tensor] | ParamCollection) -> ParamCollection:
        update_list = updates.tensors if isinstance(updates, ParamCollection) else list(updates)
        if len(update_list) != len(self.tensors):
            raise ValueError("update count must match parameter count")
        result: list[Tensor] = []
        for parameter, update in zip(self.tensors, update_list, strict=True):
            if parameter.shape != update.shape:
                raise ValueError("update shape must match parameter shape")
            result.append(parameter.detach() + update.detach())
        return ParamCollection(result)

    def scale(self, factor: float) -> ParamCollection:
        return ParamCollection([tensor.detach() * factor for tensor in self.tensors])

    def to(self, *args, **kwargs) -> ParamCollection:
        return ParamCollection([tensor.to(*args, **kwargs) for tensor in self.tensors])

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> Tensor:
        return self.tensors[index]

    def __iter__(self):
        return iter(self.tensors)


def flatten_updates(updates: Sequence[Tensor] | ParamCollection) -> Tensor:
    tensors = updates.tensors if isinstance(updates, ParamCollection) else list(updates)
    if not tensors:
        raise ValueError("at least one update tensor is required")
    return torch.cat([update.reshape(-1) for update in tensors])


def unflatten_updates(flat: Tensor, shapes: Sequence[torch.Size]) -> list[Tensor]:
    pieces: list[Tensor] = []
    offset = 0
    for shape in shapes:
        size = 1
        for dim in shape:
            size *= dim
        pieces.append(flat[offset : offset + size].reshape(shape))
        offset += size
    if offset != flat.numel():
        raise ValueError("flat update length does not match shapes")
    return pieces
