from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class TrajectoryRecord:
    """One optimizer-imitation example.

    ``features`` has shape ``[numel, feature_dim]`` and ``teacher_update`` has shape
    ``[numel]``. Records may have different ``numel`` values; callers can train on them
    one at a time or provide a task-specific collator.
    """

    features: Tensor
    teacher_update: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape [numel, feature_dim]")
        if self.teacher_update.ndim != 1:
            raise ValueError("teacher_update must be flattened to shape [numel]")
        if self.features.shape[0] != self.teacher_update.shape[0]:
            raise ValueError("features and teacher_update must describe the same elements")

    def cpu(self) -> TrajectoryRecord:
        return TrajectoryRecord(
            features=self.features.detach().cpu(),
            teacher_update=self.teacher_update.detach().cpu(),
            metadata=dict(self.metadata),
        )


class TrajectoryDataset(Dataset[TrajectoryRecord]):
    def __init__(self, records: Iterable[TrajectoryRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrajectoryRecord:
        return self.records[index]

    def save(self, path: str | Path) -> None:
        payload = [
            {
                "features": record.features.detach().cpu(),
                "teacher_update": record.teacher_update.detach().cpu(),
                "metadata": dict(record.metadata),
            }
            for record in self.records
        ]
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryDataset:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(TrajectoryRecord(**item) for item in payload)
