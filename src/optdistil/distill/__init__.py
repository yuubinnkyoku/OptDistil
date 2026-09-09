"""Trajectory collection and optimizer distillation."""

from optdistil.distill.collect import collect_teacher_step
from optdistil.distill.features import FEATURE_NAMES, build_elementwise_features
from optdistil.distill.losses import (
    DistillationLossWeights,
    direction_loss,
    distillation_loss,
    magnitude_loss,
)
from optdistil.distill.trajectory import TrajectoryDataset, TrajectoryRecord

__all__ = [
    "FEATURE_NAMES",
    "DistillationLossWeights",
    "TrajectoryDataset",
    "TrajectoryRecord",
    "build_elementwise_features",
    "collect_teacher_step",
    "direction_loss",
    "distillation_loss",
    "magnitude_loss",
]
