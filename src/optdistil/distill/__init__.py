"""Trajectory collection and optimizer distillation."""

from optdistil.distill.collect import collect_teacher_step
from optdistil.distill.features import (
    FEATURE_NAMES,
    MATRIX_FEATURE_NAMES,
    build_elementwise_features,
    build_matrix_aware_features,
)
from optdistil.distill.losses import (
    DistillationLossWeights,
    direction_loss,
    distillation_loss,
    magnitude_loss,
)
from optdistil.distill.rollout import (
    RolloutResult,
    collect_teacher_trajectory,
    evaluate_imitation,
    rollout_student,
)
from optdistil.distill.train import (
    calibrate_student_magnitude,
    magnitude_calibration_scale,
    train_student,
)
from optdistil.distill.trajectory import TrajectoryDataset, TrajectoryRecord

__all__ = [
    "FEATURE_NAMES",
    "MATRIX_FEATURE_NAMES",
    "DistillationLossWeights",
    "RolloutResult",
    "TrajectoryDataset",
    "TrajectoryRecord",
    "build_elementwise_features",
    "build_matrix_aware_features",
    "calibrate_student_magnitude",
    "collect_teacher_step",
    "collect_teacher_trajectory",
    "direction_loss",
    "distillation_loss",
    "evaluate_imitation",
    "magnitude_calibration_scale",
    "magnitude_loss",
    "rollout_student",
    "train_student",
]
