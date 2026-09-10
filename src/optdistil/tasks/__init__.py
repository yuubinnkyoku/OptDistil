"""Inner-loop optimization tasks used to train and evaluate optimizers."""

from optdistil.tasks.frozen_readout_mlp import FrozenReadoutMLPTask, make_frozen_readout_mlp

__all__ = ["FrozenReadoutMLPTask", "make_frozen_readout_mlp"]
