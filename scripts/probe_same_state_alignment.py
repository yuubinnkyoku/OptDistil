from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch
from compare_teachers import (
    LR_CANDIDATES if False else MUON_LR_CANDIDATES,
)
