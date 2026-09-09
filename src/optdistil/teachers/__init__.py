"""Teacher optimizer adapters."""

from optdistil.teachers.adamw import AdamWTeacher
from optdistil.teachers.muon import MuonTeacher, zeropower_via_newton_schulz5

__all__ = ["AdamWTeacher", "MuonTeacher", "zeropower_via_newton_schulz5"]
