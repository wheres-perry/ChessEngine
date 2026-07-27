"""Concrete evaluation components — thin re-exports of the C++ backend.

Each class below is the pybind11-bound C++ implementation; the imports here
preserve the legacy ``engine.evaluators.components`` path.
"""

from __future__ import annotations

from engine._core import moray_core as chess

_ev = chess.evaluators

MaterialComponent = _ev.MaterialComponent
PSTComponent = _ev.PSTComponent
PawnStructureComponent = _ev.PawnStructureComponent
MobilityComponent = _ev.MobilityComponent
KingSafetyComponent = _ev.KingSafetyComponent

__all__ = [
    "KingSafetyComponent",
    "MaterialComponent",
    "MobilityComponent",
    "PSTComponent",
    "PawnStructureComponent",
]
