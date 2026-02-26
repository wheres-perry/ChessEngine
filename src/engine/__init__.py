"""Single-source package metadata."""

from __future__ import annotations

__version__ = "0.1.0"

from engine.config import EngineConfig
from engine.factory import EngineRuntime, create_engine_runtime

__all__ = ["EngineConfig", "EngineRuntime", "__version__", "create_engine_runtime"]
