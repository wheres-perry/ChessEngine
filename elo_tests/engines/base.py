from __future__ import annotations

from typing import Any, Protocol


class EngineAdapter(Protocol):
    @property
    def engine_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def strength_elo(self) -> float: ...

    @property
    def draw_bias(self) -> float: ...

    def new_game(self, seed: int, opening_fen: str, side: str) -> None: ...

    def choose_move(self, state: Any, move_time_ms: int | None, depth: int | None) -> Any: ...
