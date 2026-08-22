from __future__ import annotations

from pathlib import Path
from typing import Protocol

import chess
import chess.syzygy
from src.games.contracts import TerminalOracle, WdlTarget


class SyzygyPosition(Protocol):
    @property
    def fen(self) -> str: ...


class SyzygyTerminalOracle(TerminalOracle[SyzygyPosition]):
    def __init__(self, tablebase_paths: tuple[str, ...]) -> None:
        if not tablebase_paths:
            raise ValueError('Syzygy terminal oracle requires at least one tablebase directory.')
        paths = tuple(Path(path) for path in tablebase_paths)
        missing = tuple(path for path in paths if not path.is_dir())
        if missing:
            raise ValueError(f'Syzygy tablebase directory does not exist: {missing[0]}')
        self._tablebase = chess.syzygy.open_tablebase(str(paths[0]), load_dtz=False)
        for path in paths[1:]:
            self._tablebase.add_directory(str(path), load_wdl=True, load_dtz=False)

    def probe_wdl(self, position: SyzygyPosition) -> WdlTarget | None:
        try:
            syzygy_wdl = self._tablebase.probe_wdl(chess.Board(position.fen))
        except KeyError:
            return None
        match syzygy_wdl:
            case 2:
                return WdlTarget(win=1.0, draw=0.0, loss=0.0)
            case -2:
                return WdlTarget(win=0.0, draw=0.0, loss=1.0)
            case _:
                return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def close(self) -> None:
        self._tablebase.close()
