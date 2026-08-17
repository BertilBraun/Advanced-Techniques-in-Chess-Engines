from pathlib import Path

import chess
import chess.syzygy
import pytest

from src.games.chess.syzygy import SyzygyTerminalOracle
from src.games.contracts import WdlTarget


class FakeChessPosition:
    fen = '8/8/8/8/8/8/4K3/6k1 w - - 0 1'


class FakeTablebase:
    def __init__(self, wdl: int | None) -> None:
        self.wdl = wdl
        self.added_directories: list[tuple[str, bool, bool]] = []
        self.closed = False

    def add_directory(self, directory: str, *, load_wdl: bool, load_dtz: bool) -> int:
        self.added_directories.append((directory, load_wdl, load_dtz))
        return 0

    def probe_wdl(self, board: chess.Board) -> int:
        assert board.fen() == FakeChessPosition.fen
        if self.wdl is None:
            raise KeyError('position is not covered')
        return self.wdl

    def close(self) -> None:
        self.closed = True


def _install_fake_tablebase(monkeypatch: pytest.MonkeyPatch, tablebase: FakeTablebase) -> None:
    def open_tablebase(directory: str, *, load_dtz: bool) -> FakeTablebase:
        assert Path(directory).is_dir()
        assert not load_dtz
        return tablebase

    monkeypatch.setattr(chess.syzygy, 'open_tablebase', open_tablebase)


@pytest.mark.parametrize(
    ('syzygy_wdl', 'expected'),
    (
        (-2, WdlTarget(win=0.0, draw=0.0, loss=1.0)),
        (-1, WdlTarget(win=0.0, draw=1.0, loss=0.0)),
        (0, WdlTarget(win=0.0, draw=1.0, loss=0.0)),
        (1, WdlTarget(win=0.0, draw=1.0, loss=0.0)),
        (2, WdlTarget(win=1.0, draw=0.0, loss=0.0)),
    ),
)
def test_syzygy_terminal_oracle_maps_side_to_move_wdl_without_dtz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    syzygy_wdl: int,
    expected: WdlTarget,
) -> None:
    first_path = tmp_path / 'first'
    second_path = tmp_path / 'second'
    first_path.mkdir()
    second_path.mkdir()
    tablebase = FakeTablebase(syzygy_wdl)
    _install_fake_tablebase(monkeypatch, tablebase)
    oracle = SyzygyTerminalOracle((str(first_path), str(second_path)))

    assert oracle.probe_wdl(FakeChessPosition()) == expected
    assert tablebase.added_directories == [(str(second_path), True, False)]
    oracle.close()
    assert tablebase.closed


def test_syzygy_terminal_oracle_returns_none_for_uncovered_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tablebase = FakeTablebase(None)
    _install_fake_tablebase(monkeypatch, tablebase)
    oracle = SyzygyTerminalOracle((str(tmp_path),))

    assert oracle.probe_wdl(FakeChessPosition()) is None


def test_syzygy_terminal_oracle_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='does not exist'):
        SyzygyTerminalOracle((str(tmp_path / 'missing'),))
