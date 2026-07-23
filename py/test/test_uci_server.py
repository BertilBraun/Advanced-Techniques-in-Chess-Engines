from __future__ import annotations

import io
from dataclasses import dataclass

import pytest

from src.uci.server import (
    PositionCommand,
    SearchMode,
    SearchRequest,
    UciServer,
    parse_move_time_seconds,
    parse_position,
    parse_search_mode,
)


@dataclass(frozen=True)
class FakeCandidate:
    move_uci: str


@dataclass(frozen=True)
class FakeResult:
    chosen_move_uci: str
    candidates: tuple[FakeCandidate, ...]


class FakeGame:
    def __init__(
        self,
        moves_uci: tuple[str, ...],
        result_move: str = 'e2e4',
        result_candidates_uci: tuple[str, ...] | None = None,
    ) -> None:
        self.moves_uci = list(moves_uci)
        self.result_move = result_move
        self.result_candidates_uci = (result_move,) if result_candidates_uci is None else result_candidates_uci
        self.requests: list[SearchRequest] = []

    def apply_move(self, move_uci: str) -> None:
        self.moves_uci.append(move_uci)

    def analyze(self, request: SearchRequest) -> FakeResult:
        self.requests.append(request)
        return FakeResult(
            self.result_move,
            tuple(FakeCandidate(move_uci) for move_uci in self.result_candidates_uci),
        )


class StoppableFakeGame(FakeGame):
    def analyze(self, request: SearchRequest) -> FakeResult:
        self.requests.append(request)
        if not request.stop_event.wait(timeout=2):
            raise ValueError('Test search was not stopped.')
        return FakeResult(
            self.result_move,
            tuple(FakeCandidate(move_uci) for move_uci in self.result_candidates_uci),
        )


class FakeEngine:
    def __init__(self, result_move: str = 'e2e4', stoppable: bool = False) -> None:
        self.result_move = result_move
        self.stoppable = stoppable
        self.games: list[FakeGame] = []

    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> FakeGame:
        game = (
            StoppableFakeGame(moves_uci, self.result_move) if self.stoppable else FakeGame(moves_uci, self.result_move)
        )
        self.games.append(game)
        return game


def test_parse_position_validates_and_retains_complete_history() -> None:
    assert parse_position('position startpos moves e2e4 e7e5') == PositionCommand(
        starting_fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        moves_uci=('e2e4', 'e7e5'),
    )


@pytest.mark.parametrize('command', ['position startpos moves e2e5', 'position fen broken'])
def test_parse_position_rejects_invalid_input(command: str) -> None:
    with pytest.raises(ValueError):
        parse_position(command)


@pytest.mark.parametrize(('milliseconds', 'seconds'), [(1_000, 1), (10_999, 10), (30_000, 30)])
def test_parse_move_time_accepts_contract_range(milliseconds: int, seconds: int) -> None:
    assert parse_move_time_seconds(f'go movetime {milliseconds}') == seconds


@pytest.mark.parametrize('command', ['go movetime 999', 'go movetime 30001', 'go wtime 1000'])
def test_parse_move_time_rejects_unsupported_limits(command: str) -> None:
    with pytest.raises(ValueError):
        parse_move_time_seconds(command)


def test_parse_search_mode() -> None:
    assert parse_search_mode('setoption name SearchMode value policy') is SearchMode.POLICY


def test_server_reuses_game_for_extended_history() -> None:
    engine = FakeEngine(result_move='e7e5')
    output = io.StringIO()
    server = UciServer(engine, output, io.StringIO())

    server.process('position startpos moves e2e4')
    server.process('go movetime 1000')
    server._stop_search(wait=True)
    server.process('position startpos moves e2e4 e7e5')

    assert len(engine.games) == 1
    assert engine.games[0].moves_uci == ['e2e4', 'e7e5']
    assert output.getvalue() == 'bestmove e7e5\n'


def test_server_reconstructs_on_history_divergence() -> None:
    engine = FakeEngine(result_move='e7e5')
    server = UciServer(engine, io.StringIO(), io.StringIO())
    server.process('position startpos moves e2e4')
    server.process('go movetime 1000')
    server._stop_search(wait=True)
    server.process('position startpos moves d2d4')
    server.process('go movetime 1000')
    server._stop_search(wait=True)

    assert len(engine.games) == 2
    assert engine.games[1].moves_uci == ['d2d4']


def test_server_keeps_diagnostics_off_standard_output() -> None:
    engine = FakeEngine(result_move='e2e5')
    output = io.StringIO()
    error = io.StringIO()
    server = UciServer(engine, output, error)
    server.process('go movetime 1000')
    server._stop_search(wait=True)

    assert output.getvalue() == 'bestmove a2a3\n'
    assert 'illegal move' in error.getvalue()


def test_server_uses_highest_ranked_legal_candidate() -> None:
    engine = FakeEngine(result_move='e2e5')
    output = io.StringIO()
    error = io.StringIO()
    server = UciServer(engine, output, error)
    game = FakeGame(
        (),
        result_move='e2e5',
        result_candidates_uci=('e2e5', 'd2d4', 'e2e4'),
    )
    engine.games.append(game)
    server._game = game

    server.process('go movetime 1000')
    server._stop_search(wait=True)

    assert output.getvalue() == 'bestmove d2d4\n'
    assert 'highest-ranked legal candidate' in error.getvalue()


def test_server_searches_when_draw_can_only_be_claimed_by_moving() -> None:
    engine = FakeEngine(result_move='f6g8')
    output = io.StringIO()
    server = UciServer(engine, output, io.StringIO())
    server.process('position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1')

    server.process('go movetime 1000')
    server._stop_search(wait=True)

    assert output.getvalue() == 'bestmove f6g8\n'


def test_server_stop_signals_active_analysis() -> None:
    engine = FakeEngine(stoppable=True)
    output = io.StringIO()
    server = UciServer(engine, output, io.StringIO())

    server.process('go movetime 30000')
    server.process('stop')
    server._stop_search(wait=True)

    assert engine.games[0].requests[0].stop_event.is_set()
    assert output.getvalue() == 'bestmove e2e4\n'
