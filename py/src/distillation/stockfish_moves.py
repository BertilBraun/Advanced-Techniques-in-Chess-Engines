"""Stockfish chooses the moves of distillation games while the teacher only labels the positions.

The teacher's search-free games miss the positions that decide real games: a student that imitates the teacher
on them as well as checkpoint 1026 does plays hundreds of Elo weaker. Games whose moves Stockfish chooses reach
positions like those a strong engine meets against Stockfish, at a cost of a few milliseconds of CPU per move
rather than a search tree.

Stockfish at fixed nodes is deterministic, so each move is sampled from its top MultiPV candidates weighted by
expected score; otherwise every game after the random opening plies would be the same game.
"""

from __future__ import annotations

import queue
import subprocess
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import numpy as np
from src.games.chess.contract import ChessPosition


@dataclass(frozen=True)
class ScoredMove:
    action_id: int
    expected_score: float


def sample_scored_move(moves: Sequence[ScoredMove], temperature: float, generator: np.random.Generator) -> int:
    """Samples an action with weight exp((score - best) / temperature) over expected scores in [0, 1]."""
    scores = np.asarray([move.expected_score for move in moves], dtype=np.float64)
    weights = np.exp((scores - scores.max()) / temperature)
    return moves[int(generator.choice(len(moves), p=weights / weights.sum()))].action_id


@dataclass(frozen=True)
class UciCandidate:
    move_uci: str
    score_kind: str
    score_value: int


def parse_multipv_line(line: str) -> tuple[int, UciCandidate] | None:
    """Reads the MultiPV index, score and first move from one `info` line, or None if it carries none."""
    if not line.startswith('info') or ' multipv ' not in line or ' score ' not in line or ' pv ' not in line:
        return None
    tokens = line.split()
    score = tokens.index('score')
    return int(tokens[tokens.index('multipv') + 1]), UciCandidate(
        move_uci=tokens[tokens.index('pv') + 1],
        score_kind=tokens[score + 1],
        score_value=int(tokens[score + 2]),
    )


def expected_score(candidate: UciCandidate, fen: str) -> float:
    """Stockfish's own WDL model applied to a centipawn or mate score, from the side to move."""
    fields = fen.split()
    ply = 2 * (int(fields[5]) - 1) + (1 if fields[1] == 'b' else 0)
    score = (
        chess.engine.Cp(candidate.score_value)
        if candidate.score_kind == 'cp'
        else chess.engine.Mate(candidate.score_value)
    )
    return float(score.wdl(ply=ply).expectation())


class _UciEngine:
    """A bare UCI pipe. python-chess's driver parsed and replayed every PV line Stockfish printed, which kept a
    builder's Python thread saturated while its engines sat idle; only the final line per MultiPV index is read."""

    def __init__(self, executable: Path, multi_pv: int) -> None:
        self._process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._send('uci')
        self._read_until('uciok')
        self._send('setoption name Threads value 1')
        self._send('setoption name Hash value 16')
        self._send(f'setoption name MultiPV value {multi_pv}')
        self._send('isready')
        self._read_until('readyok')

    def _send(self, command: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(command + '\n')
        self._process.stdin.flush()

    def _read_until(self, prefix: str) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if line.startswith(prefix):
                return
        raise RuntimeError(f'Stockfish exited before {prefix!r}.')

    def candidates(self, fen: str, nodes: int) -> dict[int, UciCandidate]:
        assert self._process.stdout is not None
        self._send(f'position fen {fen}')
        self._send(f'go nodes {nodes}')
        latest: dict[int, UciCandidate] = {}
        for line in self._process.stdout:
            if line.startswith('bestmove'):
                return latest
            parsed = parse_multipv_line(line)
            if parsed is not None:
                latest[parsed[0]] = parsed[1]
        raise RuntimeError('Stockfish exited before bestmove.')

    def quit(self) -> None:
        self._send('quit')
        self._process.wait(timeout=10)


class StockfishMovePool:
    """Single-threaded Stockfish processes analysing a batch of positions concurrently, one engine per thread."""

    def __init__(self, executable: Path, engines: int, nodes: int, multi_pv: int) -> None:
        if not executable.is_file():
            raise ValueError(f'Stockfish executable does not exist: {executable}')
        self._nodes = nodes
        self._idle: queue.Queue[_UciEngine] = queue.Queue()
        for _ in range(engines):
            self._idle.put(_UciEngine(executable, multi_pv))
        self._engine_count = engines
        self._executor = ThreadPoolExecutor(max_workers=engines)

    def analyse_async(self, positions: Sequence[ChessPosition]) -> list[Future[tuple[ScoredMove, ...]]]:
        """Starts the analyses so they overlap with the teacher's GPU batch; collect with .result()."""
        return [self._executor.submit(self._analyse, position) for position in positions]

    def _analyse(self, position: ChessPosition) -> tuple[ScoredMove, ...]:
        fen = position.fen
        engine = self._idle.get()
        try:
            candidates = engine.candidates(fen, self._nodes)
        finally:
            self._idle.put(engine)
        moves = tuple(
            ScoredMove(position.action_id_from_uci(candidate.move_uci), expected_score(candidate, fen))
            for _, candidate in sorted(candidates.items())
        )
        if not moves:
            raise ValueError(f'Stockfish returned no scored move for {fen}.')
        return moves

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for _ in range(self._engine_count):
            self._idle.get().quit()
