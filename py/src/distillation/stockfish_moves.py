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


class StockfishMovePool:
    """Single-threaded Stockfish processes analysing a batch of positions concurrently, one engine per thread."""

    def __init__(self, executable: Path, engines: int, nodes: int, multi_pv: int) -> None:
        if not executable.is_file():
            raise ValueError(f'Stockfish executable does not exist: {executable}')
        self._nodes = nodes
        self._multi_pv = multi_pv
        self._idle: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        for _ in range(engines):
            engine = chess.engine.SimpleEngine.popen_uci(str(executable))
            engine.configure({'Threads': 1, 'Hash': 16})
            self._idle.put(engine)
        self._engine_count = engines
        self._executor = ThreadPoolExecutor(max_workers=engines)

    def analyse_async(self, positions: Sequence[ChessPosition]) -> list[Future[tuple[ScoredMove, ...]]]:
        """Starts the analyses so they overlap with the teacher's GPU batch; collect with .result()."""
        return [self._executor.submit(self._analyse, position) for position in positions]

    def _analyse(self, position: ChessPosition) -> tuple[ScoredMove, ...]:
        engine = self._idle.get()
        try:
            board = chess.Board(position.fen)
            analysis = engine.analyse(
                board,
                chess.engine.Limit(nodes=self._nodes),
                multipv=min(self._multi_pv, board.legal_moves.count()),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
        finally:
            self._idle.put(engine)
        records = analysis if isinstance(analysis, list) else [analysis]
        moves: list[ScoredMove] = []
        for record in records:
            variation = record.get('pv')
            score = record.get('score')
            if not variation or score is None:
                continue
            expected = score.pov(board.turn).wdl(ply=board.ply()).expectation()
            moves.append(ScoredMove(position.action_id_from_uci(variation[0].uci()), float(expected)))
        if not moves:
            raise ValueError(f'Stockfish returned no scored move for {position.fen}.')
        return tuple(moves)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for _ in range(self._engine_count):
            self._idle.get().quit()
