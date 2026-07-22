from __future__ import annotations

import threading
from dataclasses import dataclass
from uuid import UUID, uuid4

import chess

from web_play.contracts import (
    AnalysisOptions,
    AnalysisResult,
    CreateGameResponse,
    GameState,
    PlayTurnRequest,
    PlayTurnResponse,
    SideToMove,
    TimedAnalysis,
)
from web_play.engine import InteractiveEngine, InteractiveGame


@dataclass
class GameSession:
    game: InteractiveGame
    starting_fen: str
    complete_moves_uci: tuple[str, ...]


class GameService:
    """In-memory reuse layer; every turn remains recoverable from its complete history."""

    def __init__(self, engine: InteractiveEngine) -> None:
        self._engine = engine
        self._sessions: dict[UUID, GameSession] = {}
        self._registry_lock = threading.Lock()
        self._search_lock = threading.Lock()

    def create_game(
        self, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> CreateGameResponse:
        with self._search_lock:
            game = self._engine.new_game(starting_fen, moves_uci)
            token = uuid4()
            session = GameSession(game, starting_fen, moves_uci)
            with self._registry_lock:
                self._sessions[token] = session
            return CreateGameResponse(game_token=token, state=_game_state(session))

    def play_turn(self, token: UUID, request: PlayTurnRequest) -> PlayTurnResponse:
        with self._search_lock:
            session = self._synchronize(token, request.starting_fen, request.moves_uci)
            complete_moves = list(session.complete_moves_uci)
            if request.human_move_uci is not None:
                session.game.apply_move(request.human_move_uci)
                complete_moves.append(request.human_move_uci)

            analysis: AnalysisResult | None = None
            engine_move_uci: str | None = None
            if not session.game.is_game_over:
                analysis = session.game.analyze(_timed_analysis(request.analysis))
                engine_move_uci = analysis.chosen_move_uci
                session.game.apply_move(engine_move_uci)
                complete_moves.append(engine_move_uci)

            session.complete_moves_uci = tuple(complete_moves)
            return PlayTurnResponse(
                state=_game_state(session),
                engine_move_uci=engine_move_uci,
                analysis=analysis,
            )

    def end_game(self, token: UUID) -> None:
        with self._registry_lock:
            self._sessions.pop(token, None)

    def _synchronize(
        self, token: UUID, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> GameSession:
        with self._registry_lock:
            current = self._sessions.get(token)
        if current is not None and (
            current.starting_fen == starting_fen
            and current.complete_moves_uci == moves_uci
        ):
            return current

        recovered = GameSession(
            game=self._engine.new_game(starting_fen, moves_uci),
            starting_fen=starting_fen,
            complete_moves_uci=moves_uci,
        )
        with self._registry_lock:
            self._sessions[token] = recovered
        return recovered


def _timed_analysis(options: AnalysisOptions) -> TimedAnalysis:
    return TimedAnalysis(
        mode=options.mode, time_limit_seconds=options.time_limit_seconds
    )


def _game_state(session: GameSession) -> GameState:
    board = chess.Board(session.game.fen)
    return GameState(
        starting_fen=session.starting_fen,
        moves_uci=session.complete_moves_uci,
        fen=session.game.fen,
        side_to_move=SideToMove.WHITE
        if board.turn is chess.WHITE
        else SideToMove.BLACK,
        game_over=session.game.is_game_over,
        result=session.game.result,
    )
