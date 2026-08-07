from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.games.chess.evaluation.types import (
    EvaluationMove,
    EvaluationTerminal,
    PairedEvaluationDecision,
    PairedEvaluationModel,
    Results,
)
from src.experiment.evaluation_protocol import GameOutcome, GameRecord, PlayerColor, ScheduledGame
from src.games.Game import Player
from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.util.tensorboard import log_text


@dataclass
class _ActivePairedGame:
    schedule_index: int
    opening_id: str
    starting_fen: str
    candidate_color: PlayerColor
    board: ChessBoard
    moves_uci: list[str]

    @property
    def candidate_player(self) -> Player:
        if self.candidate_color == PlayerColor.WHITE:
            return 1
        return -1


def _game_outcome(winner: Player | None, candidate_player: Player) -> GameOutcome:
    if winner is None:
        return GameOutcome.DRAW
    if winner == candidate_player:
        return GameOutcome.WIN
    return GameOutcome.LOSS


def play_paired_models(
    iteration: int,
    candidate_model: PairedEvaluationModel,
    opponent_model: PairedEvaluationModel,
    schedule: tuple[ScheduledGame, ...],
    maximum_game_plies: int | None,
    name: str,
) -> tuple[Results, tuple[GameRecord, ...]]:
    if not schedule:
        raise ValueError('A paired opening schedule is required.')
    if maximum_game_plies is not None and maximum_game_plies < 1:
        raise ValueError('maximum_game_plies must be positive.')

    active_games = [
        _ActivePairedGame(
            schedule_index=scheduled_game.schedule_index,
            opening_id=scheduled_game.opening_id,
            starting_fen=scheduled_game.fen,
            candidate_color=scheduled_game.candidate_color,
            board=ChessBoard.from_fen(scheduled_game.fen),
            moves_uci=[],
        )
        for scheduled_game in schedule
    ]
    completed_records: list[GameRecord | None] = [None] * len(schedule)

    while active_games:
        candidate_game_indices = [
            game_index
            for game_index, active_game in enumerate(active_games)
            if active_game.board.current_player == active_game.candidate_player
        ]
        opponent_game_indices = [
            game_index
            for game_index, active_game in enumerate(active_games)
            if active_game.board.current_player != active_game.candidate_player
        ]

        decisions_by_game_index: dict[int, PairedEvaluationDecision] = {}
        if candidate_game_indices:
            candidate_decisions = candidate_model(
                [active_games[game_index].board for game_index in candidate_game_indices]
            )
            assert len(candidate_decisions) == len(candidate_game_indices)
            decisions_by_game_index.update(zip(candidate_game_indices, candidate_decisions))
        if opponent_game_indices:
            opponent_decisions = opponent_model(
                [active_games[game_index].board for game_index in opponent_game_indices]
            )
            assert len(opponent_decisions) == len(opponent_game_indices)
            decisions_by_game_index.update(zip(opponent_game_indices, opponent_decisions))

        remaining_games: list[_ActivePairedGame] = []
        for game_index, active_game in enumerate(active_games):
            decision = decisions_by_game_index[game_index]
            match decision:
                case EvaluationTerminal():
                    completed_records[active_game.schedule_index] = GameRecord(
                        schedule_index=active_game.schedule_index,
                        opening_id=active_game.opening_id,
                        starting_fen=active_game.starting_fen,
                        candidate_color=active_game.candidate_color,
                        outcome=GameOutcome.DRAW,
                        moves_uci=tuple(active_game.moves_uci),
                    )
                    log_text(
                        f'evaluation_moves/{iteration}/{name}',
                        f'{GameOutcome.DRAW.value}:{",".join(active_game.moves_uci)}',
                    )
                    continue
                case EvaluationMove(policy):
                    pass
            if not np.all(np.isfinite(policy)) or float(np.sum(policy)) <= 0:
                raise ValueError('Evaluation move policy must be finite and have positive mass.')
            encoded_move = int(np.argmax(policy).item())
            move = CHESS_STATE_CONTRACT.decode_move(encoded_move, active_game.board)
            active_game.board.make_move(move)
            active_game.moves_uci.append(str(move))

            game_finished = active_game.board.is_game_over() or (
                maximum_game_plies is not None and len(active_game.moves_uci) >= maximum_game_plies
            )
            if not game_finished:
                remaining_games.append(active_game)
                continue

            winner = active_game.board.check_winner() if active_game.board.is_game_over() else None
            outcome = _game_outcome(winner, active_game.candidate_player)
            completed_records[active_game.schedule_index] = GameRecord(
                schedule_index=active_game.schedule_index,
                opening_id=active_game.opening_id,
                starting_fen=active_game.starting_fen,
                candidate_color=active_game.candidate_color,
                outcome=outcome,
                moves_uci=tuple(active_game.moves_uci),
            )
            log_text(
                f'evaluation_moves/{iteration}/{name}',
                f'{outcome.value}:{",".join(active_game.moves_uci)}',
            )

        active_games = remaining_games

    assert all(record is not None for record in completed_records)
    records = tuple(record for record in completed_records if record is not None)
    results = Results(
        wins=sum(record.outcome == GameOutcome.WIN for record in records),
        losses=sum(record.outcome == GameOutcome.LOSS for record in records),
        draws=sum(record.outcome == GameOutcome.DRAW for record in records),
    )
    return results, records
