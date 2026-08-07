from decimal import Decimal
from pathlib import Path

import chess

from src.games.chess.ChessBoard import ChessBoard
from src.self_play.chess_completed_game import (
    ChessCompletedGame,
    ChessMoveSelectionMode,
    ChessRepresentationMetadata,
    ChessRulesMetadata,
    ChessSearchObservation,
)
from src.self_play.completed_game import CompletedGamePublisher, SparseSearchVisit
from src.self_play.value_target import TerminationReason
from src.settings import CurrentGame
from src.train.ChessReplay import CHESS_REPLAY_IMPLEMENTATION, training_batch_loader
from src.train.Replay import ReplayMaintainer
from src.train.CreditTrainingLedger import CreditTrainingLedger
from src.train.TrainingArgs import CreditTrainingParams


def _completed_game(publisher: CompletedGamePublisher) -> ChessCompletedGame:
    moves_uci = ('f2f3', 'e7e5', 'g2g4', 'd8h4')
    board = ChessBoard()
    observations: list[ChessSearchObservation] = []
    for ply, move_uci in enumerate(moves_uci):
        move = chess.Move.from_uci(move_uci)
        legal_actions = tuple(
            sorted(CurrentGame.encode_move(candidate, board) for candidate in board.get_valid_moves())
        )
        selected_action = CurrentGame.encode_move(move, board)
        observations.append(
            ChessSearchObservation(
                ply=ply,
                model_generation=0,
                legal_action_ids=legal_actions,
                visits=(SparseSearchVisit(action_id=selected_action, visit_count=16),),
                root_value=0.0,
                selected_action_id=selected_action,
                move_selection_mode=ChessMoveSelectionMode.TEMPERATURE,
                search_budget=16,
                minimum_visit_count=0,
            )
        )
        board.make_move(move)
    return ChessCompletedGame(
        identity=publisher.reserve_identity(),
        rules=ChessRulesMetadata(),
        representation=ChessRepresentationMetadata(),
        model_generation=0,
        minimum_model_generation=0,
        created_at_seconds=1.0,
        generation_seconds=1.0,
        initial_fen=chess.STARTING_FEN,
        moves_uci=moves_uci,
        final_current_player=board.current_player,
        final_score=-1.0,
        termination_reason=TerminationReason.NATURAL,
        resignation_audit=False,
        resignation_threshold=None,
        observations=tuple(observations),
    )


def test_credit_runtime_rebuilds_fifo_and_rank_batches_after_restart(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, run_id=9, worker_id=0)
    for _ in range(3):
        publisher.publish(_completed_game(publisher))
    maintainer = ReplayMaintainer(tmp_path, CHESS_REPLAY_IMPLEMENTATION, capacity=10, sampler_seed=51)
    snapshot, _ = maintainer.maintain(10)
    parameters = CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=1,
        maximum_optimizer_steps=2,
        initial_replay_capacity_unique_positions=10,
        maximum_replay_capacity_unique_positions=10,
        replay_capacity_ramp_model_versions=1,
        retained_checkpoint_interval_steps=1,
    )
    ledger = CreditTrainingLedger(tmp_path, parameters, global_batch_size=8)

    progress = ledger.reconcile_credited_samples(snapshot.credited_samples)
    restarted, _ = ReplayMaintainer(
        tmp_path,
        CHESS_REPLAY_IMPLEMENTATION,
        capacity=10,
        sampler_seed=51,
    ).maintain(10)
    rank_zero = training_batch_loader(restarted, 0, 1, 8, 2, 0, pin_memory=False)
    rank_one = training_batch_loader(restarted, 0, 1, 8, 2, 1, pin_memory=False)

    assert progress.can_train(8)
    assert snapshot.credited_samples == restarted.credited_samples == 12
    assert snapshot.samples == restarted.samples
    assert not set(rank_zero.indices) & set(rank_one.indices)
    assert sum(len(batch) for batch in rank_zero) == 4
    assert sum(len(batch) for batch in rank_one) == 4
