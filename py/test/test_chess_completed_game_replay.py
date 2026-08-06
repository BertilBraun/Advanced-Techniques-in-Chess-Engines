from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import pytest
import torch

from src.games.chess.ChessBoard import ChessBoard
from src.self_play.chess_completed_game import (
    ChessCompletedGame,
    ChessCompletedGamePublisher,
    ChessMoveSelectionMode,
    ChessRepresentationMetadata,
    ChessRulesMetadata,
    ChessSearchObservation,
    SparseSearchVisit,
)
from src.self_play.value_target import FinalOutcome, TerminationReason
from src.settings import CurrentGame
from src.train.ChessReplay import (
    CHESS_ARCHIVE_HEADER,
    ChessReplay,
    ChessReplayMaintainer,
    ReplayPhase,
    append_chess_archive_record,
    build_chess_training_batch,
    canonical_game_payload,
    inspect_chess_archives,
    materialize_chess_game,
    read_chess_archive,
    rebuild_chess_replay,
)


FOOLS_MATE = ('f2f3', 'e7e5', 'g2g4', 'd8h4')


def completed_game(
    publisher: ChessCompletedGamePublisher,
    model_generation: int = 3,
) -> ChessCompletedGame:
    identity = publisher.reserve_identity()
    board = ChessBoard()
    observations: list[ChessSearchObservation] = []
    for ply, move_uci in enumerate(FOOLS_MATE):
        move = chess.Move.from_uci(move_uci)
        legal_actions = tuple(
            sorted(CurrentGame.encode_move(legal_move, board) for legal_move in board.get_valid_moves())
        )
        selected_action = CurrentGame.encode_move(move, board)
        alternative_action = next(action for action in legal_actions if action != selected_action)
        observations.append(
            ChessSearchObservation(
                ply=ply,
                model_generation=model_generation,
                legal_action_ids=legal_actions,
                visits=(
                    SparseSearchVisit(action_id=selected_action, visit_count=7),
                    SparseSearchVisit(action_id=alternative_action, visit_count=2),
                ),
                root_value=-0.1 * board.current_player,
                selected_action_id=selected_action,
                move_selection_mode=ChessMoveSelectionMode.TEMPERATURE,
                search_budget=9,
                minimum_visit_count=1,
                sample_weight=1.5 if ply == 0 else 1.0,
            )
        )
        board.make_move(move)
    return ChessCompletedGame(
        identity=identity,
        rules=ChessRulesMetadata(),
        representation=ChessRepresentationMetadata(),
        model_generation=model_generation,
        minimum_model_generation=model_generation,
        created_at_seconds=100.0 + identity.game_number,
        generation_seconds=2.0,
        initial_fen=chess.STARTING_FEN,
        moves_uci=FOOLS_MATE,
        final_current_player=board.current_player,
        final_score=-1.0,
        termination_reason=TerminationReason.NATURAL,
        resignation_audit=False,
        resignation_threshold=None,
        observations=tuple(observations),
    )


def test_concurrent_publishers_create_complete_unique_game_files(tmp_path: Path) -> None:
    publishers = tuple(ChessCompletedGamePublisher(tmp_path, 17, worker_id) for worker_id in range(8))

    def publish(publisher: ChessCompletedGamePublisher) -> Path:
        game = completed_game(publisher)
        return publisher.publish(game)

    with ThreadPoolExecutor(max_workers=len(publishers)) as executor:
        paths = tuple(executor.map(publish, publishers))

    assert len(set(paths)) == len(publishers)
    assert all(path.is_file() for path in paths)
    assert all(ChessCompletedGame.model_validate_json(path.read_text(encoding='utf-8')) for path in paths)
    assert not tuple((tmp_path / 'completed-games' / 'inbox').glob('*.tmp'))


def test_materializer_derives_current_chess_targets_and_source_metadata(tmp_path: Path) -> None:
    game = completed_game(ChessCompletedGamePublisher(tmp_path, 2, 0))

    samples = materialize_chess_game(game)

    assert tuple(sample.metadata.ply for sample in samples) == (3, 2, 1, 0)
    assert samples[0].value_target.final_outcome is FinalOutcome.WIN
    assert samples[1].value_target.final_outcome is FinalOutcome.LOSS
    assert samples[-1].sample_weight == 1.5
    assert samples[-1].source_model_generation == 3
    assert samples[-1].metadata.starting_fen == chess.STARTING_FEN
    assert samples[-1].metadata.moves_uci == ()
    assert samples[0].metadata.moves_uci == FOOLS_MATE[:3]


def test_archive_recovers_incomplete_final_frame(tmp_path: Path) -> None:
    publisher = ChessCompletedGamePublisher(tmp_path, 3, 0)
    first = completed_game(publisher)
    second = completed_game(publisher)
    archive = tmp_path / 'archive.games'
    append_chess_archive_record(archive, canonical_game_payload(first))
    first_size = archive.stat().st_size
    append_chess_archive_record(archive, canonical_game_payload(second))
    with archive.open('r+b') as file:
        file.truncate(archive.stat().st_size - 11)

    recovered = read_chess_archive(archive, recover_incomplete=True)

    assert recovered == (first,)
    assert archive.stat().st_size == first_size
    assert archive.read_bytes().startswith(CHESS_ARCHIVE_HEADER)


def test_restart_recovers_archived_inbox_game_exactly_once(tmp_path: Path) -> None:
    publisher = ChessCompletedGamePublisher(tmp_path, 4, 0)
    game = completed_game(publisher)
    inbox_file = publisher.publish(game)
    archive = tmp_path / 'completed-games' / 'archive' / 'model-generation-00000000000000000003.games'
    append_chess_archive_record(archive, canonical_game_payload(game))

    maintainer = ChessReplayMaintainer(tmp_path, capacity=100, sampler_seed=11)
    snapshot, _ = maintainer.maintain(100)

    assert not inbox_file.exists()
    assert snapshot.credited_samples == 4
    assert len(snapshot.samples) == 4
    assert read_chess_archive(archive) == (game,)


def test_archive_rebuild_matches_live_fifo_samples_and_credits(tmp_path: Path) -> None:
    publisher = ChessCompletedGamePublisher(tmp_path, 5, 0)
    games = tuple(completed_game(publisher) for _ in range(3))
    for game in games:
        publisher.publish(game)
    maintainer = ChessReplayMaintainer(tmp_path, capacity=9, sampler_seed=23)
    live, metrics = maintainer.maintain(9)

    rebuilt = rebuild_chess_replay(tmp_path, capacity=9, sampler_seed=23)

    assert rebuilt.samples == live.samples
    assert rebuilt.credited_samples == live.credited_samples == 12
    assert rebuilt.credited_completed_searches == live.credited_completed_searches == 108
    assert len(rebuilt.samples) == metrics.live_samples == 9
    inspection = inspect_chess_archives(tmp_path)
    assert len(inspection) == 1
    assert inspection[0].game_count == 3
    assert inspection[0].eligible_sample_count == 12


def test_fifo_phase_capacity_and_deterministic_nonoverlapping_rank_sampling(tmp_path: Path) -> None:
    publisher = ChessCompletedGamePublisher(tmp_path, 6, 0)
    replay = ChessReplay(capacity=6, sampler_seed=31)
    replay.ingest_game(completed_game(publisher))
    replay.ingest_game(completed_game(publisher))
    snapshot = replay.freeze()

    assert replay.phase is ReplayPhase.FROZEN
    assert len(snapshot.samples) == 6
    assert replay.metrics(snapshot.frozen_at_seconds).evicted_samples == 2
    with pytest.raises(RuntimeError, match='ingestion phase'):
        replay.ingest_game(completed_game(publisher))
    ranks = tuple(
        snapshot.rank_indices(
            global_step=7,
            optimizer_steps=1,
            global_batch_size=4,
            world_size=2,
            rank=rank,
        )
        for rank in range(2)
    )
    assert not set(ranks[0]) & set(ranks[1])
    assert ranks == tuple(snapshot.rank_indices(7, 1, 4, 2, rank) for rank in range(2))


def test_preallocated_batch_encoding_applies_deterministic_chess_symmetry(tmp_path: Path) -> None:
    publisher = ChessCompletedGamePublisher(tmp_path, 7, 0)
    replay = ChessReplay(capacity=10, sampler_seed=41)
    replay.ingest_game(completed_game(publisher))
    snapshot = replay.freeze()

    batch = build_chess_training_batch(snapshot, (0, 1), global_step=9, rank=0)

    assert batch.states.shape == (2, *CurrentGame.representation_shape)
    assert batch.policy_targets.shape == (2, CurrentGame.action_size)
    torch.testing.assert_close(batch.policy_targets.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(batch.sample_weights, torch.ones(2))
