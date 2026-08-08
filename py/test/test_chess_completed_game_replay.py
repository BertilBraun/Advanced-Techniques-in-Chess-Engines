from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

import src.games.chess.replay as chess_replay_module
import src.training.replay as replay_module
from src.games.chess.board import ChessBoard
from src.games.chess.completed_game import (
    ChessCompletedGame,
    ChessMoveSelectionMode,
    ChessRepresentationMetadata,
    ChessRulesMetadata,
    ChessSearchObservation,
)
from src.self_play.completed_game import CompletedGamePublisher, SparseSearchVisit
from src.games.chess.self_play import ChessSelfPlayPolicy, SelfPlayGame, SelfPlayGameMemory
from src.games.chess.self_play_statistics import SelfPlayStatistics
from src.self_play.value_target import FinalOutcome, TerminationReason
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.replay import (
    CHESS_REPLAY_IMPLEMENTATION,
    CHESS_ARCHIVE_HEADER,
    ReplayPhase,
    append_chess_archive_record,
    build_chess_training_batch,
    canonical_game_payload,
    inspect_chess_archives,
    materialize_chess_game,
    pack_chess_visits,
    read_chess_archive,
    rebuild_chess_replay,
)
from src.training.replay import Replay, ReplayMaintainer


FOOLS_MATE = ('f2f3', 'e7e5', 'g2g4', 'd8h4')


def chess_replay_maintainer(run_path: Path, capacity: int, sampler_seed: int) -> ReplayMaintainer:
    return ReplayMaintainer(run_path, CHESS_REPLAY_IMPLEMENTATION, capacity, sampler_seed)


def completed_game(
    publisher: CompletedGamePublisher,
    model_generation: int = 3,
) -> ChessCompletedGame:
    identity = publisher.reserve_identity()
    board = ChessBoard()
    observations: list[ChessSearchObservation] = []
    for ply, move_uci in enumerate(FOOLS_MATE):
        move = chess.Move.from_uci(move_uci)
        legal_actions = tuple(
            sorted(CHESS_STATE_CONTRACT.encode_move(legal_move, board) for legal_move in board.get_valid_moves())
        )
        selected_action = CHESS_STATE_CONTRACT.encode_move(move, board)
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
    publishers = tuple(CompletedGamePublisher(tmp_path, 17, worker_id) for worker_id in range(8))

    def publish(publisher: CompletedGamePublisher) -> Path:
        game = completed_game(publisher)
        return publisher.publish(game)

    with ThreadPoolExecutor(max_workers=len(publishers)) as executor:
        paths = tuple(executor.map(publish, publishers))

    assert len(set(paths)) == len(publishers)
    assert all(path.is_file() for path in paths)
    assert all(ChessCompletedGame.model_validate_json(path.read_text(encoding='utf-8')) for path in paths)
    assert not tuple((tmp_path / 'completed-games' / 'inbox').glob('*.tmp'))


def test_self_play_completion_publishes_game_instead_of_writing_samples(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 18, 2)
    self_play = object.__new__(ChessSelfPlayPolicy)
    self_play.args = SimpleNamespace()
    self_play.resolved_parameters = SimpleNamespace(greedy_after_ply=30, minimum_root_visits=0)
    self_play.completed_game_publisher = publisher
    self_play.statistics = SelfPlayStatistics()
    self_play.iteration = 0
    self_play.num_searches_per_turn = 8
    game = SelfPlayGame(identity=publisher.reserve_identity())
    game.acknowledge_model_version(0)
    for ply, move_uci in enumerate(FOOLS_MATE):
        move = chess.Move.from_uci(move_uci)
        selected_action = CHESS_STATE_CONTRACT.encode_move(move, game.board)
        game.memory.append(SelfPlayGameMemory(game.board.copy(), [(selected_action, 8)], 0.0, ply, 0, 8, True))
        game = game.expand(move)

    self_play._add_training_data(game, -1.0, TerminationReason.NATURAL)

    inbox_files = tuple((tmp_path / 'completed-games' / 'inbox').glob('*.json'))
    assert len(inbox_files) == 1
    published = ChessCompletedGame.model_validate_json(inbox_files[0].read_text(encoding='utf-8'))
    assert published.moves_uci == FOOLS_MATE
    assert len(published.observations) == 4
    assert self_play.statistics.stats.num_samples == 0


def test_materializer_derives_current_chess_targets_and_source_metadata(tmp_path: Path) -> None:
    game = completed_game(CompletedGamePublisher(tmp_path, 2, 0))

    samples = materialize_chess_game(game)

    assert tuple(sample.metadata.ply for sample in samples) == (3, 2, 1, 0)
    assert samples[0].value_target.final_outcome is FinalOutcome.WIN
    assert samples[1].value_target.final_outcome is FinalOutcome.LOSS
    assert samples[-1].sample_weight == 1.5
    assert samples[-1].source_model_generation == 3
    assert samples[-1].visits.dtype == np.uint16
    assert samples[-1].visits.shape == (2, 2)
    assert not samples[-1].visits.flags.writeable
    assert game.initial_fen == chess.STARTING_FEN
    assert game.moves_uci == FOOLS_MATE


@pytest.mark.parametrize(
    'visits',
    (
        ((-1, 1),),
        ((65_536, 1),),
        ((0, 0),),
        ((0, 65_536),),
    ),
)
def test_packed_chess_visits_reject_values_outside_uint16(visits: tuple[tuple[int, int], ...]) -> None:
    with pytest.raises(ValueError, match='uint16'):
        pack_chess_visits(visits)


def test_ineligible_fast_search_is_archived_and_counted_without_earning_credit(tmp_path: Path) -> None:
    game = completed_game(CompletedGamePublisher(tmp_path, 20, 0))
    fast_observation = game.observations[1].model_copy(update={'sample_eligible': False, 'search_budget': 3})
    game = game.model_copy(update={'observations': (game.observations[0], fast_observation, *game.observations[2:])})
    replay = Replay(CHESS_REPLAY_IMPLEMENTATION, capacity=10, sampler_seed=2)

    credited = replay.ingest_game(game)
    snapshot = replay.freeze()

    assert len(game.observations) == len(game.moves_uci)
    assert credited == snapshot.credited_samples == 3
    assert snapshot.credited_completed_searches == 30


def test_resignation_materializes_final_unplayed_search_position(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 22, 0)
    game = completed_game(publisher)
    board = ChessBoard()
    for move_uci in FOOLS_MATE[:3]:
        board.make_move(chess.Move.from_uci(move_uci))
    legal_actions = tuple(sorted(CHESS_STATE_CONTRACT.encode_move(move, board) for move in board.get_valid_moves()))
    terminal_observation = ChessSearchObservation(
        ply=3,
        model_generation=3,
        legal_action_ids=legal_actions,
        visits=(SparseSearchVisit(action_id=legal_actions[0], visit_count=9),),
        root_value=-0.9,
        selected_action_id=None,
        move_selection_mode=ChessMoveSelectionMode.TERMINAL,
        search_budget=9,
        minimum_visit_count=0,
    )
    game = game.model_copy(
        update={
            'moves_uci': FOOLS_MATE[:3],
            'final_current_player': board.current_player,
            'final_score': -1.0,
            'termination_reason': TerminationReason.RESIGNATION,
            'observations': (*game.observations[:3], terminal_observation),
        }
    )

    samples = materialize_chess_game(game)

    assert samples[0].metadata.ply == len(game.moves_uci)
    assert samples[0].value_target.final_outcome is FinalOutcome.LOSS


def test_archive_recovers_incomplete_final_frame(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 3, 0)
    first = completed_game(publisher)
    second = completed_game(publisher)
    archive = tmp_path / 'archive.games'
    append_chess_archive_record(archive, canonical_game_payload(first), ingestion_sequence=0)
    first_size = archive.stat().st_size
    append_chess_archive_record(archive, canonical_game_payload(second), ingestion_sequence=1)
    with archive.open('r+b') as file:
        file.truncate(archive.stat().st_size - 11)

    recovered = read_chess_archive(archive, recover_incomplete=True)

    assert recovered == (first,)
    assert archive.stat().st_size == first_size
    assert archive.read_bytes().startswith(CHESS_ARCHIVE_HEADER)


def test_restart_recovers_archived_inbox_game_exactly_once(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 4, 0)
    game = completed_game(publisher)
    inbox_file = publisher.publish(game)
    archive = tmp_path / 'completed-games' / 'archive' / 'model-generation-00000000000000000003.games'
    append_chess_archive_record(archive, canonical_game_payload(game), ingestion_sequence=0)

    maintainer = chess_replay_maintainer(tmp_path, capacity=100, sampler_seed=11)
    snapshot, _ = maintainer.maintain(100)

    assert not inbox_file.exists()
    assert snapshot.credited_samples == 4
    assert len(snapshot.samples) == 4
    assert read_chess_archive(archive) == (game,)


def test_archive_rebuild_matches_live_fifo_samples_and_credits(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 5, 0)
    games = tuple(completed_game(publisher) for _ in range(3))
    for game in games:
        publisher.publish(game)
    maintainer = chess_replay_maintainer(tmp_path, capacity=9, sampler_seed=23)
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


def test_restart_materializes_only_the_newest_frames_needed_for_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = CompletedGamePublisher(tmp_path, 23, 0)
    for model_generation in range(5):
        publisher.publish(completed_game(publisher, model_generation=model_generation))
    chess_replay_maintainer(tmp_path, capacity=20, sampler_seed=29).maintain(20)
    original_reader = replay_module.read_frame_payload
    read_sequences: list[int] = []

    def track_read(frame_index: chess_replay_module.ChessArchiveFrameIndex) -> bytes:
        read_sequences.append(frame_index.ingestion_sequence)
        return original_reader(frame_index)

    monkeypatch.setattr(replay_module, 'read_frame_payload', track_read)

    restarted, _ = chess_replay_maintainer(tmp_path, capacity=8, sampler_seed=29).maintain(8)

    assert read_sequences == [3, 4]
    assert restarted.credited_samples == 20
    assert tuple(sample.source_model_generation for sample in restarted.samples) == (3, 3, 3, 3, 4, 4, 4, 4)


def test_archive_ingestion_sequence_preserves_late_older_generation_order(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 21, 0)
    publisher.publish(completed_game(publisher, model_generation=2))
    maintainer = chess_replay_maintainer(tmp_path, capacity=20, sampler_seed=5)
    maintainer.maintain(20)
    publisher.publish(completed_game(publisher, model_generation=1))

    live, _ = maintainer.maintain(20)
    rebuilt = rebuild_chess_replay(tmp_path, capacity=20, sampler_seed=5)

    assert tuple(sample.source_model_generation for sample in live.samples) == (2, 2, 2, 2, 1, 1, 1, 1)
    assert rebuilt.samples == live.samples


def test_fifo_phase_capacity_and_deterministic_nonoverlapping_rank_sampling(tmp_path: Path) -> None:
    publisher = CompletedGamePublisher(tmp_path, 6, 0)
    replay = Replay(CHESS_REPLAY_IMPLEMENTATION, capacity=6, sampler_seed=31)
    replay.ingest_game(completed_game(publisher))
    replay.ingest_game(completed_game(publisher))
    snapshot = replay.freeze()

    assert replay.phase is ReplayPhase.FROZEN
    assert len(snapshot.samples) == 6
    metrics = replay.metrics(snapshot.frozen_at_seconds)
    assert metrics.evicted_samples == 2
    assert snapshot.encoded_state_value_overhead_bytes > 0
    assert snapshot.projected_review_capacity_bytes >= snapshot.projected_capacity_bytes
    assert metrics.encoded_state_value_overhead_bytes == snapshot.encoded_state_value_overhead_bytes
    assert metrics.projected_review_capacity_bytes == snapshot.projected_review_capacity_bytes
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


def test_preallocated_batch_encoding_applies_deterministic_chess_symmetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = CompletedGamePublisher(tmp_path, 7, 0)
    replay = Replay(CHESS_REPLAY_IMPLEMENTATION, capacity=10, sampler_seed=41)
    replay.ingest_game(completed_game(publisher))
    snapshot = replay.freeze()

    def never_mirror(sampler_seed: int, global_step: int, rank: int, sample_position: int) -> bool:
        del sampler_seed, global_step, rank, sample_position
        return False

    def always_mirror(sampler_seed: int, global_step: int, rank: int, sample_position: int) -> bool:
        del sampler_seed, global_step, rank, sample_position
        return True

    monkeypatch.setattr(chess_replay_module, 'sample_is_mirrored', never_mirror)
    unaugmented = build_chess_training_batch(snapshot, (0, 1), global_step=9, rank=0)
    monkeypatch.setattr(chess_replay_module, 'sample_is_mirrored', always_mirror)
    batch = build_chess_training_batch(snapshot, (0, 1), global_step=9, rank=0)

    assert batch.states.shape == (2, *CHESS_STATE_CONTRACT.game.representation_shape)
    assert batch.policy_targets.shape == (2, CHESS_STATE_CONTRACT.action_size)
    torch.testing.assert_close(batch.policy_targets.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(batch.sample_weights, torch.ones(2))
    torch.testing.assert_close(batch.states, torch.flip(unaugmented.states, dims=(3,)))
    torch.testing.assert_close(
        batch.policy_targets[:, chess_replay_module.CHESS_MIRROR_ACTION_MAP],
        unaugmented.policy_targets,
    )
