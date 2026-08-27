from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pytest
import src.replay.materialization_worker as materialization_worker
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import PackedPlaneLayout, PackedPlanePayload, RepresentationDimensions
from src.replay.columnar import flatten_column_views
from src.replay.contracts import ReplaySample
from src.replay.dispatch import worker_source_file_name, worker_source_file_names
from src.replay.encoding import encode_replay_columns
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.materialization_worker import (
    MaterializationReport,
    MaterializationSettings,
    MaterializationWorker,
)
from src.replay.shard import (
    ReplayShardReader,
    replay_shard_identity,
    replay_shard_manifest_path,
)
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
)
from src.training.targets import NextPolicyHeadLayout, TrainingTargetLayout
from src.util.generation_schedule import ConstantSchedule


@dataclass(frozen=True)
class _Position:
    actions: tuple[int, ...] = ()


class _State(GameStateContract[_Position]):
    def __init__(self) -> None:
        packed = PackedPlaneLayout(board_size=1, binary_plane_count=1, scalar_count=0)
        self._representation = RepresentationDimensions(
            channels=1,
            rows=1,
            columns=1,
            binary_channels=(0,),
            scalar_channels=(),
            packed_planes=packed,
        )

    @property
    def name(self) -> str:
        return 'parallel-materialization-test'

    @property
    def action_size(self) -> int:
        return 3

    @property
    def representation(self) -> RepresentationDimensions:
        return self._representation

    def initial_position(self) -> _Position:
        return _Position()

    def legal_action_ids(self, position: _Position) -> tuple[int, ...]:
        return () if len(position.actions) == 4 else (0, 1, 2)

    def child_position(self, position: _Position, action_id: int) -> _Position:
        if action_id not in self.legal_action_ids(position):
            raise ValueError('Action is not legal.')
        return _Position((*position.actions, action_id))

    def is_irreversible_transition(self, position: _Position, action_id: int, child: _Position) -> bool:
        del position, child
        return action_id == 2

    def current_player(self, position: _Position) -> Player:
        return Player.FIRST if len(position.actions) % 2 == 0 else Player.SECOND

    def natural_terminal_wdl(self, position: _Position) -> WdlTarget | None:
        return WdlTarget(win=0.0, draw=0.0, loss=1.0) if len(position.actions) == 4 else None

    def adjudicated_wdl(self, position: _Position, reason: TerminationReason) -> WdlTarget:
        del position, reason
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)

    def encode_network_input(self, position: _Position) -> PackedPlanePayload:
        return self.packed_plane_layout.value(len(position.actions).to_bytes(8, byteorder='little'))

    @property
    def augmentation_count(self) -> int:
        return 1

    def transform_decoded_states(
        self,
        states: npt.NDArray[np.float32],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> None:
        if len(states) != len(augmentation_indices) or np.any(augmentation_indices != 0):
            raise ValueError('Only identity augmentation is supported.')

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        if augmentation_index != 0:
            raise ValueError('Only identity augmentation is supported.')
        return action_id


STATE = _State()
VALUE_DISCOUNT = ConstantSchedule[float](value=1.0)
LAYOUT = ReplayLayout(
    packed_planes=STATE.packed_plane_layout,
    targets=TrainingTargetLayout(
        action_size=STATE.action_size,
        wdl_size=3,
        auxiliary_heads=(NextPolicyHeadLayout(kind='next_policy', action_size=STATE.action_size, ply_offset=1),),
    ),
    maximum_policy_entries=2,
    maximum_legal_actions=STATE.maximum_legal_action_count,
)


def _game(game_number: int, *, full_search: bool = True) -> CompletedSelfPlayGame:
    actions = (0, 1, 0, 2)
    observations = tuple(
        SearchObservation(
            ply=ply,
            model_generation=2 + game_number,
            policy_target_visits=SearchVisitCounts(
                action_ids=(selected_action, (selected_action + 1) % STATE.action_size),
                visit_counts=(10, 3),
            ),
            root_value=0.25,
            highest_visited_child_action_id=selected_action,
            highest_visited_child_visit_count=10,
            highest_visited_child_q=0.2,
            selected_action_id=selected_action,
            full_search=full_search,
            sample_weight=1.0,
            search_budget=13,
            network_root_value=0.1,
            policy_correction=0.2,
            value_correction=0.075,
            search_correction_target=0.2,
            predicted_search_correction=0.15,
            starting_visits=0,
            final_visits=13,
            stop_reason=SearchStopReason.FIXED_LIMIT,
            learned_gate_evaluated=False,
        )
        for ply, selected_action in enumerate(actions)
    )
    return CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=3,
            process_instance_id=UUID('38c8809f-a49d-4d98-8da5-034614893665'),
            game_number=game_number,
        ),
        created_at_seconds=100.0 + game_number,
        generation_seconds=1.5,
        action_ids=actions,
        observations=observations,
        final_wdl=WdlTarget(win=0.0, draw=0.0, loss=1.0),
        termination_reason=TerminationReason.NATURAL,
    )


SETTINGS = MaterializationSettings(
    shard_maximum_games=32,
    shard_target_source_bytes=16 * 1024 * 1024,
    staging_shard_limit=96,
    maximum_policy_entries=LAYOUT.maximum_policy_entries,
)
WORKER_INDEX = 2


def _worker(tmp_path: Path, **overrides: int) -> MaterializationWorker[_Position]:
    worker_path = tmp_path / 'worker-2'
    staging_path = tmp_path / 'staging'
    rejected_path = tmp_path / 'rejected'
    for directory in (worker_path, staging_path, rejected_path):
        directory.mkdir(parents=True, exist_ok=True)
    return MaterializationWorker(
        WORKER_INDEX,
        worker_path,
        staging_path,
        rejected_path,
        STATE,
        None,
        LAYOUT,
        VALUE_DISCOUNT,
        False,
        replace(SETTINGS, **overrides),
    )


def _place(worker: MaterializationWorker[_Position], counter: int, game: CompletedSelfPlayGame) -> Path:
    path = worker.worker_path / worker_source_file_name(counter, game.identity.file_name)
    path.write_bytes(game.model_dump_json().encode())
    return path


def _place_malformed(worker: MaterializationWorker[_Position], counter: int, game_number: int) -> Path:
    identity = _game(game_number).identity
    path = worker.worker_path / worker_source_file_name(counter, identity.file_name)
    path.write_bytes(b'{not-json')
    return path


def _expected_samples(games: tuple[CompletedSelfPlayGame, ...]) -> tuple[ReplaySample, ...]:
    return tuple(
        sample
        for game in games
        for sample in materialize_completed_game(
            game, STATE, None, LAYOUT.targets, LAYOUT.maximum_policy_entries, VALUE_DISCOUNT
        ).samples
    )


def test_sealed_shard_materializes_every_game_and_preserves_counter_order(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    games = (_game(0), _game(1, full_search=False), _game(2))
    for counter, game in enumerate(games):
        _place(worker, counter, game)
    expected_samples = _expected_samples(games)

    report = worker.materialize_once()

    assert report == MaterializationReport(WORKER_INDEX, 3, 0, 1, len(expected_samples))
    assert not tuple(worker.worker_path.iterdir())
    identity = replay_shard_identity(LAYOUT.digest, WORKER_INDEX, 0, 2)
    with ReplayShardReader.open(replay_shard_manifest_path(worker.staging_path, identity), LAYOUT) as reader:
        expected_columns = encode_replay_columns(LAYOUT, expected_samples)
        for actual, expected in zip(
            flatten_column_views(LAYOUT, reader.columns),
            flatten_column_views(LAYOUT, expected_columns),
            strict=True,
        ):
            np.testing.assert_array_equal(actual.values, expected.values)
        assert tuple(metadata.source.counter for metadata in reader.manifest.games) == (0, 1, 2)
        assert tuple(metadata.source.identity for metadata in reader.manifest.games) == tuple(
            game.identity for game in games
        )
        assert tuple((metadata.row_start, metadata.row_count) for metadata in reader.manifest.games) == (
            (0, 4),
            (4, 4),
            (8, 4),
        )


def test_malformed_game_is_rejected_and_its_shard_mates_still_seal(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    _place(worker, 0, _game(0))
    malformed = _place_malformed(worker, 1, 1)
    _place(worker, 2, _game(2))

    report = worker.materialize_once()

    assert report is not None
    assert (report.materialized_games, report.rejected_games, report.sealed_shards) == (2, 1, 1)
    assert not tuple(worker.worker_path.iterdir())
    assert [path.name for path in worker.rejected_path.iterdir()] == [malformed.name]
    identity = replay_shard_identity(LAYOUT.digest, WORKER_INDEX, 0, 2)
    with ReplayShardReader.open(replay_shard_manifest_path(worker.staging_path, identity), LAYOUT) as reader:
        assert tuple(metadata.source.counter for metadata in reader.manifest.games) == (0, 2)


def test_a_batch_of_only_malformed_games_seals_nothing_and_does_not_stall(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    _place_malformed(worker, 0, 0)
    _place_malformed(worker, 1, 1)

    report = worker.materialize_once()

    assert report == MaterializationReport(WORKER_INDEX, 0, 2, 0, 0)
    assert not tuple(worker.worker_path.iterdir())
    assert len(tuple(worker.rejected_path.iterdir())) == 2
    assert worker.materialize_once() is None

    _place(worker, 2, _game(2))
    resumed = worker.materialize_once()

    assert resumed is not None
    assert resumed.sealed_shards == 1


def test_an_unparseable_file_name_is_rejected_without_touching_the_other_games(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    (worker.worker_path / 'not-a-dispatched-name.json').write_bytes(b'{}')
    _place(worker, 0, _game(0))

    report = worker.materialize_once()

    assert report is not None
    assert (report.materialized_games, report.rejected_games) == (1, 0)
    assert [path.name for path in worker.rejected_path.iterdir()] == ['not-a-dispatched-name.json']


def test_seal_failure_rejects_the_whole_batch_without_stalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker(tmp_path)
    _place(worker, 0, _game(0))
    _place(worker, 1, _game(1))

    def fail_seal(*arguments: object, **keywords: object) -> None:
        del arguments, keywords
        raise OSError('forced seal failure')

    monkeypatch.setattr(materialization_worker, 'write_replay_shard', fail_seal)
    report = worker.materialize_once()

    assert report == MaterializationReport(WORKER_INDEX, 0, 2, 0, 0)
    assert not tuple(worker.worker_path.iterdir())
    assert len(tuple(worker.rejected_path.iterdir())) == 2
    assert not tuple(worker.staging_path.glob('*.replay-shard.json'))


def test_a_worker_killed_between_seal_and_unlink_does_not_seal_the_games_twice(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    games = (_game(0), _game(1))
    paths = tuple(_place(worker, counter, game) for counter, game in enumerate(games))
    first = worker.materialize_once()
    assert first is not None and first.sealed_shards == 1
    # The kill leaves the sealed shard in staging and every consumed source still in place.
    for path, game in zip(paths, games, strict=True):
        path.write_bytes(game.model_dump_json().encode())

    repeated = worker.materialize_once()

    assert repeated == MaterializationReport(WORKER_INDEX, 2, 0, 0, 0)
    assert not tuple(worker.worker_path.iterdir())
    assert len(tuple(worker.staging_path.glob('*.replay-shard.json'))) == 1


def test_batches_are_bounded_by_game_count(tmp_path: Path) -> None:
    worker = _worker(tmp_path, shard_maximum_games=2)
    for counter in range(5):
        _place(worker, counter, _game(counter))

    counts = []
    while (report := worker.materialize_once()) is not None:
        counts.append(report.materialized_games)

    assert counts == [2, 2, 1]


def test_batches_are_bounded_by_source_bytes(tmp_path: Path) -> None:
    worker = _worker(tmp_path, shard_target_source_bytes=1)
    for counter in range(3):
        _place(worker, counter, _game(counter))

    counts = []
    while (report := worker.materialize_once()) is not None:
        counts.append(report.materialized_games)

    assert counts == [1, 1, 1]


def test_a_full_staging_directory_pauses_the_worker_without_losing_games(tmp_path: Path) -> None:
    worker = _worker(tmp_path, shard_maximum_games=1, staging_shard_limit=2)
    for counter in range(4):
        _place(worker, counter, _game(counter))

    sealed = 0
    while worker.materialize_once() is not None:
        sealed += 1

    assert sealed == 2
    assert len(worker_source_file_names(worker.worker_path)) == 2
