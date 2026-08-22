from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
import numpy.typing as npt
import pytest
import src.replay.parallel_materialization as parallel_materialization
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import PackedPlaneLayout, PackedPlanePayload, RepresentationDimensions
from src.replay.columnar import flatten_column_views
from src.replay.encoding import encode_replay_columns
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.parallel_materialization import stage_replay_shard
from src.replay.shard import (
    InboxGameOrder,
    PendingReplayShardManifest,
    ReplayShardGameMetadata,
    ReplayShardReader,
    ReplayShardSourceGame,
    replay_shard_manifest_path,
    write_replay_shard,
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


def _write_game(inbox_path: Path, game: CompletedSelfPlayGame) -> ReplayShardSourceGame:
    payload = game.model_dump_json().encode()
    path = inbox_path / game.identity.file_name
    path.write_bytes(payload)
    return ReplayShardSourceGame(
        identity=game.identity,
        order=InboxGameOrder(modified_at_ns=100 + game.identity.game_number, file_name=path.name),
        source_size=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _pending(inbox_path: Path, games: tuple[CompletedSelfPlayGame, ...]) -> PendingReplayShardManifest:
    return PendingReplayShardManifest.create(LAYOUT, 7, tuple(_write_game(inbox_path, game) for game in games))


def _stage(pending: PendingReplayShardManifest, inbox_path: Path, staging_path: Path) -> None:
    stage_replay_shard(pending, inbox_path, staging_path, STATE, None, LAYOUT, VALUE_DISCOUNT, False)


def test_batched_shard_matches_independent_materialization_and_preserves_game_order(tmp_path: Path) -> None:
    inbox_path = tmp_path / 'inbox'
    staging_path = tmp_path / 'staging'
    inbox_path.mkdir()
    games = (_game(0), _game(1, full_search=False), _game(2))
    pending = _pending(inbox_path, games)
    expected_materialized = tuple(
        materialize_completed_game(
            game,
            STATE,
            None,
            LAYOUT.targets,
            LAYOUT.maximum_policy_entries,
            VALUE_DISCOUNT,
        )
        for game in games
    )
    expected_samples = tuple(sample for game in expected_materialized for sample in game.samples)

    result = stage_replay_shard(
        pending,
        inbox_path,
        staging_path,
        STATE,
        None,
        LAYOUT,
        VALUE_DISCOUNT,
        False,
    )

    assert result.sequence == 7
    assert result.shard_identity == pending.shard_identity
    assert result.row_count == len(expected_samples)
    assert result.game_count == 3
    assert not tuple(inbox_path.iterdir())
    with ReplayShardReader.open(replay_shard_manifest_path(staging_path, pending.shard_identity), LAYOUT) as reader:
        expected_columns = encode_replay_columns(LAYOUT, expected_samples)
        for actual, expected in zip(
            flatten_column_views(LAYOUT, reader.columns),
            flatten_column_views(LAYOUT, expected_columns),
            strict=True,
        ):
            np.testing.assert_array_equal(actual.values, expected.values)
        assert tuple(metadata.source for metadata in reader.manifest.games) == pending.games
        assert tuple((metadata.row_start, metadata.row_count) for metadata in reader.manifest.games) == (
            (0, 4),
            (4, 0),
            (4, 4),
        )
        assert tuple(metadata.observations for metadata in reader.manifest.games) == tuple(
            game.observations for game in games
        )


@pytest.mark.parametrize('changed_field', ('size', 'hash', 'identity'))
def test_source_claim_mismatch_deletes_nothing_and_seals_nothing(tmp_path: Path, changed_field: str) -> None:
    inbox_path = tmp_path / 'inbox'
    staging_path = tmp_path / 'staging'
    inbox_path.mkdir()
    game = _game(0)
    pending = _pending(inbox_path, (game,))
    path = inbox_path / game.identity.file_name
    if changed_field == 'identity':
        payload = game.model_copy(update={'identity': _game(1).identity}).model_dump_json().encode()
        path.write_bytes(payload)
        source = pending.games[0].model_copy(
            update={'source_size': len(payload), 'source_sha256': hashlib.sha256(payload).hexdigest()}
        )
        pending = PendingReplayShardManifest.create(LAYOUT, 7, (source,))
    elif changed_field == 'size':
        path.write_bytes(path.read_bytes() + b' ')
    else:
        payload = bytearray(path.read_bytes())
        payload[-1] = ord(' ')
        path.write_bytes(payload)

    with pytest.raises(ValueError):
        _stage(pending, inbox_path, staging_path)

    assert path.exists()
    assert not replay_shard_manifest_path(staging_path, pending.shard_identity).exists()


def test_malformed_later_game_deletes_no_source_and_seals_nothing(tmp_path: Path) -> None:
    inbox_path = tmp_path / 'inbox'
    staging_path = tmp_path / 'staging'
    inbox_path.mkdir()
    first = _game(0)
    malformed_identity = _game(1).identity
    malformed_payload = b'{not-json'
    first_source = _write_game(inbox_path, first)
    malformed_path = inbox_path / malformed_identity.file_name
    malformed_path.write_bytes(malformed_payload)
    malformed_source = ReplayShardSourceGame(
        identity=malformed_identity,
        order=InboxGameOrder(modified_at_ns=101, file_name=malformed_path.name),
        source_size=len(malformed_payload),
        source_sha256=hashlib.sha256(malformed_payload).hexdigest(),
    )
    pending = PendingReplayShardManifest.create(LAYOUT, 7, (first_source, malformed_source))

    with pytest.raises(ValueError):
        _stage(pending, inbox_path, staging_path)

    assert tuple(sorted(path.name for path in inbox_path.iterdir())) == tuple(
        sorted((first.identity.file_name, malformed_identity.file_name))
    )
    assert not replay_shard_manifest_path(staging_path, pending.shard_identity).exists()


def test_seal_failure_keeps_every_inbox_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inbox_path = tmp_path / 'inbox'
    staging_path = tmp_path / 'staging'
    inbox_path.mkdir()
    games = (_game(0), _game(1))
    pending = _pending(inbox_path, games)

    def fail_seal(*arguments: object, **keywords: object) -> None:
        del arguments, keywords
        raise OSError('forced seal failure')

    monkeypatch.setattr(parallel_materialization, 'write_replay_shard', fail_seal)
    with pytest.raises(OSError, match='forced seal failure'):
        _stage(pending, inbox_path, staging_path)

    assert tuple(sorted(path.name for path in inbox_path.iterdir())) == tuple(
        sorted(game.identity.file_name for game in games)
    )
    assert not replay_shard_manifest_path(staging_path, pending.shard_identity).exists()


def test_sealed_shard_with_undeleted_inbox_is_recovered_without_rematerializing(tmp_path: Path) -> None:
    inbox_path = tmp_path / 'inbox'
    staging_path = tmp_path / 'staging'
    inbox_path.mkdir()
    games = (_game(0), _game(1))
    pending = _pending(inbox_path, games)
    materialized = tuple(
        materialize_completed_game(
            game,
            STATE,
            None,
            LAYOUT.targets,
            LAYOUT.maximum_policy_entries,
            VALUE_DISCOUNT,
        )
        for game in games
    )
    metadata = []
    row_start = 0
    for source, game, materialized_game in zip(pending.games, games, materialized, strict=True):
        metadata.append(
            ReplayShardGameMetadata(
                source=source,
                row_start=row_start,
                row_count=len(materialized_game.samples),
                length_plies=len(game.action_ids),
                termination_reason=game.termination_reason,
                is_resignation_continuation=game.is_resignation_continuation,
                final_wdl=game.final_wdl,
                observations=game.observations,
                policies_truncated=materialized_game.policies_truncated,
                retained_visit_mass=materialized_game.retained_visit_mass,
                discarded_visit_mass=materialized_game.discarded_visit_mass,
            )
        )
        row_start += len(materialized_game.samples)
    samples = tuple(sample for game in materialized for sample in game.samples)
    sealed = write_replay_shard(staging_path, LAYOUT, pending, encode_replay_columns(LAYOUT, samples), tuple(metadata))

    result = stage_replay_shard(
        pending,
        inbox_path,
        staging_path,
        STATE,
        None,
        LAYOUT,
        VALUE_DISCOUNT,
        False,
    )

    assert result.row_count == sealed.row_count
    assert not tuple(inbox_path.iterdir())
