from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from src.experiment.configuration import load_experiment_configuration_json
from src.games.composition import ConfiguredGame, create_game_implementation
from src.games.contracts import GameStateContract, TerminalOracle
from src.replay.encoding import encode_replay_columns
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.shard import (
    PendingReplayShardManifest,
    ReplayShardGameMetadata,
    ReplayShardReader,
    replay_shard_data_path,
    replay_shard_manifest_path,
    write_replay_shard,
)
from src.self_play.completed_game import CompletedSelfPlayGame
from src.util.generation_schedule import FloatGenerationSchedule

PositionT = TypeVar('PositionT')


@dataclass(frozen=True)
class SealedReplayShard:
    sequence: int
    shard_identity: str
    row_count: int
    game_count: int


def stage_replay_shard(
    pending: PendingReplayShardManifest,
    inbox_path: Path,
    staging_path: Path,
    state: GameStateContract[PositionT],
    terminal_oracle: TerminalOracle[PositionT] | None,
    layout: ReplayLayout,
    value_discount_per_ply: FloatGenerationSchedule,
    censor_remaining_game_length_on_cut_games: bool,
) -> SealedReplayShard:
    if pending.layout_digest != layout.digest:
        raise ValueError('Pending replay shard layout does not match the materialization worker.')
    existing = _sealed_replay_shard(pending, staging_path, layout)
    if existing is not None:
        _remove_consumed_inbox_games(pending, inbox_path, verify_sources=True)
        return existing

    data_path = replay_shard_data_path(staging_path, pending.shard_identity)
    data_path.unlink(missing_ok=True)
    samples = []
    metadata = []
    next_row = 0
    for source in pending.games:
        inbox_file = inbox_path / source.order.file_name
        payload = _read_claimed_source(inbox_file, source.source_size, source.source_sha256)
        game = CompletedSelfPlayGame.model_validate_json(payload)
        if game.identity != source.identity or inbox_file.name != game.identity.file_name:
            raise ValueError(f'Completed-game identity does not match its replay shard claim: {inbox_file}')
        materialized = materialize_completed_game(
            game,
            state,
            terminal_oracle,
            layout.targets,
            layout.maximum_policy_entries,
            value_discount_per_ply,
            censor_remaining_game_length_on_cut_games=censor_remaining_game_length_on_cut_games,
        )
        row_count = len(materialized.samples)
        samples.extend(materialized.samples)
        metadata.append(
            ReplayShardGameMetadata(
                source=source,
                row_start=next_row,
                row_count=row_count,
                length_plies=len(game.action_ids),
                termination_reason=game.termination_reason,
                is_resignation_continuation=game.is_resignation_continuation,
                final_wdl=game.final_wdl,
                observations=game.observations,
                policies_truncated=materialized.policies_truncated,
                retained_visit_mass=materialized.retained_visit_mass,
                discarded_visit_mass=materialized.discarded_visit_mass,
            )
        )
        next_row += row_count
    columns = encode_replay_columns(layout, tuple(samples))
    sealed = write_replay_shard(staging_path, layout, pending, columns, tuple(metadata))
    _remove_consumed_inbox_games(pending, inbox_path, verify_sources=False)
    return SealedReplayShard(
        sequence=sealed.sequence,
        shard_identity=sealed.shard_identity,
        row_count=sealed.row_count,
        game_count=len(sealed.games),
    )


def _sealed_replay_shard(
    pending: PendingReplayShardManifest,
    staging_path: Path,
    layout: ReplayLayout,
) -> SealedReplayShard | None:
    manifest_path = replay_shard_manifest_path(staging_path, pending.shard_identity)
    if not manifest_path.exists():
        return None
    with ReplayShardReader.open(manifest_path, layout) as reader:
        manifest = reader.manifest
        if (
            manifest.sequence != pending.sequence
            or manifest.shard_identity != pending.shard_identity
            or tuple(game.source for game in manifest.games) != pending.games
        ):
            raise ValueError('Sealed replay shard does not match its pending claim.')
        return SealedReplayShard(
            sequence=manifest.sequence,
            shard_identity=manifest.shard_identity,
            row_count=manifest.row_count,
            game_count=len(manifest.games),
        )


def _remove_consumed_inbox_games(
    pending: PendingReplayShardManifest,
    inbox_path: Path,
    verify_sources: bool,
) -> None:
    for source in pending.games:
        inbox_file = inbox_path / source.order.file_name
        if not inbox_file.exists():
            continue
        if verify_sources:
            _read_claimed_source(inbox_file, source.source_size, source.source_sha256)
        inbox_file.unlink()


def _read_claimed_source(inbox_file: Path, source_size: int, source_sha256: str) -> bytes:
    payload = inbox_file.read_bytes()
    if len(payload) != source_size:
        raise ValueError(f'Completed-game source size changed after replay shard claim: {inbox_file}')
    if hashlib.sha256(payload).hexdigest() != source_sha256:
        raise ValueError(f'Completed-game source hash changed after replay shard claim: {inbox_file}')
    return payload


_worker_game: ConfiguredGame | None = None
_worker_layout: ReplayLayout | None = None


def initialize_materialization_worker(configuration_json: str, maximum_policy_entries: int) -> None:
    global _worker_game, _worker_layout
    _worker_game = create_game_implementation(load_experiment_configuration_json(configuration_json))
    _worker_layout = ReplayLayout(
        packed_planes=_worker_game.state.packed_plane_layout,
        targets=_worker_game.target_layout,
        maximum_policy_entries=maximum_policy_entries,
        maximum_legal_actions=_worker_game.state.maximum_legal_action_count,
    )


def stage_replay_shard_path(
    pending: PendingReplayShardManifest,
    inbox_path: Path,
    staging_path: Path,
) -> SealedReplayShard:
    if _worker_game is None or _worker_layout is None:
        raise RuntimeError('Replay materialization worker has not been initialized.')
    return stage_replay_shard(
        pending,
        inbox_path,
        staging_path,
        _worker_game.state,
        _worker_game.terminal_oracle,
        _worker_layout,
        _worker_game.value_discount_per_ply,
        _worker_game.censor_remaining_game_length_on_cut_games,
    )
