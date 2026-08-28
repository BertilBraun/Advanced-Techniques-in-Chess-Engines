from __future__ import annotations

import heapq
import os
from dataclasses import dataclass
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Generic, TypeVar

from src.experiment.configuration import load_experiment_configuration_json
from src.games.composition import create_game_implementation
from src.games.contracts import GameStateContract, TerminalOracle
from src.replay.contracts import ReplaySample
from src.replay.dispatch import (
    parse_worker_source_file_name,
    worker_directory_path,
    worker_source_file_names,
)
from src.replay.encoding import encode_replay_columns
from src.replay.layout import ReplayLayout
from src.replay.materialization import materialize_completed_game
from src.replay.shard import (
    ReplayShardGameMetadata,
    ReplayShardSourceGame,
    SealedReplayShardManifest,
    read_sealed_replay_shard_manifest,
    replay_shard_identity,
    sealed_replay_shard_manifest_paths,
    write_replay_shard,
)
from src.self_play.completed_game import CompletedSelfPlayGame, GameIdentity
from src.util.generation_schedule import FloatGenerationSchedule
from src.util.log import log

PositionT = TypeVar('PositionT')


@dataclass(frozen=True)
class MaterializationReport:
    worker_index: int
    materialized_games: int
    rejected_games: int
    sealed_shards: int
    sealed_rows: int


@dataclass(frozen=True)
class MaterializationSettings:
    shard_maximum_games: int
    shard_target_source_bytes: int
    staging_shard_limit: int
    maximum_policy_entries: int


@dataclass(frozen=True)
class _SourceCandidate:
    counter: int
    identity: GameIdentity
    path: Path


@dataclass(frozen=True)
class _MaterializedGame:
    candidate: _SourceCandidate
    metadata: ReplayShardGameMetadata
    samples: tuple[ReplaySample, ...]


class MaterializationWorker(Generic[PositionT]):
    """Consumes one worker directory in counter order, sealing what it can and rejecting what it cannot."""

    def __init__(
        self,
        worker_index: int,
        worker_path: Path,
        staging_path: Path,
        rejected_path: Path,
        state: GameStateContract[PositionT],
        terminal_oracle: TerminalOracle[PositionT] | None,
        layout: ReplayLayout,
        value_discount_per_ply: FloatGenerationSchedule,
        censor_remaining_game_length_on_cut_games: bool,
        settings: MaterializationSettings,
    ) -> None:
        self.worker_index = worker_index
        self.worker_path = worker_path
        self.staging_path = staging_path
        self.rejected_path = rejected_path
        self.state = state
        self.terminal_oracle = terminal_oracle
        self.layout = layout
        self.value_discount_per_ply = value_discount_per_ply
        self.censor_remaining_game_length_on_cut_games = censor_remaining_game_length_on_cut_games
        self.settings = settings

    def materialize_once(self) -> MaterializationReport | None:
        if self._staging_is_full():
            return None
        batch = self._next_batch()
        if not batch:
            return None
        shard_identity = replay_shard_identity(
            self.layout.digest, self.worker_index, batch[0].counter, batch[-1].counter
        )
        sealed = read_sealed_replay_shard_manifest(self.staging_path, shard_identity)
        if sealed is not None:
            return self._adopt_sealed_shard(batch, sealed)
        return self._materialize_batch(batch)

    def _staging_is_full(self) -> bool:
        return len(sealed_replay_shard_manifest_paths(self.staging_path)) >= self.settings.staging_shard_limit

    def _next_batch(self) -> tuple[_SourceCandidate, ...]:
        names = heapq.nsmallest(self.settings.shard_maximum_games, worker_source_file_names(self.worker_path))
        batch: list[_SourceCandidate] = []
        batch_bytes = 0
        for name in names:
            candidate = self._candidate(name)
            if candidate is None:
                continue
            try:
                source_bytes = candidate.path.stat().st_size
            except OSError:
                continue
            if batch and batch_bytes + source_bytes > self.settings.shard_target_source_bytes:
                break
            batch.append(candidate)
            batch_bytes += source_bytes
        return tuple(batch)

    def _candidate(self, file_name: str) -> _SourceCandidate | None:
        path = self.worker_path / file_name
        try:
            counter, completed_game_file_name = parse_worker_source_file_name(file_name)
            identity = GameIdentity.from_file_name(completed_game_file_name)
        except ValueError as error:
            self._reject(path, error)
            return None
        return _SourceCandidate(counter=counter, identity=identity, path=path)

    def _adopt_sealed_shard(
        self,
        batch: tuple[_SourceCandidate, ...],
        sealed: SealedReplayShardManifest,
    ) -> MaterializationReport:
        ingested_counters = {game.source.counter for game in sealed.games}
        rejected = tuple(candidate for candidate in batch if candidate.counter not in ingested_counters)
        for candidate in batch:
            if candidate.counter in ingested_counters:
                candidate.path.unlink(missing_ok=True)
        self._reject_all(rejected, ValueError('rejected before its shard was sealed'))
        return MaterializationReport(
            worker_index=self.worker_index,
            materialized_games=len(sealed.games),
            rejected_games=len(rejected),
            sealed_shards=0,
            sealed_rows=0,
        )

    def _materialize_batch(self, batch: tuple[_SourceCandidate, ...]) -> MaterializationReport:
        materialized: list[_MaterializedGame] = []
        rejected: list[_SourceCandidate] = []
        next_row = 0
        for candidate in batch:
            outcome = self._materialize_one(candidate, next_row)
            if outcome is None:
                rejected.append(candidate)
                continue
            materialized.append(outcome)
            next_row += outcome.metadata.row_count
        if not materialized:
            self._reject_all(tuple(rejected), ValueError('no game in the batch could be materialized'))
            return MaterializationReport(self.worker_index, 0, len(rejected), 0, 0)
        try:
            samples = tuple(sample for outcome in materialized for sample in outcome.samples)
            sealed = write_replay_shard(
                self.staging_path,
                self.layout,
                self.worker_index,
                batch[0].counter,
                batch[-1].counter,
                encode_replay_columns(self.layout, samples),
                tuple(outcome.metadata for outcome in materialized),
            )
        except Exception as error:  # noqa: BLE001
            self._reject_all(batch, error)
            return MaterializationReport(self.worker_index, 0, len(batch), 0, 0)
        # Rejected sources stay in place until the shard is sealed so a restart re-derives the same
        # counter span, and with it the same shard identity.
        self._reject_all(tuple(rejected), ValueError('could not be materialized'))
        for outcome in materialized:
            outcome.candidate.path.unlink(missing_ok=True)
        return MaterializationReport(
            worker_index=self.worker_index,
            materialized_games=len(materialized),
            rejected_games=len(rejected),
            sealed_shards=1,
            sealed_rows=sealed.row_count,
        )

    def _materialize_one(self, candidate: _SourceCandidate, row_start: int) -> _MaterializedGame | None:
        try:
            game = CompletedSelfPlayGame.model_validate_json(candidate.path.read_bytes())
            if game.identity != candidate.identity:
                raise ValueError('Completed-game identity does not match its file name.')
            materialized = materialize_completed_game(
                game,
                self.state,
                self.terminal_oracle,
                self.layout.targets,
                self.layout.maximum_policy_entries,
                self.value_discount_per_ply,
                censor_remaining_game_length_on_cut_games=self.censor_remaining_game_length_on_cut_games,
            )
            metadata = ReplayShardGameMetadata(
                source=ReplayShardSourceGame(identity=candidate.identity, counter=candidate.counter),
                created_at_seconds=game.created_at_seconds,
                generation_seconds=game.generation_seconds,
                action_ids=game.action_ids,
                row_start=row_start,
                row_count=len(materialized.samples),
                length_plies=len(game.action_ids),
                termination_reason=game.termination_reason,
                is_resignation_continuation=game.is_resignation_continuation,
                resignation_threshold=game.resignation_threshold,
                final_wdl=game.final_wdl,
                observations=game.observations,
                policies_truncated=materialized.policies_truncated,
                retained_visit_mass=materialized.retained_visit_mass,
                discarded_visit_mass=materialized.discarded_visit_mass,
            )
        except Exception as error:  # noqa: BLE001
            log(f'Replay worker {self.worker_index} cannot materialize {candidate.path.name}: {error}')
            return None
        return _MaterializedGame(candidate=candidate, metadata=metadata, samples=tuple(materialized.samples))

    def _reject_all(self, rejected: tuple[_SourceCandidate, ...], error: BaseException) -> None:
        for candidate in rejected:
            self._reject(candidate.path, error)

    def _reject(self, path: Path, error: BaseException) -> None:
        self.rejected_path.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, self.rejected_path / path.name)
        except OSError:
            path.unlink(missing_ok=True)
        log(f'Replay worker {self.worker_index} rejected {path.name}: {error}')


def run_materialization_worker(
    experiment_configuration_json: str,
    completed_games_path: Path,
    worker_index: int,
    settings: MaterializationSettings,
    poll_interval_seconds: float,
    report_queue: Queue[MaterializationReport],
    stop_event: Event,
) -> None:
    game = create_game_implementation(load_experiment_configuration_json(experiment_configuration_json))
    layout = ReplayLayout(
        packed_planes=game.state.packed_plane_layout,
        targets=game.target_layout,
        maximum_policy_entries=settings.maximum_policy_entries,
        maximum_legal_actions=game.state.maximum_legal_action_count,
    )
    worker = MaterializationWorker(
        worker_index,
        worker_directory_path(completed_games_path, worker_index),
        completed_games_path / 'staging',
        completed_games_path / 'rejected',
        game.state,
        game.terminal_oracle,
        layout,
        game.value_discount_per_ply,
        game.censor_remaining_game_length_on_cut_games,
        settings,
    )
    while not stop_event.is_set():
        report = worker.materialize_once()
        if report is None:
            if stop_event.wait(poll_interval_seconds):
                return
            continue
        report_queue.put(report)
