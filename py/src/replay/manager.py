from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

from pydantic import Field, model_validator
from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract, TerminalOracle
from src.replay.configuration import ReplayConfiguration
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.parallel_materialization import (
    SealedReplayShard,
    initialize_materialization_worker,
    stage_replay_shard,
    stage_replay_shard_path,
)
from src.replay.shard import (
    InboxGameOrder,
    PendingReplayShardManifest,
    ReplayShardGameMetadata,
    ReplayShardReader,
    ReplayShardSourceGame,
    SealedReplayShardManifest,
    replay_shard_data_path,
    replay_shard_manifest_path,
)
from src.replay.store import ReplayStore
from src.self_play.completed_game import GameIdentity, SearchObservation, TerminationReason
from src.self_play.resignation import ResignationCalibrator
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import FloatGenerationSchedule
from src.util.log import log

PositionT = TypeVar('PositionT')
DISPATCH_INTERVAL_SECONDS = 1.0
_QUEUE_FILE = 'shard-queue.json'
_LEGACY_STAGED_ROWS_SUFFIX = '.rows.npy'
_LEGACY_STAGED_METADATA_SUFFIX = '.meta.json'
_MINIMUM_PENDING_SHARD_LIMIT = 32
_STAGED_SHARD_LIMIT_FACTOR = 3


@dataclass(frozen=True)
class IngestedCompletedGame:
    length_plies: int
    termination_reason: TerminationReason
    observations: tuple[SearchObservation, ...] = ()


@dataclass(frozen=True)
class ReplayIngestion:
    games_ingested: int
    samples_added: int
    live_samples: int
    evicted_samples: int
    policies_truncated: int
    retained_visit_mass: int
    discarded_visit_mass: int
    elapsed_seconds: float
    completed_games: tuple[IngestedCompletedGame, ...]

    @property
    def samples_per_second(self) -> float:
        return self.samples_added / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


class ReplayShardQueue(FrozenModel):
    schema_version: Literal[1] = 1
    layout_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    next_sequence: int = Field(ge=0)
    pending: tuple[PendingReplayShardManifest, ...] = ()

    @model_validator(mode='after')
    def validate_pending(self) -> ReplayShardQueue:
        sequences = tuple(claim.sequence for claim in self.pending)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError('Replay shard queue claims must have unique increasing sequences.')
        if sequences and sequences != tuple(range(sequences[0], self.next_sequence)):
            raise ValueError('Replay shard queue claims must form one contiguous sequence.')
        if sequences and self.next_sequence != sequences[-1] + 1:
            raise ValueError('Replay shard queue next sequence must follow its final claim.')
        sources = tuple(source for claim in self.pending for source in claim.games)
        identities = tuple(source.identity.archive_key for source in sources)
        file_names = tuple(source.order.file_name for source in sources)
        orders = tuple(source.order.key for source in sources)
        if len(set(identities)) != len(identities) or len(set(file_names)) != len(file_names):
            raise ValueError('Replay shard queue source games must be globally unique.')
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise ValueError('Replay shard queue source games must preserve global FIFO order.')
        return self


class _MaterializationDispatcher:
    def __init__(self, manager: ReplayManager[PositionT]) -> None:
        self._manager = manager
        self.on_sealed: Callable[[SealedReplayShard], None] = lambda sealed: None
        self.poll_interval_seconds = DISPATCH_INTERVAL_SECONDS
        self._pending: dict[Future[SealedReplayShard], int] = {}
        self._retry_after: dict[int, float] = {}
        self._notified: set[int] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.failed_game_count = 0

    def start(self) -> None:
        assert self._thread is None
        self._thread = threading.Thread(target=self._run, name='replay-materialization-dispatcher', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._pending:
            wait(tuple(self._pending))
            self._collect_finished()

    def dispatch_once(self) -> None:
        with self._lock:
            try:
                self._collect_finished()
                self._manager._allocate_claims()
                active = set(self._pending.values())
                with self._manager._lock:
                    claims = self._manager._queue.pending
                for claim in claims:
                    sealed: SealedReplayShard | None = None
                    submit_claim = False
                    stage_inline = False
                    with self._manager._lock:
                        if claim not in self._manager._queue.pending:
                            continue
                        if claim.sequence < self._manager.store.state.append_sequence:
                            continue
                        if self._manager._is_sealed(claim):
                            if claim.sequence not in self._notified:
                                sealed = self._manager._sealed_result(claim)
                        elif (
                            claim.sequence not in active
                            and self._retry_after.get(claim.sequence, 0.0) <= time.monotonic()
                        ):
                            executor = self._manager.materialization_executor
                            submit_claim = executor is not None
                            stage_inline = executor is None
                    if sealed is not None:
                        self._notify_sealed(sealed)
                    elif submit_claim:
                        executor = self._manager.materialization_executor
                        assert executor is not None
                        try:
                            future = executor.submit(
                                stage_replay_shard_path,
                                claim,
                                self._manager.inbox_path,
                                self._manager.staging_path,
                            )
                        except BaseException as error:
                            self._manager._set_fatal_materialization_error(
                                RuntimeError(
                                    f'Replay shard executor submission failed for sequence {claim.sequence}: {error}'
                                )
                            )
                        else:
                            self._pending[future] = claim.sequence
                    elif stage_inline:
                        self._stage_inline(claim)
                self._collect_finished()
            except BaseException as error:
                self._manager._set_fatal_materialization_error(
                    RuntimeError(f'Replay materialization dispatcher failed: {error}')
                )

    def drain(self) -> None:
        while True:
            self.dispatch_once()
            if self._manager._fatal_materialization_error is not None:
                return
            if (
                not self._pending
                and not self._manager._has_unsealed_claims()
                and (not self._manager._unclaimed_inbox_files() or not self._manager._has_claim_capacity())
            ):
                return
            time.sleep(0.01)

    def _stage_inline(self, claim: PendingReplayShardManifest) -> None:
        try:
            sealed = self._manager._stage_shard_inline(claim)
        except Exception as error:  # noqa: BLE001
            self._record_failure(claim, error)
            return
        self._retry_after.pop(claim.sequence, None)
        self._notify_sealed(sealed)

    def _collect_finished(self) -> None:
        for future in [candidate for candidate in self._pending if candidate.done()]:
            sequence = self._pending.pop(future)
            error = future.exception()
            if error is not None:
                self._record_failure(self._manager._claim(sequence), error)
                continue
            self._retry_after.pop(sequence, None)
            self._notify_sealed(future.result())

    def _notify_sealed(self, sealed: SealedReplayShard) -> None:
        try:
            self.on_sealed(sealed)
        except BaseException as error:
            self._manager._set_fatal_materialization_error(
                RuntimeError(f'Replay shard callback failed for sequence {sealed.sequence}: {error}')
            )
            return
        self._notified.add(sealed.sequence)

    def _record_failure(self, claim: PendingReplayShardManifest, error: BaseException) -> None:
        self.failed_game_count += len(claim.games)
        if isinstance(error, ValueError | FileNotFoundError):
            self._manager._set_fatal_materialization_error(
                RuntimeError(f'Replay shard {claim.sequence} is not materializable: {error}')
            )
            log(f'Fatal replay materialization failure for shard {claim.sequence}: {error}')
            return
        retry_seconds = min(max(self.poll_interval_seconds, 0.1), 5.0)
        self._retry_after[claim.sequence] = time.monotonic() + retry_seconds
        log(f'Transient replay materialization failure for shard {claim.sequence}: {error}')

    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            self.dispatch_once()


class ReplayManager(Generic[PositionT]):
    def __init__(
        self,
        completed_games_path: Path,
        store: ReplayStore,
        state: GameStateContract[PositionT],
        configuration: ReplayConfiguration,
        value_discount_per_ply: FloatGenerationSchedule,
        resignation_calibrator: ResignationCalibrator | None,
        terminal_oracle: TerminalOracle[PositionT] | None,
        experiment_configuration: ExperimentConfiguration | None,
        censor_remaining_game_length_on_cut_games: bool = False,
    ) -> None:
        if store.layout.packed_planes != state.packed_plane_layout:
            raise ValueError('Replay layout does not match the game packed-plane representation.')
        if store.layout.targets.action_size != state.action_size:
            raise ValueError('Replay layout does not match the game action count.')
        if store.layout.maximum_policy_entries != configuration.maximum_policy_entries:
            raise ValueError('Replay layout does not match replay policy retention configuration.')
        if store.state.maximum_capacity != configuration.maximum_capacity:
            raise ValueError('Replay file does not match replay maximum capacity configuration.')
        self.inbox_path = completed_games_path / 'inbox'
        self._inbox_scanner = InboxScanner(self.inbox_path, '.json')
        self.staging_path = completed_games_path / 'staging'
        self.queue_path = completed_games_path / _QUEUE_FILE
        for directory in (self.inbox_path, self.staging_path):
            directory.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.state = state
        self.configuration = configuration
        self.value_discount_per_ply = value_discount_per_ply
        self.resignation_calibrator = resignation_calibrator
        self.terminal_oracle = terminal_oracle
        self.censor_remaining_game_length_on_cut_games = censor_remaining_game_length_on_cut_games
        self._lock = threading.RLock()
        self._sealed_manifest_cache: dict[str, SealedReplayShardManifest] = {}
        self._fatal_materialization_error: RuntimeError | None = None
        if configuration.materialization_processes > 1 and experiment_configuration is None:
            raise ValueError('Parallel replay materialization requires the experiment configuration.')
        self._queue = self._load_queue()
        self._recover_directories()
        self.materialization_executor = (
            None
            if configuration.materialization_processes == 1
            else ProcessPoolExecutor(
                max_workers=configuration.materialization_processes,
                initializer=initialize_materialization_worker,
                initargs=(experiment_configuration.model_dump_json(), configuration.maximum_policy_entries),
            )
        )
        self._dispatcher = _MaterializationDispatcher(self)

    @classmethod
    def open(
        cls,
        run_path: Path,
        state: GameStateContract[PositionT],
        layout: ReplayLayout,
        configuration: ReplayConfiguration,
        model_generation: int,
        value_discount_per_ply: FloatGenerationSchedule,
        terminal_oracle: TerminalOracle[PositionT] | None,
        resignation_calibrator: ResignationCalibrator | None = None,
        experiment_configuration: ExperimentConfiguration | None = None,
        censor_remaining_game_length_on_cut_games: bool = False,
    ) -> ReplayManager[PositionT]:
        replay_path = run_path / 'replay.bin'
        if replay_path.exists():
            store = ReplayStore.open(replay_path, layout)
        else:
            store = ReplayStore.create(
                replay_path, layout, configuration.maximum_capacity, configuration.capacity_at(model_generation)
            )
        try:
            return cls(
                run_path / 'completed-games',
                store,
                state,
                configuration,
                value_discount_per_ply,
                resignation_calibrator,
                terminal_oracle,
                experiment_configuration,
                censor_remaining_game_length_on_cut_games,
            )
        except BaseException:
            try:
                store.close()
            except BaseException:
                pass
            raise

    @property
    def live_samples(self) -> int:
        return self.store.state.size

    @property
    def inbox_depth(self) -> int:
        return len(self.inbox_files_by_modification_time())

    @property
    def staging_depth(self) -> int:
        return sum(len(self._sealed_manifest(claim).games) for claim in self._queue.pending if self._is_sealed(claim))

    @property
    def materialization_failures(self) -> int:
        return self._dispatcher.failed_game_count

    def total_materialized_samples(self) -> int:
        self.raise_if_materialization_failed()
        with self._lock:
            return self.store.total_appended_rows + sum(
                self._sealed_manifest(claim).row_count
                for claim in self._queue.pending
                if claim.sequence >= self.store.state.append_sequence and self._is_sealed(claim)
            )

    def start_materialization(
        self, on_staged: Callable[[SealedReplayShard], None], poll_interval_seconds: float = DISPATCH_INTERVAL_SECONDS
    ) -> None:
        self._dispatcher.on_sealed = on_staged
        self._dispatcher.poll_interval_seconds = poll_interval_seconds
        self._dispatcher.start()

    def materialize_available_games(self, on_staged: Callable[[SealedReplayShard], None]) -> None:
        self._dispatcher.on_sealed = on_staged
        self._dispatcher.drain()
        self.raise_if_materialization_failed()

    def append_staged_games(self, model_generation: int) -> ReplayIngestion:
        self.raise_if_materialization_failed()
        started_at = time.perf_counter()
        with self._lock:
            before = self.store.state
            self.store.set_logical_capacity(self.configuration.capacity_at(model_generation))
            claims = self._contiguous_sealed_claims()
            if not claims:
                # Nothing was written since the previous append, and msync of the whole store
                # mapping costs ~0.2 s per gigabyte while holding this lock.
                after = self.store.state
                return ReplayIngestion(
                    0,
                    0,
                    after.size,
                    after.evicted_rows - before.evicted_rows,
                    0,
                    0,
                    0,
                    time.perf_counter() - started_at,
                    (),
                )
            readers: list[ReplayShardReader] = []
            try:
                for claim in claims:
                    readers.append(self._open_claim(claim))
                metadata = tuple(game for reader in readers for game in reader.manifest.games)
                for reader in readers:
                    self.store.append_columns(reader.columns, reader.manifest.shard_identity)
                self.store.flush()
            finally:
                for reader in readers:
                    reader.close()
            self._observe_resignation_games(metadata)
            self._finalize_committed_claims(claims)
            after = self.store.state
            return ReplayIngestion(
                len(metadata),
                sum(game.row_count for game in metadata),
                after.size,
                after.evicted_rows - before.evicted_rows,
                sum(game.policies_truncated for game in metadata),
                sum(game.retained_visit_mass for game in metadata),
                sum(game.discarded_visit_mass for game in metadata),
                time.perf_counter() - started_at,
                tuple(
                    IngestedCompletedGame(game.length_plies, game.termination_reason, game.observations)
                    for game in metadata
                ),
            )

    def raise_if_materialization_failed(self) -> None:
        with self._lock:
            error = self._fatal_materialization_error
        if error is not None:
            raise error

    def description(self) -> ReplayDescription:
        state = self.store.state
        return ReplayDescription(
            path=self.store.path,
            head=state.head,
            size=state.size,
            logical_capacity=state.logical_capacity,
            maximum_capacity=state.maximum_capacity,
            layout=self.store.layout,
        )

    def close(self) -> None:
        self._dispatcher.stop()
        if self.materialization_executor is not None:
            self.materialization_executor.shutdown()
        self.store.close()

    def inbox_files_by_modification_time(self) -> tuple[Path, ...]:
        return self._inbox_scanner.scan()

    def _load_queue(self) -> ReplayShardQueue:
        if not self.queue_path.exists():
            queue = ReplayShardQueue(
                layout_digest=self.store.layout.digest, next_sequence=self.store.state.append_sequence
            )
            self._save_queue(queue)
            return queue
        queue = ReplayShardQueue.model_validate_json(self.queue_path.read_text(encoding='utf-8'))
        if queue.layout_digest != self.store.layout.digest:
            raise ValueError('Replay shard queue layout does not match the replay store.')
        if queue.next_sequence < self.store.state.append_sequence:
            raise ValueError('Replay shard queue sequence precedes the replay store.')
        first_sequence = queue.pending[0].sequence if queue.pending else queue.next_sequence
        if first_sequence > self.store.state.append_sequence:
            raise ValueError('Replay shard queue begins after the replay store append sequence.')
        return queue

    def _save_queue(self, queue: ReplayShardQueue) -> None:
        write_text_atomically(self.queue_path, queue.model_dump_json() + '\n')

    def _allocate_claims(self) -> None:
        with self._lock:
            if self._fatal_materialization_error is not None:
                return
            queue_snapshot = self._queue
            claimed = {source.order.file_name for claim in queue_snapshot.pending for source in claim.games}
            last_claimed_key = max(
                (source.order.key for claim in queue_snapshot.pending for source in claim.games),
                default=None,
            )
            available = tuple(path for path in self.inbox_files_by_modification_time() if path.name not in claimed)
            claim_slots = self._claim_slots(queue_snapshot)
        batches: list[tuple[ReplayShardSourceGame, ...]] = []
        fatal_error: RuntimeError | None = None
        candidate_index = 0
        while candidate_index < len(available) and len(batches) < claim_slots:
            batch: list[ReplayShardSourceGame] = []
            batch_bytes = 0
            while (
                candidate_index < len(available) and len(batch) < self.configuration.materialization_shard_maximum_games
            ):
                candidate = available[candidate_index]
                try:
                    status = candidate.stat()
                    # A game renamed into the inbox can surface after a younger file was already
                    # claimed; refresh its timestamp so it re-enters the FIFO at the tail instead
                    # of violating the queue's global order.
                    if last_claimed_key is not None and (status.st_mtime_ns, candidate.name) <= last_claimed_key:
                        os.utime(candidate)
                        self._inbox_scanner.invalidate(candidate.name)
                        candidate_index += 1
                        continue
                    if (
                        batch
                        and batch_bytes + status.st_size > self.configuration.materialization_shard_target_source_bytes
                    ):
                        break
                    identity = GameIdentity.from_file_name(candidate.name)
                    source = ReplayShardSourceGame(
                        identity=identity,
                        order=InboxGameOrder(modified_at_ns=status.st_mtime_ns, file_name=candidate.name),
                        source_size=status.st_size,
                        source_sha256=_file_sha256(candidate),
                    )
                except (OSError, ValueError) as error:
                    fatal_error = RuntimeError(f'Completed game cannot be claimed in FIFO order: {candidate}: {error}')
                    break
                candidate_index += 1
                batch.append(source)
                batch_bytes += status.st_size
            if batch:
                batches.append(tuple(batch))
            if fatal_error is not None:
                break
            if not batch:
                break
        with self._lock:
            if self._queue != queue_snapshot:
                return
            pending = list(queue_snapshot.pending)
            next_sequence = queue_snapshot.next_sequence
            claimed_now = {source.order.file_name for claim in pending for source in claim.games}
            for batch in batches:
                for source in batch:
                    path = self.inbox_path / source.order.file_name
                    try:
                        status = path.stat()
                    except OSError:
                        return
                    if (
                        source.order.file_name in claimed_now
                        or status.st_size != source.source_size
                        or status.st_mtime_ns != source.order.modified_at_ns
                    ):
                        return
                    claimed_now.add(source.order.file_name)
                pending.append(PendingReplayShardManifest.create(self.store.layout, next_sequence, batch))
                next_sequence += 1
            queue = ReplayShardQueue(
                layout_digest=self.store.layout.digest,
                next_sequence=next_sequence,
                pending=tuple(pending),
            )
            if queue != self._queue:
                self._save_queue(queue)
                self._queue = queue
            if fatal_error is not None:
                self._set_fatal_materialization_error(fatal_error)
                self._dispatcher.failed_game_count += 1
                log(str(fatal_error))

    def _claim_slots(self, queue: ReplayShardQueue) -> int:
        append_sequence = self.store.state.append_sequence
        outstanding = tuple(claim for claim in queue.pending if claim.sequence >= append_sequence)
        unsealed = sum(not self._is_sealed(claim) for claim in outstanding)
        materialization_limit = max(_MINIMUM_PENDING_SHARD_LIMIT, 2 * self.configuration.materialization_processes)
        # Sealed shards wait on the single-threaded appender; counting them as in-flight work
        # idles every materialization worker for the whole duration of an append.
        staging_limit = _STAGED_SHARD_LIMIT_FACTOR * materialization_limit
        return min(materialization_limit - unsealed, staging_limit - len(outstanding))

    def _unclaimed_inbox_files(self) -> tuple[Path, ...]:
        with self._lock:
            claimed = {source.order.file_name for claim in self._queue.pending for source in claim.games}
        return tuple(path for path in self.inbox_files_by_modification_time() if path.name not in claimed)

    def _has_claim_capacity(self) -> bool:
        with self._lock:
            return self._claim_slots(self._queue) > 0

    def _has_unsealed_claims(self) -> bool:
        with self._lock:
            return any(
                claim.sequence >= self.store.state.append_sequence and not self._is_sealed(claim)
                for claim in self._queue.pending
            )

    def _set_fatal_materialization_error(self, error: RuntimeError) -> None:
        with self._lock:
            if self._fatal_materialization_error is None:
                self._fatal_materialization_error = error

    def _claim(self, sequence: int) -> PendingReplayShardManifest:
        return next(claim for claim in self._queue.pending if claim.sequence == sequence)

    def _stage_shard_inline(self, claim: PendingReplayShardManifest) -> SealedReplayShard:
        return stage_replay_shard(
            claim,
            self.inbox_path,
            self.staging_path,
            self.state,
            self.terminal_oracle,
            self.store.layout,
            self.value_discount_per_ply,
            self.censor_remaining_game_length_on_cut_games,
        )

    def _is_sealed(self, claim: PendingReplayShardManifest) -> bool:
        return replay_shard_manifest_path(self.staging_path, claim.shard_identity).exists()

    def _open_claim(self, claim: PendingReplayShardManifest) -> ReplayShardReader:
        # _sealed_manifest already validated this manifest against the claim; re-parsing its
        # embedded search observations here costs more than the row copy it guards.
        return ReplayShardReader.open(
            replay_shard_manifest_path(self.staging_path, claim.shard_identity),
            self.store.layout,
            verify_data_hash=False,
            manifest=self._sealed_manifest(claim),
        )

    def _sealed_result(self, claim: PendingReplayShardManifest) -> SealedReplayShard:
        manifest = self._sealed_manifest(claim)
        return SealedReplayShard(manifest.sequence, manifest.shard_identity, manifest.row_count, len(manifest.games))

    def _sealed_manifest(self, claim: PendingReplayShardManifest) -> SealedReplayShardManifest:
        cached = self._sealed_manifest_cache.get(claim.shard_identity)
        if cached is not None:
            return cached
        path = replay_shard_manifest_path(self.staging_path, claim.shard_identity)
        manifest = SealedReplayShardManifest.model_validate_json(path.read_text(encoding='utf-8'))
        if (
            manifest.layout_digest != self.store.layout.digest
            or manifest.sequence != claim.sequence
            or tuple(game.source for game in manifest.games) != claim.games
        ):
            raise ValueError('Sealed replay shard does not match its durable claim.')
        self._sealed_manifest_cache[claim.shard_identity] = manifest
        return manifest

    def _contiguous_sealed_claims(self) -> tuple[PendingReplayShardManifest, ...]:
        by_sequence = {claim.sequence: claim for claim in self._queue.pending}
        sequence = self.store.state.append_sequence
        claims = []
        while (claim := by_sequence.get(sequence)) is not None and self._is_sealed(claim):
            claims.append(claim)
            sequence += 1
        return tuple(claims)

    def _observe_resignation_games(self, metadata: tuple[ReplayShardGameMetadata, ...]) -> None:
        if self.resignation_calibrator is None:
            return
        with self.resignation_calibrator.calibration_batch() as calibration_batch:
            for game in metadata:
                calibration_batch.observe_game_record(
                    archive_key=game.source.identity.archive_key,
                    is_resignation_continuation=game.is_resignation_continuation,
                    termination_reason=game.termination_reason,
                    final_wdl=game.final_wdl,
                    length_plies=game.length_plies,
                    observations=game.observations,
                )

    def _finalize_committed_claims(self, claims: tuple[PendingReplayShardManifest, ...]) -> None:
        committed = {claim.shard_identity for claim in claims}
        for claim in claims:
            self._validate_leftover_sources(claim.games)
        queue = ReplayShardQueue(
            layout_digest=self._queue.layout_digest,
            next_sequence=self._queue.next_sequence,
            pending=tuple(claim for claim in self._queue.pending if claim.shard_identity not in committed),
        )
        self._save_queue(queue)
        self._queue = queue
        for claim in claims:
            self._delete_shard_artifacts(claim.shard_identity, claim.games)

    def _validate_leftover_sources(self, sources: tuple[ReplayShardSourceGame, ...]) -> None:
        for source in sources:
            inbox_file = self.inbox_path / source.order.file_name
            if not inbox_file.exists():
                continue
            if inbox_file.stat().st_size != source.source_size or _file_sha256(inbox_file) != source.source_sha256:
                raise ValueError('Leftover completed-game source does not match its committed replay shard.')

    def _delete_shard_artifacts(self, identity: str, sources: tuple[ReplayShardSourceGame, ...]) -> None:
        for source in sources:
            (self.inbox_path / source.order.file_name).unlink(missing_ok=True)
        self._sealed_manifest_cache.pop(identity, None)
        replay_shard_manifest_path(self.staging_path, identity).unlink(missing_ok=True)
        replay_shard_data_path(self.staging_path, identity).unlink(missing_ok=True)

    def _recover_directories(self) -> None:
        legacy = tuple(self.staging_path.glob(f'*{_LEGACY_STAGED_ROWS_SUFFIX}')) + tuple(
            self.staging_path.glob(f'*{_LEGACY_STAGED_METADATA_SUFFIX}')
        )
        if legacy:
            raise ValueError('Legacy per-game replay staging exists; an explicit replay migration is required.')
        for directory in (self.inbox_path, self.staging_path):
            for temporary in directory.glob('.*.tmp'):
                temporary.unlink(missing_ok=True)
        committed_claims = tuple(
            claim for claim in self._queue.pending if claim.sequence < self.store.state.append_sequence
        )
        committed_metadata = tuple(
            game for claim in committed_claims if self._is_sealed(claim) for game in self._sealed_manifest(claim).games
        )
        self._observe_resignation_games(committed_metadata)
        if committed_claims:
            self._finalize_committed_claims(committed_claims)
        claimed = {claim.shard_identity for claim in self._queue.pending}
        for manifest_path in self.staging_path.glob('*.replay-shard.json'):
            identity = manifest_path.name.removesuffix('.replay-shard.json')
            if identity not in claimed:
                manifest = SealedReplayShardManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
                if (
                    manifest.layout_digest != self.store.layout.digest
                    or manifest.shard_identity != identity
                    or manifest.sequence >= self.store.state.append_sequence
                ):
                    raise ValueError('Sealed replay shard is not owned by the durable shard queue.')
                self._observe_resignation_games(manifest.games)
                sources = tuple(game.source for game in manifest.games)
                self._validate_leftover_sources(sources)
                self._delete_shard_artifacts(identity, sources)
        for data_path in self.staging_path.glob('*.replay-shard.bin'):
            identity = data_path.name.removesuffix('.replay-shard.bin')
            if not replay_shard_manifest_path(self.staging_path, identity).exists():
                data_path.unlink(missing_ok=True)
        for claim in self._queue.pending:
            if self._is_sealed(claim):
                self._stage_shard_inline(claim)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class InboxScanner:
    """Orders inbox files by modification time without re-stat()ing the whole backlog every cycle."""

    def __init__(self, directory: Path, suffix: str) -> None:
        self.directory = directory
        self.suffix = suffix
        self._modified_at_by_name: dict[str, int] = {}

    def invalidate(self, file_name: str) -> None:
        self._modified_at_by_name.pop(file_name, None)

    def scan(self) -> tuple[Path, ...]:
        try:
            entries = tuple(os.scandir(self.directory))
        except FileNotFoundError:
            self._modified_at_by_name.clear()
            return ()
        known = self._modified_at_by_name
        modified_at_by_name: dict[str, int] = {}
        for entry in entries:
            name = entry.name
            if not name.endswith(self.suffix):
                continue
            # A completed game is renamed into place fully written, so a name's timestamp is
            # immutable for as long as the name exists; only re-queued files are invalidated.
            cached = known.get(name)
            if cached is not None:
                modified_at_by_name[name] = cached
                continue
            try:
                modified_at_by_name[name] = entry.stat().st_mtime_ns
            except OSError:
                continue
        self._modified_at_by_name = modified_at_by_name
        ordered = sorted(modified_at_by_name.items(), key=lambda item: (item[1], item[0]))
        return tuple(self.directory / name for name, _ in ordered)
