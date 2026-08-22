from __future__ import annotations

import hashlib
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
from src.replay.store import ReplayAppendPlan, ReplayAppendTransaction, ReplayStore, plan_replay_append_chain
from src.self_play.completed_game import GameIdentity, SearchObservation, TerminationReason
from src.self_play.resignation import ResignationCalibrator
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.generation_schedule import FloatGenerationSchedule
from src.util.log import log

PositionT = TypeVar('PositionT')
DISPATCH_INTERVAL_SECONDS = 1.0
_QUEUE_FILE = 'shard-queue.json'
_APPEND_FILE = 'last-append.json'
_RECEIPT_SUFFIX = '.ingestion-receipt.json'
_LEGACY_STAGED_ROWS_SUFFIX = '.rows.npy'
_LEGACY_STAGED_METADATA_SUFFIX = '.meta.json'
_MINIMUM_PENDING_SHARD_LIMIT = 32


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
    receipt_identities: tuple[str, ...] = ()

    @property
    def samples_per_second(self) -> float:
        return self.samples_added / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


class ReplayIngestionReceipt(FrozenModel):
    schema_version: Literal[1] = 1
    receipt_identity: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_generation: int = Field(ge=0)
    shard_identities: tuple[str, ...]
    append_sequence_after: int = Field(ge=0)
    games_ingested: int = Field(ge=0)
    samples_added: int = Field(ge=0)
    live_samples: int = Field(ge=0)
    evicted_samples: int = Field(ge=0)
    policies_truncated: int = Field(ge=0)
    retained_visit_mass: int = Field(ge=0)
    discarded_visit_mass: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    completed_games: tuple[IngestedCompletedGame, ...]


class PendingReplayResize(FrozenModel):
    logical_capacity: int = Field(gt=0)
    evicted_rows_before: int = Field(ge=0)
    model_generation: int = Field(ge=0)


class ReplayShardQueue(FrozenModel):
    schema_version: Literal[1] = 1
    layout_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    next_sequence: int = Field(ge=0)
    pending: tuple[PendingReplayShardManifest, ...] = ()
    pending_resize: PendingReplayResize | None = None

    @model_validator(mode='after')
    def validate_pending(self) -> ReplayShardQueue:
        sequences = tuple(claim.sequence for claim in self.pending)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError('Replay shard queue claims must have unique increasing sequences.')
        if sequences and sequences != tuple(range(sequences[0], self.next_sequence)):
            raise ValueError('Replay shard queue claims must form one contiguous sequence.')
        if sequences and self.next_sequence != sequences[-1] + 1:
            raise ValueError('Replay shard queue next sequence must follow its final claim.')
        return self


class ReplayAppendRecovery(FrozenModel):
    schema_version: Literal[1] = 1
    layout_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_generation: int = Field(ge=0)
    shard_manifest_files: tuple[str, ...] = Field(min_length=1)
    plans: tuple[ReplayAppendPlan, ...] = Field(min_length=1)
    receipt_identity: str = Field(pattern=r'^[0-9a-f]{64}$')
    evicted_rows_before: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_chain(self) -> ReplayAppendRecovery:
        if len(self.shard_manifest_files) != len(self.plans):
            raise ValueError('Replay append recovery requires one plan per shard manifest.')
        if any(Path(file_name).name != file_name for file_name in self.shard_manifest_files):
            raise ValueError('Replay append recovery shard files must be basenames.')
        expected_files = tuple(
            replay_shard_manifest_path(Path(), plan.transaction_identity).name for plan in self.plans
        )
        if self.shard_manifest_files != expected_files:
            raise ValueError('Replay append recovery shard files do not match its transactions.')
        if any(left.after != right.before for left, right in zip(self.plans, self.plans[1:])):
            raise ValueError('Replay append recovery plans must form one state chain.')
        expected_receipt = _receipt_identity(
            self.model_generation,
            tuple(plan.transaction_identity for plan in self.plans),
        )
        if self.receipt_identity != expected_receipt:
            raise ValueError('Replay append recovery receipt identity is invalid.')
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
        if isinstance(error, ValueError):
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
        self.staging_path = completed_games_path / 'staging'
        self.receipts_path = completed_games_path / 'reporting-receipts'
        self.queue_path = completed_games_path / _QUEUE_FILE
        self.append_manifest_path = completed_games_path / _APPEND_FILE
        for directory in (self.inbox_path, self.staging_path, self.receipts_path):
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
        append_path = run_path / 'completed-games' / _APPEND_FILE
        queue_path = run_path / 'completed-games' / _QUEUE_FILE
        pending_resize = False
        if queue_path.exists():
            pending_resize = (
                ReplayShardQueue.model_validate_json(queue_path.read_text(encoding='utf-8')).pending_resize is not None
            )
        if replay_path.exists():
            store = (
                ReplayStore.open_for_recovery(replay_path, layout)
                if append_path.exists() or pending_resize
                else ReplayStore.open(replay_path, layout)
            )
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
            logical_capacity = self.configuration.capacity_at(model_generation)
            if self.store.state.logical_capacity != logical_capacity:
                pending_resize = self._queue.pending_resize or PendingReplayResize(
                    logical_capacity=logical_capacity,
                    evicted_rows_before=before.evicted_rows,
                    model_generation=model_generation,
                )
                pending_resize = PendingReplayResize(
                    logical_capacity=logical_capacity,
                    evicted_rows_before=pending_resize.evicted_rows_before,
                    model_generation=model_generation,
                )
                queue = ReplayShardQueue(
                    layout_digest=self._queue.layout_digest,
                    next_sequence=self._queue.next_sequence,
                    pending=self._queue.pending,
                    pending_resize=pending_resize,
                )
                self._save_queue(queue)
                self._queue = queue
            self.store.set_logical_capacity(logical_capacity)
            if self.store.state != before:
                self.store.flush()
            claims = self._contiguous_sealed_claims()
            if not claims:
                after = self.store.state
                return ReplayIngestion(
                    0,
                    0,
                    after.size,
                    0 if self._queue.pending_resize is not None else after.evicted_rows - before.evicted_rows,
                    0,
                    0,
                    0,
                    time.perf_counter() - started_at,
                    (),
                )
            readers = tuple(self._open_claim(claim) for claim in claims)
            try:
                plans = plan_replay_append_chain(
                    self.store.state,
                    tuple(
                        ReplayAppendTransaction(reader.manifest.row_count, reader.manifest.shard_identity)
                        for reader in readers
                    ),
                )
                receipt_identity = _receipt_identity(model_generation, tuple(claim.shard_identity for claim in claims))
                recovery = ReplayAppendRecovery(
                    layout_digest=self.store.layout.digest,
                    model_generation=model_generation,
                    shard_manifest_files=tuple(reader.manifest_path.name for reader in readers),
                    plans=plans,
                    receipt_identity=receipt_identity,
                    evicted_rows_before=(
                        self._queue.pending_resize.evicted_rows_before
                        if self._queue.pending_resize is not None
                        else before.evicted_rows
                    ),
                )
                write_text_atomically(self.append_manifest_path, recovery.model_dump_json() + '\n')
                return self._complete_append_recovery(recovery, started_at, readers)
            except BaseException:
                for reader in readers:
                    reader.close()
                raise

    def pending_ingestion_receipts(self) -> tuple[ReplayIngestionReceipt, ...]:
        receipts = tuple(self._load_receipt(path) for path in self.receipts_path.glob(f'*{_RECEIPT_SUFFIX}'))
        return tuple(sorted(receipts, key=lambda receipt: receipt.append_sequence_after))

    def acknowledge_ingestion_receipts(self, receipt_identities: tuple[str, ...]) -> None:
        for identity in receipt_identities:
            if len(identity) != 64 or any(character not in '0123456789abcdef' for character in identity):
                raise ValueError('Replay ingestion receipt identity must be a lowercase SHA-256 digest.')
            self._receipt_path(identity).unlink(missing_ok=True)

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
        return _files_by_modification_time(self.inbox_path, '*.json')

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
        if not self.append_manifest_path.exists():
            first_uncommitted = queue.pending[0].sequence if queue.pending else queue.next_sequence
            if first_uncommitted != self.store.state.append_sequence:
                raise ValueError('Replay shard queue does not begin at the replay store append sequence.')
        return queue

    def _save_queue(self, queue: ReplayShardQueue) -> None:
        write_text_atomically(self.queue_path, queue.model_dump_json() + '\n')

    def _allocate_claims(self) -> None:
        with self._lock:
            if self._fatal_materialization_error is not None:
                return
            queue_snapshot = self._queue
            claimed = {source.order.file_name for claim in queue_snapshot.pending for source in claim.games}
            available = tuple(path for path in self.inbox_files_by_modification_time() if path.name not in claimed)
            maximum_pending = max(_MINIMUM_PENDING_SHARD_LIMIT, 2 * self.configuration.materialization_processes)
            claim_slots = maximum_pending - sum(
                claim.sequence >= self.store.state.append_sequence for claim in queue_snapshot.pending
            )
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
                pending_resize=queue_snapshot.pending_resize,
            )
            if queue != self._queue:
                self._save_queue(queue)
                self._queue = queue
            if fatal_error is not None:
                self._set_fatal_materialization_error(fatal_error)
                self._dispatcher.failed_game_count += 1
                log(str(fatal_error))

    def _unclaimed_inbox_files(self) -> tuple[Path, ...]:
        with self._lock:
            claimed = {source.order.file_name for claim in self._queue.pending for source in claim.games}
        return tuple(path for path in self.inbox_files_by_modification_time() if path.name not in claimed)

    def _has_claim_capacity(self) -> bool:
        with self._lock:
            outstanding = sum(claim.sequence >= self.store.state.append_sequence for claim in self._queue.pending)
            maximum_pending = max(_MINIMUM_PENDING_SHARD_LIMIT, 2 * self.configuration.materialization_processes)
            return outstanding < maximum_pending

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
        reader = ReplayShardReader.open(
            replay_shard_manifest_path(self.staging_path, claim.shard_identity),
            self.store.layout,
            verify_data_hash=False,
        )
        if (
            reader.manifest.sequence != claim.sequence
            or tuple(game.source for game in reader.manifest.games) != claim.games
        ):
            reader.close()
            raise ValueError('Sealed replay shard does not match its durable claim.')
        return reader

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

    def _complete_append_recovery(
        self,
        recovery: ReplayAppendRecovery,
        started_at: float,
        opened_readers: tuple[ReplayShardReader, ...] | None = None,
    ) -> ReplayIngestion:
        receipt_path = self._receipt_path(recovery.receipt_identity)
        final_state = recovery.plans[-1].after
        if receipt_path.exists() and self.store.state == final_state:
            receipt = self._load_receipt(receipt_path)
            expected_shards = tuple(plan.transaction_identity for plan in recovery.plans)
            if (
                receipt.receipt_identity != recovery.receipt_identity
                or receipt.shard_identities != expected_shards
                or receipt.append_sequence_after != final_state.append_sequence
                or receipt.model_generation != recovery.model_generation
                or receipt.samples_added != sum(plan.row_count for plan in recovery.plans)
                or receipt.live_samples != final_state.size
                or receipt.evicted_samples != final_state.evicted_rows - recovery.evicted_rows_before
                or receipt.games_ingested != len(receipt.completed_games)
            ):
                raise ValueError('Replay ingestion receipt does not match append recovery.')
            self._cleanup_committed_recovery(recovery)
            return self._ingestion_from_receipt(receipt)
        readers = opened_readers or tuple(
            ReplayShardReader.open(self.staging_path / file_name, self.store.layout)
            for file_name in recovery.shard_manifest_files
        )
        try:
            if len(readers) != len(recovery.plans):
                raise ValueError('Replay append recovery shard and plan counts differ.')
            for reader, plan in zip(readers, recovery.plans, strict=True):
                if reader.manifest.shard_identity != plan.transaction_identity:
                    raise ValueError('Replay append recovery transaction does not match its shard.')
            self.store.reapply_append_plan_chain(
                tuple((reader.columns,) for reader in readers),
                recovery.plans,
            )
            if self.store.state != final_state:
                raise ValueError('Replay append recovery did not reach its expected final state.')
            self.store.flush()
            metadata = tuple(game for reader in readers for game in reader.manifest.games)
            self._observe_resignation_games(metadata)
            receipt = self._receipt(recovery, metadata, started_at)
            write_text_atomically(receipt_path, receipt.model_dump_json() + '\n')
        finally:
            for reader in readers:
                reader.close()
        self._cleanup_committed_recovery(recovery)
        return self._ingestion_from_receipt(receipt)

    def _receipt(
        self,
        recovery: ReplayAppendRecovery,
        metadata: tuple[ReplayShardGameMetadata, ...],
        started_at: float,
    ) -> ReplayIngestionReceipt:
        state = self.store.state
        return ReplayIngestionReceipt(
            receipt_identity=recovery.receipt_identity,
            model_generation=recovery.model_generation,
            shard_identities=tuple(plan.transaction_identity for plan in recovery.plans),
            append_sequence_after=state.append_sequence,
            games_ingested=len(metadata),
            samples_added=sum(plan.row_count for plan in recovery.plans),
            live_samples=state.size,
            evicted_samples=state.evicted_rows - recovery.evicted_rows_before,
            policies_truncated=sum(game.policies_truncated for game in metadata),
            retained_visit_mass=sum(game.retained_visit_mass for game in metadata),
            discarded_visit_mass=sum(game.discarded_visit_mass for game in metadata),
            elapsed_seconds=time.perf_counter() - started_at,
            completed_games=tuple(
                IngestedCompletedGame(game.length_plies, game.termination_reason, game.observations)
                for game in metadata
            ),
        )

    @staticmethod
    def _ingestion_from_receipt(receipt: ReplayIngestionReceipt) -> ReplayIngestion:
        return ReplayIngestion(
            receipt.games_ingested,
            receipt.samples_added,
            receipt.live_samples,
            receipt.evicted_samples,
            receipt.policies_truncated,
            receipt.retained_visit_mass,
            receipt.discarded_visit_mass,
            receipt.elapsed_seconds,
            receipt.completed_games,
            (receipt.receipt_identity,),
        )

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

    def _cleanup_committed_recovery(self, recovery: ReplayAppendRecovery) -> None:
        committed = {plan.transaction_identity for plan in recovery.plans}
        for claim in self._queue.pending:
            if claim.shard_identity not in committed:
                continue
            for source in claim.games:
                inbox_file = self.inbox_path / source.order.file_name
                if not inbox_file.exists():
                    continue
                if inbox_file.stat().st_size != source.source_size or _file_sha256(inbox_file) != source.source_sha256:
                    raise ValueError('Leftover completed-game source does not match its committed replay shard.')
                inbox_file.unlink()
        for identity in committed:
            self._sealed_manifest_cache.pop(identity, None)
            replay_shard_manifest_path(self.staging_path, identity).unlink(missing_ok=True)
            replay_shard_data_path(self.staging_path, identity).unlink(missing_ok=True)
        queue = ReplayShardQueue(
            layout_digest=self._queue.layout_digest,
            next_sequence=self._queue.next_sequence,
            pending=tuple(claim for claim in self._queue.pending if claim.shard_identity not in committed),
            pending_resize=None,
        )
        self._save_queue(queue)
        self._queue = queue
        self.append_manifest_path.unlink(missing_ok=True)

    def _receipt_path(self, identity: str) -> Path:
        return self.receipts_path / f'{identity}{_RECEIPT_SUFFIX}'

    def _load_receipt(self, path: Path) -> ReplayIngestionReceipt:
        receipt = ReplayIngestionReceipt.model_validate_json(path.read_text(encoding='utf-8'))
        if path.name != self._receipt_path(receipt.receipt_identity).name:
            raise ValueError('Replay ingestion receipt file name does not match its identity.')
        return receipt

    def _recover_directories(self) -> None:
        legacy = tuple(self.staging_path.glob(f'*{_LEGACY_STAGED_ROWS_SUFFIX}')) + tuple(
            self.staging_path.glob(f'*{_LEGACY_STAGED_METADATA_SUFFIX}')
        )
        if legacy:
            raise ValueError('Legacy per-game replay staging exists; an explicit replay migration is required.')
        for directory in (self.inbox_path, self.staging_path, self.receipts_path):
            for temporary in directory.glob('.*.tmp'):
                temporary.unlink(missing_ok=True)
        if self._queue.pending_resize is not None:
            self.store.set_logical_capacity(self._queue.pending_resize.logical_capacity)
            self.store.flush()
        claimed = {claim.shard_identity for claim in self._queue.pending}
        for manifest_path in self.staging_path.glob('*.replay-shard.json'):
            identity = manifest_path.name.removesuffix('.replay-shard.json')
            if identity not in claimed:
                raise ValueError('Sealed replay shard is not owned by the durable shard queue.')
        for data_path in self.staging_path.glob('*.replay-shard.bin'):
            identity = data_path.name.removesuffix('.replay-shard.bin')
            if identity in claimed and not replay_shard_manifest_path(self.staging_path, identity).exists():
                data_path.unlink(missing_ok=True)
        for claim in self._queue.pending:
            if self._is_sealed(claim):
                self._stage_shard_inline(claim)
        if self.append_manifest_path.exists():
            recovery = ReplayAppendRecovery.model_validate_json(self.append_manifest_path.read_text(encoding='utf-8'))
            if recovery.layout_digest != self.store.layout.digest:
                raise ValueError('Replay append recovery layout does not match the store.')
            self._complete_append_recovery(recovery, time.perf_counter())


def _receipt_identity(model_generation: int, shard_identities: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in (str(model_generation), *shard_identities):
        encoded = value.encode('ascii')
        digest.update(len(encoded).to_bytes(8, byteorder='little'))
        digest.update(encoded)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _files_by_modification_time(directory: Path, pattern: str) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    modified_at_by_path: dict[Path, int] = {}
    for path in directory.glob(pattern):
        try:
            modified_at_by_path[path] = path.stat().st_mtime_ns
        except OSError:
            continue
    return tuple(sorted(modified_at_by_path, key=lambda path: (modified_at_by_path[path], path.name)))
