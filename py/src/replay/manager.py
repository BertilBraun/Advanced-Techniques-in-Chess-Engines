from __future__ import annotations

import multiprocessing
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from queue import Empty
from typing import Generic, TypeVar

from src.experiment.configuration import ExperimentConfiguration
from src.games.contracts import GameStateContract, TerminalOracle
from src.replay.configuration import ReplayConfiguration
from src.replay.description import ReplayDescription
from src.replay.dispatch import (
    COMPLETED_GAME_SUFFIX,
    InboxDispatcher,
    parse_worker_source_file_name,
    worker_directory_paths,
    worker_source_file_names,
)
from src.replay.layout import ReplayLayout
from src.replay.materialization_worker import (
    MaterializationReport,
    MaterializationSettings,
    MaterializationWorker,
    run_materialization_worker,
)
from src.replay.shard import (
    MANIFEST_SUFFIX,
    ReplayShardGameMetadata,
    ReplayShardReader,
    SealedReplayShardManifest,
    replay_shard_data_path,
    replay_shard_manifest_path,
    sealed_replay_shard_manifest_paths,
)
from src.replay.store import ReplayStore
from src.self_play.completed_game import SearchObservation, TerminationReason
from src.self_play.resignation import ResignationCalibrator
from src.util.generation_schedule import FloatGenerationSchedule
from src.util.log import log, warn

PositionT = TypeVar('PositionT')
DISPATCH_INTERVAL_SECONDS = 1.0
_LEGACY_STAGED_ROWS_SUFFIX = '.rows.npy'
_LEGACY_STAGED_METADATA_SUFFIX = '.meta.json'
_LEGACY_QUEUE_FILE = 'shard-queue.json'


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


class _RejectionRateAlarm:
    """Bounds the fraction of discarded games so that "skip a bad game" cannot become "skip every game"."""

    def __init__(self, window_games: int, rate_ceiling: float) -> None:
        self.window_games = window_games
        self.rate_ceiling = rate_ceiling
        self._outcomes: deque[bool] = deque(maxlen=window_games)

    @property
    def rejection_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    def observe(self, materialized_games: int, rejected_games: int) -> None:
        self._outcomes.extend([False] * materialized_games)
        self._outcomes.extend([True] * rejected_games)

    def breached(self) -> bool:
        return len(self._outcomes) == self.window_games and self.rejection_rate > self.rate_ceiling


class _MaterializationSupervisor:
    """Owns the dispatcher thread and one long-lived process per worker directory."""

    def __init__(self, manager: ReplayManager[PositionT], poll_interval_seconds: float) -> None:
        self._manager = manager
        self.poll_interval_seconds = poll_interval_seconds
        self._context = multiprocessing.get_context('spawn')
        self._stop_event: EventType = self._context.Event()
        self._report_queue: Queue[MaterializationReport] = self._context.Queue()
        self._processes: dict[int, SpawnProcess] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        assert self._thread is None
        for worker_index in range(len(self._manager.worker_paths)):
            self._start_worker(worker_index)
        self._thread = threading.Thread(target=self._run, name='replay-materialization-supervisor', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for process in self._processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        self._processes.clear()
        self._manager.drain_worker_reports()
        self._report_queue.close()

    def drain_reports(self) -> None:
        while True:
            try:
                report = self._report_queue.get_nowait()
            except (Empty, OSError, ValueError):
                return
            self._manager.record_materialization_report(report)

    def _start_worker(self, worker_index: int) -> None:
        configuration_json = self._manager.experiment_configuration_json
        assert configuration_json is not None
        process = self._context.Process(
            target=run_materialization_worker,
            args=(
                configuration_json,
                self._manager.completed_games_path,
                worker_index,
                self._manager.materialization_settings,
                self.poll_interval_seconds,
                self._report_queue,
                self._stop_event,
            ),
            name=f'replay-materialization-worker-{worker_index}',
            daemon=True,
        )
        process.start()
        self._processes[worker_index] = process

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._manager.dispatch_once()
                self.drain_reports()
                self._supervise_workers()
            except BaseException as error:  # noqa: BLE001
                self._manager.set_fatal_materialization_error(
                    RuntimeError(f'Replay materialization supervisor failed: {error}')
                )
                return
            self._stop_event.wait(self.poll_interval_seconds)

    def _supervise_workers(self) -> None:
        for worker_index, process in tuple(self._processes.items()):
            if process.is_alive():
                continue
            warn(f'Replay materialization worker {worker_index} died with code {process.exitcode}; restarting.')
            process.join(timeout=1.0)
            self._start_worker(worker_index)


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
        self.completed_games_path = completed_games_path
        self.inbox_path = completed_games_path / 'inbox'
        self.staging_path = completed_games_path / 'staging'
        self.rejected_path = completed_games_path / 'rejected'
        self.worker_paths = worker_directory_paths(completed_games_path, configuration.materialization_processes)
        for directory in (
            self.inbox_path,
            self.staging_path,
            self.rejected_path,
            *self.worker_paths,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.state = state
        self.configuration = configuration
        self.value_discount_per_ply = value_discount_per_ply
        self.resignation_calibrator = resignation_calibrator
        self.terminal_oracle = terminal_oracle
        self.censor_remaining_game_length_on_cut_games = censor_remaining_game_length_on_cut_games
        self.experiment_configuration_json = (
            None if experiment_configuration is None else experiment_configuration.model_dump_json()
        )
        self.materialization_settings = MaterializationSettings(
            shard_maximum_games=configuration.materialization_shard_maximum_games,
            shard_target_source_bytes=configuration.materialization_shard_target_source_bytes,
            staging_shard_limit=configuration.materialization_staging_shard_limit,
            maximum_policy_entries=configuration.maximum_policy_entries,
        )
        self._lock = threading.RLock()
        self._fatal_materialization_error: RuntimeError | None = None
        self._rejection_alarm = _RejectionRateAlarm(
            configuration.materialization_rejection_window_games,
            configuration.materialization_rejection_rate_ceiling,
        )
        self.rejected_games = 0
        self._sealed_manifest_cache: dict[str, SealedReplayShardManifest] = {}
        self._recover_directories()
        self._dispatcher = InboxDispatcher(
            self.inbox_path, self.worker_paths, configuration.materialization_inbox_rename_cap
        )
        self._inline_workers: dict[int, MaterializationWorker[PositionT]] = {}
        self._supervisor: _MaterializationSupervisor | None = None

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
        return len(_completed_game_names(self.inbox_path)) + sum(
            len(worker_source_file_names(path)) for path in self.worker_paths
        )

    @property
    def staging_depth(self) -> int:
        return sum(len(manifest.games) for manifest in self._staged_manifests())

    @property
    def materialization_failures(self) -> int:
        return self.rejected_games

    @property
    def rejection_rate(self) -> float:
        return self._rejection_alarm.rejection_rate

    def total_materialized_samples(self) -> int:
        self.raise_if_materialization_failed()
        with self._lock:
            return self.store.total_appended_rows

    def start_materialization(self, poll_interval_seconds: float = DISPATCH_INTERVAL_SECONDS) -> None:
        if self.experiment_configuration_json is None:
            raise ValueError('Replay materialization workers require the experiment configuration.')
        assert self._supervisor is None
        self._supervisor = _MaterializationSupervisor(self, poll_interval_seconds)
        self._supervisor.start()

    def materialize_available_games(self) -> None:
        """Runs the dispatcher and every worker loop in this process until nothing more can be sealed."""
        while True:
            self.dispatch_once()
            progressed = False
            for worker_index in range(len(self.worker_paths)):
                worker = self._inline_worker(worker_index)
                while (report := worker.materialize_once()) is not None:
                    self.record_materialization_report(report)
                    progressed = True
            if not progressed:
                self.raise_if_materialization_failed()
                return

    def dispatch_once(self) -> int:
        return self._dispatcher.dispatch_once()

    def drain_worker_reports(self) -> None:
        if self._supervisor is not None:
            self._supervisor.drain_reports()

    def record_materialization_report(self, report: MaterializationReport) -> None:
        with self._lock:
            self.rejected_games += report.rejected_games
            self._rejection_alarm.observe(report.materialized_games, report.rejected_games)
            if self._rejection_alarm.breached():
                self.set_fatal_materialization_error(
                    RuntimeError(
                        f'Replay materialization rejected {self._rejection_alarm.rejection_rate:.1%} of the last '
                        f'{self._rejection_alarm.window_games} games, above the configured ceiling of '
                        f'{self._rejection_alarm.rate_ceiling:.1%}.'
                    )
                )

    def append_staged_games(self, model_generation: int) -> ReplayIngestion:
        self.raise_if_materialization_failed()
        self.drain_worker_reports()
        started_at = time.perf_counter()
        with self._lock:
            before = self.store.state
            self.store.set_logical_capacity(self.configuration.capacity_at(model_generation))
            manifests = self._staged_manifests()
            if not manifests:
                # Nothing was written since the previous append, and msync of the whole store
                # mapping costs ~0.2 s per gigabyte while holding this lock.
                after = self.store.state
                elapsed_seconds = time.perf_counter() - started_at
                return ReplayIngestion(
                    0, 0, after.size, after.evicted_rows - before.evicted_rows, 0, 0, 0, elapsed_seconds, ()
                )
            readers: list[ReplayShardReader] = []
            try:
                for manifest in manifests:
                    readers.append(self._open_staged_shard(manifest))
                metadata = tuple(game for reader in readers for game in reader.manifest.games)
                for reader in readers:
                    self.store.append_columns(reader.columns, reader.manifest.shard_identity)
                self.store.flush()
            finally:
                for reader in readers:
                    reader.close()
            self._observe_resignation_games(metadata)
            for manifest in manifests:
                self._delete_shard_artifacts(manifest.shard_identity)
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

    def set_fatal_materialization_error(self, error: RuntimeError) -> None:
        with self._lock:
            if self._fatal_materialization_error is None:
                self._fatal_materialization_error = error
                log(str(error))

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

    @contextmanager
    def training_snapshot(self) -> Iterator[ReplayDescription]:
        with self._lock:
            yield self.description()

    def close(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()
            self._supervisor = None
        self.store.close()

    def _inline_worker(self, worker_index: int) -> MaterializationWorker[PositionT]:
        worker = self._inline_workers.get(worker_index)
        if worker is None:
            worker = MaterializationWorker(
                worker_index,
                self.worker_paths[worker_index],
                self.staging_path,
                self.rejected_path,
                self.state,
                self.terminal_oracle,
                self.store.layout,
                self.value_discount_per_ply,
                self.censor_remaining_game_length_on_cut_games,
                self.materialization_settings,
            )
            self._inline_workers[worker_index] = worker
        return worker

    def _staged_manifests(self) -> tuple[SealedReplayShardManifest, ...]:
        manifests = []
        for manifest_path in sealed_replay_shard_manifest_paths(self.staging_path):
            identity = manifest_path.name.removesuffix(MANIFEST_SUFFIX)
            # A sealed manifest never changes, and re-parsing its embedded search observations on
            # every coordinator loop iteration costs more than the append it precedes.
            manifest = self._sealed_manifest_cache.get(identity)
            if manifest is None:
                try:
                    manifest = SealedReplayShardManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
                except (OSError, UnicodeError, ValueError) as error:
                    raise ValueError(f'Sealed replay shard manifest is invalid: {manifest_path}') from error
                if manifest.layout_digest != self.store.layout.digest:
                    raise ValueError('Sealed replay shard layout does not match the replay store.')
                self._sealed_manifest_cache[identity] = manifest
            manifests.append(manifest)
        return tuple(manifests)

    def _open_staged_shard(self, manifest: SealedReplayShardManifest) -> ReplayShardReader:
        # The manifest was parsed and layout-checked above; re-parsing its embedded search
        # observations here costs more than the row copy it would guard.
        return ReplayShardReader.open(
            replay_shard_manifest_path(self.staging_path, manifest.shard_identity),
            self.store.layout,
            verify_data_hash=False,
            manifest=manifest,
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

    def _delete_shard_artifacts(self, shard_identity: str) -> None:
        self._sealed_manifest_cache.pop(shard_identity, None)
        replay_shard_manifest_path(self.staging_path, shard_identity).unlink(missing_ok=True)
        replay_shard_data_path(self.staging_path, shard_identity).unlink(missing_ok=True)

    def _recover_directories(self) -> None:
        legacy = tuple(self.staging_path.glob(f'*{_LEGACY_STAGED_ROWS_SUFFIX}')) + tuple(
            self.staging_path.glob(f'*{_LEGACY_STAGED_METADATA_SUFFIX}')
        )
        if legacy:
            raise ValueError('Legacy per-game replay staging exists; an explicit replay migration is required.')
        if (self.completed_games_path / _LEGACY_QUEUE_FILE).exists():
            raise ValueError('A legacy replay shard queue exists; an explicit replay migration is required.')
        self._assert_one_filesystem()
        for directory in (self.inbox_path, self.staging_path, *self.worker_paths):
            for temporary in directory.glob('.*.tmp'):
                temporary.unlink(missing_ok=True)
        for data_path in self.staging_path.glob('*.replay-shard.bin'):
            identity = data_path.name.removesuffix('.replay-shard.bin')
            if not replay_shard_manifest_path(self.staging_path, identity).exists():
                data_path.unlink(missing_ok=True)
        self._return_orphaned_worker_directories()

    def _assert_one_filesystem(self) -> None:
        # A cross-device rename silently degrades to copy+unlink, which is the whole cost the
        # per-worker dispatch exists to avoid.
        devices = {path.stat().st_dev for path in (self.inbox_path, self.staging_path, *self.worker_paths)}
        if len(devices) != 1:
            raise ValueError('Replay inbox, worker and staging directories must share one filesystem.')

    def _return_orphaned_worker_directories(self) -> None:
        for entry in self.completed_games_path.glob('worker-*'):
            if not entry.is_dir() or entry in self.worker_paths:
                continue
            for file_name in worker_source_file_names(entry):
                _, completed_game_file_name = parse_worker_source_file_name(file_name)
                os.replace(entry / file_name, self.inbox_path / completed_game_file_name)
            try:
                entry.rmdir()
            except OSError:
                warn(f'Orphaned replay worker directory {entry} is not empty; leaving it in place.')


def _completed_game_names(directory: Path) -> tuple[str, ...]:
    try:
        with os.scandir(directory) as entries:
            return tuple(entry.name for entry in entries if entry.name.endswith(COMPLETED_GAME_SUFFIX))
    except FileNotFoundError:
        return ()
